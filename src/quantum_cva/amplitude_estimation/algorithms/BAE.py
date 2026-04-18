'''
Bayesian quantum amplitude estimation.
'''
import sys
import numpy as np
import scipy.optimize as opt
from copy import deepcopy
import re 

from quantum_cva.amplitude_estimation.algorithms.samplers import get_sampler
from quantum_cva.amplitude_estimation.utils.running import BAERunData
from quantum_cva.amplitude_estimation.utils.running import PrintManager
from quantum_cva.amplitude_estimation.utils.misc import  b10str, kwarg_str, round_if_float, dict_info, sigdecstr
from quantum_cva.amplitude_estimation.utils.mydataclasses import MeasurementData

NDIGITS = 4

class BAE():
    
    def __init__(self, model, Tc_precalc, Tcrange):
        '''
        Parameters
        ----------
        model : object
            Model for measurements and likelihood evaluations.
        Tc_precalc : bool or float
            Whether to estimate the coherence time; if it is a float, it 
            provides a value to be used directly.
        Tcrange : tuple
            Lower and upper bounds for the coherence time.

        Attributes
        ----------
        model : object
            Model for measurements and likelihood evaluations.
        Tc_precalc : float
            Whether to estimate the coherence time; if it is a float, it 
            provides a value to be used directly.
        Tcrange : tuple
            Lower and upper bounds for the coherence time.
        Tc_est : float
            Estimated value of the coherence time.
        data : MeasurementData
            Data from the measurements.
        cmin : float
            Lower bound for the control.
        cmax : float
            Upper bound for the control.
        double : bool
            Whether to double the control range in the next iteration.
        exp_list: [int]
            Iterations at which the search range was expanded.
        pman : PrintManager
            Print manager.
        '''
        global param
        self.model = model
        self.Tc_precalc = Tc_precalc
        self.Tcrange = Tcrange
        self.Tc_est = None
        self.data = None
        self.cmin = None
        self.cmax = None
        self.double = False
        self.ctrls_list = []
        self.exp_list = []
        self.pman = PrintManager()

    def adapt_inference(self, 
                        sampler,
                        strat: dict,
                        maxPT: int = None,
                        print_evol = False,
                        plot_all = False):

        '''
        Performs adaptive inference.

        Warms up to offline measurements with m = 0, then performs adaptive
        ones until a total of 'maxPT' queries has been reached.

        During the adaptive phase, we search in an expanding window that is
        doubled if chosen control has been within the 'erefs' highest
        for 'ethr' times. If a finite Tc is considered, expansion is
        capped at 'max_ctrl'.

        Parameters
        ----------
        sampler : object
            Sampler object.
        strat : dict
            Dictionary with strategy parameters. The keys are:
                - wNs: int, number of classical warm up measurements.
                - Ns: int, number of measurements per iteration in the adaptive 
                phase.
                - exp_refs: int, the number of highest values that count as hits 
                    to  trigger an expansion.
                - exp_thr: int, number of times a value must be within the top
                    'exp_refs' values for expansion to be capped.
        maxPT : int, optional
            Maximum number of queries. The default is None.
        print_evol : bool, optional
            Whether to print evolution data. The default is False.
        plot_all : bool, optional
            Whether to plot all data. The default is False.

        Returns
        -------
        lists
            Lists consecutive means, standard deviations, and cumulative probing 
            times.
        '''
        rd = BAERunData(sampler, self.probing_time)

        # Items will be removed from the dictionary during each run.
        astrat = deepcopy(strat)

        # Classical warm up, and Tc warm up if applicable.
        self.inference_warmup(astrat.pop("wNs"), astrat.pop("TNs"), sampler, rd)

        # Adaptive phase.
        self.adaptive_phase(sampler, astrat, maxPT, rd)

        self.results(sampler, "Online")
        return rd.get_lists()
    
    def adaptive_phase(self, sampler, strat, maxPT, rd):
        Ns = strat.pop("Ns")
        s = (f"> Adaptive phase. Will perform {Ns} shot measurements up to Nq ="
             f" {b10str(maxPT)}.")
        s += f"\n> Strat: {dict_info(strat)}"
        self.pman.print1st(s, "adaptive_phase")

        maxs = 0
        while rd.latest_CPT < maxPT:
            ctrl_opt, max_flag = self.choose_control(sampler, **strat)
            self.ctrls_list.append(ctrl_opt)
            if max_flag:
                maxs += 1
                if maxs > strat["ethr"] and not self.capped:
                    self.double = True

            outcome = self.model.measure(ctrl_opt, Ns)
            self.data.append_datum(ctrl_opt, outcome, Ns)
            sampler.update_latest(self.data)

            rd.add_iteration_data(ctrl_opt, Ns)

        print(f"\n> Estimation interrupted after {len(rd)} measurements"
              f" due to Nq = {b10str(rd.latest_CPT)} >= {b10str(maxPT)}.")
    
    def inference_warmup(self, wNs, TNs, sampler, rd):
        if wNs > 0:
            # Warm up and save data into BAERunData object. 
            ctrl, Ns = self.warmup(sampler, wNs, TNs)
            mean, std = rd.add_iteration_data(ctrl, Ns)
            print(f"> Post warm-up stats: {mean = :.4f}, {std = :.4f}.")
        else:
            # Create empty dataset to append to.
            self.data = MeasurementData([], [], [])

    def set_Tc_est(self, Tc_est):
        self.Tc_est = Tc_est
        self.model.set_Tc_est(Tc_est)

    def warmup(self, sampler, wNs, TNs):
        self.warmup_Tc(TNs)
        ctrl, Ns = self.classical_warmup(sampler, wNs)
        return ctrl, Ns
    
    def warmup_Tc(self, TNs):
        if not isinstance(self.Tc_precalc, bool):
            # It's a fixed value known_Tc; use it.
            Tc_est = self.Tc_precalc
            self.set_Tc_est(Tc_est)
            print(f"> Assuming given Tc = {Tc_est}. [warm_up]")
        elif self.Tc_precalc:
            Tc_est = self.learn_Tc(TNs)
            self.set_Tc_est(Tc_est)

    def classical_warmup(self, sampler, wNs):
        if wNs == 0:
            return

        s = f"> Warming up with {wNs} classical shots."
        self.pman.print1st(s, "warm_up")

        ctrls = self.gather_data(wNs, how = "classical")
        assert len(ctrls) == 1; ctrl = ctrls[0]

        data = self.data
        sampler.update_latest(data)
        return ctrl, wNs

    def learn_Tc(self, TNs, rng = (0, 1)):
        '''
        off_shots: what fraction of Tc_Ns to be used in non adaptive
        measurements (initial learning phase).
        off_rng: control range for the offline measurements, as a percentage
        of Tcmax.
        '''
        def get_T_ctrls(revert = True):
            '''
            Parameters for the linspace function. Not necessarily integers.
            '''
            Tcmax = self.Tcrange[1]
            start = int(rng[0]*Tcmax)
            stop = int(rng[1]*Tcmax)
            ctrls = np.linspace(start, stop, TNs)
            if revert:
                ctrls = ctrls[::-1]
            return ctrls

        print("> Learning Tc.")
        sampler_kwargs = {"Npart": 2000, "thr": 1, "c": 1, "var": "Tc", "ut": "var"}
        Tcsampler = get_sampler("RWM", self.model, sampler_kwargs)

        ctrls = get_T_ctrls()
        s = ("> Tc sampler: " + dict_info(sampler_kwargs) + "\n")
        s += (f"> {TNs} controls for Tc in: {ctrls[0]}-{ctrls[-1]}.")
        self.pman.print1st(s, "learn_Tc")

        Tcdata = MeasurementData([], [], [])
        for ctrl in ctrls:
            outcome = self.model.measure(ctrl, 1, var="Tc")
            Tcdata.append_datum(ctrl, outcome, 1)
            Tcsampler.update_seq(Tcdata)

        Tmean, Tstd = Tcsampler.mean_and_std()

        print(f"> Estimated Tc based on {TNs} shots: "
          f"{Tmean:.0f} ± {Tstd:.0f}.")
        self.Tc_est = Tmean

        return Tmean

    def choose_control(self, *args, **kwargs):
        grid, ctrl_opt, max_flag = self.optimize_control(*args, **kwargs)
        self.print_grid_info(grid, **kwargs)
        return ctrl_opt, max_flag

    def optimize_control(self, sampler, k = 1, Nevals = 100,
                         ethr = 3, erefs = 3, cap = True,
                         capk = 1, stoch = True):
        '''
        The search range is a set of integers initially spaced by 'k' and
        going from 1 to k*Nevals.
        '''
        if self.cmin is None:
            # First time optimizing.
            self.init_opt(k, Nevals, cap, capk)

        if self.double:
            self.double_grid()

        if stoch:
            grid = np.round(np.random.uniform(self.cmin, self.cmax, int(Nevals)))
            ctrl_opt = self.discrete_optimization(grid, sampler)
            # Find the 'erefs' largest grid point.
            thr = np.partition(grid, -erefs)[-erefs]
            max_flag = True if ctrl_opt >= thr else False
        else:
            grid = np.round(np.linspace(self.cmin, self.cmax, num = int(Nevals)))
            ctrl_opt = self.discrete_optimization(grid, sampler)
            max_flag = True if ctrl_opt in grid[-erefs:] else False

        
        return grid, ctrl_opt, max_flag

    def init_opt(self, k, Nevals, cap, capk):
        self.cmin, self.cmax = 1, int(k*Nevals)
        self.k = k
        self.cap = cap
        self.capk = capk

    def double_grid(self):
        self.exp_list.append(self.data.non_classical_len())
        print(f"> Upping search range from {self.cmin}-{self.cmax}. ",
              end = "")
        if self.capped:
            # Previously capped.
            return
        # Uncapped vs. capped limits.
        self.cmin, self.cmax = self.cmax, self.cmax*2
        if self.capped:
            # Cap now.
            self.set_capped_lims()
        self.double = False

    def discrete_optimization(self, grid, sampler):
        ims = self.objective_function(grid, sampler)
        iopt = np.argmin(ims)
        ctrl_opt = grid[iopt]
        # print(f"> Optimal control {ctrl_opt} with utility {ims[iopt]}.")
        return ctrl_opt

    def objective_function(self, x, sampler):
        '''
        Function to be maximized. Considers all data gathered so far, which
        is usually intended in online estimation.
        '''
        return -sampler.expected_utilities(x, self.data)

    @property
    def capped(self):
        '''
        Window is capped if a finite Tc is considered, 'cap' is true, and
        maxctrl has been exceeded.
        '''
        is_capped = self.Tc_precalc and self.cap and self.cmax >= self.maxctrl
        return is_capped

    def set_capped_lims(self):
        print(f"Would be {self.cmin}-{self.cmax}; permanently capped"
        f" due to Tc={self.Tc_est:.0f}. ")

        self.cmin, self.cmax = int(self.maxctrl/2), self.maxctrl

        print(f"New range: {self.cmin}-{self.cmax}.")

    @property
    def maxctrl(self):
        '''
        Maximum control considered when searching for the optimal choice.
        '''
        return int(self.capk*self.Tc_est)

    def cumul_probing_times(self, ctrls, Ns_list):
        if isinstance(Ns_list, int):
            Ns_list = [Ns_list for i in range(len(ctrls))]

        PT_per_meas = [self.probing_time(ctrl)*Ns for ctrl, Ns
                       in zip(ctrls, Ns_list)]
        cumul_PTs = np.cumsum(PT_per_meas)
        return cumul_PTs

    def probing_time(self, ctrl):
        PT = 2*ctrl+1
        return PT

    def gather_data(self, Ns, how = "classical"):
        if how == "classical":
            ctrls, Nsshots = [0], [Ns]
        if how == "exp":
            ctrls = self.exp_controls(Ns)
            Nsshots = [1 for i in range(Ns)]

        data = self.model.create_data(deepcopy(ctrls), Nsshots)
        self.data = data

        print(f"> Measured {Ns} shots data for {how} controls.")
        return ctrls

    def print_grid_info(self, grid, **kwargs):
        if self.pman.is_first("optimize_control"):
            s = ("> Optimized experimental controls on a grid over range "
                 f"[{self.cmin}, {self.cmax}].\n"
                 f"> Working with {kwarg_str(self.optimize_control)}.")
            # gridstart = ", ".join([str(x) for x in grid[:3]])
            # gridend = ", ".join([str(x) for x in grid[-3:]])
            # s += ("\n> Grid: " + gridstart + ",..., " + gridend +  ".")
            self.pman.print1st(s, "optimize_control")

    def data_warn(f):
        def wrapper(self, *args, **kwargs):
            if self.data is None:
                print("> To perform estimation, I need data. Gather it using"
                      "the 'gather_data' method and get back to me."
                      f"[BAE.{f.__name__}]")
                return
            return f(self, *args, **kwargs)
        return wrapper

    @data_warn
    def mle(self, Nevals = 10e3, finish = None):
        def objective_function(param):
            return -self.model.batch_likelihood(param, self.data)

        info = "without" if finish is None else "with"
        print(f"> Testing brute force MLE on the data ({info} Nelder Mead; "
              f"Nevals = {Nevals})... [test_mle]")

        if finish is None:
            param_opt = opt.brute(objective_function, [(0,1)], Ns=Nevals,
                                  finish = finish)
        else:
            param_opt = opt.brute(objective_function, [(0,1)], Ns=Nevals,
                                  finish = finish)[0]
        print("> MLE: ", param_opt)
        return param_opt

    @data_warn
    def off_inference(self, sampler, batch = False):
        '''
        Offline inference (pre-determined controls).
        '''
        data = self.data
        method = ("Batch-updating a grid " if batch else
                  f"Running SMC-{sampler.str}")
        s = f"> {method} on the data... [BAE.off_inference]"
        self.pman.print1st(s, "off_inference")

        if batch:
            sampler.batch_update(data)
        else:
            sampler.update_seq(data)

        return self.results(sampler, "Offline")

    def exp_controls(self, Nmeas):
        ctrls = [k for k in range(Nmeas)]

        toprint = ", ".join(str(round_if_float(ctrl)) for ctrl in ctrls[:5])
        if len(ctrls)>1:
          toprint += ",..., " + str(round_if_float(ctrls[-1]))

        print(f"> Determined experimental controls: {toprint}."
              " [BAE.exp_controls]")
        return ctrls

    def results(self, sampler, strat):
        self.pman.print1st("=================================", "results")
        sampler.print_stats()
        mean, std = sampler.mean_and_std()
        print(f"> {strat} SMC-{sampler.str} estimate: a = "
              f"{sigdecstr(mean, NDIGITS)} ± {sigdecstr(std, NDIGITS)}")
        return mean, std

def fix_aBAE_label(execdata):
    s = execdata.extra_info
    match = re.search(r",ut=(\w{3})", s)
    if match:
        result = match.group(1)
        if result=="ESS":
            old = "BAE"
            new = "aBAE"
            execdata.label = new
            for d in execdata.estdata.unpack_data():
                if d:
                    d[new] = d.pop(old)