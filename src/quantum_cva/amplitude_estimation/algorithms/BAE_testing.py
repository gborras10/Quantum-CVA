'''
Class for testing BAE.
'''
import numpy as np
import matplotlib.pyplot as plt
from quantum_cva.amplitude_estimation.algorithms.BAE import BAE
from quantum_cva.amplitude_estimation.algorithms.samplers import get_sampler
from quantum_cva.amplitude_estimation.utils.models import QAEmodel
from quantum_cva.amplitude_estimation.utils.plotting import process_and_plot, plot_single_run
from quantum_cva.amplitude_estimation.utils.mydataclasses import EstimationData, ExecutionData
from quantum_cva.amplitude_estimation.utils.misc import (print_centered, dict_str, sigdecstr, k_largest_tuples, 
                        k_smallest_tuples, b10str, dict_info, lprint)
from quantum_cva.amplitude_estimation.utils.running import Runner, BAERunsData

NDIGITS = 4

class TestBAE():
    def __init__(self, a, Tc_opts, strat, maxPT, sampler_str, sampler_kwargs,
                 silent = False, save = True, show = True):
        self.a = a
        # Keep full Tc dicts just for the prints; organize rest of info into
        # other attributes.
        self.Tc_opts = Tc_opts
        self.Tc = Tc_opts["Tc"]
        self.Tcrange = Tc_opts["range"]
        self.strat = strat
        self.maxPT = maxPT
        self.Tc_precalc = Tc_opts["Tc_precalc"]
        self.known_Tc = Tc_opts["known_Tc"]
        self.sampler_str = sampler_str
        self.sampler_kwargs = sampler_kwargs
        self.silent = silent
        self.save = save
        self.show = show

    def param_str(self):
        a_str = (self.rand_pstr(self.a) if isinstance(self.a,tuple)
                 else str(self.a))
        Tc_str = (self.rand_pstr(self.Tc) if isinstance(self.Tc,tuple)
                 else str(self.Tc))
        s = f"a={a_str};Tc={Tc_str}"
        return s

    @staticmethod
    def rand_pstr(param):
        return f"[{param[0]},{param[1]}]"

    @property
    def local_a(self):
        '''
        For each run, the real amplitude parameter will be 'local_a'.

        The 'a' attribute is always constant, and can hold:

        - A permanent value for 'a'. In that case, all runs will use it;
        'local_a' is equivalent to 'a'.

        - A tuple. In that case, each run will sample an amplitude at random
        in the interval given by the tuple.
        '''
        if isinstance(self.a, tuple):
            amin, amax = self.a
            a = np.random.uniform(amin,amax)
            print(f"> Sampled a = {a} (theta = {QAEmodel.theta_from_a(a)}).")
            return a
        else:
            return self.a

    @property
    def local_Tc(self):
        if isinstance(self.Tc, tuple):
            Tcmin, Tcmax = self.Tc
            Tc = np.random.uniform(Tcmin,Tcmax)
            print(f"> Sampled Tc = {Tc}.")
            return Tc
        else:
            return self.Tc

    def sqe_evolution_multiple(self, nruns, redirect = 0):
        '''
        Gets the evolution of the squared error with the iteration number, for
        multiple runs. All the data is joined together in a non-nested list,
        because what matters is the order. e.g.

        nqs_1 = [1, 3] | sqes_1 = [A1, B1]
        nqs_2 = [2, 5] | sqes_2 = [A2, B2]
        -> nqs_all = [1, 3, 2, 5] | sqes_all = [A1, B1, A2, B2]

        The results are then processed to get working averages (obtained by
        binning the nqs and averaging both them and the associated sqes within
        bins) and plotted. Straightforward averaging cannot be done because the
        nqs are not constant among runs, even for the same iteration.

        Also, the square root is taken when processing. The reason we don't
        use the RMSE from the outset is that we may want to do calculations
        with MSE, then take the root in the end (to mimic the usual "average
        the square -> take the root", but with curve fits/...).
        
        '''
        def print_info():
            info = ["Bayesian adaptive QAE"]
            info.append("- scaling of the estimation error with Nq")

            info.append("\n~ Exec details ~")
            info.append(f"{self.param_str()} | runs = {nruns} | "
                        f"nqs/PT = {b10str(self.maxPT)}")

            info.append("\n~ Strategy ~")
            info.append(dict_info(self.strat, sep = " |"))

            info.append("\n~ Sampler ~")
            info.append(dict_info(self.sampler_kwargs, sep = " |"))

            info.append("\n~ Tc info ~")
            info.append(dict_info(self.Tc_opts, sep = " |"))

            print_centered(info)

        print(f"> Will test {nruns} runs of 'Bayesian QAE'.")
        if not self.silent:
            print_info()
        rdata = BAERunsData()
        runner = Runner(f = self.sqe_evolution, nruns = nruns,
                        process_fun = rdata.add_run_data, redirect = redirect,
                        silent = self.silent, save = self.save)

        nruns = runner.run()

        full, final = rdata.get_lists()
        _, final_dscr = rdata.get_descriptors()

        if nruns == 0:
            return 
        
        if not self.silent:
            self.print_stats_several(final, final_dscr)
            
        raw_estdata = self.create_estdata(*full)
        process_and_plot(raw_estdata, save = self.save, show = self.show)

        if self.save:
            exdata = self.create_execdata(raw_estdata, nruns)
            exdata.save_to_file()

    def sqe_evolution(self, i, debug = False):
        '''
        Perform a single run of Bayesian adaptive QAE, and return lists of the
        numbers of queries, errors and standard deviations (ordered by step).
        '''
        a = self.local_a; Tc = self.local_Tc

        M =  QAEmodel(a, Tc = Tc, Tcrange = self.Tcrange)

        Tc_precalc = self.Tc_precalc
        # Tc_precalc is False if Tc to be ignored; True if to be considered.
        # known_Tc is whether to estimate it from scratch or get an input value.
        if self.Tc_precalc and self.known_Tc:
            Tc_precalc = Tc
        # At this point Tc_precalc is False if Tc to be ignored; True if to be
        # estimated; or a float, non-Boolean, giving an estimate for Tc to use
        # directly.

        Est = BAE(M, Tc_precalc, self.Tcrange)
        sampler = get_sampler(self.sampler_str, M, self.sampler_kwargs)
        means, stds, nqs = Est.adapt_inference(sampler, self.strat,
                                               maxPT = self.maxPT)
        nsqes = [(est/a - 1)**2 for est in means]
        nstds = [sqe/mean for sqe,mean in zip(stds, means)]

        ctrls = Est.ctrls_list
        devs = [np.abs(est - a) for est in means]
        # for i, (ctrl, dev) in enumerate(zip(ctrls,devs)):
        #    print(f"> Iteration {i}: ctrl = {ctrls[i]}, error = {devs[i]}.")

        print("> Final (normalized) RMSE: ", sigdecstr(nsqes[-1]**0.5, NDIGITS))
        print("> Final (normalized) std : ", sigdecstr(nstds[-1], NDIGITS))

        # For debugging.
        if debug and nsqes[-1]**0.5 > 1e-4:
            print("> Larger than 1e-4!")
            el, rl, accl = Est.exp_list, sampler.resampled_list, sampler.acc_rates
            essl = [ess/sampler.Npart for ess in sampler.ess_list]
            self.show_single_run(sampler, nqs, nstds, nsqes, el, rl, accl, essl,
                                 f"run {i} (bad)")
        return nqs, nsqes, nstds
    
    def show_single_run(self, sampler, nqs, nstds, nsqes, el, rl, accl, essl, 
                        title):
        nerrs = [nsqe**0.5 for nsqe in nsqes]
        plot_single_run(nqs, nstds, nerrs, rl, self.strat["wNs"], 
                        self.strat["Ns"], el,  accl, essl, title = title)
        plt.show()

        print("> Expanded at iterations: ", end = "")
        lprint(el)
        sampler.print_lists()
        sampler.plot_particles(ttl_xtra = f"- {title}")

    def print_stats_several(self, ls, dscrs):
        for l, dscr in zip(ls, dscrs):
            self.print_stats(l, dscr)

    @staticmethod
    def print_stats(l, dscr, print_all = False):
        '''
        l: list of quantities, e.g. errors, ordered by increasing iteration #.
        dscr: descriptor for the quantities.
        print_all: whether to print every item or just summary stats.
        '''
        lenum = list(enumerate(l))
        if print_all:
            print(f"> List of {dscr}s: ", lenum)
        print(f"> Mean {dscr}:   ", sigdecstr(np.mean(l), NDIGITS))
        print(f"> Median {dscr}: ", sigdecstr(np.median(l), NDIGITS))
        if len(l)>3:
            print(f"> 3 largest {dscr}s:  ", k_largest_tuples(lenum, 3,
                                                              sortby = 1))
            print(f"> 3 smallest {dscr}s: ", k_smallest_tuples(lenum, 3,
                                                               sortby = 1))

    def create_estdata(self, nqs, sqes, stds):
        '''
        Organize information into a EstimationData object.
        '''
        raw_estdata = EstimationData()
        raw_estdata.add_data("BAE", nqs = nqs, lbs = None, errs = sqes,
                             stds = stds)
        return raw_estdata

    def create_execdata(self, raw_estdata, nruns):
        '''
        Organize information into a ExecutionData object.
        '''
        sampler_params = dict_str(self.sampler_kwargs)
        sampler_info = f"{self.sampler_str}({sampler_params})"
        strat_info = f"STRAT=({dict_str(self.strat)})"
        extra = f"{strat_info},{sampler_info}"
        Ns = (self.strat["wNs"], self.strat["Ns"])
        exdata = ExecutionData(self.param_str(), raw_estdata, nruns,
                               Ns, label="BQAE",
                               extra_info = extra)
        return exdata

    def create_dummy_data(self):
        '''
        Generate quick dummy data for testing plots, save to files, etc.
        '''

        print("> USING DUMMY DATA.")
        Nq_warmup = self.strat["wNs"]
        nqs_all = np.linspace(Nq_warmup, self.maxPT, 100)
        sqes_all = 10/nqs_all

        # Add some noise to make the runs distinguishable.
        sqes_all = [np.random.uniform(0.5, 1.5)*sqe for sqe in sqes_all]

        sqes_warmup = [sqes_all[0]]
        return sqes_warmup, nqs_all, sqes_all

def test_evol(save, show):
    # If a is a tuple, it's picked at random from the interval for each run. 
    # If it is a number, it will be used direcly.
    # Tcrange could be different from Tc if we want to fix Tc but have a 
    # wider prior for the Tc estimation.
    Tcrange = None # (2000, 5000)
    a = 0.17 # (0,1)  
    Tc = Tcrange 
    maxPT = 10**8
    nruns = 1
    sampler_str = "RWM"

    Tc_opts = {"Tc": Tc,
                "Tc_precalc": True if Tc else False,
                "known_Tc": False,
                "range": Tcrange}

    # Strategy for the adaptive optimization.
    strat = {"wNs": 10,
                "Ns": 1,
                "TNs": 500,
                "k": 2,
                "Nevals": 50,
                "erefs": 3,
                "ethr": 3,
                "cap": False,
                "capk": 2}
    # Sampler arguments.
    sampler_kwargs = {"Npart": 2000,
                        "thr": 0.5,
                        "var": "theta",
                        "ut": "var",
                        "log": True,
                        "res_ut": False,
                        "plot": False}
    if sampler_str=="RWM":
        sampler_kwargs["c"] = 2.38
    if sampler_str=="LW":
        sampler_kwargs["a_LW"] = 0.98

    Test = TestBAE(a, Tc_opts, strat, maxPT, sampler_str,
                        sampler_kwargs, save = save, show = show)

    Test.sqe_evolution_multiple(nruns)

if __name__ == "__main__":
    save = False
    show = True
    test_evol(save, show)