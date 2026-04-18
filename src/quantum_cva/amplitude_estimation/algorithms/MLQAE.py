'''
Maximum likelihood amplitude estimation.
'''

import numpy as np
import sys
import importlib
# from qiskit import Aer, execute, QuantumCircuit

try:
    from google.colab import files
    sys.path.append('/content/drive/Othercomputers/Laptop/Phd/3.Scripts') 
    using_colab = True
except ModuleNotFoundError:
     using_colab = False

from quantum_cva.amplitude_estimation.algorithms.QAA import CheckingFunction
from quantum_cva.amplitude_estimation.algorithms.QAE import BayesianQAE, TesterQAE
from quantum_cva.amplitude_estimation.utils.mydataclasses import EstimationData, ExecutionData
from quantum_cva.amplitude_estimation.utils.plotting import process_and_plot
from quantum_cva.amplitude_estimation.utils.misc import single_warning, print_centered, logspace
from quantum_cva.amplitude_estimation.utils.running import ProgressBar
from quantum_cva.amplitude_estimation.utils.models import QAEmodel

reload = False
if reload:
    importlib.reload(sys.modules["algorithms.QAE"])

# Number of decimal cases to be used when printing estimates.
ndigits = 5

class MLQAE(BayesianQAE):
    '''
   ============================================================================
    Class for quantum amplitude estimation without phase estimation [2].
    It performs maximum likelihood estimation on 'a', based on measurement 
    data extracted from an ensemble of Grover circuits. 

    Attributes
    ----------
    encoding_qs: int
        The number of qubits required to represent prospective solutions.
    function: CheckingFunction
        The target function, for which we mean to find inputs that
        evaluate to 1.
    circuit: QuantumCircuit
        The quantum circuit on which to act (to apply or create operators,
        including measurement operators). 
    n_qubits: int
        the number of qubits for the circuit. This is the number of qubits
        required for prospective solutions + at least 1 aux qubit for the
        phase oracle (2 if derandomizing; the same effect could be achieved
        without the extra qubit by generalizing Q as per [1], but that would
        require solving a trigonometric equation).
    Agate: Gate
        The superposition-creating operator, to be used for initialization 
        as well as when constructing the Grover operator. It represents a 
        "best guess" probability distribution.
    Ainvgate: Gate
        The inver of Agate.
    Ncircs: Gate
        The number of different circuits to measure.
    nshots: int
        How many times to prepare and measure each circuit.
    seq: str
        How to choose the number of Grover iterations m_k for each circuit k: 
        "LIS" for "linearly increasing sequence (m_k = k) or "EIS" for 
        "exponentially increasing sequence" (m_k = 2**(k-1) if k!=0 else 0).
    ms: [int]
        The list of numbers of Grover iterations, one for each circuit; i.e,. 
        for the kth circuit Q^ms[k] is applied. These are determined by 
        "seq".
    Nq_calc: str, optional
        How to compute the number of queries. "cumul" to consider all queries
        across all (parallel) circuits; "last" to consider only the 
        largest sequential number of queries (corresponding to the last 
        circuit).
   ============================================================================
   '''
    def __init__(self, encoding_qs, function, Ncircs, nshots, seq="EIS",
                 Nq_calc = "cumul", silent = False):
        # super().__init__(encoding_qs, function)
        # By construction, a circuit attribute is still necessary to build and 
        # export gates. An auxiliary qubit is needed for the standard oracle
        # if the validity of the solution is to be evaluated in the circuit.
        # From the parent class, we already have n_qubits = encoding_qs + 1;
        # this aux qubit can be reused so 'n_qubits' is the same.
        # self.circuit = QuantumCircuit(self.n_qubits) 
        self.Ncircs = Ncircs
        self.nshots = nshots
        self.seq = seq
        self.evals = MLQAE.def_evals(self.seq, self.Ncircs)
        self.silent = silent

        self.ms = self.coefs_sequence(seq, Ncircs, self.evals)
        self.Nq_calc = Nq_calc
        if not silent:
            nqstr = "cumulative" if Nq_calc=="cumul" else "sequential max"
            print(f"> Instanced MLE-QAE with {Ncircs} circuits for {nshots} "
                  f"shots each, following a {self.seq} strategy. Number of "
                  f"queries: {nqstr}. [MLQAE.__init__]")
        
    @staticmethod
    def def_evals(seq, Ncircs, LIS_points = 15):
        # Ncircs = maximum number of circuits used (last iteration).
        # LIS has smaller query different between circuits, no need to run all.
        # Space to get ~ LIS_points points evenly spaced on a logscale.
        # Actually will be less than LIS_points because repeats removed.
        if seq == "LIS":
            evals = logspace(0, Ncircs, LIS_points).astype(int)
            evals = np.sort(list(set(evals)))
            # evals = np.array(range(Ncircs))
        else:
            evals = np.array(range(Ncircs))
        return evals
        
    @staticmethod
    def coefs_sequence(seq, Ncircs, evals = None):
        if evals is None:
            evals = [k for k in range(Ncircs)]
        if seq=="LIS":
            # Need to evaluate all integer ms up to maximum.
            ms = [k for k in range(evals[-1]+1)]
        if seq=="EIS":
            ms = [0] + [2**k for k in evals[:-1]]
        return ms
    
    def maximize_likelihood(self, hs, finish = True, evals = 1e3, 
                            excfrom = None, silent = True):
        return super().maximize_likelihood(self.ms[:excfrom], hs[:excfrom], 
                                           finish = True, evals = evals,
                                           silent = silent)
        
    def estimate_amplitude(self, model = None, allsteps = False, 
                           show = False, silent = False, nevals = 1e3):
        if not silent:
            info = "Qiskit's simulator" if amp is None else "numpy.binomial"
            print(f"> Using {info} for sampling. [MLQAE.estimate_amplitude]")
        if model is None:
            qc_list = self.create_circuits()
            hs = self.run_circuits(qc_list)
        else:
            # Not the most efficient but MLAE is the only algorithm evaluating
            # for a fixed list of controls so not sure if worth changing.
            hs = [model.measure(m, self.nshots) for m in self.ms]
            
        if show:
            self.plot_likelihood(hs)
            self.plot_likelihood(hs, log=False)
            print("> I would like to let you know I am plotting 2 figures."
                  " [MLQAE.estimate_amplitude]")
          
        if not allsteps:
            theta_est = self.maximize_likelihood(hs, evals = nevals)
            a_est = np.sin(theta_est**2)
        else:
            # List of intermediate estimates based on the cumulative data.
            if show:
                [self.plot_likelihood(hs[:k+1], ttl_extra=f" - step {k}") 
                 for k in range(self.Ncircs)]
                
            theta_ests = [self.maximize_likelihood(hs, excfrom = k+1, 
                                                   evals = nevals)
                          for k in self.evals] 
            
            a_ests = [np.sin(theta_est)**2 for theta_est in theta_ests]
            theta_est, a_est = theta_ests[-1], a_ests[-1]
        
        if not silent:
            t_est = a_est*2**self.encoding_qs
            theta_round = round(theta_est,ndigits)
            
            print(f"> The MLE of theta is θ={theta_round}, meaning "
                  f"a≈{round(a_est,ndigits)} and t={round(t_est)}. "
                  "[MLQAE.estimate_amplitude]")
        
        return a_ests if allsteps else a_est
    
    def numpy_sampling(self, amp):
        return super().numpy_sampling(amp, self.ms)
    
    def create_circuits(self, check = "classical", show = False):
        return super().create_circuits(self.ms, check = check, show = show)
         
    def run_circuits(self, qc_list):
        sim = Aer.get_backend('aer_simulator')
        results = execute(qc_list, sim, shots = self.nshots).result()
        # Get a list of outcome dictionaries, one for each circuit.
        counts = results.get_counts() 
        
        # Distinguish between list of outcome dictionaries (multiple circuits)
        # and single outcome dictionary (single circuit).
        sample_dict = counts if isinstance(counts, dict) else counts[0]
        # Take one example key to find the number of measured qubits. 
        sample_key = list(sample_dict.keys())[0]

        if len(sample_key) == 1:
            # Single measured qubit; hit if outcome==1. 
            hs = self.count_hits(counts)
        else:
            # Measured full register; hit if f(outcome)==1.
            hs = self.evaluate_counts(counts)
        return hs
        
    def error_bound(self, a):
        nqs = self.Nqueries_evol()
        CR = self.Cramer_Rao_evol(a)
        return(nqs, CR)
    
    def Cramer_Rao_evol(self, a):
        CRevol = [self.Cramer_Rao_lower_bound(a, upto=k) 
               for k in range(1,self.Ncircs+1)]
        return CRevol
        
    def Cramer_Rao_lower_bound(self, a, upto = None):
        lb = np.sqrt(1 / self.Fisher_info(a, upto))
        return lb
    
    def Fisher_info_check_LIS(self, a):
        for k in range(self.Ncircs):
            print(self.Fisher_info(a,upto=k+1),
                  self.Fisher_info_LIS(a,upto=k+1))
    
    def Fisher_info(self, a, upto = None):
        # Calculates Fisher information at the k='upto' iteration.
        upto = self.Ncircs if upto is None else upto
        Fi = np.sum([self.nshots*(2*m+1)**2
                     for m in self.ms[:upto]])/(a*(1-a))
        return Fi
    
    def Fisher_info_LIS(self, a, upto = None):
        # Analytical calculation. Note that M + 1 is the number of circuits, 
        # as per [2]. The same can be done for EIS. 
        M = self.Ncircs-1 if upto is None else upto-1
        Fi = self.nshots/(a*(1-a)) * (2*M+3)*(M+1)*(2*M+1)/3
        return Fi
    
    def Nqueries_evol(self):
        nqs = [self.Nqueries_from_ms(self.ms[:k+1], self.nshots, 
                                     self.Nq_calc) for k in self.evals]
        return nqs
    
    @staticmethod
    def Ncircs_from_Nqueries(Nq_target, nshots, seq, Nq_calc):
        # Calculate the number of circuits that will get the number of queries
        # closest to Nq_target.
        Nq, Ncircs = 0, 0
        Nq = 0
        while Nq <= Nq_target:
            prev_Nq, prev_Ncircs = Nq, Ncircs
            Ncircs += 1
            ms = MLQAE.coefs_sequence(seq, Ncircs)
            Nq = MLQAE.Nqueries_from_ms(ms, nshots, Nq_calc)
        if abs(Nq-Nq_target) <= abs(prev_Nq-Nq_target):
            return Ncircs, Nq
        else:
            return prev_Ncircs, prev_Nq 
    
    @staticmethod
    def Nqueries_from_ms(ms, nshots, Nq_calc, upto = None):
        # Static to serve 'Ncircs_from_Nqueries'.
        if upto is not None:
            ms = ms[:upto] 
            
        if Nq_calc=="last":
            # Use information pertaining to last circuit alone
            Nq = nshots*(2*ms[-1]+1)
        elif Nq_calc=="cumul":
            Nq = np.sum([nshots*(2*m+1) for m in ms])
        else:
            raise ValueError("Nq_calc must be either 'last' or 'cumul'.")
        return Nq
    
    def plot_likelihood(self, hs, xcoord = "a", log = True, atol = 1e-6,
                        ttl_extra = ""):
        title = (f"Maximum likelihood amplitude estimation ({self.seq})" 
                 + ttl_extra)
        xlabel = "Amplitude" if xcoord == "a" else "Angle (radians)"
        ylabel = ("Logl" if log else "L") + "ikelihood"

        super().plot_likelihood(self.ms, hs, xcoord = xcoord, log = log, 
                              atol = atol, title = title, xlabel=xlabel, 
                              ylabel=ylabel)

class TestMLQAE(TesterQAE):
    '''
    For 'final_result()', a must be a float. For other functions, it can be 
    "rand"; in that case, it will be picked at random. 
    
    The two stand-alone (non auxiliary) methods are:
    - final_result: runs QAE once for a given 'a' and shows the results: 
    likelihoods plots and numerical results.
    - sqe_evolution_multiple: gets estimation errors by averaging over as many 
    runs as wished, with fixed or random 'a'. Can do "LIS", "EIS", or both. 
    '''
    def __init__(self, a, Tc, nshots,  Ncircs = None, Nq_calc = "cumul",
                 silent = False):
        self.a = a
        self.Tc = Tc
        self.nshots = nshots
        self.Ncircs = Ncircs
        self.Nq_calc = Nq_calc
        self.silent = silent
        if silent:
            # log(0) may occur, but it's fine to ignore.
            np.seterr(divide='ignore')
        else:
            np.seterrcall(single_warning)
            np.seterr(divide='call')
       
    def final_result(self, Ncircs, seq, n = None, t = None, simulator = True):
        '''
        Perform maximum likelihood QAE, plot the likelihood and loglikelihood,
        and print the estimation results.
        If n and t are given, a is calculated from them. If not, self.a is used.
        '''
        assert (n is not None and t is not None) or simulator is True, \
            "using Qiskit's simulator requires defining 'n' and 't'."
        if n is not None and t is not None:
            assert self.a == t/2**n, "'a' must agree with the inputs for n, t."
        self.Ncircs = Ncircs
        self.print_final_result_info(seq, n, t)
            
        # Checking function only needed if actually using the Qiskit simulator.
        f = CheckingFunction(n,t,silent=True) if simulator else None
        MLQAE_instance = MLQAE(n, f, self.Ncircs, self.nshots, seq = seq,
                               silent = self.silent)
        
        if simulator:
            MLQAE_instance.estimate_amplitude(show = True)
        else:
            MLQAE_instance.estimate_amplitude(amp = self.a, show = True)
            
    def print_final_result_info(self, seq, n, t):
        print("> Will test maximum likelihood quantum amplitude estimation "
              "(without QPE), and present the estimation results and likelihood "
              f"plots. Strategy for the Grover applications: {seq}. [TestMLQAE]")
        
        nstr = "N/A" if n is None else n
        tstr = "N/A" if t is None else t
        theta = np.arcsin(np.sqrt(self.a))
        
        info = [f"MLQAE estimation single ({seq})"]
        info.append((f"  n={nstr} | t={tstr} | a≈{round(self.a,ndigits)} | "
                     f"θ≈{round(theta,ndigits)}"))
        info.append(f"Ncircs = {self.Ncircs} | nshots = {self.nshots}")
        print_centered(info)
        
    def sqe_evolution_multiple(self, nruns, Nq_target, seqs, nevals, save = True):
        '''
        Perform maximum likelihood QAE for LIS and/or EIS (whichever string(s)
        are in 'seqs') and plot the evolution of the MSE with the number of 
        queries to A.
        
        If self.a is 'rand', the amplitude is picked at random for each run. If 
        not, it's kept constant and equal to self.a.
        
        If seqs includes both "EIS" and "LIS", it is advised that it does so by 
        this order. Otherwise the maximum numbers of queries for EIS vs LIS may 
        be quite unmatched: we want to use EIS' effective Nqueries as LIS' 
        target to avoid wide gaps between their x span in the plots. EIS has  
        coarse granularity, whereas LIS is pretty thin grained and can get 
        close to any target.
        '''
        self.print_sqe_evolution_info(seqs, nruns, Nq_target, nevals)
        # Effective nruns for each seq (could be < runs if KeyboardInterrupt).
        enruns = []
        
        # Create common EstimationData instance to store both results.
        estdata = EstimationData()
        for i,seq in enumerate(seqs):
            estdata, r = self.rmse_evolution(seq, Nq_target, nruns, estdata, 
                                             nevals = nevals[i])
            enruns.append(r)
            
            if r > 0:
                # Update Nq_target to previous sequence's so they're closer.
                Nq_target = estdata.Nq_dict[seq][-1]
        
        # Create a string specifying the runs used for each strategy in seq. 
        runstr = ";".join([f"{runs}({seq})" for runs,seq in zip(enruns, seqs)])
        
        ed = ExecutionData(self.param_str, estdata, runstr, self.nshots, 
                           label = "MLQAE", extra_info = f"nevals_{nevals}" +
                           f"Nq_{self.Nq_calc}≈10^{round(np.log10(Nq_target),1)}")
        if save:
            ed.save_to_file()
            
        process_and_plot(estdata, processing = "averaging", save = save)
    
    def rmse_evolution(self, seq, Nq_target, nruns, estdata, nevals = 1e3, 
                       bounds = False):
        ''' 
        Auxiliary function for 'sqe_evolution_multiple'.
        
        Performs several runs of estimation, and averages the squared error
        for each iteration (which given 'seq' determines the number of queries) 
        to get the MSE.
        
        Returns an EstimationData instance containing the evolution of the root
        mean squared error, sqrt(MSE), with the  iteration number, among other 
        things (number of queries per step, and Cramér Rao bound values if 
        applicable). The average is taken over multiple runs of MLQAE. 
        '''
        self.Ncircs, actual_Nq = MLQAE.Ncircs_from_Nqueries(Nq_target, 
                                                            self.nshots, seq, 
                                                            self.Nq_calc)
        if not self.silent:
            print(f"> Will test a {seq} strategy using {self.Ncircs} circuits "
                f"(~10^{round(np.log10(actual_Nq),1)} queries). "
                "[rmse_evolution]")
            
        sqe_by_run = []
        pb = ProgressBar(nruns)
        for i in range(nruns): 
            pb.update()
            try:
                MLQAE_instance, sqe_evol = self.sqe_evolution_single(seq, 
                                                                     nevals = nevals)
                sqe_by_run.append(sqe_evol)
            except KeyboardInterrupt:
                print(f"\n> Keyboard interrupt at run {i}. "
                      "Will present partial results if possible.")
                print("> Breaking from cycle... [rmse_evolution]")
                nruns = i
                break
        
        if nruns > 0:
            # err_per_step = estimation_errors(sqe_by_run, stat = "mean")
            
            # The lower bounds can only be calculated for fixed a. 
            if bounds and not isinstance(self.a, tuple):
                nqs, lbs = MLQAE_instance.error_bound(self.a) 
            else:
                nqs, lbs = MLQAE_instance.Nqueries_evol(), None
                
            estdata.add_data(seq, nqs = nqs, lbs = lbs, errs = sqe_by_run)
        
        return estdata, nruns
    
    def sqe_evolution_single(self, seq, nevals = 1e3):
        ''' 
        Auxiliary function for 'MLQAE_evol_avg'.
        
        Returns the evolution of the squared error with the iteration number,
        for a single run of MLQAE. Also returns the created MLQAE instance.
        '''
        # n and CheckingFunction not needed, use analytical calculations and
        # multinomial sampling since it's simpler. Pick n = 1 because the
        # circuits are created by default... Maybe consider fixing?
        n = 1
        f = None
        MLQAE_instance = MLQAE(n, f, self.Ncircs, self.nshots, seq = seq, 
                               Nq_calc = self.Nq_calc, silent = True)
        
        a = self.local_a
        Tc = self.local_Tc
        M =  QAEmodel(a, Tc = Tc)
        a_ests = MLQAE_instance.estimate_amplitude(model = M, 
                                                   allsteps = True, 
                                                   silent = True, 
                                                   nevals = nevals)
        sqe_evol = [(est/a-1)**2 for est in a_ests]
        return MLQAE_instance, sqe_evol
    
    def print_sqe_evolution_info(self, seqs, nruns, Nq_target, nevals):
        '''
        Auxiliary function for 'sqe_evolution_multiple'.
        '''
        print(f"> Will test {nruns} runs of 'Maximum Likelihood QAE' ",
              f"({", ".join(seqs)}).")
        
        # theta = "rand" if self.a=="rand" else np.arcsin(np.sqrt(self.a))
        
        info = ["MLQAE error plot averaged (" +  "/".join(seqs) + ")"]
        info.append(f"a={self.a} | Tc={self.Tc} | nevals = {nevals}")
        info.append(f"runs = {nruns} | nshots = {self.nshots} | " 
                    f"Nq_target = 10^{round(np.log10(Nq_target),1)}")
        if not self.silent:
            print_centered(info)
    
def test(which):
    if which == 0:
        '''
        Test a single run of MLQAE for fixed parameters, and plot and print the
        final results.
        '''
        nshots = 100
        Ncircs = 8
        seq = "LIS"
        sim = True
        if sim:
            # If using Qiskit's simulator, 'a' must be defined by 'n' and 't' 
            # due to the circuit implementation.
            n = 4
            t = 3
            a = t/2**n
        else:
            # If using analytical calculations + noise, define 'a' directly
            # while leaving 'n' and 't' blank.
            a = 0.3
        test = TestMLQAE(a, nshots)
        test.final_result(Ncircs, seq, n, t, simulator = True)
    elif which==1:
        # a can be a float, or "rand" for picking 'a' at random for each run.
        a = (0,1)
        Tc = None # (2000, 5000)
        nshots = 100
        nruns = 100
        Nq_target = 5*10**6
        # Use less evaluations for LIS because the landscape is more regular
        # (and the cost higher, because there are more data for a given Nq).
        nevals = [5e4, 5e3]
        seqs = ["EIS", "LIS"]
        Nq_calc = "cumul" # 'cumul' or 'last'.
        test = TestMLQAE(a, Tc, nshots, Nq_calc = Nq_calc, silent = False)
        test.sqe_evolution_multiple(nruns, Nq_target, seqs, nevals, 
                                    save = True)
    
if __name__ == "__main__":
    test(1)