"""
Simpler quantum amplitude estimation.
"""
import numpy as np
import importlib
import sys

from quantum_cva.amplitude_estimation.algorithms.QAE import TesterQAE
from quantum_cva.amplitude_estimation.utils.plotting import process_and_plot
from quantum_cva.amplitude_estimation.utils.misc import print_centered, expb10
from quantum_cva.amplitude_estimation.utils.mydataclasses import EstimationData, ExecutionData
from quantum_cva.amplitude_estimation.utils.running import ProgressBar
from quantum_cva.amplitude_estimation.utils.models import QAEmodel

try:
    from google.colab import files
    using_colab = True
except:
     using_colab = False
     
reload = True
if reload and using_colab:
    importlib.reload(sys.modules["utils.plotting"])
    importlib.reload(sys.modules["utils.quantum_ops"])
    importlib.reload(sys.modules["utils.misc"])
    importlib.reload(sys.modules["utils.binning"])
    importlib.reload(sys.modules["src.utils.mydataclasses"])
    
Ndigits = 5

first_SQAE = True
class SQAE:
    def __init__(self, model, nshots, formula, threshold = 0.5,
                 silent = False):
        self.model = model
        self.nshots = nshots
        self.formula = formula
        self.threshold = threshold
        self.silent = silent
        
    def estimate(self):
        p1, kend = self.Hadamard_tests()
        if self.formula == 0:
            a = self.calculate_formula0(p1, kend)
        if self.formula == 1:
            a = self.calculate_formula1(p1, kend)
        if self.formula == 2:
            a = self.calculate_formula2(p1, kend)
            
        Nq = SQAE.Nqueries(kend, self.nshots)
        if not self.silent:
            print("> SQAE: estimation completed, results below.")
            print(f"> a     =  {round(a, Ndigits)}")
            print(f"> Nq = {Nq}")
        return Nq, a
        
    @staticmethod
    def calculate_formula0(p1, k):
        '''
        Invert the probability of 1 directly using the definition of amplitude.
        '''
        theta = np.arcsin(p1**0.5)*2**-k
        a = np.sin(theta)**2
        return a
    
    @staticmethod
    def calculate_formula1(p1, k):
        '''
        Inversion formula of [Wie19].
        '''
        # 'p_k' is defined as p_k := p0 - p1.
        p_k = 1 - 2*p1
        theta = 2**-(k + 1)*np.arccos(p_k)
        a = np.sin(theta)**2
        return a
        
    @staticmethod
    def calculate_formula2(p1, k):
        '''
        Recursive formula of [Wie19].
        '''
        # 'p_k' is defined as p_k := p0 - p1.
        p_k = 1 - 2*p1
        # The last iteration of the cycle calculates p_k for k==0.
        while k >= 1:
            k -= 1
            p_k = np.sqrt((1+p_k)/2)
        a = (1 - p_k)/2
        return a
        
    def Hadamard_tests(self):
        '''
        Perform Hadamard tests on the Grover operator with an exponent  
        increasing exponentially as 2**k, until the percentage of measured
        1 outcomes crosses 'threshold'. Return this percentage (>=0.5) and the 
        number of iterations it took to reach it.
        '''
        p1 = -1
        k = -1
        while p1 < self.threshold:
            k += 1
            p1 = self.Hadamard_test(k)
        return p1, k
        
    def Hadamard_test(self, k, shotnoise = True, Tc = None):
        if shotnoise:
            p1 = self.model.hadamard_test(k, self.nshots)/self.nshots
        else:
            p1 = self.model.hadamard_test(k)
        return p1
    
    @staticmethod
    def Nqueries(kend, nshots):
        Nq = SQAE.Napps_from_kend(kend)*nshots
        return Nq
    
    @staticmethod
    def Nqueries_noiseless(a, nshots):
        Napps = SQAE.Napps_from_a_noiseless(a)
        # Multiply by the number of times each circuit is repeated.
        Nq = int(Napps*nshots)
        return Nq
    
    '''
    The following methods are static so they can be carried out without an 
    instance; particularly, we can calculate nshots given Nq, and only after
    initialize the SQAE instance using this nshots. 
    '''
    @staticmethod
    def nshots_from_Nq_noiseless(a, Nq_target):
        Napps = SQAE.Napps_from_a_noiseless(a)
        nshots = max(int(Nq_target/Napps), 1)
        return nshots
    
    @staticmethod
    def Napps_from_a_noiseless(a):
        '''
        Number of unique applications of the Grover operator across all 
        circuits, given the amplitude and assuming no shot noise (so the 
        termination occurs deterministically at a fixed k_end).
        '''
        kx = np.log2(np.pi/(4*a**0.5))
        kend = np.ceil(kx)
        Napps = SQAE.Napps_from_kend(kend)
        return Napps
    
    @staticmethod
    def Napps_from_kend(kend):
        '''
        Number of unique applications of the Grover operator across all
        circuits, given kend (cycle runs k from 0 to kend) . 
        
        Same as expected Nq if nshots=1.
        '''
        # Napps of the A operator. If Grover, Napps = 2**(kend+1)-1. Each 
        # circuit uses A twice + 1 as many times as G. See blog post 23/08/22.
        Napps = 2**(kend+2) + kend
        return Napps
        
    
class TestSQAE(TesterQAE):
    def __init__(self, a, Tc, nshots, formula, threshold, silent = False):
        self.a = a
        self.Tc = Tc
        self.nshots = nshots
        self.formula = formula
        self.threshold = threshold
        self.silent = silent
        
    def single_run(self):
        sq = SQAE(self.theta, self.nshots, self.formula, self.threshold, 
                  self.silent)
        Nq, err = sq.estimate()
        return Nq, err
        
    def sqe_evolution(self, Nq_start, Nq_target, plot = True):
        nqs = []
        sqes = []
        
        Nq_curr_target = Nq_start
        while Nq_curr_target <= Nq_target:
            a = self.local_a
            Tc = self.local_Tc
            M =  QAEmodel(a, Tc = Tc)
            
            nshots = SQAE.nshots_from_Nq_noiseless(a, Nq_curr_target)
            sq = SQAE(M, nshots, self.formula, self.threshold,
                      silent = self.silent if Nq_curr_target == Nq_start \
                        else True)
            Nq_actual, a_est = sq.estimate()
            
            nqs.append(Nq_actual)
            sqes.append((a_est/a - 1)**2)

            Nq_curr_target *= 2
            
        if plot:
            errs = [sqe**0.5 for sqe in sqes]
            estdata = EstimationData()
            estdata.add_data(f"SQAE #{self.formula}", nqs = nqs, lbs = None, 
                             errs = errs)
            plot_err_vs_Nq(estdata, exp_fit = False)
        return nqs, sqes
    
    def sqe_evolution_multiple(self, nruns, Nq_start, Nq_target, save = True):

        print(f"> Will test {nruns} runs of 'Simpler QAE'.")
        nqs_all = []
        sqes_all = []
        pb = ProgressBar(nruns)
        for i in range(nruns):
            pb.update()
            try:
                nqs, sqes = self.sqe_evolution(Nq_start, Nq_target, 
                                               plot = False)
    
                nqs_all.extend(nqs)
                sqes_all.extend(sqes)
                
            except KeyboardInterrupt:
                print(f"\n> Keyboard interrupt at run {i}. "
                      "Will present partial results if possible.")
                print("> Breaking from cycle... [sqe_evolution_multiple]")
                nruns = i
                break

        estdata = EstimationData()
        estdata.add_data(f"SQAE #{self.formula}", nqs = nqs_all, lbs = None, 
                         errs = sqes_all)
        
        ed = ExecutionData(self.param_str, estdata, nruns, self.nshots, 
                           label = f"SQAE_f{self.formula}", 
                           extra_info = f"Nq≈{{10^{expb10(Nq_start)}.."
                           f"10^{expb10(Nq_target)}}},thr={self.threshold}")
        if save:
            ed.save_to_file()
        
        process_and_plot(estdata, save = save)
        self.print_info()
        
    def print_info(self):
        info = [" Simpler Quantum Amplitude Estimation "]
        info.append(f"a={self.a} | Tc={self.Tc}")
        info.append(f"nshots = {self.nshots} | threhsold = {self.threshold} ")
        info.append(f"inversion formula: {self. formula}")
        
        if not isinstance(self.a, tuple):
            info.append(f"Noiseless Nq would be: SQAE.Nqueries_noiseless(self.a, self.nshots)")
            
        if not self.silent:
            print_centered(info)
        
def test(which):
    a = (0,1)
    Tc = (2000, 10000)
    nshots = 100
    formula = 2
    threshold = 0.5
    test = TestSQAE(a, Tc, nshots, formula, threshold, silent = False)
    
    if which == 0:
        test.single_run()
    if which == 1:
        Nq_start = 200
        Nq_target = 10**5
        test.sqe_evolution(Nq_start, Nq_target)
    if which == 2:
        nruns = int(1e2) # 10**5
        Nq_start = 500
        Nq_target = 10**10
        test.sqe_evolution_multiple(nruns, Nq_start, Nq_target, save = True)  

if __name__ == "__main__": 
    test(2)