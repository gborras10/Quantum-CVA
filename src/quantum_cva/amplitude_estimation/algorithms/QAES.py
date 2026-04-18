# -*- coding: utf-8 -*-
'''
Quantum amplitude estimation simplified.
'''

import numpy as np
import importlib
import sys

from quantum_cva.amplitude_estimation.algorithms.QAE import TesterQAE
from quantum_cva.amplitude_estimation.utils.misc import closest_odd_int, expb10, print_centered
from quantum_cva.amplitude_estimation.utils.mydataclasses import EstimationData, ExecutionData
from quantum_cva.amplitude_estimation.utils.plotting import process_and_plot
from quantum_cva.amplitude_estimation.utils.models import QAEmodel
from quantum_cva.amplitude_estimation.utils.running import ProgressBar

reload = False
if reload:
    importlib.reload(sys.modules["utils.plotting"])
    importlib.reload(sys.modules["utils.models"])
    importlib.reload(sys.modules["utils.misc"])
    importlib.reload(sys.modules["utils.binning"])
    #importlib.reload(sys.modules["src.utils.mydataclasses"])

class QAES():
    def __init__(self, M, epsilon, alpha, silent = False):
        self.model = M
        # The constant to divide 'a' by. With this rescaling, theta < 0.001,
        # which is required by the algorithm. We then multiply back. In 
        # practice this can be achieved by rotating an ancilla qubit. 
        self.div = 1001**2
        self.model.rescale(1001**2)
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.silent = silent 
        
    def pre_processing(self):
        Nq = 0
        k = 0
        p1 = 0
        while p1 < 0.95:
            rk = np.floor(1.05**k)
            if rk % 2 == 0:
                # rk must be odd.
                rk -= 1
                
            # rk is the "Grover factor"; apply Grover operator (rk-1)/2 times.
            m = self.Napps_from_r(rk)
            nshots = np.ceil(5000*np.log(5/self.alpha))
            p1 = self.model.measure(m, nshots)/nshots
            
            Nq += m*nshots
            k += 1
        return Nq, k
    
    def exp_refinement(self, kend):
        Nq = 0
        th_min = 0.9*1.05**-kend
        th_max = 1.65*th_min
        
        t = 0
        while th_max > (1 + self.epsilon/5)*th_min:
            delta_th = th_max - th_min
            k = round(th_min/(2*delta_th))
            rt = closest_odd_int(np.pi*k/th_min)
            m = self.Napps_from_r(rt)
            
            alpha_t = self.alpha*self.epsilon/65 * 0.9**(-t) 
            nshots = 250*np.log(alpha_t**-1)
            
            p1 = self.model.measure(m, nshots)/nshots
            
            gamma = th_max/th_min - 1
            if p1 > 0.12:
                th_min = th_max/(1 + 0.9*gamma)
            else:
                th_max = (1 + 0.9*gamma)*th_min
            
            Nq += m*nshots
            t += 1
            
        return Nq, th_min, th_max
    
    def estimate(self):
        Nq1, kend = self.pre_processing()
        Nq2, th_min, th_max = self.exp_refinement(kend)
        Nq = Nq1 + Nq2
        a_est = self.div*np.sin(th_max)**2
        return Nq, a_est
            
    @staticmethod
    def Napps_from_r(r):
        '''
        'r' is the Grover factor multiplying theta. Napps is the number of 
        Grover iterations.
        '''
        m = (r-1)/2
        return m
   
class TestQAES(TesterQAE):
    def __init__(self, a, Tc, alpha, silent = False):
        self.a = a
        self.Tc = Tc
        self.alpha = alpha
        self.silent = silent
        
    def final_result(self, epsilon):
        def print_info():
            info = ["QAES estimation single"]
            info.append((f"  ε={epsilon} | α={self.alpha} | a={self.a} | "
                         f"θ≈{np.arcsin(self.a**0.5)}"))
            print_centered(info)
            
        qs = QAES(self.a, epsilon, self.alpha)
        Nq, a_est = qs.estimate()
        return Nq, a_est
    
    def rmse_evol_single(self, epsmin, epsmax):
        '''
        Perform estimation for a fixed amplitude, progressively decreasing the
        relative tolerance (which requires increasing the number of queries).
        '''
        nqs, sqes = [], []
        epsilon = epsmax
        a = self.local_a
        Tc = self.local_Tc
        while epsilon > epsmin:
            # Two stage algorithm, so it must be run once for each query number
            # (no intermediate results).
            M = QAEmodel(a, Tc = Tc)
            qs = QAES(M, epsilon, self.alpha, silent = self.silent)
            Nq, a_est = qs.estimate()
            sqe = (a_est/a-1)**2
            
            nqs.append(Nq)
            sqes.append(sqe)
            
            epsilon = epsilon*0.8    
        return nqs, sqes
    
    def sqe_evolution_multiple(self, reps, epsmin, epsmax, save = True):
        def print_info():
            info = ["QAES estimation RMSE scaling"]
            info.append(f"  ε in {{10^{expb10(epsmin)},10^{expb10(epsmax)}}} "
                         f"| α={self.alpha} | a={self.a} ")
            info.append(f"| reps = {reps}")
            print_centered(info)
        
        print(f"> Will test {reps} runs of 'QAE, simplified'.")
        if not self.silent:
            print_info()
        
        pb = ProgressBar(reps)
        nqs_all, sqes_all = [], []
        try:
            for i in range(reps):
                pb.update()
                nqs, sqes = self.rmse_evol_single(epsmin, epsmax)
                
                nqs_all.extend(nqs)
                sqes_all.extend(sqes)
        except KeyboardInterrupt:
            print(f"\n> Keyboard interrupt at run {i}. "
                      "Will present partial results if possible.")
            print("> Breaking from cycle... [sqe_evolution_multiple]")
            reps = i
            
        self.handle_results(nqs_all, sqes_all, reps, epsmin, epsmax, save)
        
    def handle_results(self, nqs_all, sqes_all, reps, epsmin, epsmax, save):
        estdata = EstimationData()
        estdata.add_data("QAES", nqs = nqs_all, lbs = None, errs = sqes_all)
        
        ed = ExecutionData(self.a, estdata, reps, "NA", 
                           label = "QAES", 
                           extra_info = f"eps∈{{10^{expb10(epsmin)},"
                           f"10^{expb10(epsmax)}}}, alpha={self.alpha}")
        if save:
            ed.save_to_file()
       
        process_and_plot(estdata, save = save)
   
def test(which):
    if which == 0:
        '''
        Test a single estimation run. 
        '''
        a = 0.34
        epsilon = 1e-5
        alpha = 0.001
        test = TestQAES(a, alpha)
        _, a_est = test.final_result(epsilon)
        print(f"> Estimated a = {round(a_est, -expb10(epsilon)+1)}.")
        rerr = a_est/a-1
        print(f"> Relative error: {rerr:.1e}.")
    if which == 1:
        a = (0,1)
        Tc = None # (2000, 5000)
        epsmax = 1e-2
        epsmin = 1e-6
        alpha = 0.05
        runs = 1
        test = TestQAES(a, Tc,  alpha)
        test.sqe_evolution_multiple(runs, epsmin, epsmax, save = True)
        
if __name__ == "__main__":
    test(1)