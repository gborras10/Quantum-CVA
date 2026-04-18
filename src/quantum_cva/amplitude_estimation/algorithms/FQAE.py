'''
Faster quantum amplitude estimation.
'''

import numpy as np

import importlib, sys

from quantum_cva.amplitude_estimation.algorithms.QAE import TesterQAE
from quantum_cva.amplitude_estimation.utils.plotting import process_and_plot, plot_err_evol
from quantum_cva.amplitude_estimation.utils.mydataclasses import EstimationData, ExecutionData
from quantum_cva.amplitude_estimation.utils.models import QAEmodel
from quantum_cva.amplitude_estimation.utils.misc import print_centered
from quantum_cva.amplitude_estimation.utils.running import ProgressBar
from quantum_cva.amplitude_estimation.utils.binning import process_raw_estdata

NDIGITS = 3 

class FQAE():
    def __init__(self, model, delta_c, kmax, rfactor = 16):
        '''
        Rescale by a factor of 16 to get 0 < sqrt(a) < 1/4. Note that in the 
        paper, they use a_FAE := sqrt(a_usu), where a_usu^2 is the probability
        of success and a_FAE would be the amplitude in the usual sense.
        '''
        self.model = model
        self.rfactor = 16
        self.model.rescale(rfactor)
        self.delta_c = delta_c
        self.kmax = kmax
        
    def estimate(self):
        thmin, thmax = 0, 0.252
        Nq1, k_x, th_x, thmax = self.first_stage(thmin, thmax)
        if k_x < self.kmax - 1:
            Nq2, thmin, thmax = self.second_stage(k_x, th_x, thmax)
        else: 
            Nq2 = 0
        
        Nq = Nq1 + Nq2
        th_est = (thmin + thmax)/2
        a_est = self.rfactor*np.sin(th_est)**2
        return Nq, a_est
        
    def first_stage(self, thmin, thmax):
        nshots = np.ceil(1944*np.log(2/self.delta_c))
        k, Nq = 0, 0
        # j from the paper is = k + 1, and in {1, kmax}. Here k in {0, kmax-1}.
        while k < self.kmax:
            m = 2**k
            p1 = self.model.measure(m, nshots)/nshots
            Nq += (2*m+1)*nshots
            c = 1 - 2*p1
            cmin, cmax = self.Chernoff_bounds(c, nshots)
            thmin, thmax = list(map(lambda c: self.theta_from_c(c, m), 
                                    [cmax, cmin]))
            
            # The j-dependant part of the factor multiplying theta in the 
            # double-angle cosine for the next iteration. This factor is 2r
            # for r the Grover factor. We want cos(2r*theta) with 2r*theta < pi
            # r_next = 2*2^(k+1)
            next_factor = 4*2**(k+1)
            if next_factor*thmax >= 3*np.pi/4:
                # Can't assure invertibility in the following iteration.
                break
            k += 1
            
        k_x = k
        th_x = (thmin + thmax)/2
        return Nq, k_x, th_x, thmax
            
    def second_stage(self, k_x, th_x, thmax):
        nshots = np.ceil(972*np.log(2/self.delta_c))
        k = k_x + 1
        Nq = 0
        while k < self.kmax:
            m1, m2 = 2**k, 2**k+2**k_x
            cos1 = 1 - 2*self.model.measure(m1, nshots)/nshots
            cos2 = 1 - 2*self.model.measure(m2, nshots)/nshots
            Nq += (2*m1+1)*nshots + (2*m2+1)*nshots
            sin1 = (cos1*np.cos(2**k_x*th_x) - cos2)/np.sin(2**k_x*th_x)
            rho = np.arctan2(sin1, cos1)
            
            x = (2**(k+2)+2)*thmax - rho + np.pi/3
            n = np.floor(x/2/np.pi)
            thmin = (2*np.pi*n + rho - np.pi/3)/(2**(k+2)+2)
            thmax = (2*np.pi*n + rho + np.pi/3)/(2**(k+2)+2)
            k += 1
            
        return Nq, thmin, thmax
        
    @staticmethod
    def theta_from_c(c, m):
        K = 2*(2*m+1) 
        theta = np.arccos(c)/K
        return theta
                       
    def Chernoff_bounds(self, c, Ns):
        halfwidth = np.sqrt(np.log(2/self.delta_c)*12/Ns)
        cmin = max(-1, c - halfwidth)
        cmax = min(1, c + halfwidth)
        return cmin, cmax
    
class TestFQAE(TesterQAE):
    def __init__(self, a, Tc, delta_c, silent = False):
        self.a = a
        self.Tc = Tc
        self.delta_c = delta_c
        self.silent = silent
        if not silent:
            info = [" Faster Quantum Amplitude Estimation (by Nakaji)"]
            info.append(f"a = {self.a} | Tc = {self.Tc} | {delta_c = } ")
            print_centered(info)
        
    def single_run(self, kmax):
        model = QAEmodel(self.a)
        fq = FQAE(model, self.delta_c, kmax)
        Nq, a = fq.estimate()
        err = a - self.a
        if not self.silent:
            print(f"> Tested {kmax =}.")
            print(f"> Estimated {a = } ({err = :.2e}) using {Nq:.1e} queries.")
            
    def sqe_evolution(self, kmax_start, kmax_end, plot = True):
        nqs = []
        sqes = []
        
        a = self.local_a
        Tc = self.local_Tc
            
        kmax = kmax_start
        while kmax <= kmax_end:
            M =  QAEmodel(a, Tc = Tc)
            
            fq = FQAE(M, self.delta_c, kmax)
            Nq, a_est = fq.estimate()
            
            nqs.append(Nq)
            sqes.append((a_est/a - 1)**2)

            kmax += 1
        
        if sqes[-1] > 0.2 and not self.silent:
            print(f"\n> Squared error error was over 0.2.")
            print(f"> Details: A {a = }, {a_est = } sqe = {sqes[-1]}.")

        if plot:
            errs = [sqe**0.5 for sqe in sqes]
            estdata = EstimationData()
            estdata.add_data("FQAE ", nqs = nqs, lbs = None, 
                             errs = errs)
            plot_err_evol(estdata, exp_fit = False)
        return nqs, sqes
    
    def sqe_evolution_multiple(self, nruns, kmax_start, kmax_end, 
                               save = True):

        print(f"> Will test {nruns} runs of 'Faster QAE'.")
        nqs_all = []
        sqes_all = []
        pb = ProgressBar(nruns)
        for i in range(nruns):
            pb.update()
            try:
                nqs, sqes = self.sqe_evolution(kmax_start, kmax_end,
                                               plot = False)
                
                nqs_all.extend(nqs)
                sqes_all.extend(sqes)
                
                # Ensure silence after first run (otherwise too many prints).
                self.silent = True
                
            except KeyboardInterrupt:
                print(f"\n> Keyboard interrupt at run {i}. "
                      "Will present partial results if possible.")
                print("> Breaking from cycle... [sqe_evolution_multiple]")
                nruns = i
                break
        # Save raw data.
        label = "FQAE"
        raw_estdata = EstimationData()
        raw_estdata.add_data(label, nqs = nqs_all, lbs = None, 
                         errs = sqes_all)
        
        ed = ExecutionData(self.param_str, raw_estdata, nruns, nshots = "NA", 
                           label = "FQAE", 
                           extra_info = f"kmax≈{{{kmax_start}.."
                           f"{kmax_end}}},delta_c={self.delta_c}")
        if save:
            ed.save_to_file()
            
        process_and_plot(raw_estdata, save = save)
    
def test(which):
    if which == 0:
        a = 0.007
        global rth 
        rth= np.arcsin((a/4)**0.5)
        print("th",rth)
        delta_c = 0.01
        kmax = 10
        test = TestFQAE(a, delta_c)
        test.single_run(kmax)
    if which == 1:
        a = 0.4
        delta_c = 0.01
        kmax_start, kmax_end = 1, 10
        test = TestFQAE(a, delta_c)
        test.sqe_evolution(kmax_start, kmax_end)
    if which == 2:
        # 0.49606640829040916 -> bad
        a = (0, 1)
        Tc = (2000, 5000)
        delta_c = 0.01
        nruns = int(1e2)
        kmax_start, kmax_end = 1, 18
        test = TestFQAE(a, Tc, delta_c)
        test.sqe_evolution_multiple(nruns, kmax_start, kmax_end, save = True)
         
if __name__ == "__main__":
    test(2)