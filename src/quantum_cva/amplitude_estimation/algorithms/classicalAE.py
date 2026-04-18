'''
Amplitude estimation by classical Monte Carlo averaging.
'''

import numpy as np
import sys
import importlib

from quantum_cva.amplitude_estimation.algorithms.QAE import TesterQAE
from quantum_cva.amplitude_estimation.utils.misc import print_centered, expb10
from quantum_cva.amplitude_estimation.utils.models import QAEmodel
from quantum_cva.amplitude_estimation.utils.plotting import process_and_plot
from quantum_cva.amplitude_estimation.utils.mydataclasses import EstimationData, ExecutionData

reload = False
if reload:
    importlib.reload(sys.modules["utils.models"])

class classicalAE():
    def __init__(self, nshots, model):
        self.nshots = nshots
        self.model = model
        
    def estimate(self):
        # Classical can only measure for m=0 (no amplification).
        hits = self.model.measure(m = 0, nshots = self.nshots)
        a_est = hits/self.nshots
        return a_est
        
    @property
    def Nqueries(self):
        # Each shot requires a single query to A for initialization. In this
        # case, same as the queries to the oracle (for checking).
        return self.nshots
    
    @staticmethod
    def nshots_from_Nq(Nqueries):
        # 1-to-1 correspondence.
        return Nqueries
    
    @staticmethod
    def Nq_from_nshots(nshots):
        # 1-to-1 correspondence.
        return nshots
        
class testCAE(TesterQAE):
    def __init__(self, a, silent = False):
        self.a = a
        self.silent = silent
    
    def estimation(self, nshots):
        
        CAE = classicalAE(nshots, QAEmodel(self.a))
        a_est = CAE.estimate()
        print(f"> Estimated a = {a_est}. [test_CAE.estimation]")
        
    def rmse_given_nshots(self, nshots, nruns):
        '''
        Run CAE 'nshots' times in the same setting, namely using the same 
        number of shots. Use the resulting list of squared errors to compute 
        the RMSE.
        '''
        sqerrs = []
        for _ in range(nruns):
            a = self.local_a
            CAE = classicalAE(nshots, QAEmodel(a))
            a_est = CAE.estimate()
            sqerr = (a_est/a - 1)**2
            sqerrs.append(sqerr)
            
        # Nq is constant among all runs, depends only on nshots. So pick any.
        Nq = CAE.Nqueries
            
        sqe = np.mean(sqerrs)
        return Nq, sqe
        
    def sqe_evolution_multiple(self, Nq_start, Nq_target, nruns, save = True):
        '''
        Note that in this case any target Nqueries can be achieved exactly, 
        by setting nshots to said target.
        '''
        def print_info():
            info = ["Classical AE"]
            info.append("- scaling of the estimation error with Nq")
            info.append(f"a = {self.a} | runs = 10^{expb10(nruns)}")
            info.append(f"nshots={{10^{expb10(Nsmin)}..10^{expb10(Nsmax)}}} → "
                        f"Nq={{10^{expb10(Nq_start)}.. 10^{expb10(Nq_target)}}}")
            print_centered(info)

        print(f"> Will test {nruns} runs of 'classical AE'.")
        nqs, sqes_by_run = [], []
        Nsmin = classicalAE.nshots_from_Nq(Nq_start)
        Nsmax = classicalAE.nshots_from_Nq(Nq_target)
        if not self.silent:
            print_info()
        
        for i in range(nruns):
            nshots = Nsmin
            sqes = []
            while nshots <= Nsmax:
                Nq, sqe = self.rmse_given_nshots(nshots, 1)
                
                if i == 0:
                    nqs.append(Nq)
                sqes.append(sqe)
                nshots *= 5
                
            sqes_by_run.append(sqes) 
        
        estdata = EstimationData()
        estdata.add_data("classical", nqs = nqs, lbs = None, errs = sqes_by_run)
        pestdata = process_and_plot(estdata, processing = "averaging", 
                                    save = save)
        if save:
            ed = ExecutionData(self.a, pestdata, nruns, 
                               f"{{10^{expb10(Nsmin)}"
                               f"..10^{expb10(Nsmax)}}}", 
                               label = "classical_AE")
            ed.save_to_file()

def test():
    
    a_real = (0, 1)
    Nq_start = 5*10**1
    Nq_target = 10**8
    nruns = 10**2
    test = testCAE(a_real, silent = False)
    test.sqe_evolution_multiple(Nq_start, Nq_target, nruns, save = False)
   
if __name__ == "__main__":
    test()
