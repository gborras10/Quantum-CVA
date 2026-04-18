'''
Amplitude estimation by Gaussian rejection filtering.
'''
import numpy as np
from scipy.stats import circvar
from scipy.stats import vonmises, norm, uniform
import matplotlib.pyplot as plt

from quantum_cva.amplitude_estimation.utils.models import QAEmodel
from quantum_cva.amplitude_estimation.utils.mydataclasses import MeasurementData, EstimationData
from quantum_cva.amplitude_estimation.utils.misc import estimation_errors
from quantum_cva.amplitude_estimation.utils.plotting import plot_err_vs_Nq

test0 = False
if test0:
    mu = 0.77
    kappa = 0.1
    Nsamples = int(1e6)
    
    xs = vonmises.rvs(kappa, loc = mu, size=Nsamples)
    print(xs)
    params = vonmises.fit(xs, fscale=1)
    print(params)

class WGRF():
    
    def __init__(self, model, Nsamples, dist = "vonmises"):
        self.model = model
        self.Nsamples = Nsamples
        self.curr_mean = None
        self.curr_std = None
        self.dist = dist
        
    def update_single(self, m, outcome, nshots):
        if self.curr_mean is None:
            locs = uniform.rvs(size = self.Nsamples)
            if self.dist == "vonmises":
                locs = locs*2*np.pi - np.pi
        elif self.dist == "norm":
            locs = norm.rvs(loc = self.curr_mean,
                              scale = self.curr_std,
                              size = self.Nsamples)
        elif self.dist == "vonmises":
            locs = vonmises.rvs(self.curr_std,
                                loc = self.curr_mean,
                                size = self.Nsamples)
            #print(locs)
            # locs = list(map(abs, locs))
            ## print("2", locs)
            
        likelihood = self.model.likelihood 
        
        if self.dist == "norm":
            probs = [likelihood(loc, m, outcome, nshots) for loc in locs]
        elif self.dist == "vonmises":
            a_from_loc = lambda loc: np.sin((loc + np.pi)/4)**2
            probs = [likelihood(a_from_loc(loc), m, outcome, nshots) 
                     for loc in locs]
            
        pmax = max(probs)
        # print("pmax", pmax)
        locs = [loc for loc, prob in zip(locs, probs) 
                if uniform.rvs() < prob/pmax]
        #print(len(locs))
        
        if self.dist == "norm":
            self.curr_mean, self.curr_std = norm.fit(locs)
        elif self.dist == "vonmises":
            self.curr_mean, self.curr_std = vonmises.fit(locs, fscale = 1)[1::-1]
            #print(self.curr_mean, self.curr_std)
        
    def update_multiple(self, data: MeasurementData):
        a_ests = []
        for i in range(len(data)):
            self.update_single(data.ctrls[i], data.outcomes[i], data.Nsshots[i])
            a_from_loc = lambda loc: np.sin((loc + np.pi)/4)**2
            a_ests.append(a_from_loc(self.curr_mean))
        return a_ests

def test(): 
    a = 0.99
    model = QAEmodel(a)
    
    Nmeas = 100
    nshots = 50
    Nsshots = [nshots for _ in range(Nmeas)]
    ms = [0 for _ in range(Nmeas)]
    
    Nsamples = 1000
    
    nqs = [(2*m+1)*nshots for m, nshots in zip(ms, Nsshots)]
    nqs = np.cumsum(nqs)
    
    runs = 3
    sqes_by_run = []
    for i in range(runs):
      try:
        rf = WGRF(model, Nsamples)
        data = model.create_data(ms, Nsshots)
        a_ests = rf.update_multiple(data)
        sqes = [(a_est-a)**2 for a_est in a_ests]
        sqes_by_run.append(sqes)
        print(f"> Run {i} completed, estimated {a_ests[-1]}.")
      except KeyboardInterrupt:
        break
      # except:
      #   print("Error, will ignore.")
        
    sqes_by_step = estimation_errors(sqes_by_run)
    estdata = EstimationData()
    estdata.add_data("GRF", nqs = nqs, lbs = None, errs = sqes_by_step)
    plot_err_vs_Nq(estdata)

        
test()
# Sometimes fails due to negative a!