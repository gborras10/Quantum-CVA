'''
Iterative quantum amplitude estimation.
'''

import sys
import importlib
import numpy as np

from quantum_cva.amplitude_estimation.utils.misc import print_centered, expb10
from quantum_cva.amplitude_estimation.utils.running import ProgressBar
from quantum_cva.amplitude_estimation.utils.mydataclasses import EstimationData, ExecutionData
from quantum_cva.amplitude_estimation.utils.models import QAEmodel
from quantum_cva.amplitude_estimation.utils.plotting import process_and_plot
from quantum_cva.amplitude_estimation.algorithms.QAE import TesterQAE

Ndigits = 5

class IQAE:
    def __init__(self, model, epsilon, alpha, nshots, modified = False,
                 ci = "chernoff", cos = True, silent = False):
        self.model = model
        self.epsilon = epsilon 
        self.alpha = alpha
        self.nshots = nshots
        self.modified = modified
        self.ci = ci
        self.silent = silent
        # Whether to use the cosine likelihood function as opposed to the sine.
        # sin^2((2k+1)*theta) vs [1-cos((4k+2)*theta]/2.
        self.cos = cos
        assert ci == "chernoff", \
            "Only Chernoff bounds implemented."
            
        if not silent:
            mstr = "Modified " if modified else ""
            info = [mstr + "Iterative Quantum Amplitude Estimation"]
            info.append(f"a = {round(model.amplitude, Ndigits)} | epsilon = "
                        f"{epsilon} | nshots = {nshots}")
            info.append(f"alpha = {alpha} | CI: {ci}")
            likelihood = ("1-cos/(K*theta)2, K = 4k+2" if cos 
                          else "sin^2(K*theta), K = 2k+2")
            info.append(f"likelihood: {likelihood}")
            print_centered(info)
        
    def estimate(self):
        '''
        Sequentially refine the confidence interval by choosing amplification 
        factors, measuring the associated circuits, and learning from the data.
        
        Note that the amplification factor K / number of Grover iterations k
        can be kept the same across several iterations. In that case, the data
        (1 outcomes, numbers of shots) from said iterations is combined into a
        compound tuple (observed relative frequency, number of shots). 
        
        We call a set of consecutive iterations with the same k a 'round'.
        '''
        kprev = 0
        Nq = 0
        # Need to store each round's measurement data across its iterations.
        Ns, hits_all = [], []
        first_half = True
        theta_min, theta_max = 0, np.pi/2
        while theta_max - theta_min > 2*self.epsilon:
            k, first_half = self.find_next_k(kprev, first_half, theta_min, 
                                             theta_max)
            K = self.K_from_k(k)
            
            if k != kprev:
                # New k, new round - start over.
                Ns, hits_all = [], []
            
            if self.modified:
                Kmax = np.pi/4/self.epsilon
                alpha_i = 2*self.alpha*K/(3*Kmax)
                N = 2*np.log(2/alpha_i)/(np.sin(np.pi/21)**2*np.sin(8*np.pi/21)**2)
            # No overshooting condition.
            elif (K if self.cos else 2*K) > np.ceil(self.Lmax/self.epsilon):
                N = np.ceil(self.nshots*self.Lmax/self.epsilon/K/10)
            else:
                N = self.nshots
            Ns.append(N)
                
            # Get number of '1' (correct) outcomes.
            hits = self.model.measure(k, N)
            hits_all.append(hits)
            # Queries to A.
            Nq += N*(2*k+1)
            
            # The effective quantities consider all iterations in the round.
            N = sum(Ns)
            p1 = sum(hits_all)/N
            
            # Process the information resulting from this round (so far).
            a_min, a_max = self.confidence_interval(p1, N)
            Ktheta_lims = (self.Ktheta_from_meas(a_min, first_half),
                           self.Ktheta_from_meas(a_max, first_half))
            # Second region -> P(1|theta) is decreasing function of theta.
            if not first_half:
                Ktheta_lims = Ktheta_lims[::-1]
                
            # Need the prior knowledge to roughly place theta: amplification 
            # may render part of it an inconsequential multiple of 2*pi.
            theta_lims = (theta_min, theta_max)
            theta_min, theta_max = self.refine_boundaries(theta_lims, 
                                                          Ktheta_lims, K)
            kprev = k
            
        a_min = self.a_given_theta(theta_min)
        a_max = self.a_given_theta(theta_max)
        a_est = (a_min + a_max)/2
        err = (a_max - a_min)/2
        
        if not self.silent:
            print("> IQAE: estimation completed, results below.")
            print(f"> amplitude   =  {round(a_est, Ndigits)}")
            print(f"> uncertainty =  {round(err, Ndigits)}")
            print(f"> Nq = {int(Nq)}")
            
        return Nq, a_est
        
    def find_next_k(self, k, first_half, theta_min, theta_max, r = 2):
        '''
        We want K as large as possible while not introducing ambiguity, so
        we'll start high and reduce it as much as needed. 
        
        Additionally, we stop if the tentative K gets too close to the 
        previous one. In that case, instead of changing K, we'll repeat the 
        previous measurement and join the data together.
        '''
        # The previous iteration K.
        Kprev = self.K_from_k(k)
        # Upper roof for K that allows theta_min and theta_max to coexist in 
        # the same injective section of the likelihood domain (-> invertible).
        max_injective_interval = np.pi if self.cos else np.pi/2
        Kmax = np.floor(max_injective_interval/(theta_max - theta_min))
        # K must be of the form in 'K_from_k' for integer k -> subtract excess.
        Kmax -= (Kmax - 2) % 4 if self.cos else (Kmax - 1) % 2

        K = Kmax
        while K > r*Kprev:
            same_half = self.check_same_half(K*theta_min, K*theta_max)
            if same_half:
                k = self.k_from_K(K)
                first_half = True if same_half == "1" else False
                break
            K -= 4 if self.cos else 2
        return k, first_half
        
    def refine_boundaries(self, theta_lims, Ktheta_lims, K):
        '''
        Refine the boundaries 'theta_lims' on theta using the boundaries 
        'Ktheta_lims' on K*theta.
        '''
        # Solid knowledge that roughly places theta on the unit circle.
        theta_min, theta_max = theta_lims
        # More precise estimates of the limits of the net/"fractional" 
        # (wrt. period) angle, K*theta mod 2pi. Good for refining previous 
        # knowledge on theta.
        Ktheta_frac_min, Ktheta_frac_max = Ktheta_lims 
        
        # print(f"{K}, {K*theta_min % (2*np.pi)}, {Ktheta_frac_min}, {K*theta_max  % (2*np.pi)}, {Ktheta_frac_max}")

        new_theta_min = self.refine_boundary(theta_min, Ktheta_frac_min, K)
        new_theta_max = self.refine_boundary(theta_max, Ktheta_frac_max, K)
        
        # If the previous boundaries are tighter than the new ones, keep them.
        if new_theta_min > theta_min:
            theta_min = new_theta_min
        if new_theta_max < theta_max:
            theta_max = new_theta_max
            
        return theta_min, theta_max
            
    def refine_boundary(self, theta_lim, Ktheta_frac_lim, K):
        '''
        Refine a minimum or maximum boundary 'theta_lim', given a minimum or 
        maximum for the fractional part 'Ktheta_frac_lim'.
        
        The 2 first inputs should correspond to the same extrema, i.e. be  
        either both minima or both maxima.
        '''
        # Angle accumulated by the periods completed by K*theta_lim ("integer 
        # part" wrt. period). Inconsequential for K*theta, but not for the real 
        # angle theta.
        Ktheta_int = self.full_cycles_acc_distance(K*theta_lim, self.period)
        new_theta_lim = (Ktheta_int + Ktheta_frac_lim)/K
        return new_theta_lim
    
    @staticmethod
    def full_cycles_acc_distance(total_distance, period):
        '''
        Calculate the distance accumulated by the complete cycles/"full laps" 
        within 'total_distance', for a function of period 'period'.
        
        Example:
            distance_completed_cycles(4.7*pi, 2*pi) = 4*pi
        '''
        laps = np.floor(total_distance/period)
        acc_distance = laps*period
        return acc_distance
            
    def confidence_interval(self, p1, N):
        '''
        Construct a confidence interval around the observed 'probability'
        (relative frequency) of 1, which estimates the amplitude.
        '''
        if self.ci == "chernoff":
            epsilon_a = (np.log(2*self.T/self.alpha)/2/N)**0.5
            a_min = max(0, p1 - epsilon_a)
            a_max = min(1, p1 + epsilon_a)
        
        return a_min, a_max
    
    def check_same_half(self, theta1, theta2):
        '''
        Return False if the two angles are not in the same half of the working 
        domain we want to invert theta in, and otherwise a string indicating 
        the half they're both in.
        '''
        half1 = self.which_half(theta1)
        half2 = self.which_half(theta2)
        if half1 == half2:
            return half1
        else:
            return False
           
    def which_half(self, theta):
        '''
        In which half of the working domain does theta lie: first, or second?
        We can invert the likelihood function in each of these regions.
        
        For self.cos = True, this means [0, pi[ vs. [pi, 2pi[.
        For self.cos = False, this means [0, pi/2[ vs. [pi/2, pi[.
        '''
        frontier = self.period/2
        if theta % self.period < frontier:
            return "1"
        else:
            return "2"
            
    def Ktheta_from_meas(self, p1, first_half):
        '''
        Estimate (K*theta mod period) from the observed probability of 
        measuring 1.
        
        The likelihood function is only invertible in half of the considered
        region, but with the 'first_half' flag we can cover is entirety. If it 
        indicates that the angle is in the 2nd injective region, we can find 
        the angle from that region which gives the same p1.
        
        Since the likelihood function starts retracing back its steps halfway 
        through the period, that angle is 'original angle' away from the 
        period, so subtract the former from the latter.
        '''
        Ktheta = np.arccos(1-2*p1) if self.cos else np.arcsin(p1**0.5)
        if not first_half:
            Ktheta = self.period - Ktheta
        return Ktheta
    
    @property
    def period(self):
        period = 2*np.pi if self.cos else np.pi
        return period
        
    @staticmethod
    def a_given_theta(theta):
        a = np.sin(theta)**2
        return a
    
    @staticmethod
    def theta_given_a(a):
        theta = np.arcsin(a**0.5)
        return theta
    
    def K_from_k(self, k):
       K = 4*k+2 if self.cos else 2*k+1
       return K
    
    def k_from_K(self, K):
        k = (K - 2)/4 if self.cos else (K - 1)/2
        return k
            
    @property
    def T(self):
        T = np.ceil(np.log2(np.pi/8/self.epsilon))
        return T
    
    @property
    def Lmax(self):
        if self.ci == "chernoff":
            Lmax = np.arcsin(2*np.log(2*self.T/self.alpha)/self.nshots)
        return Lmax
    
class TestIQAE(TesterQAE):
    def __init__(self, a, Tc, nshots, alpha, modified, ci, silent = False):
        self.a = a
        self.Tc = Tc
        self.nshots = nshots
        self.alpha = alpha
        self.modified = modified
        self.ci = ci
        self.silent = silent
        
    def single_run(self, epsilon):
        model = QAEmodel(self.a)
        iq = IQAE(model, epsilon, self.alpha, self.nshots, self.modified, self.ci)
        iq.estimate()
        
    def sqe_evolution(self, eps_start, eps_end, silent, plot = True):
        nqs = []
        sqes = []
        epsilon = eps_start
        while epsilon > eps_end:
            a = self.local_a
            Tc = self.local_Tc
            model =  QAEmodel(a, Tc = Tc)
            
            # Only not silent for the first iteration otherwise it's too much.
            iq = IQAE(model, epsilon, self.alpha, self.nshots, self.modified, 
                      self.ci, silent = silent if epsilon == eps_start else True)
            Nq, a_est = iq.estimate()
            
            nqs.append(Nq)
            sqes.append((a_est/a - 1)**2)

            epsilon *= 0.5
            
        if plot:
            errs = [sqe**0.5 for sqe in sqes]
            estdata = EstimationData()
            mchar = "m" if self.modified else ""
            estdata.add_data(f"{mchar}IQAE - {self.ci}", nqs = nqs, lbs = None, 
                             errs = errs)
            process_and_plot(estdata)
        return nqs, sqes
    
    def sqe_evolution_multiple(self, nruns, eps_start, eps_end, save = True):

        print(f"> Will test {nruns} runs of 'Iterative QAE'.")
        nqs_all = []
        sqes_all = []
        pb = ProgressBar(nruns)
        for i in range(nruns):
            pb.update()
            try:
                nqs, sqes = self.sqe_evolution(eps_start, eps_end, 
                                               silent = self.silent \
                                                if i==0 else True, 
                                               plot = False)
                
                nqs_all.extend(nqs)
                sqes_all.extend(sqes)
                
            except KeyboardInterrupt:
                print(f"\n> Keyboard interrupt at run {i}. "
                      "Will present partial results if possible.")
                print("> Breaking from cycle... [sqe_evolution_multiple]")
                nruns = i
                break

        # Save raw data.
        mchar = "m" if self.modified else ""
        label = f"{mchar}IQAE - {self.ci}"
        raw_estdata = EstimationData()
        raw_estdata.add_data(label, nqs = nqs_all, lbs = None, 
                         errs = sqes_all)
        mchar = "m" if self.modified else ""
        ed = ExecutionData(self.param_str, raw_estdata, nruns, self.nshots, 
                           label = f"{mchar}IQAE_{self.ci[:4]}", 
                           extra_info = f"eps≈{{10^{expb10(eps_start)}.."
                           f"10^{expb10(eps_end)}}},alpha={self.alpha}")
        if save:
            ed.save_to_file()

        process_and_plot(raw_estdata, save = save)
     
def test(which):
    ci = "chernoff"
    modified = False
    if which == 0:
        '''
        Single estimation run, print end result..
        '''
        a = 0.3
        alpha = 0.05
        nshots = 100
        epsilon = 1e-2
        test = TestIQAE(a, nshots, alpha, modified, ci)
        test.single_run(epsilon)
    if which == 1:
        '''
        Plot evolution of the error.
        '''
        a = 0.3
        alpha = 0.05
        nshots = 100
        eps_start, eps_end = 1e-1, 1e-5
        test = TestIQAE(a, nshots, alpha, modified, ci)
        test.sqe_evolution(eps_start, eps_end, silent = False)
    if which == 2:
        '''
        Run several times and plot evolution of the RMSE.
        '''
        a = (0,1)
        Tc = None # (2000, 5000)
        alpha = 0.05
        nshots = 100
        eps_start, eps_end = 1e-1, 4e-7
        test = TestIQAE(a, Tc, nshots, alpha, modified, ci, silent = True)
        nruns = int(1e2)
        test.sqe_evolution_multiple(nruns, eps_start, eps_end, save = True)

if __name__ == "__main__": 
    test(2)
