'''
Emulate canonical QAE without actually running the circuits, by using
the knowledge of the solution to compute the outcome probabilities 
analytically. Binomial sampling can be used to inject shot noise.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from qiskit.visualization import plot_histogram
import scipy.optimize as opt
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 300
import sys
import importlib


from quantum_cva.amplitude_estimation.utils.running import ProgressBar
from quantum_cva.amplitude_estimation.utils.misc import (outcome_dist_to_dict, rem_negligible_items,
                        k_largest_tuples, print_centered, expb10)
from quantum_cva.amplitude_estimation.utils.plotting import barplot_from_data, process_and_plot
from quantum_cva.amplitude_estimation.utils.mydataclasses import QAEspecs, EstimationData, ExecutionData

reload = False
if reload:
    importlib.reload(sys.modules["utils.models"])

def estimation_table():
    '''
    Print a table with the main variables associated with canonical QAE, for
    varying numbers of auxiliary qubits 'm' in the QFT register.
    '''
    n = 3
    t = 1
    
    table = []
    for m in range(1,12):
        qd = QAEspecs(n, t, m)
        info = [m]
        info.append(qd.x0())
        info.extend(qd.closest_outcomes())
        est_thetas = qd.estimated_thetas()
        est_thetas = [th for th in est_thetas if isinstance(th, float)]
        info.append(min(est_thetas))
        info.append(qd.estimated_a())
        info.append(qd.estimated_Nsol())
        
        table.append(info)
        
    a = t/2**n
    th = np.arcsin(a**0.5)
    table.append(["REAL", "-", "-", "-", th, a, a*2**n])
        
    df = pd.DataFrame(table, columns = ['m', 'x0 = Mθ/π', 'out1', 'out2',
                                        'θ_est', 'a_est', 'Nsol'])
    
    for col in ['θ_est', 'a_est', 'Nsol']:
        df[col] = df.apply(lambda row: round(row[col], 5), axis = 1)
    
    df.set_index('m', inplace=True)
    print(df)
    #print(df.to_latex())

def QPE_probability_of(y, phi, M):
    '''
    Calculate the probability of outcome 'y' in phase estimation, given 
    'phi' the real angle (i.e. eigenvalue exp(i*2*pi*phi)) and M the QFT order. 
    '''
    # 'd' is the error in the discretized estimate for 'phi' produced by  
    # outcome 'y': phi = y/M + d.
    
    d = phi - y/M
    d = d % 1
    p = 1 if d==0 else np.sin(M*d*np.pi)**2/(M*np.sin(d*np.pi))**2
    return p 

class QAEemulator():
    def __init__(self, theta_real, m, nshots, silent = False, show = True):
        self.theta_real = theta_real
        self.a = np.sin(theta_real)**2
        self.nshots = nshots
        self.m = m
        self.silent = silent
        
        if show:
            fig, ax = plt.subplots(1, figsize=(12,8))
            self.fig = fig
            self.ax = ax
            # A secondary y axis may be necessary, and if so should be 
            # accessible.
            self.ax2 = None
        
        if not silent:
            print("> Created QAEemulator instance with:")
            print("> theta = %.4f" % theta_real)
            print("> a = %.4f." % self.a)
            M = 2**m
            print(f"> m={m} qubits, so M={M}.")
            x0 = theta_real*M/np.pi
            print("> Exact case result #1: ", x0)
            print("> Exact case result #2: ", M-x0)
        
    def estimate(self, plist = None, MLE = True, Nevals=100, ret_range = False):
        def objective_function(a):
            th = np.arcsin(a**0.5)
            return -self.likelihood(plist, th, log = True)
        
        if plist is None:
            plist = self.probability_list(self.nshots)
            
        if MLE:
            # Perform MLE on the results.
            rng = self.search_range_from_plist(plist)
            rngs = np.array([rng], dtype=object)
            a_est = opt.brute(objective_function, ranges=rngs, Ns=Nevals, 
                                      finish = True)
        else:
            # Output the result produced most frequent outcome, as in 
            # conventional QAE.
            largest_tuple = k_largest_tuples(plist, k=1, sortby=1)[0]
            most_freq_outcome = largest_tuple[0]
            a_est = self.amp_from_outcome(most_freq_outcome)
        
        if not self.silent:
            method = ("(MLE by grid + Nelder-Mead)" 
                      if MLE else "(conventional QAE)")
            print(f"> Determined amplitude: a={round(a_est,4)} {method}. "
                  "[QAEemulator.estimate]")
        
        if MLE and ret_range:
            return a_est, rng
        return a_est
    
    def search_range_from_plist(self, plist):
        '''
        Finds upper and lower bounds for the search range of the MLE. They are
        the 2 amplitudes representable in 2^m which are closest to the
        canonical QAE estimate sin^2(pi*x/M), i.e. the contiguous grid points. 
        '''
        # Find the most frequent outcome.
        largest_tuple = k_largest_tuples(plist, k=1, sortby=1)[0]
        most_freq_outcome = largest_tuple[0]
        # Calculate the points halfway between "grid point" estimates from 
        # canonical QAE: the ones immediately to the left and right of the mode.
        if most_freq_outcome > 0:
            b1 = self.amp_from_outcome(most_freq_outcome-1)
        else:
            b1 = self.amp_from_outcome(most_freq_outcome)
        if most_freq_outcome < 2**self.m-1:
            b2 = self.amp_from_outcome(most_freq_outcome+1)
        else: 
            b2 = self.amp_from_outcome(most_freq_outcome)
        # If outcome<M/2, theta is in [0,pi/2[ so b1 is the lower estimate: 
        # smaller argument, smaller sine. If outcome>M/2, theta is in ]pi/2,pi]
        # and the opposite happens. If outcome = pi/2, they're the same.
        lb = min(b1,b2)
        hb = max(b1,b2)
        return lb, hb
    
    def search_range_from_plist_wrong(self, plist):
        '''
        Finds upper and lower bounds for the search range of the MLE. They are
        the 2 amplitudes representable in 2^m which are closest to the
        canonical QAE estimate sin^2(pi*x/M), i.e. the contiguous grid points. 
        '''
        # print(plist)
        # plist = [(self.amp_from_outcome(o),f) for o,f in plist]
        # print(plist)
        pdict = outcome_dist_to_dict(plist, 
                                     fun = lambda o: self.amp_from_outcome(o))
        amps = list(map(tuple, pdict.items()))
        
        most_freq_outcomes = k_largest_tuples(amps, k=2, sortby=1)
        b1, b2 = most_freq_outcomes[0][0], most_freq_outcomes[1][0]
        lb = min(b1,b2)
        hb = max(b1,b2)
        return lb, hb
        
    def clean_graph(self):
        fig, ax = plt.subplots(1, figsize=(12,8))
        self.ax = ax
        self.ax2 = None

    def probability_of(self, y, theta = None):
        '''
        Calculate the probability of outcome y given angle theta. If theta is
        None, the 'theta_real' attribute is used.
        In QAE we phase-estimate one of two angles (with equal probabilities): 
        2*theta and 2*pi-2*theta, in exp(i*angle). The corresponding 'phis' in 
        exp(i*2*pi*theta) are theta/pi, and (pi-theta)/pi.
        The thetas are associated with the eigenvalues of the Grover operator, 
        theta being the Grover angle.
        '''
        if theta is None:
            theta = self.theta_real
        M = 2**self.m
        phi0 = theta/np.pi
        phi1 = 1 - phi0
        p0 = QPE_probability_of(y, phi0, M)
        
        # Unnecessary, same result.
        # if self.a == 0 or self.a == 1:
        #     return p0
        
        p1 = QPE_probability_of(y, phi1, M)
        p = (p0+p1)/2
        return p

    def probability_list(self, nshots):
        '''
        Returns a list of tuples (outcome, probability) with the probability of 
        QAE outputting the binary encoding of 'outcome' given the real phase  
        theta in exp(i*2*theta). If nshots, shot noise is introduced using 
        multinomial sampling, the probabilities are replaced by numbers of 
        occurences. 
        '''
        M = 2**self.m
        plist = [(y,self.probability_of(y)) for y in range(M)]
        if not self.silent:
            print("> Computed list of (outcome, exact probability) tuples.")
        
        if nshots:
            ps = np.random.multinomial(nshots, [p for o,p in plist])
            plist = list(zip([o for o,p in plist], ps))
            if not self.silent:
                print(f"> Generated {nshots} samples using a"
                      " multinomial distribution.")
        return plist

    def measurement_data(self, plist, xaxis):
        '''
        Returns a dictionary whose keys are QAE outcomes and whose values are 
        either probabilities (if nshots is None) or numbers of occurences among 
        nshots.
        '''
        if xaxis=="outcomes":
            # Use the binary outcomes as xaxis for the histogram.
            pdict = outcome_dist_to_dict(plist)
        elif xaxis=="a":
            # Use the amplitudes as keys. We don't know if we're measuring 
            # theta = M*pi/M or pi-M*pi/M, but it doesn't matter because they
            # result in the same amplitude (squared sine).s
            pdict = outcome_dist_to_dict(
                plist, lambda o: self.amp_from_outcome(o, rounding = True))
        else:
            raise ValueError("xaxis string must be either 'a' or 'outcomes'.")
        return pdict

    def barplot_and_likelihood(self, log = False, plist = None,
                               ret = "plist"):
        logstr = "log-" if log else ""
        self.ax.set_title(f"QAE: bar chart and {logstr}likelihood", 
                          fontsize=18, pad=15)
        
        plist = self.barplot(self.nshots, xaxis = "a", plist = plist)
        a_est = self.plot_likelihood(plist, log)
        self.ax.axvline(self.a, linestyle="dashed", color="black",
                        label="exact amplitude")
        
        # Join together the legends of ax1 and ax2.
        lines, labels = self.ax.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax2.legend(lines + lines2, labels + labels2, 
                        loc="upper right", fontsize=14, framealpha=0.8)
        if ret=="plist":
            return plist
        if ret=="MLE":
            return a_est
    
    def barplot(self, xaxis, plist = None):
        '''
        'xaxis' defines the horizontal labels for the histogram: the binary 
        outcomes if 'outcomes', their associated amplitudes if 'a'. 
        '''
        # Title the figure if it hasn't been titled already, e.g. by
        # barplot_and_likelihood.
        if self.ax.get_title()=='':
            self.ax.set_title("QAE: bar chart", fontsize=18, pad=15)
        
        if plist is None:
            plist = self.probability_list(self.nshots)
            
        res_dict = self.measurement_data(plist, xaxis)
        barplot_from_data(res_dict, self.ax,  
                          label = "relative frequency")
        self.ax.set_xlabel("amplitude" if xaxis=="a" else "outcome", 
                           fontsize=16, style="italic", labelpad=10)
        self.ax.set_ylabel("relative frequency", fontsize=16, style="italic", 
                           labelpad=10)
        plt.xticks(rotation=45, ha="right")
        return plist
    
    def plot_hist_Qiskit(self, xaxis):
        plist = self.probability_list(self.nshots)
        res_dict = self.measurement_data(plist, xaxis)
        rem_negligible_items(res_dict)
        plot_histogram(res_dict, color="lightgray", figsize=(11.5,8), 
                       ax=self.ax)
        plt.title("Analytical results - QAE", fontsize=14, pad=15)
        
        # Print the most likely outcome.
        mostfreq = max(res_dict, key=res_dict.get)
        print(f"> The most frequent result was {mostfreq}. [plot_hist_Qiskit]")
        return plist
        
    def likelihood(self, plist, theta, log):
        Ls = []
        for o, hits in plist:
            Ls.append(hits*np.log(self.probability_of(o, theta=theta)))
            
        # To avoid VisibleDeprecationWarning.
        # Ls = np.array(Ls)#, dtype=object)
        # L = np.sum(Ls)
        L = sum(Ls)
        if not log:
            L = np.exp(L)
        return L

    def amp_from_outcome(self, o, rounding = True):
        th = np.pi*o/2**self.m
        # Complementary angle due to oracle implementation.
        a = np.sin(th)**2
        if rounding: 
            a = round(a,4)
        return a
    
    def plot_likelihood(self, plist, log, Npoints = 5000, 
                        reference_point = None):
        # Place the grid on [0,pi] since the amplitude is the same for theta
        # vs. 2pi-theta
        
        thetas = np.linspace(0, np.pi, Npoints)[1:]
        amps = [np.sin(theta)**2 for theta in thetas]

        Ls = [self.likelihood(plist, theta, log) for theta in thetas]

        imax = np.argmax(Ls)
        a_est, rng = self.estimate(plist = plist, ret_range = True)
        
        # For the loglikelihood, plot raw values. For the likelihood, normalize.
        if not log:
            # Normalize for plotting: P(D) \approx V/M* \sum Ls
            Z = sum(Ls)/Npoints
                
            for i,_ in enumerate(Ls):
                Ls[i] = Ls[i]/Z

        self.ax2 = self.ax.twinx()
        label = "log-likelihood" if log else "likelihood"
        self.ax2.plot(amps, Ls, linewidth = 1.5, color='black', label = label) 
        self.ax2.scatter(a_est, Ls[imax], label = "MLE", marker="*", s=15**2, 
                    color="crimson")
        self.ax2.set_ylabel(label, fontsize=16, style="italic", labelpad=10)
        if log:
            self.ax2.set_ylim(-400)
            
        # Shade the search range.
        xs = np.linspace(*rng, 10)
        self.ax.fill_between(xs, y1=0, y2=1, 
                              color="crimson", alpha=0.1, 
                              transform=self.ax.get_xaxis_transform())
            
        return a_est
    
    @staticmethod
    def Nq_from_m(m, nshots, b10 = False):
        '''
        Number of queries to A, as a function of m.
        If expb10, a number x is returned such that 10**x approximates Nq.
        '''
        # Nq = 2*M+1 for each circuit execution (to A; M queries to the oracle).
        Nq_per_shot = 2*(2**m)+1
        Nq = Nq_per_shot*nshots
        if b10:
            return expb10(Nq)
        else:
            return Nq
    
    @staticmethod
    def m_from_Nq(Nq_target, nshots):
        '''
        Return the number of Grover iterations 'm' that will result in a number
        of queries closest to Nq_target, should 'nshots' shots be used per 
        circuit.
        '''
        # Nq = 2*M+1 for each circuit execution (to A; M queries to the oracle).
        m_float = np.log2(Nq_target/nshots - 1) - 1
        # Actual number of iterations must be a float.
        m_closest = int(round(m_float))
        return m_closest
    
class TestQAE():
    def __init__(self, a, m, nshots, silent = False):
        self.a = a
        self.m = m
        self.nshots = nshots
        self.silent = silent
                
    def estimation(self, MLE, complementary = False):
        '''
        Canonical QAE, eventually enhanced with MLE. 
        
        If 'complementary', use the complementary angle. Due to the oracle
        circuit, my gate-based QAE learns the complementary angle.
        '''
        theta = np.arcsin((self.a)**0.5)
        theta = np.pi/2 - theta if complementary else theta

        QAEe = QAEemulator(theta, self.m, self.nshots, show = False)
        QAEe.estimate(plist = None, MLE = MLE)
        
    def barplot(self, complementary = False, Qiskit = False):
        '''
        Produce barplot(s) using the QAE emulator. Either make a Qiskit 
        "histogram", or 4 different customized bar plots:
        - With the outcomes as x labels;
        - With the amplitudes as x labels;
        - With the amplitudes as x labels, juxtaposed with the likelihood;
        - With the amplitudes as x labels, juxtaposed with the loglikelihood.
        
        Note that it doesn't make sense to plot the likelihood over a barplot
        with outcomes as labels. The xaxis must be a parameter. 
        
        Also, Qiskit's histogram maker can convert labels to amplitudes too, 
        but it always spaces themout evenly, so the amplitudes look weird and 
        plotting the likelihood on top doesn't look very good.
        
        If 'complementary', use the complementary angle. Due to the oracle
        circuit, my gate-based QAE learns the complementary angle.
        '''
        theta = np.arcsin((self.a)**0.5)
        theta = np.pi/2 - theta if complementary else theta
                                
        QAEe = QAEemulator(theta, self.m, self.nshots)
        
        if Qiskit:
            QAEe.plot_hist_Qiskit(xaxis = "outcomes")
        else:
            plist = QAEe.barplot(xaxis = "outcomes")
            QAEe.clean_graph()
            plist = QAEe.barplot(xaxis = "a")
            QAEe.clean_graph()
            QAEe.barplot_and_likelihood(plist = plist)
            QAEe.clean_graph()
            QAEe.barplot_and_likelihood(log = True, plist=plist)
            
    def sqe_evolution_multiple(self, mmin, Nq_target, nruns, MLE = True, save = True):
        '''
        Run canonical QAE (eventually enhanced with MLE) several times for each
        'm' in a sequence of 'ms', to get the root mean square error (RMSE) as
        a function of 'm' - and thus of the number of queries, since Nq=f(m).

        This implementation is somewhat opposite to other QAE algorithms.
        The others do independent runs under the same circumstances, producing
        a list whose elements are lists of estimation errors for specific runs.
        The outter list is over the different runs. The inner lists are 
        sqe_by_run.
        Here we produce a list whose elements are lists of estimation errors 
        for specific Nqs. The outter list is over the different Nqs. The inner
        lists are sqe_by_step (step = Nq).
        This could be made more consistent by instead of doing nruns for each m,
        doing nruns each of which would be a sequence of executions of QAE for 
        each m. Then the results could be treated as in MLAE.
        But this is less natural, because in MLAE intermediate results are 
        available within the run for continuous Nqs(progressive learning), 
        whereas in QAE one must run the algorithm completely for each m and then
        start over. 
        But it would have 2 benefits: one, more accurate and compact progress 
        bar. Two, interrupting execution could still produce results with a 
        smaller number of runs.
        '''
        def print_info():
            info = ["Canonical QAE"]
            info.append("- scaling of the estimation error with Nq")
            info.append(f"a={self.a} | runs = {nruns} | nshots = {self.nshots}"
                        f" | MLE: {'Yes' if MLE else 'No'}")
            info.append(f"m={{{mmin}..{mmax}}} → Nq in [10^"
                        f"{QAEemulator.Nq_from_m(mmin, self.nshots, b10 = True)}, 10^"
                        f"{QAEemulator.Nq_from_m(mmax, self.nshots, b10 = True)}] "
                        f"(target max is 10^{expb10(Nq_target)}).")
            if not self.silent:
                print_centered(info)
        
        assert mmin>=0, "negative number of qubits makes no sense"
        
        mmax = QAEemulator.m_from_Nq(Nq_target, self.nshots)
        print(f"> Will test {nruns} runs of 'canonical QAE'.")
        if not self.silent:
            print_info()
        
        ms = list(range(mmin,mmax+1, 2))
        nqs = [QAEemulator.Nq_from_m(m, self.nshots) for m in ms]
        sqe_by_step = []
        for m in ms:
            try:
                self.m = m
                print(f"> Testing {nruns} runs of QAE with {self.m} aux qubits...")
                sqes = self.sqes_given_m(nruns, MLE)
                sqe_by_step.append(sqes)
            except KeyboardInterrupt:
                print(f"\n> Keyboard interrupt at m={m}. "
                      "Will present partial results if possible.")
                print("> Breaking from cycle... [QAE_sqe_scaling]")
                mmax = m
                break
            
        if mmax == mmin:
            return
        
        estdata = EstimationData()
        estdata.add_data("canonical", nqs = nqs, lbs = None, errs = sqe_by_step)
        ed = ExecutionData(self.a, estdata, nruns, self.nshots, 
                           label = "canonical_QAE_"
                           f"{'MLE' if MLE else 'conv'}_multiple",
                           extra_info = f"m={{{mmin}..{mmax}}}")
        if save:
            ed.save_to_file()
            
        if all([e==0 for e in sqes]):
            print("> All errors are 0, so no plots for you! I would guess you "
                  "ran an exact case of conventional QAE, e.g. a=0.5 for m>1.")
        else:
            process_and_plot(estdata, processing = "averaging2", save = save)
            # plot_err_evol("RMSE", estdata, exp_fit = False)
    

    
    def sqes_given_m(self, nruns, MLE):
        '''
        Run canonical QAE (eventually enhanced with MLE) several times for 
        fixed 'm' (the current self.m), to get the root mean squared error.
        
        If self.a is 'rand', the amplitude is picked at random. If not, self.a
        is used direcly.
        '''
        sqes = [] 
        pb = ProgressBar(nruns)
        for i in range(nruns):
            pb.update()

            theta_real = np.arcsin((a := self.local_a)**0.5)
            QAEe = QAEemulator(theta_real, self.m, self.nshots, 
                               silent = True, show = False)
            # MLE = QAEe.barplot_and_likelihood(log=True, ret ="MLE")
            a_est = QAEe.estimate(MLE = MLE)
            sqe = (a_est/a-1)**2
            # print("as", a_est, a, sqe)
            sqes.append(sqe)
            
        return sqes
    
    @property
    def local_a(self):
        '''
        For each run, the real amplitude parameter will be 'local_a'.
        
        The 'a' attribute is always constant, and can hold:
            
        - A permanent value for 'a'. In that case, all runs will use it;
        'local_a' is equivalent to 'a'. 
        
        - The string "rand". In that case, each run will sample an amplitude
        at random, which in general will differ between runs. 
        '''
        if isinstance(self.a, tuple):
            amin, amax = self.a
            a = np.random.uniform(amin,amax)
        else:
            a = self.a
        return a

def test(which):
    if which==0:
        '''
        Emulate a QAE bar plot, Qiskit style and with binary outcomes as 
        labels.
        '''
        a = 1/9
        m = 2
        nshots = None # None for exact probabilities.
        Test = TestQAE(a, m, nshots)
        Test.barplot(Qiskit = True)
    elif which==1:
        '''
        Test all (4) possible combinations of customized barplots. Se method
        for details.
        '''
        a = 0.3
        m = 3
        nshots = 50
        Test = TestQAE(a, m, nshots)
        Test.barplot()
    elif which==2:
        '''
        Plot a table with QAE results for varying size of the QFT register.
        '''
        estimation_table()
    elif which==3:
        '''
        Perform QAE and get the estimator by MLE.
        '''
        a = 1/9
        m = 2
        nshots = 100
        # Whether to use MLE on the QAE outcomes or the conventional approach.
        MLE = True
        Test = TestQAE(a, m, nshots)
        Test.estimation(MLE)
        #test_QAE_estimation(nshots)
    elif which==4:
        '''
        Compute the mean squared error over multiple runs with the same 'm'
        and either a selected 'a' or random ones. 
        '''
        # a can be a float, or 'rand' for picking 'a' at random for each run.
        a = 0.5 
        m = 4
        nshots = 100
        nruns = 2
        MLE = True
        Test = TestQAE(a, m, nshots)
        mse = Test.sqe_evolution_avg(nruns, MLE)
        print(f"> MSE ({nruns} runs): ", mse)
    elif which==5:
        '''
        Compute the mean squared error over multiple runs for a sequence of 
        (increasing) values of 'm' and either a selected 'a' or random ones. 
        Plot the scaling of the estimation errors with the number of queries.
        '''
        # a can be a float, or 'rand' for picking 'a' at random for each run.
        a =  (0,1)
        mmin = 2
        Nq_target = 10**6
        nruns = 100
        nshots = 100
        Test = TestQAE(a, mmin, nshots)
        Test.sqe_evolution_multiple(mmin, Nq_target, nruns, MLE = True, save = True)
        
if __name__ == "__main__":
    #test(5)
    print(QAEemulator.m_from_Nq(1e6, 1))