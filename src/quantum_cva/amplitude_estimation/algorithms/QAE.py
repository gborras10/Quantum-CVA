'''
Algorithms for quantum amplitude estimation. The QAE class implements the 
canonical approach. The BayesianQAE class is used for MLQAE, and Bayesian 
inference based algorithms.
'''
import sys
import importlib

try:
    from google.colab import files
    using_colab = True
except:
     using_colab = False

if using_colab:
    # Mount drive (left menu, third icon) before running for this to work.
    sys.path.append('/content/drive/Othercomputers/Laptop/Phd/3.Scripts') 

# from qiskit import Aer, execute
# from qiskit.circuit.library import QFT
# from qiskit import QuantumCircuit
# from qiskit.visualization import plot_histogram   

# When running on Google Colab, modules must be explicitly reloaded to get the
# latest version (or restart the runtime).
reload = False
if reload and using_colab:
    importlib.reload(sys.modules["utils.quantum_ops"])
    importlib.reload(sys.modules["utils.misc"])
  
import math, numpy as np
import scipy.optimize as opt
try: 
    from IPython.display import display
except:
    pass
    
# Custom modules.
from quantum_cva.amplitude_estimation.algorithms.QAA import QAA, CheckingFunction
from quantum_cva.amplitude_estimation.utils.plotting import plot_graph, plot_warn
from quantum_cva.amplitude_estimation.utils.misc import binary_to_decimal 

ndigits = 5

first_maximize_likelihood = True
class QAE(QAA):
    '''
   ============================================================================
    Base class for quantum amplitude estimation. It implements the standard
    fault-tolerant algorithm, which estimates the phase θ=arcsin(a^0.5) using
    QPE [1].

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
    aux_qs: int
        The number of qubits for the auxiliary register. This determines 
        the precision digits in the output: they are precisely 'aux_qs'. This
        also determines the QFT order as QFT_2^aux_qs, as well as the number
        of (controlled) Grover operator  applications which is 'aux_qs'
        "best guess" probability distribution.
   ============================================================================
   '''
    def __init__(self, encoding_qs, aux_qs, function):
        super().__init__(encoding_qs, function, nsol = None,
                         derandomize = False, omit_aux = False)
        self.aux_qs = aux_qs
        # Create a circuit for QAE. The aux_qs register will be acted on by the 
        # QFT and measured into a classical register.
        self.circuit = QuantumCircuit(self.n_qubits + aux_qs, aux_qs) 
        
    def estimate_amplitude(self, show = False, silent = False):     
        n, aux_qs = self.n_qubits, self.aux_qs
        qlist_aux, qlist_main = [*range(aux_qs)], [*range(aux_qs, aux_qs+n)] 
        self.A(qlist = qlist_main[:-1]) # Last qubit is oracle aux so exclude.
        self.Hadamard_transform(qlist_aux)
        self.controlled_Q_ladder(qlist_aux, qlist_main)
        self.IQFT(qlist_aux)
        
        # Classical register has same indices as qlist_aux.
        meas_info = (qlist_aux, qlist_aux) 
        self.circuit.measure(meas_info[0], meas_info[1])
        if show:
            self.show(fold = 26)
            print("> I would like to let you know I am plotting a figure."
                  " [QAE.estimate_amplitude]")

        counts = self.run_circuit(how = "hist", 
                                  meas_info = [], info=" - QAE")
        a_est = self.amplitude_from_data(counts, silent)
        return a_est
        
    def amplitude_from_data(self, counts, silent):
        outcome1 = max(counts, key=counts.get)
        del counts[outcome1]
        outcome2 = max(counts, key=counts.get)
        theta1, a1 = self.interpret_phase(outcome1)
        theta2, a2 = self.interpret_phase(outcome2)
        assert math.isclose(a1,a2), f"Beware that the two most likely "\
            f"outcomes do not produce consistent estimates for 'a': outcome "\
                f"{outcome1} yields a={a1} but outcome {outcome2} yields a={a2}."
        t_est = a1* 2**self.encoding_qs
        
        if not silent:
            theta_round = round(np.abs(np.pi/2-theta1),ndigits)
            print(f"> Estimated θ≈{theta_round}, meaning "
                  f"a≈{round(a1,ndigits)} and t={round(t_est)}={[t_est]}. "
                  "[QAE.amplitude_from_data]")   
        return a1
        
    def interpret_phase(self, outcome):
        # Compute initial probability of success (i.e. the squared amplitude of  
        # A|0>'s projection onto the 'good' subspace, and re
        
        # Number of counting qubits = log_2 of the estimate precision digits. 
        m = len(outcome) 
        r = binary_to_decimal(outcome)
        # Phase estimation gives x=ϕ*2^m where the eigenvalue is e^{i2πϕ},
        # and we're estimating e^{i2θ}.
        theta = np.pi*r/2**m
        # In [1], a=sin(θ)^2. Here a is the complementary probability because 
        # the implemented operator is symmetric relative to their definition.
        # We do not implement the minus sign in Q = -AS_0A†S_f, so it works 
        # as though as we're inferring the complementary angle to theta. See
        # blog post from 10-11 May 2022 or PhD notebook p. 17.
        a = np.cos(theta)**2 
        return theta, a
    
    def controlled_Q_ladder(self, controls, target_reg):
        for i,c in enumerate(controls):
            for j in range(2**i):
                self.cQ([c], target_reg)
        
    def cQ(self, c, ts):
        Q = self.get_gate(rep = 1, include_init = False)
        cQ = Q.control()
        self.circuit.append(cQ, [c] + ts)
        
    def IQFT(self, qlist):
        self.QFT(qlist, inverse = True)
    
    def QFT(self, qlist, inverse = False):
        QFTgate = QFT(len(qlist), inverse = inverse)
        self.circuit.append(QFTgate, qlist)
        
    def run(self, info = None):
        sim = Aer.get_backend('aer_simulator')
        result = execute(self.circuit, sim).result()
        counts = result.get_counts()
        ttl = "Aer simulation results" + ("" if info is None else f" - {info}")
        hist = plot_histogram(counts, 
                               color='lightgray', figsize=(11.5,8))
        hist.suptitle(ttl, y=0.93, fontsize=18)
        display(hist)
        
first_BayesianQAE = True
class BayesianQAE(QAA):     
    def maximize_likelihood(self, ms, hs, finish = True, evals = 1e3,
                            silent = False):
        def objective_function(theta):
            # For scipy.optimize.brute, the objective function must have the 
            # free parameter as first argument, and the target should be the
            # minumum rather than the maximum. Also, using the loglikelihood
            # has some numerical advantages.
            return -self.loglikelihood(theta, ms, hs)
        global first_maximize_likelihood
        if first_maximize_likelihood and not silent:
            print(f"> Using brute force optimization on {int(round(evals))} "
            "grid points " + ("+ Nelder-Mead " if finish else "") 
            + "for finding the MLE. [BayesianQAE.maximize_likelihood]")
            first_maximize_likelihood = False
        
        interval = (0, np.pi/2)
        # 'Finish': whether to improve the result using Nelder–Mead.
        result = opt.brute(objective_function, [interval], Ns=evals, 
                           finish=opt.fmin if finish else None)
        theta_opt = result[0] if finish else result
        # print(np.sin(theta_opt)**2)
        return theta_opt
    
    def numpy_sampling(self, amp, ms, Tc = None):
        theta = np.arcsin(np.sqrt(amp))
        
        hs = []
        for m in ms:
            arg = (2*m+1)*theta
            p1 = np.sin(arg)**2
            
            global first_BayesianQAE
            if Tc is not None:
                exp = np.exp(-m/Tc)
                p1 = exp*p1 + (1 - exp)/2
                
                if first_BayesianQAE:
                    print(f"> Using discrete coherence time Tc = {Tc}."
                          " Please change this into using the model class soon."
                          " [BayesianQAE.numpy_sampling]")
                    first_BayesianQAE = False
            
            h = np.random.binomial(self.nshots, p1)
            hs.append(h)
        return hs
    
    def create_circuits(self, ms, check = "classical", show = False):
        '''
        'check': how to check whether the state measured after the applications
        of Q is a solution. If 'classical', measure on the computational basis
        feed the bit string as an input to the checking function, and take the
        output (0/1). If 'oracle', apply a standard oracle, measure the 
        auxiliary qubit (last), and take the single bit (0/1).
        '''
        print(f"> Outcome checking strategy: {check}. [MLQAE.create_circuits]")
        
        qc_list = []
        for k in range(self.Ncircs):
            qc = QuantumCircuit(self.n_qubits, 
                                self.encoding_qs if check=="classical" else 1) 
            self.A(qc=qc)
            self.append_Q(qc, times = ms[k])
            if check == "classical":
                main_indices = list(range(self.encoding_qs))
                qc.measure(main_indices, main_indices)
            if check == "oracle":
                self.Uf(qc)
                qc.measure(self.n_qubits-1, 0)
            qc_list.append(qc)
        if show:
            self.plot_circs(qc_list)
        return qc_list
    
    @staticmethod
    def count_hits(counts):
        # Count the number of 1 outcomes (measured a good state).
        if isinstance(counts, dict): 
            # Dictionary with the counts of a single executed circuit.
            hs = [counts.get("1",0)]
        elif isinstance(counts, list): 
            # List of dictionaries, each with the counts of 1 circuit.
            hs = [counts_k.get("1",0) for counts_k in counts]
        return hs
        
    def evaluate_counts(self, counts):
        if isinstance(counts, dict): 
            # Dictionary with the counts of a single executed circuit.
            hk = 0
            for key in list(counts.keys()):
                hk += counts[key] if self.function.evaluate(key)==1 else 0
            hs = [hk]
        elif isinstance(counts, list): 
            # List of dictionaries, each with the counts of 1 circuit.
            hs = [self.evaluate_counts(counts_k)[0] for counts_k in counts]
        return hs
    
    def plot_likelihood(self, ms, hs, xcoord = "a", log = True, atol = 1e-6,
                        title = "", xlabel="", ylabel=""):
        def is_significant(l, lmax, log):
            if log:
                return l-lmax > np.log(atol)
            else:
                return abs(l/lmax) > atol
        
        xs = np.linspace(0,np.pi/2,10000)[1:]
        ys = [self.loglikelihood(x, ms, hs)  for x in xs] if log else \
            [self.likelihood(x, ms, hs)  for x in xs]
            
        if all([math.isclose(y,0, abs_tol=1e-300) for y in ys]):
            print("> All the y values you want me to plot are roughly zero. "
                  + ("" if log else "Maybe try your luck with a log scale?")
                  + "[BayesianQAE.plot_likelihood]") 
            return
        # Remove negligible likelihoods (considering scale) in order to
        # close in on the graph.
        ymax = max(ys)
        xs = [xs[k] for k in range(len(xs)) if is_significant(ys[k], ymax,log)]
        ys = [ys[k] for k in range(len(ys)) if is_significant(ys[k], ymax,log)]
        if xcoord == "a":
            xs = np.sin(xs)**2
            
        plot_graph(xs, ys, startat0=None if log else "y",
                    title=title, xlabel=xlabel, ylabel=ylabel)
        
    def likelihood(self, theta, ms, hs):
        def kth_likelihood(k):
            arg = (2*ms[k]+1)*theta
            Lk = np.sin(arg)**(2*hs[k])*np.cos(arg)**(2*(self.nshots-hs[k]))
            return Lk
        Lks = [kth_likelihood(k) for k in range(len(hs))]
        L = np.prod(Lks)
        return L
    
    def loglikelihood(self, theta, ms, hs):
        def kth_loglikelihood(k):
            arg = (2*ms[k]+1)*theta
            lk = 0
            # Use absolute values of log argument because we're bringing down
            # the even exponents.
            
            # If there are no "1" outcomes, the likelihood  associated
            # with them is irrelevant and may cause log(0) issues 
            # unnecessarily. So only compute it if it makes sense.
            if hs[k]!=0:
                lk += 2*hs[k]*np.log(np.abs(np.sin(arg)))
            # Same for "0" outcomes.
            if (self.nshots-hs[k])!=0:
                lk += 2*(self.nshots-hs[k])*np.log(np.abs(np.cos(arg)))
            return lk
        lks = [kth_loglikelihood(k) for k in range(len(hs))]
        #print("lks", lks)
        l = np.sum(lks)
        return l
            
    def Uf(self, qc):
        Uf = self.function.standard_oracle()
        qc.append(Uf, [q for q in range(self.n_qubits)])
            
    def append_Q(self, qc, times = 1):
        # Appends Q to a given circuit, vs. apply_Q which applies it to 
        # self.circuit. 
        Q = self.get_gate(rep=1, include_init=False)
        for i in range(times):
            qc.append(Q, list(range(self.n_qubits)))
            
    @plot_warn
    def plot_circs(self, qc_list):
        for qc in qc_list:
            self.show(qc=qc)
            
class TesterQAE():
    
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
            # print(f"> Sampled a = {a}.")
            return a
        else:
            return self.a

    @property
    def local_Tc(self):
        if isinstance(self.Tc, tuple):
            Tcmin, Tcmax = self.Tc
            Tc = np.random.uniform(Tcmin,Tcmax)
            # print(f"> Sampled Tc = {Tc}.")
            return Tc
        else:
            return self.Tc
        
    @property
    def param_str(self):
        a_str = (self.rand_pstr(self.a) if isinstance(self.a,tuple)
                 else str(self.a))
        Tc_str = (self.rand_pstr(self.Tc) if isinstance(self.Tc,tuple)
                 else str(self.Tc))
        s = f"a={a_str};Tc={Tc_str}"
        return s
    
    @staticmethod
    def rand_pstr(param):
        return f"rand[{param[0]},{param[1]}]"

def test_QAE(simulator = True):
    '''
    Test the canonical quantum amplitude estimation algorithm.
    '''
    counting_qs = 3
    print(f"> Will test canonical quantum amplitude estimation with "
          f"{counting_qs} auxiliary qubits for QPE. [test_QAE]")
    print(f"> QAE | aux_qs = {counting_qs}")
    n, t = 3, 1
    a = t/2**n
    theta = np.arcsin(np.sqrt(a))
    print(f"> n={n} | t={t} | a≈{round(a,ndigits)} | θ≈{round(theta,ndigits)}")
    f = CheckingFunction(n,t)#, ws=[[0,0,0,1],[0,1,0,0]])
    QAE_instance = QAE(n, counting_qs, f)   
    QAE_instance.estimate_amplitude(show=True)
    
# test_QAE()