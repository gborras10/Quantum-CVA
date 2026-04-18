'''
Algorithms related to quantum amplitude estimation (QAA). The CheckingFunction 
class manages a generalized Grover oracle. QAA is a base class for QAA based 
algorithms.
'''
try:
    from google.colab import files
    using_colab = True
except:
     using_colab = False

import numpy as np
use_qiskit = False
if use_qiskit:
  from qiskit import QuantumCircuit
  from utils.quantum_ops import aer_simulate, multi_cz
import random
import sys
import importlib
try: 
    from IPython.display import display
except:
    pass

if using_colab:
    # Mount drive (left menu, third icon) before running for this to work.
    sys.path.append('/content/drive/Othercomputers/Laptop/Phd/3.Scripts') 

from quantum_cva.amplitude_estimation.utils.misc import binary_to_decimal

# When running on Google Colab, modules must be explicitly reloaded to get the
# latest version (or restart the runtime).
reload = False
if reload and using_colab:
    importlib.reload(sys.modules["utils.quantum_ops"])
    importlib.reload(sys.modules["utils.misc"])

ndigits = None

class CheckingFunction:
    '''
   ============================================================================
    Creates and manages a function {0,1,...,2^n} -> {0,1} that partitions a 
    set of strings into 'good' (solutions/winner states such that f(w)=1) and
    'bad' elements. Includes the implementation of standard and phase oracles
    as quantum circuits.


    Attributes:
    ----------
    n: int
        The number of bits.
    t: int
        The number of solutions.
    winners: [[int]]
        The list of solutions (each a list of qubits with little endian
        ordering).
    winners: [[int]]
        The list of solutions (each a list of bits with little endian
        ordering).
   ============================================================================
   '''
    
    def __init__(self, n, t, ws = None, silent = False):
        signature = "[CheckingFunction.__init__]"
        self.n, self.t = n, t
        print("> Please don't forget I'm technically missing a pi phase. "
              "If the measurements aren't as you expect, that's probably why! "
              " The QAE algorithm knows of this, so the estimates should "
              "still be correct.")
        if not silent:
            print(f"> Initializing a checking function for n={n} bits "
                  f"and t={t} solutions picked at random " 
                  f"(a≈{round(t/2**n,ndigits)}).")
        
        if ws is not None:
            # The winner states are given as input; use them.
            t_ = len(ws)
            if t != t_:
                print(f"> Warning: number of winner states given differs from "
                      f"number of solutions. Will replace t={t} with {t_}.")
                t = t_
            for i,w in enumerate(ws):
                s = ''.join(map(str,w[::-1])) # Big endian for s.
                print(f"> Good state {i+1} of {t} (input): {w} = 0b{s} "
                      f"(= {binary_to_decimal(w[::-1])}).")
            self.winners = ws
            if not silent:
                print(signature)
            return
        
        # Draw the winners at random from all possible strings.
        ws = []
        for i in range(t):
            while True:
                w = [random.randint(0,1) for q in range(n)]
                # Qubit list; w[0] considered l.s.b. as per Qiskits' LE choice.
                if w not in ws:
                    break
            if not silent:
                s = ''.join(map(str,w[::-1])) # Big endian for s.
                print(f"> Good state {i+1} of {t} has been drawn: {w} = 0b{s} "
                      f"(= {binary_to_decimal(w[::-1])}).")
                # Little endian order for lists, "big endian" for strings.
            ws.append(w)
        self.winners = ws
        if not silent:
            print(signature)
            
    def evaluate(self, string):
        # Check whether 0b'string' is a solution.
        l = [int(char) for char in string[::-1]]
        if l in self.winners:
            return 1
        else:
            return 0
        
    def standard_oracle(self, show = False):
        # Does Uf|x>|0> -> |x>|f(x)>.
        n_ = self.n + 1 # Add one qubit to write the output to.
        def string_to_ones(qc, w):
            for i in range(self.n):
                if w[i]==0:
                    qc.x(i)
            
        def reflect_ones(qc):
            # Apply X to last qubit, conditionally on all others' being 1. 
            qc.mct(list(range(n_-1)), n_-1)

        qc = QuantumCircuit(n_)
        ws = self.winners
        
        for w in ws:
            string_to_ones(qc, w) 
            reflect_ones(qc)
            string_to_ones(qc, w)
            
        if show:
            display(qc.draw('mpl')) 
            print("> I would like to let you know I am plotting a figure."
                  " [CheckingFunction.standard_oracle]")
        Uf = qc.to_gate(label = "$U_f$")
        return Uf
    
    def test_standard_oracle(self):
# =============================================================================
#       The statevector's rows in which the main register is a solution should  
#       have the auxiliary qubit in state 1. For all others, it should be zero.
# =============================================================================
        n = self.n
        aux_qs = 1 
        n_ = n + aux_qs
        qc = QuantumCircuit(n_)
        for q in range(n):
            qc.h(q)
        Uf = self.standard_oracle(show=True)
        qc.append(Uf, list(range(n_)))
        aer_simulate(qc, 
                     how="statevec",
                     info = " (standard oracle on uniform superposition;"
                     f" the msq should be 1 iff the {n} lsq are a solution)")
    
    def phase_oracle(self, show = False, 
                     extra_literal = False, connective="and"):
# =============================================================================
#         Marks some states, considered solutions, with a π phase (reflection).
#         The default version reflects $n$ qubit states |x> for which f(x)=1.
#         If extra_literal is True, the gate will act an additional qubit q_n.
#         If the connective is "and", it will reflect states satisfying:
#                                   f(x) = 1 ∧ q_n = 1
#         , i.e. it restricts the solutions by imposing an extra condition.
#         If it is "or", it will reflect states satisfying:
#                                   f(x) = 1 ∨ q_n = 0
#         , i.e. it extends the solutions by allowing an alternative criterion.
#         In general, only the "and" version is pertinent. The "or" idea is
#         just for testing; see blog post from 10-11/05/22, which also explains
#         the assymetry in q_n == 0/1.
# =============================================================================
            
        # Function evaluation qubits.
        n = self.n 
        # We need at least one extra qubit to implement the phase kickback, and
        # another if an extra literal is to be considered. Their indices will
        # be n_1-1 and n_-2 respectively (equivalently n+1 and n).
        # If connective=="and", this difference in n_ is all that's needed to 
        # enforce the extra condition.
        # add = 2 if extra_literal else 1 
        # new version: no extra qubit, consider phase gate directly 2024
        add = 1 if extra_literal else 0
        n_ = n + add
        def string_to_ones(qc, w):
            # Transforms binary strings we want to reflect into the |1>^n 
            # state, so that a phase can be added to them by phase kickback
            # using a mCX.
            for i in range(n):
                if w[i]==0:
                    qc.x(i)
            # The extra_literal qubit will also be a control for the mCX, but
            # we don't need to act on that qubit here. If the connective is 
            # "and", we precisely want to condition on 1. If it is "or", we'll
            # still control the mCX on q_n==1: if it is 0, we won't mark the
            # states yet, but rather handle them separately by adding a phase
            # whenever q_n=0 (using a simple CX). Otherwise we will have  
            # clashing reflections, and 2 reflections make identity.
            
        def reflect_ones(qc):
            # Initialize msb (aux for phase kickback) to |->.
            # qc.x(n_-1); qc.h(n_-1)
            # Apply X to last qubit, conditionally on all others' being 1. This
            # includes the extra_literal qubit if applicable, since in that 
            # case n_-2 is its index. 
            # qc.mct(list(range(n_-1)), n_-1) 
            #qc.h(n_-1); qc.x(n_-1)
            # changed 2024
            qc.mcp(np.pi, [i for i in range(n_-1)], n_-1)
            

        qc = QuantumCircuit(n_)
        ws = self.winners
        
        for w in ws:
            string_to_ones(qc, w) 
            reflect_ones(qc)
            string_to_ones(qc, w)
            
        if extra_literal and connective=="or": 
            # Mark as solutions all states s.t. q_n is |0>.
            qc.x(n_-2)
            qc.x(n_-1); qc.h(n_-1)
            qc.cx(n_-2,n_-1)
            qc.h(n_-1); qc.x(n_-1)
            qc.x(n_-2)
            
        if show:
            display(qc.draw('mpl')) 
            print("> I would like to let you know I am plotting a figure."
                  " [CheckingFunction.phase_oracle]")
        Sf = qc.to_gate(label = "$S_f$")
        return Sf
    
    def test_phase_oracle(self, extra_literal = False):
# =============================================================================
#       The statevector's rows corresponding to good solution(s) should have 
#       opposite signs to the bad ones.
# =============================================================================
        n = self.n
        aux_qs = 2 if extra_literal else 1 
        n_ = n + aux_qs
        qc = QuantumCircuit(n_)
        for q in range(n):
            qc.h(q)
        if extra_literal:
            qc.h(n) # Get the extra condition qubit on a superposition to test.
        Sf = self.phase_oracle(extra_literal = extra_literal, show = True)
        qc.append(Sf, list(range(n_)))
        aer_simulate(qc, 
                     how="statevec",
                     omit = [(0,'0')], # Keep extra_literal qubit for assessment.
                     info = " (phase oracle on uniform superposition; there "
                     f"should be a minus sign iff the {n} lsq are a solution)")

    def brute_force(self, print_info = True):
        i = 0; n = self.n
        while True:
            attempt = [random.randint(0,1) for q in range(n)]
            if print_info:
                print(f"> Brute force attempt {i}: {attempt}", end="")
            output = self.evaluate(attempt)
            if output == 1:
                print(" <- Sucess." if print_info else "")
                break
            print("", end="\n")
            i += 1
        return attempt

class QAA:
    '''
    ============================================================================
    Base class for quantum amplitude amplification (QAA) [1] based algorithms. 
    Includes quantum circuit implementations for all relevant Grover 
    operators (namely the 2 key reflections). Also supports derandomization 
    features for Grover search.

    Attributes
    ----------
    encoding_qs: int
        The number of qubits required to represent prospective solutions.
    function: CheckingFunction
        The target function, for which we mean to find inputs that
        evaluate to 1.
    nsol: int or None, optional
        The number of solutions, or None if this is not known. 
    derandomize: bool, optional
        Whether to apply derandomization using an auxiliary qubit. Only works 
        if the required gates 'B' have been defined. 
    n_qubits: int
        The number of qubits for the circuit. This is the number of qubits
        required for prospective solutions + at least 1 aux qubit for the
        phase oracle (2 if derandomizing; the same effect could be achieved
        without the extra qubit by generalizing Q as per [1], but that would
        require solving a trigonometric equation).
    omit_aux: bool
        Whether to omit the auxiliary qubits from the measurement results,
        leaving only the main register outcomes. 
    Agate: Gate
        The superposition-creating operator, to be used for initialization 
        as well as when constructing the Grover operator. It represents a 
        "best guess" probability distribution.
    Ainvgate: Gate
        The inverse of Agate.
   ============================================================================
   '''
    def __init__(self, encoding_qs, function, 
                 nsol = None,
                 derandomize = False, 
                 omit_aux = True):
        self.encoding_qs = encoding_qs
        self.function = function
        self.nsol = nsol
        if derandomize and nsol is None:
            print("> Derandomization is not possible when the number of "
                  "solutions is not given. ")
        self.derandomize = derandomize and nsol is not None
        self.n_qubits = encoding_qs + (1 if self.derandomize else 0) 
        self.Agate, self.Ainvgate = self.create_As()
        
        if omit_aux:
            # omit msb (aux for phase oracle) from results if in state '0'
            # (if '1' there's some problem).
            # Likewise, omit derandomizing qubit if in the good state ('1').
            # Supression will act on binary strings, so it's not little endian.
            # To omit msb, use index 0.
            phase_aux_ind, derand_aux_ind = 0, 1
            self.omit = [(phase_aux_ind,'0')] + \
                ([(derand_aux_ind,'1')] if self.derandomize else []) 
        else:
            self.omit = None
    
    def reflect_w(self, barriers = True):
        n, f, circuit = self.n_qubits, self.function, self.circuit
        Sf = f.phase_oracle(extra_literal = self.derandomize)
        circuit.append(Sf, [q for q in range(n)])
        if barriers:
            self.barrier()  
    
    def reflect_A(self, barriers = True):
        # Reflects A|0>.
        self.Ainv()
        if self.derandomize:
            self.Binv()
        if barriers:
            self.barrier()
        self.reflect_0()
        if barriers:
            self.barrier()
        self.A()
        if self.derandomize:
            self.B()
        if barriers:
            self.barrier()
                
    def reflect_0(self, show = False):
        circuit = self.circuit
        qs = self.encoding_qs + (1 if self.derandomize else 0)
        # The auxiliary derandomizing qubit must be reflected too.
        qc = QuantumCircuit(qs)
        
        for q in range(qs):
            qc.x(q)
        
        # qc.append(multi_cz(qs), [q for q in range(qs)])
        qc.mcp(np.pi, [i for i in range(qs-1)], qs-1)
        
        for q in range(qs):
            qc.x(q)
            
        if show:
            display(qc.draw('mpl'))
            print("> I would like to let you know I am plotting a figure."
                  " [QAE.reflect_0]")
            
        S0 = qc.to_gate(label="$S_0$")
        circuit.append(S0, [q for q in range(qs)])
           
    def apply_Q(self, barriers = True):
        self.reflect_w(barriers = barriers)
        self.reflect_A(barriers = barriers)
        
    def return_Q(self):
        qc = self.circuit.copy() # Save the current circuit.
        self.circuit = QuantumCircuit(self.n_qubits) # Create new circuit.
        self.apply_Q(barriers = False) # Build Q on self.circuit.
        Q = self.circuit.to_gate(label="Q") # Export circuit to gate.
        self.circuit = qc # Restore the original circuit.
        return Q
    
    def iterate_Q(self, rep, barriers = True):
        for i in range(rep):
            self.apply_Q(barriers = barriers)
    
    def initialize(self, barriers = True):
        self.A()
        if self.derandomize:
            self.B()
        if barriers:
            self.barrier()
            
    def A(self, qc = None, qlist = None):
        en = self.encoding_qs
        qc = self.circuit if qc is None else qc
        qlist = [q for q in range(en)] if qlist is None else qlist
        qc.append(self.Agate, qlist)
        
    def Ainv(self):
        en = self.encoding_qs
        self.circuit.append(self.Ainvgate, [q for q in range(en)])
        
    def create_As(self, type = "Hadamard"):
        en = self.encoding_qs
        qc = QuantumCircuit(en)
        if type == "Hadamard": 
            self.Hadamard_transform([*range(en)], qc)
        else:
            print(f"> The requested type {type} is not defined. "
                  "[QAA.create_As]")
        A = qc.to_gate(label="A")
        Agate = A
        Ainvgate = A.inverse(); Ainvgate.label = "$A^{-1}$"
        return Agate, Ainvgate
    
    def Hadamard_transform(self, qlist, qc = None):
        qc = self.circuit if qc is None else qc
        for i in qlist:
            qc.h(i) 
            
    def single_iter_circ(self):
        qc = QuantumCircuit(self.n_qubits)
        unitary = self.get_gate(rep = 1)
        qc.append(unitary, list(range(self.n_qubits)))
        return qc
            
    def get_gate(self, rep = 1, include_init = True): 
        # m is the intended number of applications of Q.
        # Always includes application of Q, may include initialization.
        qc = self.circuit.copy() # Save the current circuit.
        self.circuit = QuantumCircuit(self.n_qubits) # Create new circuit.
        if include_init:
            self.initialize(barriers = False)
        self.iterate_Q(rep, barriers = False) # Build Q on self.circuit.
        C = self.circuit.to_gate(label="QAA" if include_init else "Q") 
        # Export circuit to gate.
        self.circuit = qc # Restore the original circuit.
        return C
   
    def barrier(self):
        self.circuit.barrier()
         
    def run_circuit(self, how = "hist", info = None, 
                    meas_info = None):
        circuit, n = self.circuit, self.n_qubits
        if self.omit is not None:
            # Convert indices to little endian when printing for consistency.
            omit_clauses = ['q' + str(n-cond[0]-1) + ' = \'' + cond[1] + '\''
                            for cond in self.omit]
            omit_clauses = ', '.join(omit_clauses)
            print(f"> Warning: will omit qubit(s) {omit_clauses} when "
                  "presenting results. [QAA.run_circuit]")
            
        counts = aer_simulate(circuit, 
                              how = how,
                              meas_info = meas_info,
                              info = info,
                              omit = self.omit)
        return counts
    
    def show(self, qc = None, fold = None):
        # Default fold is 25
        circuit = self.circuit if qc is None else qc
        display(circuit.draw('mpl', fold = fold,
                             style={'displaycolor': 
                                    {'A': ('#ffc7f1', '#043812'),
                                     '$A^{-1}$': ('#ffc7f1', '#043812'),
                                     '$S_0$': ('#d1d1d1', '#043812'),
                                     '$S_f$': ('#f47aff','#043812'),
                                     'IQFT': ('#98c1dd','#043812'),
                                     'Q': ('#FF8080','#043812')},}))
            
def test_oracle():
    '''
    Test the standard and phase oracle implementations.
    '''
    print("> Will test the standard and phase oracles. [test_oracle]")
    n, t = 3, 3
    a = t/n**2
    theta = np.arcsin(np.sqrt(a))
    print(f"> n={n} | t={t} | a≈{round(a,ndigits)} | θ≈{round(theta,ndigits)}")
    f = CheckingFunction(n, t)
    f.test_standard_oracle()
    f.test_phase_oracle()

# test_oracle()