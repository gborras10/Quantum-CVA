'''
Generalized Grover algorithm for arbitrary functions and numbers of solutions.
'''

import sys

try:
    from google.colab import files
    using_colab = True
except:
     using_colab = False

from qiskit import QuantumCircuit
import numpy as np
import importlib
from operator import itemgetter
import random

if using_colab:
    # Mount drive (left menu, third icon) before running for this to work.
    sys.path.append('/content/drive/Othercomputers/Laptop/Phd/3.Scripts') 
    
# Custom modules.
from quantum_cva.amplitude_estimation.algorithms.QAA import QAA, CheckingFunction
from quantum_cva.amplitude_estimation.utils.quantum_ops import aer_measure, aer_simulate
from quantum_cva.amplitude_estimation.utils.misc import binary_to_decimal

# When running on Google Colab, modules must be explicitly reloaded to get the
# latest version (or restart the runtime).
reload = True
if reload and using_colab:
    importlib.reload(sys.modules["algorithms.QAA"])
    importlib.reload(sys.modules["utils.quantum_ops"])
    importlib.reload(sys.modules["utils.misc"])
    
ndigits = 5

class GroverSearch(QAA):
    '''
   ============================================================================
    Class for performing quantum search (which is enhanced by quantum 
    amplitude amplification) [1]. Implements the standard algorithm,
    derandomization features (applicable when the amplitude is known), and an
    adaptation of the algorithm for unknown amplitude / 'nsol'.

    Attributes
    ----------
    encoding_qs: int
        The number of qubits required to represent prospective solutions.
    function: CheckingFunction
        The target function, for which we mean to find inputs that
        evaluate to 1.
    nsol: int or None, optional
        The number of solutions, or None if this is not known. Default is 
        None.
    circuit: QuantumCircuit
        The quantum circuit on which to act (to apply or create operators,
        including measurement operators). 
    derandomize: bool, optional
        Whether to apply derandomization using an auxiliary qubit. Only works 
        if the number of solutions 'nsol' is given.
    n_qubits: int
        the number of qubits for the circuit. This is the number of qubits
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
    Bgate: Gate
        The derandomization operator, which extends A to the auxiliary qubit.
    Binvgate: Gate
        The inverse of Bgate.
   ============================================================================
   '''
    def __init__(self, encoding_qs, function, nsol = None, 
                 derandomize = True, omit_aux = True):
        super().__init__(encoding_qs, function, nsol, derandomize, omit_aux)
        self.circuit = QuantumCircuit(self.n_qubits)
        # The B gates will be created later when/if necessary.
        self.Bgate, self.Binvgate = None, None
        
    def search(self, m = None, c = 1.5):
        signature = "[GroverSearch.search]"
        en, nsol = self.encoding_qs, self.nsol
        def compute_m(print_info = True):
            a = nsol/2**en
            theta = np.arcsin(np.sqrt(a))
            m_ideal = np.pi/(4*theta) - 1/2 
            # So that (2m+1)θ = π/2, but may not be int.
            m = int(np.floor(np.pi/(4*theta)))
            print("> Computed optimal iteration number: ")
            print(f"> a  ≈ {round(a,3)} | θ ≈ {round(theta/np.pi,2)}π | "
                  f"m = {m} = ⌊{np.pi/(4*theta)}⌋ = [{m_ideal}]")
            if self.derandomize:
                m_ = int(np.ceil(m_ideal))
                theta_ = np.pi/(4*m_+2)
                a_ = np.sin(theta_)**2

                print(f"> Derandomizing: a_ ≈ {round(a_,3)} | "
                      f"θ ≈ {round(theta_/np.pi,2)}π | "
                      f"m = {m_} = ⌈{m_ideal}⌉")
                m = m_
                self.assign_Bs(a_, a)
            return(m)
        
        if m is None:
            if nsol is None:
                print("> Cannot compute optimal operator exponent due to "
                      "unknown number of solutions. "
                      "Switching to exponential search of 'm'.",
                      signature)
                self.QSearch_unknown_t()
                return
            m = compute_m()
            print(signature)
        else:
            print(f"> Using given number of applications m = {m}. ", signature)
        self.initialize()
        self.iterate_Q(m)
        counts = self.run_circuit(info = " - amplitude amplification")
        self.solutions_from_data(counts)
        
    def QSearch_unknown_t(self, c = 1.2):
        f = self.function; signature = "[GroverSearch.QSearch_unknown_t]"
        def declare_winner(w):
            print(f"> Success! Found solution {w} at exponent m={M} "
                  f"(iteration #{l}).", signature)
        print("> Working under the assumption of unknown number "
              "of solutions. Will increase the operator exponent "
              f"exponentially with base c={c}. ", signature)
        l = 0
        self.c = c
        while True:     
            l += 1; M = int(np.ceil(c**l))
            self.circuit.A()
            outcome = aer_measure(self.circuit, omit = self.omit)
            output = f.evaluate(outcome)
            if output == 1:
                declare_winner(outcome); return
            else:
                self.circuit = QuantumCircuit(self.n_qubits)
                self.initialize()
                k = random.randint(1,M); self.apply_Q(m = k)
                outcome = aer_measure(self.circuit, omit = self.omit)
                output = f.evaluate(outcome)
                if output == 1:  
                    declare_winner(outcome); return
                self.circuit = QuantumCircuit(self.n_qubits)
    
    def check_rotation(self, a_, a):
        print("> Testing the rotation realized by B: [check_rotation]")
        r = a_/a
        target = {}
        target['0'] = np.sqrt(1-r)
        target['1'] = np.sqrt(r)
        print("Target coefficients: ")
        print(target)
        
        B = self.B_gate(a_, a)
        qc = QuantumCircuit(1)
        qc.append(B,[0])
        aer_simulate(qc,how = "statevec")
        
    def B(self):
        en = self.encoding_qs
        self.circuit.append(self.Bgate, [en])
       
    def Binv(self):
        en = self.encoding_qs
        self.circuit.append(self.Binvgate, [en])
        
    def assign_Bs(self, a_, a):
        self.Bgate, self.Binvgate = self.create_Bs(a_,a)  
            
    def create_Bs(self, a_, a):
        Bgate = self.B_gate(a_, a)
        Binvgate = Bgate.inverse(); Binvgate.label = "$B^{-1}$"
        return Bgate, Binvgate  
            
    def B_gate(self, a_, a):
        # To be applied along with A if derandomizing.
        # Rotate |0> -> sqrt(1-a_/a)|0> + sqrt(a_/a)|1>
        # When derandomizing, 'a_' will be slightly smaller than 'a'.
        qc = QuantumCircuit(1)
        
        qc.h(0)
            
        rth = self.B_ry_angle(a_, a)
        qc.ry(rth,0)

        B = qc.to_gate(label='$B$')
        return B
    
    def B_ry_angle(self, a_, a):
        # Compute the angle such that <0|RY|+> = sqrt(1-a_/a).
        r = a_/a
        angle = 2*r-1   
        rth = np.arcsin(angle)
        return rth
    
    def solutions_from_data(self, counts):
        # Sort outcomes by decreasing order of frequency.
        kv_list = [(k, v) for k, v in counts.items()]
        sorted_list = sorted(kv_list, key=itemgetter(1), reverse=True)
        
        shots = sum([kv[1] for kv in kv_list])
        print("> These were the obtained outcomes and their relative "
              f"frequencies (for {shots} experiments):")
        for (outcome, freq) in sorted_list:
            print(f"- 0b{outcome} ({binary_to_decimal(outcome)}): "
                  f"{round(freq/shots,2)}")
        print("[GroverSearch.solutions_from_data]")
        

def test_search():
    '''
    Test the generalized Grover search implementation.
    '''
    derandomize=True
    print("> Will test" + (" (derandomized)" if derandomize else "") 
          + " Grover search. [test_search]")
    n, t = 4, 2
    a = t/n**2; theta = np.arcsin(np.sqrt(a))
    print(f"> n={n} | t={t} | a≈{round(a,ndigits)} | θ≈{round(theta,ndigits)}")
    f = CheckingFunction(n,t)#, ws=[[0,0,0,1],[0,1,0,0]])
    Grover = GroverSearch(n, f, nsol = t, derandomize = derandomize)
    Grover.search()
    
# test_search()