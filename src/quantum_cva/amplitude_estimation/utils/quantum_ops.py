'''
Some quantum operations and auxiliary functions, implemented using Qiskit. 
'''
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from IPython.display import display
from qiskit.visualization import plot_histogram
import numpy as np
from pprint import pprint

def simulate(circ, shots = 1024, full_output = False):
    simulator = Aer.get_backend('aer_simulator')
    tcirc = transpile(circ, simulator)
    job = simulator.run(tcirc, shots=shots)
    r = job.result()
    if full_output:
        return r
    
    counts = r.get_counts(circ)
    return counts

def aer_measure(qc, omit = None):
    '''
   ============================================================================
    Performs a measurement (single shot) on a circuit and returns the result,
    possibly suppressing some qubits. If the qubits to be suppressed are not
    in the correct states, it prints a message in the console, but suppresses
    them still. The Aer simulator is used as the backend for the execution.
    
    Parameters
    ----------
    qc: QuantumCircuit
        The quantum circuit to be measured.
    omit: [(int,str)]
        A list of qubits to suppress from the outcome, and their respective
        states. Can be empty if the states are to be left unaltered. We don't
        simply not measure them because we want to make sure they're in the 
        expected states. Default is None.
        
    Returns
    -------
    outcome_red: string
        The result as a binary string, possibly with some qubits omitted.
   ============================================================================
   '''
    qc.measure_all()
    sim = Aer.get_backend('aer_simulator')
    # result = execute(qc, sim, shots = 1).result()
    # counts = result.get_counts()
    counts = simulate(qc)
    
    outcome = list(counts.keys())[0]
    if omit is not None:
        outcome_red, error = remove_q(outcome, omit)
    if error:
        omit_clauses = ['q' + str(cond[0]) + ' = \'' + cond[1] + '\''
                        for cond in omit]
        omit_clauses = ', '.join(omit_clauses)
        print(f"> ERROR: The qubits you wanted to omit don't seem to be in the "
              f"states you intended. The measurement result is {outcome} "
              f"but you said {omit_clauses}. [aer_measure]")
    return outcome_red

def aer_simulate(qc, 
                 how = "hist", 
                 meas_info = None,
                 omit = None, 
                 info = "",
                 atol = None):
    '''
   ============================================================================
    Executes a circuit and presents the results as a histogram or a 
    statevector, possibly suppressing some qubits. If the qubits to be 
    suppressed are not in the correct states, it prints a message in the 
    console and does not suppress them. The Aer simulator is used as the 
    backend for the execution.
    
    Parameters
    ----------
    qc: QuantumCircuit
        The quantum circuit to be measured.
    how: str , optional
        How to present the results: 
        - "counts" (return counts), or
        - "statevec" (print statevector), or 
        - "hist" (plot histogram).  
        Default is "statevec".
    meas_info: ([int],[int]) or [], optional
        A tuple containing the list of qubits to be measured and the list of
        classical bits to store the result to, in the 0 and 1 positions 
        respectively. They should have the same length and correspond to 
        valid registers in 'qc'. If None, all qubits will be measured into 
        a classical register added for the purpose. If empty, it will be 
        assumed the circuit already measures the intended qubits. Only used if 
        type=="hist". Default is None.
    omit: [(int,str)], optional
        A list of qubits to suppress from the results, and their respective
        states. We don't simply not measure them because we want to make sure
        they're in the expected states. Default is None.
    info: str, optional
        Some annotation to include in the title of the histogram or when
        printing the statevector on the console. Default is "".
    atol: float, optional
        The threshold amplitude for considering a state. If None, will use
        the one given by the atol attribute of Qiskit's Statevector class.
        Default is None.
   ============================================================================
   '''
    def omit_qs(res: dict):
        error = 0
        new_res = res.copy()
        for outcome in list(res):
            truncated_outcome, status = remove_q(outcome, omit)
            error = max(error, status)
            new_res[truncated_outcome] = new_res.pop(outcome)
        if error:
            print("> ERROR: Qubit(s) to be omitted are not in the correct "
                  "state(s). I'll just leave them all be so you can fix "
                  "this mess. [aer_simulate.omit_qs]")
            return res
        return new_res
    
    qc_ = qc.copy()
    # sim = Aer.get_backend('aer_simulator')
    
    if how == "counts" or how == "hist":
        if meas_info is None:
            qc_.measure_all()
        elif meas_info==[]:
            # Measurements already included in the circuit.
            pass 
        else:
            qc_.measure(meas_info[0], meas_info[1])
        # result = execute(qc_, sim).result()
        # counts = result.get_counts()
        counts = simulate(qc_)
        
        if omit is not None:
            counts = omit_qs(counts)
            
        if how == "counts":
            return counts
        elif how == "hist":
            ttl = "Aer simulation results" + info
            hist = plot_histogram(counts, 
                                   color='lightgray', figsize=(11.5,8))
            hist.suptitle(ttl, y=0.93, fontsize=18)
            display(hist)
            print("> I would like to let you know I am plotting a figure."
                  " [aer_simulate]")
            return counts
        
    if how == "statevec":
        qc_.save_statevector()
        result = simulate(qc_, full_output = True)
        vec = result.get_statevector()
        atol = vec.atol if atol is None else atol
        d = vec.to_dict()
        for outcome in list(d):
            if np.abs(d[outcome]) < atol:
                d.pop(outcome)
        if omit is not None:
            d = omit_qs(d)
        print(f"> Non-negligible (amp > {atol}) statevector coefficients" 
              + info + ":")
        pprint(d)
        print("[aer_simulate]")
        return
    
    print(f"«{how}» is not a valid option so I did nothing. [aer_simulate]")

def remove_q(string, which):
    '''
   ============================================================================
    Removes qubits from a string representing a state, and signals a problem
    if the removed qubits' states don't match the intended ones.
    
    Parameters
    ----------
    string: str
        The state from which to suppress a qubit.
    which: [(int,str)]
        A list of tuples, where each tuple contains the index of one qubit to 
        be removed + the state it is expected to be in (str of length 1). It 
        is assumed that the states of the qubits to be removed are fixed 
        and known.
        
    Returns
    -------
    string: int
        The decimal value of the string.
    status: int
        Signals whether the removed qubit was in the correct state. It is 1 
        if string[which[i]] != which[i] for any i, and 0 otherwise.
   ============================================================================
   '''
    status = 0 
    indices = [cond[0] for cond in which]
    states = [cond[1] for cond in which]
    indices.sort()
    for i,pos in enumerate(indices):
        curr_pos = pos - i # Compensate for the already removed chars.
        if string[curr_pos] != states[pos]:
            status = 1
        string = string[:curr_pos] + string[curr_pos+1:]
    return string, status

def multi_cz(n: int, show=False):
    '''
   ============================================================================
    Produces a multi-controlled Z-gate. This gate has a symmetric effect on 
    all its inputs: it adds a pi phase to the |1...1> state, while leaving 
    others invariant.
    
    Parameters
    ----------
    n: int
        The number of qubits the gate will act on.
    show: bool, optional
        Whether to plot the circuit. Default is False.
        
    Returns
    -------
    mcZ: Gate
        A multi-controlled Z gate acting on n qubits.
   ============================================================================
   '''
    qc = QuantumCircuit(n)
    ctrls, tg = [q for q in range(n-1)], n-1
    qc.h(tg)
    qc.mct(ctrls, tg) # Multi-controlled Toffoli, made available by Qiskit.
    qc.h(tg)
    mcZ = qc.to_gate(label="$mcZ$")
    if show:
        display(qc.draw('mpl'))
        print("> I would like to let you know I am plotting a figure."
              " [multi_cz]")
    return mcZ

def sample_from(p, nshots):
    '''
    Take nshots from a binomial distribution with probability of success p,
    and return the relative frequency of successful measurementes. I.e., 
    introduce sampling noise.
    '''
    hits = np.random.binomial(nshots, p)
    f = hits/nshots
    return f