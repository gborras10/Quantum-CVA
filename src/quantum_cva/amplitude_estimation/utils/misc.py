'''
Miscellaneous utility functions.
'''
import importlib
import inspect
import os, sys
from operator import itemgetter
from copy import deepcopy
from itertools import count
import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout   
        
class Iterator():
    '''
    If indices is None, it eternally increments i (in practice, we'll handle 
    breaking from a "while True" cycle by some other means).
    If it is a list, it yields its consecutive elements until they're over, 
    then (-1) times the number of elements. 
    In both cases, the final value of i is the number of iterations/'advances'.
    '''
    def __init__(self, indices = None):
        self.it = count(0) if indices is None else iter(indices) 
        self.length = None if indices is None else len(indices)
        
    def advance(self):
        try:
            i = next(self.it)
        except StopIteration:
            i = -self.length
        return i
    
first_single_warning = True
def single_warning(errtype, status):
    '''
   ============================================================================

    Prints a warning on the first occurrence of some selected errors. It does
    not warn again after that, regardless of how many such occurrences are
    encountered.
    
    Parameters
    ----------
    errtype: str
        The type of error.
    status: bool, optional
        A status flag provided by numpy. 
   ============================================================================
   '''
    global first_single_warning
    if errtype=="divide by zero" and first_single_warning:
        first_single_warning = False
        print("> Warning: divide(s) by zero encountered. Probably a log(0) "
              "in a loglikelihood somewhere for a parameter with 0 probability."
              " I will not warn again about the same issue. [single_warning]")

def outcome_dist_to_dict(plist, fun = None):
    '''
    Format a list describing a distribution of outcomes as a dictionary, to be 
    e.g. plotted as a histogram. The list's items are expected to be
    (outcome, relative frequency) tuples. If a function 'fun' is given, the 
    keys of the dictionary are processed using it. If not, they are converted
    to same-length binary strings.
    
    Example:
    ps = [10, 20, 30, 40] --> {'00': 10, '01': 20, '10': 30, '11': 40}.
    
    '''
    # Calculate m, minimum number of bits required to represent all outcomes.
    if not fun:
        M = max([outcome for outcome,p in plist]) + 1
        m = int(np.ceil(np.log2(M)))
        fun = lambda o: (bin(o)[2:]).zfill(m)
        
    plist = [(fun(x),f) for x,f in plist]
    ps_dict = {x: 0 for x, _ in plist}
    for o, p in plist:
        # Represent the keys in binary, ignore '0b' prefix..
        #bstr = bin(o)[2:]
        # Pad with 0s, get same-length binary strings (e.g. for hist order).
        #key = bstr.zfill(m)
        ps_dict[o] += p
        
    return ps_dict

def binary_to_decimal(string):
    '''
   ============================================================================
    Converts a binary string to decimal. Since it is a string, the msb always
    comes first. The string can be given as a list, in which case big endian 
    ordering is used (msb @ index 0). This is unlike Qiskit's qreg lists
    
    Parameters
    ----------
    string: str or [int]
        The string to be converted.
        
    Returns
    -------
    r: int
        The decimal value of the string.
   ============================================================================
   '''
    n = len(string)
    r = sum([int(string[n-1-i]) * 2**i for i in range(n)])
    return r

def sigdecstr(x, Nsigfigs):
    '''
    Return a string of x with the fractional part rounded to 'sigfigs' 
    significant decimals. If x is 0, return '0'.
    
    sigdecstr(5, 2) = 5.00
    sigdecstr(5.01, 2) = 5.01
    sigdecstr(0.01, 2) = 0.010
    sigdecstr(5.111e-5, 2) = 5.11e-5
    
    '''
    assert Nsigfigs > 0
    if x == 0:
        return str(0)
    
    # Divide into integer and fractionary parts. 
    i, f = divmod(x, 1)

    if np.isclose(i, 0):
        # Calculate number of leading non-significant zeros in the fractional. 
        nzeros = int(np.ceil(-np.log10(abs(f))) - 1)
        if nzeros >= 4:
            # Use expb10 notation fo ease of reading, as usual in python.
            return f"{myround(f*10**(nzeros+1), Nsigfigs)}e-{nzeros + 1}"
        
        ndigits = nzeros + Nsigfigs 
        return str(myround(f, ndigits))
    
    # Otherwise leading zeros in the fractional are significant.
    ndigits = Nsigfigs 
    return str(int(i)) + str(myround(f, ndigits))[1:]
    

def myround(x, Ndig):
    # To do {:.nf}.format() for arbitrary n. It works like 'round', but
    # perserves zeros when printing, i.e. format(1.000,2) = 1.00 while 
    # round(1.000,2) = 1.0.
    xf = f"{{:.{Ndig}f}}".format(x)
    return xf
    
def round_if_float(x, cases=1):
    if isinstance(x,int):
        return x
    if isinstance(x, float):
        return round(x) if cases==1 else round(x, cases)

def initialize_modules():
    # We only want to do this once, to register only initial sys.modules.
    if 'initial_modules' not in globals():
        global initial_modules
        # Register the current sys.modules.
        initial_modules = set(sys.modules.values())
        
first_reload_custom_modules = True
def reload_custom_modules(method = 1):
    global first_reload_custom_modules, initial_modules
    if method == 0:
        for module in set(sys.modules.values()) - initial_modules:
            try:
                importlib.reload(module)
            except Exception:
                if first_reload_custom_modules:
                    print("> I'm ignoring some weird errors regarding sys.modules "
                          "not in sys.modules(). If you get weird errors, maybe try "
                          "restarting the Colab kernel? [reload_custom_modules]")
                    first_reload_custom_modules = False
        importlib.reload(sys.modules["utils.misc"])
    elif method == 1:
        # Just hardcode it, it's not worth the trouble.
        custom_mods = ("algorithms.adaptiveQAE", "algorithms.estimation", 
                "algorithms.Grover", "algorithms.QAA", "algorithms.QAE", 
                "algorithms.samplers", "utils.binning", "src.utils.mydataclasses", 
                "utils.misc", "utils.models", "utils.plotting", 
                "utils.quantum_ops")
        for mod in custom_mods:
            if mod in sys.modules.keys():
                importlib.reload(sys.modules[mod])

def k_largest_tuples(l, k, sortby=0):
    '''
    Return the k tuples whose 'sortby'th element are largest in a list.
    '''
    assert k <= len(l), (f"List of length {len(l)} has no {k}th largest "
                        "element.")
    largest = sorted(l, key=itemgetter(sortby), reverse=True)[:k]
    return largest

def k_smallest_tuples(l, k, sortby=0):
    '''
    Return the k tuples whose 'sortby'th element are smallest in a list.
    '''
    assert k <= len(l), (f"List of length {len(l)} has no {k}th smallest "
                        "element.")
    smallest = sorted(l, key=itemgetter(sortby))[:k]
    return smallest

def k_largest(l, k):
    '''
    Return the k largest elements in a list.
    '''
    assert k <= len(l), (f"List of length {len(l)} has no {k}th largest "
                        "element.")
    largest = sorted(l, reverse=True)[:k]
    return largest

def k_smallest(l, k):
    '''
    Return the k tuples whose 'sortby'th element are smallest in a list.
    '''
    assert k <= len(l), (f"List of length {len(l)} has no {k}th smallest ")
    smallest = sorted(l)[:k]
    return smallest

def kth_largest(l, k):
    '''
    Return the kth largest value in a list.
    '''
    assert k <= len(l), (f"List of length {len(l)} has no {k}th largest "
                        "element.")
    l_ = deepcopy(l)
    for i in range(k-1):
        l_.remove(max(l_))
    return max(l_)

def closest_odd_int(x):
    '''
    Return the odd integer closest to 'x'.
    '''
    xr = np.round(x)
    if xr % 2 == 0:
        '''
        One of the 2 nearest integers is odd. If the closest one is not, get 
        the other.
        '''
        xr = xr - np.sign(xr-x)
    return xr

def estimation_errors(sqe_lists, stat, by_step = False):
    '''
    If by_step is False:
    Calculate the average estimation error sqrt(MSE) ordered by step, given
    a list of squared error evolutions ordered by run.
    sqe_lists[i]: list with the squared errors of run #i, ordered by step.
    sqe_by_step[i]: list with the errors of step #i, ordered by run.

    Otherwise:
    sqe_lists is sqe_by_step already
    '''
    if not by_step:
        sqe_by_step = list(zip(*sqe_lists))
    else:
        sqe_by_step = sqe_lists
    # Estimation error ε = E[(â-a)^2]^0.5

    if stat == "mean":
        f = lambda x: np.mean(x)
    if stat == "median":
        f = lambda x: np.median(x)
        
    err_per_step = [np.sqrt(f(step_errors))
                            for step_errors in sqe_by_step]
    return err_per_step

def print_centered(slist, w = 62):
    '''
    Print the strings in slist by order, centering them in the console (w 
    chars wide) and printing lines before and after.
    
    If any has a len > 62, split into smaller segments, if possible divided 
    at "|".
    
    If any starts by a line break, print the line break before centering.
    '''
    print("==============================================================")
    for s in slist:
        if s[0] == "\n":
            print("\n", end = "")
            s = s[1:]
        if len(s) <= w:
            print(s.center(w))
        else:
            segments = split_string_vbar(s, w)
            for seg in segments:
                print(seg.center(w))
        
    print("==============================================================")
    
def split_string_vbar(s, maxlen):
    '''
    Split a string into a list of segments of length up to max_length; if 
    possible divide at vertical bar  "|". 
    '''
    segments = []
    start = 0
    length = len(s)

    while start < length:
        end = min(start + maxlen, length)
        latest_vbar = s.rfind('|', start, end)
        
        if end - start < maxlen:
            segments.append(s[start:])
            break 
        if latest_vbar != -1:
            segments.append(s[start:latest_vbar])
            start = latest_vbar + 1
        else:
            segments.append(s[start:end])
            start = end
    
    return segments
    
    
def closest_comma(s, pos):
    comma_is = [i for i, char in enumerate(s) if char == ',']
    
    if not comma_is:
        return None 
    
    cc = min(comma_is, key=lambda x: abs(x - pos))
    return cc
    
def rem_negligible_items(d, atol = 5*1e-3):
    # Delete dictionary keys whose values are < atol.
    print(f"> Warning: deleting dictionary items with values < {atol}.")

    for key in list(d.keys()):
        if d[key] < atol:
            del d[key]
    
def b10str(n):
    return f"10^{expb10(n)}" 
    
def expb10(n):
    '''
    Return the integer 'e' for which 10^e is closest to n.
    '''
    e = np.log10(n)
    return round(e, 2)

def truncated_normal(mean, std, range, Nsamples):
    '''
    Return a array containing 'Nsamples' samples from a Gaussian with the given
    mean and standard deviation truncated to the domain given by 'range'. 
    '''
    samples = norm.rvs(loc = mean, scale = std, size = Nsamples)
    while True:
        try:
            i_neg = next(i for i, s in enumerate(samples) 
                         if s < range[0] or s > range[1])
            samples = np.delete(samples, i_neg)
            new_sample = norm.rvs(loc = mean, scale = std)
            samples = np.insert(samples, i_neg, new_sample)
        except StopIteration:
            break

    return samples

def get_truncated_normal(mean, sd, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def kwarg_str(mycallable = None):
    '''
    If mycallable is none, assume it's a function calling. In that case the 
    callable can be easily obtained because it is in the global namespace.
    For methods pass self.method.
    '''
    caller_frame = inspect.currentframe().f_back  
    if mycallable is None:
        mycallable = caller_frame.f_globals[caller_frame.f_code.co_name]
    
    signature = inspect.signature(mycallable)
    kwargs = kwargs_to_dict(signature, caller_frame)
    kstr = dict_str(kwargs, spaces = True)
    return kstr

def kwargs_to_dict(signature, caller_frame = None):
    '''
    Return a dictionary with the kwargs of caller function,
    or another given by the caller_frame argument.
    '''
    if caller_frame is None:
        caller_frame = inspect.currentframe().f_back

    lvars = caller_frame.f_locals
    kwargs = filter_kwargs(signature, caller_frame, lvars)
    return kwargs

def filter_kwargs(signature, caller_frame, lvars):
    '''
    def f(x, a=1, b=2, c=3, **kwargs):
          d = 1
          kwargs = filter_kwargs()
          
    f(1, a=3, z=10) 
    
    'kwargs' will be a dict with all kwargs (passed explicitly or through 
    **kwargs) and their default values: {'a': 3, 'b': 2, 'c': 3, 'z': 10}

    '''
    # Select local variables that are keyword arguments passed explicitly. 
    kwargs = {var: lvars[var] for var in lvars
              if var in signature.parameters 
              and signature.parameters[var].default != inspect.Parameter.empty}
    # Add kwargs passed by **kwargs.
    if "kwargs" in lvars:
        kwargs.update(lvars["kwargs"])
    return kwargs

def dict_str(d, spaces = False, shorten = True):
    '''
    d = {'a':1, 'b':2} -> 'a=1,b=2' or 'a = 1, b = 2'
    '''
    s = d.__str__()[1:-1]
    s = s.replace("'", "")
    if shorten:
        s = s.replace("True", "T")
        s = s.replace("False", "F")
        s = s.replace("None", "N")
    if not spaces:
        s = s.replace(" ", "")
        s = s.replace(":", "=")
    else:
        s = s.replace(":", " =")
    return s

def dict_info(d, sep = ","):
    '''
    d = {'a':1, 'b':2} -> a = 1, b = 2'
    '''
    s = d.__str__()[1:-1]
    s = s.replace("'", "")
    s = s.replace(":", " =")
    if sep != ",":
        s = s.replace(",", sep)
    return s

def thin_list(l, k, mn):
    '''
    Only keep every kth element, but present at least mn points, or all if <mn.
    '''
    # len(l)/k > mn.
    k = min(k, 
            max(1,int(len(l)/mn)))
    return l[::k]

def logspace(mn, mx, N):
    assert mn >= 0 
    if mn == 0:
        mn = 1
        l = np.array([0])
        N -= 1
    else:
        l = np.array([])
    l = np.concatenate([l, np.logspace(np.log10(mn), np.log10(mx), N)])
    return l

def logsumexp(ws):
    C = ws.max()
    lse = np.log(np.sum(np.exp(ws - C))) + C
    return lse

def np_invalid_catch(f, largs, errdesc, cleanup, caller, proceed = True):
    ''' 
    Run f(args) and catch invalid runtime warnings from numpy. If they don't 
    match errdesc, warn; if they do, execute cleanup. If no warnings, return 
    value. If proceed is True, return calculation after warning and clean up
    despite warning. Otherwise, require input to continue (for debugging, to
    pause when it happens).
    
    If cleanup returns a value, use that value as calculation instead of 
    calculating it and suppressing warnings.
    
    errdesc is a list of descriptions for which the clean up should be performed.
    
    warn is a list of errdesc for which the error should raise warning
    not error.
    
    Don't print arguments for underflows, usually they're harmless.
    
    '''
    def fun(throw = False):
        if throw:
            r = f(*largs)
        else:
            with np.errstate(all = "ignore"):
                r = f(*largs)
        return r
    
    # Ignore underflows because they're expected and don't have weird behavior
    # (just round to zero).
    with np.errstate(divide = "raise", over = "raise", invalid = "raise",
                     under = "ignore"):
        try:
            r = fun(throw = True)
        except FloatingPointError as e:
            if e.args[0] not in errdesc:
                print(">", e.args[0].capitalize(), "[np_invalid_catch].")
                print(f"> Caller function: {caller}.")
                
                if e.args[0][:10] != " underflow":
                    # print(f"10 first: «{e.args[0][:10]}»")
                    print("> Arguments were: ")
                    print_items(largs)
                            
                if not proceed:
                    print("> Enter to continue.", end = "")
                    input()
                    
                r = fun(throw = False)
            else:
                print(">", e.args[0].capitalize(), f"in {caller}.", end ="")
                print(" Executing clean up...")
                r = cleanup(*largs)
                if r is None:
                    r = fun(throw = False)
                if not proceed:
                    print("> Cleanup done. Enter to continue.", end = "")
                    input()
        return r
    
def print_items(l):
    '''
    Print a list of items such that items that are lists are not truncated.
    '''
    for i,x in enumerate(l):
        print(f"* item #{i}:", end = " ")
        if isinstance(x, list):
            lprint(x)
        else:
            print(x)
            
def lprint(l):
    '''
    Print full list (no truncation).
    '''
    print(', '.join(map(str, l)))
    
def print_1st_err(f, args, argstr, ls = [], strs = []):
    '''
    Print first x in args for which f(x) raises an error. Require input to
    return (for debugging).
    
    ls and strs are lists and the corresponding descriptors. If provided,
    print the info for the items in the same positions as x in args.

    '''
    assert len(ls) == len(strs)
    for tple in zip(args, *ls):
        try:
            f(tple[0])
        except:
            vals = [f"{desc} = {tple[i]}" 
                    for i, desc in enumerate([argstr] + strs)]
            print("> ", end = "")
            print(*vals, sep = "; ")
            print("> Printed 1st error. Enter to continue. [print_1st_err]", end = "")
            input()
            return
        
class PrintSeqTable:
    '''
    Print a table sequentially. Init gets the header labels, subsequence calls
    to print_row will print the values for each, assuming they are ordered as
    the labels.
    
    E.g. input:
    ls = ["AAA", "B", "CCCCCCCC", "DDDDD"]
    st = PrintSeqTable(ls)
    st.print_row([1,2,3,4])
    st.print_row([1,20])
    
    Output:
             AAA      |       B       |    CCCCCCCC   |     DDDDD     
    (1)       1       |       2       |       3       |       4       
    (2)       1       |       20      |               |            
    '''
    def __init__(self, ls, chars = 62):
        '''
        ls: a list of strings describing each column.
        '''
        self.chars = chars
        self.counter = 0
        self.ncols = len(ls)
        self.ilen = int(chars/len(ls))
        self.print_row(ls)
           
    def print_row(self, ls):
        if self.counter != 0:
            row = f"({self.counter})"
        else:
            # Header.
            row = "   "
        self.counter += 1
            
        row += self.get_row(ls)
        print(row)
        
    def get_row(self, ls):
        ls = self.truncate_and_center(ls)
        row = self.join_row(ls)
        return row
        
    def truncate_and_center(self, ls):
        for i in range(self.ncols-len(ls)):
            # If some elements are missing, fill with empty space to still
            # print bars diving rows.
            ls.append("".center(self.ilen))
            
        ls = [str(x)[:self.ilen-2].center(self.ilen) for x in ls]
        return ls
    
    def join_row(self, ls):
        row = "|".join(ls)
        return row
   