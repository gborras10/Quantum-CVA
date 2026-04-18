'''
Classes and functions related to executions (multiple runs of the same algorithm
for statistics).
'''
import time
import timeit
from time import perf_counter
from typing import Callable
from dataclasses import dataclass, field
import numpy as np 

from quantum_cva.amplitude_estimation.utils.files import PrintsToFile
from quantum_cva.amplitude_estimation.algorithms.samplers import SMCsampler

class Timed:
    '''
    To use as a decorator that accepts parameters. 
    '''
    def __init__(self, units = "s", extra_info = "", f = lambda x: x):
        self.units = units
        self.info = extra_info
        self.f = f

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            timer = Timer(extra_info = self.info)
            result = func(*args, **kwargs)
            timer.stop(units = self.units, f = self.f)
            return result

        return wrapper
    
class Runner():
    '''
    Run a given function f with given arguments nruns times, time the 
    executions, and process the resulting outputs using a given function.

    The first argument of the function should be an integer, serving as a label
    for the iteration number. This is so the called function can have access to
    its own label.
    
    The function'process_fun' will be applied to the results of each run. This
    is expected to be the method of a dataclass that organizes the information
    from the runs, taking the outputs given by one execution of f. 
    
    A progress bar is always printed. Apart from that, the redirect argument 
    changes what happens to the other prints - those done by 'f', the prints 
    signaling each iteration's  start and conclusion, and the timer prints.
    
        - For redirect = 0, all prints are shown in the console as usual.
        - For redirect = 1, the prints for the first run are shown in the 
        console, but all others are saved to a file. This useful for testing,
        to see what is happening without repeating it hundreds of times.
        - For redirect = 2, all prints are saved to a file.
    '''
    def __init__(self, f, nruns, process_fun = None, redirect = 0, 
                 silent = False, save = True):
        '''
        Untimed: just print the progress bar. 
        '''
        self.f = f
        self.nruns = nruns
        if process_fun is None:
            self.process_fun = lambda *args: None
        else:
            self.process_fun = process_fun
        self.redirect = redirect
        self.pb = ProgressBar(nruns)
        assert redirect in [0, 1, 2]
        # If redirect is 1, this will be done only after the second run.
        if redirect == 2:
            self.tofile = PrintsToFile("run_logs", silent)
        self.silent = silent
        self.save = save
        
    def run(self, *args, **kwargs):
        return self.run_timed(*args, **kwargs)
            
    # Can't use decorator because we must need self.nruns for the average.
    def run_timed(self, *args, **kwargs):
        timer_all = Timer(extra_info = "average per run", silent = self.silent)
        r = self.run_untimed(*args, **kwargs)
        
        if self.nruns != 0:
            timer_all.stop(units = "s", f = lambda x: x/self.nruns)
        
        if self.redirect == 2  or (self.redirect == 1 and self.nruns > 1):
            if self.save:
                self.tofile.save_file()
            
        return r
        
    def run_untimed(self, *args, **kwargs):
        for i in range(self.nruns):
            try:
                r = self.run_one(i, *args, **kwargs)
                if isinstance(r, tuple):
                    self.process_fun(*r)
                else:
                    self.process_fun(r)
                
                self.pb.update()
            except KeyboardInterrupt:
                print(f"> Keyboard interrupt at run {i}. ")
                print("> Breaking from cycle... [Runner.run]")
                self.nruns = i
                break
    
        return self.nruns
    
    def run_one(self, i, *args, **kwargs):
        if self.redirect == 0 or (self.redirect == 1 and i == 0):
            return self.run_one_aux(i, *args, **kwargs)
        if self.redirect == 1 and i == 1:
            self.tofile = PrintsToFile("run_logs")
        if self.redirect in [1, 2]:
            with self.tofile:
                return self.run_one_aux(i, *args, **kwargs)
            
    @Timed(extra_info = "entire run")
    def run_one_aux(self, i, *args, **kwargs):
        print(f"> Run {i}.")
        r = self.f(i, *args, **kwargs)
        print(f"> Run {i} has been completed.")
        return r

class ProgressBar():
    '''
    Prints up to mx progress updates spaced throughout a loop.
    '''
    def __init__(self, runs, mx = 20):
        # Run before the loop.
        
        self.runs = runs
        # 'i' tracks the iteration number.
        self.i = 0
        # 'counter' tracks the percentage completed. 
        self.counter = 0
        self.progress_interval = 100/runs if runs < mx else 100/mx
        self.mx = mx
        
    def update(self):
        # Place in the end of the loop. Prints 0% only after the first 
        # iteration and 100% before the last one, but it's good enough.
        i, runs = self.i, self.runs
        
        if i==0:
            print("|0%",end="|")
        if runs < self.mx or (i%(runs/self.mx)<1):
            self.counter += self.progress_interval
            print(round(self.counter),"%",sep="",end="|")
        self.i += 1
        if i == runs-1:
            print("") # For newline.

MULTIPLIER = {'s': 1,
              'ms': 1e3,
              'us':  1e6}
    
class Timer():
    def __init__(self, silent = False, extra_info = ""):
        self.ti = perf_counter()
        self.silent = silent
        self.extra_str = " (" + extra_info + ")"
        
    def stop(self, units = "ms", f = lambda x: x):
        '''
        f: function to process the result before printing (apart from units).
        '''
        tf = perf_counter()
        Dt = tf - self.ti 
        if not self.silent:
            print(f"> Time taken{self.extra_str}: "
                  f"{round(f(Dt)*MULTIPLIER[units])}{units}. [Timer.stop]")
        return Dt
    
class RelativeTimer():
    def __init__(self):
        self.ti = perf_counter()
        self.reference = None
        
    def new(self):
        self.ti = perf_counter()
        
    def stop(self):
        tf = perf_counter()
        Dt = tf - self.ti 
        
        if self.reference is None:
            self.reference = Dt
            print(f"> Time elapsed: {Dt*1e-3}ms.")
            print("  (To be used as reference.)")
        else:
            print(f"> Time elapsed: {Dt*1e-3}ms.")
            print(f"  ({round(self.reference/Dt,1)}x faster than the reference.)")
        return Dt

def time_and_print(f):
    # Time a function and print the time. Call e.g. print_time(lambda: 10**4)
    # But it seems like perf_counter is more accurate?
    t = timeit.Timer(f)
    print("Time taken:", t.timeit(number=1000))
   
"""
                Tests below for Runner (which uses ProgressBar, Timer).
"""
     
@dataclass
class RunData:
    n: list[int] = field(default_factory=list)
    
    def append(self, n):
        self.n.append(n)
        
    def get_list(self):
        return self.n
    
class Test():
    def __init__(self):
        self.counter = 0
        
    def run(self):
        rd = RunData()
        runner = Runner(f = self.inc, nruns = 20, process_fun = rd.append)
        runner.run(2)
        l = rd.get_list()
        print("Result: ", l)
    
    def inc(self, num):
        time.sleep(0.1) 
        prev = self.counter
        self.counter += num
        print(f"Counter incremented from {prev} to {self.counter}.")
        return(self.counter)

if __name__ == "__main__":
    t = Test()
    t.run()

class PrintManager():
    '''
    To print only the first time a function calls it. Also adds signature.
    '''
    def __init__(self):
        self.callers = [] 

    def add_caller(self, caller):
        self.callers.append(caller)
    
    def print1st(self, s, fname):
        if fname not in self.callers:
            print(s + f" [{fname}]")
            self.add_caller(fname)

    def is_first(self, fname):
        if fname not in self.callers:
            self.add_caller(fname)
            return True
        
        return False
       
@dataclass
class BAERunsData:
    '''
    For managing the data of BAE runs.
    '''
    nqs: list[float] = field(default_factory=list)
    sqes: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)
    
    # For last iteration. Just for convenience; also stored in previous group.
    frmses: list[float] = field(default_factory=list)
    fstds: list[float] = field(default_factory=list)
    
    def add_run_data(self, nqs, sqes, stds):
        self.nqs.extend(nqs)
        self.sqes.extend(sqes)
        self.stds.extend(stds)
        
        self.frmses.append(sqes[-1]**0.5)
        self.fstds.append(stds[-1])
        
    def get_lists(self):
        full = (self.nqs, self.sqes, self.stds)
        final = (self.frmses, self.fstds)
        return full, final

    def get_descriptors(self):
        full =  ["Numbers of queries for all iterations",
                  "Squared errors for all iterations",
                  "Standard deviations for iterations >=1"]
        final = ["final estimation error",
                 "final standard deviation"]
        return full, final
    


@dataclass
class BAERunData:
    '''
    For managing the data of a single BAE run.
    '''
    # Sampler for calculating the means and standard deviations.
    sampler: SMCsampler
    # Fuction that given a control, calculates the probing time.
    PTfun: Callable[[float], float]
    # SMC means.
    means: list[float] = field(default_factory=list)
    # SMC standard deviations.
    stds: list[float] = field(default_factory=list)
    # Cumulative probing times.
    CPTs: np.ndarray = field(default_factory=list)
    
    def add_iteration_data(self, ctrl, nshots):
        mean, std = self.sampler.mean_and_std()
        self.means.append(mean); self.stds.append(std)
        self.CPTs.append(self.latest_CPT + self.PTfun(ctrl)*nshots)
        return mean, std
    
    def get_lists(self):
        return self.means, self.stds, self.CPTs
    
    @property
    def latest_CPT(self):
        return 0 if len(self.CPTs)==0 else self.CPTs[-1]
    
    def __len__(self):
        l = len(self.means)
        assert l == len(self.stds) == len(self.CPTs)
        return l