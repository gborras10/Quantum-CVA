'''
Auxiliary plotting functions.
'''

import numpy as np
import scipy.optimize as opt
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from copy import deepcopy
from itertools import chain
import os, pytz
from datetime import datetime

from quantum_cva.amplitude_estimation.utils.binning import process_raw_estdata
from quantum_cva.amplitude_estimation.utils.misc import estimation_errors, thin_list, logspace
# from src.utils.mydataclasses import EstimationData
from quantum_cva.amplitude_estimation.utils.files import data_from_file

NDIGITS = 5

class Plotter():
    
    def __init__(self, log = True):
        fig, ax = plt.subplots(1,figsize=(10,6))
        self.ax = ax
            
        if log:
            plt.xscale('log'); plt.yscale('log')
    
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        ax.spines['right'].set_color('lightgray')
        ax.spines['top'].set_color('lightgray')
        
        ax.grid(which='both')
    
    def scatter(self, xs, ys):
        print("H", len(xs), len(ys))
        self.ax.scatter(xs, ys, marker="*", color='crimson', s=400,
                        edgecolors='black', linewidth=1)
        
    def scatter_by_groups(self, grouped_points, ypower = 1):
        colors = []
        for group in grouped_points:
            # Get a new color for this group.
            new_color = ('#%06X' % np.random.randint(0, 0xFFFFFF))
            while new_color in colors: 
                new_color = ('#%06X' % np.random.randint(0, 0xFFFFFF))
            colors.append(new_color)
            
            xs = [point[0] for point in group]
            ys = [point[1]**ypower for point in group]
            self.ax.scatter(xs, ys, color = new_color, alpha = 0.4)
        
    def line(self, xs, ys):
        # Sort by order of first tuple element (x).
        sorted_pairs = sorted(zip(xs,ys))
        # zip(*l) where l = [(x0,y0), (x1,y1)] combines same-index elements   
        # among all (tuple) inputs, producing a [(x0, x1), (y0, y1)] generator.
        xs, ys = [list(tuple) for tuple in zip(*sorted_pairs)] 
        self.ax.plot(xs, ys)
        
    def curve(self, f, xrange, npoints, style="-"):
        xs = np.linspace(xrange, npoints)
        ys = f(xs)
        self.ax.plot(xs, ys, style, color = "black")
        
    def vertical_lines(self, xs):
        for x in xs:
            self.ax.axvline(x, linestyle = '--', color = 'crimson') 

    def set_labels(self, xlabel, ylabel):
        self.ax.set_xlabel(xlabel, fontsize=16, style="italic", labelpad=10)
        self.ax.set_ylabel(ylabel, fontsize=16, style="italic", labelpad=10)
    

def plot_single_run(nqs, stds, errs, rl, wNs, Ns, el, accl, essl, title):
    fig, ax = get_logplot(ylabel = None, title = title, return_fig = True)
    fig.subplots_adjust(right=0.75)

    ax.scatter(nqs, stds, label="standard deviation")
    ax.scatter(nqs, errs, label="true deviation")

    rnqs, accl, essl = fix_sampler_lists(rl, nqs, wNs, Ns, accl, essl)

    ax2 = ax.twinx()
    ax2.set_ylabel("MCMC acceptance rate")
    ax2.set_ylim(0, 1)
    ax3 = ax.twinx()
    ax3.set_ylabel("ESS")
    ax3.set_ylim(0, 1)
    ax3.spines.right.set_position(("axes", 1.1))

    ax2.scatter(rnqs, accl, marker = "*", s = 120, color = "green",
                label='MCMC acc rate')
        
    ax3.scatter(nqs, essl, marker = "v", color = "gray",
                label='ESS')

    for i,rNq in enumerate(rnqs):
        label = 'resampled' if i==0 else None
        plt.axvline(x = rNq, color = 'tab:red', label = label, 
                    linestyle = 'dashed')

    for i,v in enumerate(el):
        label = 'expanded' if i==0 else None
        plt.axvline(x = nqs[int(v/Ns)], color = 'tab:purple', label = label, 
                    linestyle = 'dotted', linewidth = 2.5)

    combine_legends([ax, ax2, ax3])
    plt.show()
    safe_save_fig("single_run")

def safe_save_fig(filename):
    timestamp = datetime.now(pytz.timezone('Portugal')).strftime("%d_%m_%Y_%H_%M")
    filename = filename + "_" + timestamp

    i = 0
    while os.path.exists(f'{filename}({i}).png'):
        i += 1

    filename = f'{filename}({i}).png'
    plt.savefig(filename)
    print(f"> Saved figure '{filename}'.")

def combine_legends(axs):
    handles_labels_tuples = [ax.get_legend_handles_labels() for ax in axs]

    handles, labels = zip(*handles_labels_tuples)
    handles = [x for l in handles for x in l]
    labels = [x for l in labels for x in l]

    plt.legend(handles, labels, loc="lower left", fontsize=12, framealpha=0.8)

def fix_sampler_lists(rl, nqs, wNs, Ns, accl, essl):
    '''
    Fix resampler lists according to the numbers of shots (for the warm-up and 
    others). 
    
    A "measurement" considers 1 control for N >= 1 of shots, and nqs has the 
    ordered numbers of queries per measurement. However, the shots are 
    considered in independent iterations by the sampler. We want to condense
    this information to match the measurements.

    We thus divide these iterations into groups, one for each measurement.
    We consider to have resampled for a measurement if resampling occurred at
    any shot. Additionally, the resampling statistics are averaged among groups.

    rl is the list of iterations at the end of which resampling occurred.
    nqs is the list of cumulative queries for all iterations.
    wNs is the warm-up number of shots.
    accl is the ordered list of the acceptance rates for the resampling 
    occurrences.
    essl is the list of effective sample sizes for all iterations.


    So e.g. if we have 

    wNs = 10, Ns = 1
    nqs = [Nq0 = 10, ..., Nq19] 
    rl = [2, 5, 11, 15]
    accl = [A, B, C, D]

    Return:
    rnqs = [Nq0, Nq11, Nq15]
    accl = [(A+B)/2, C, D]
    '''
    Nmeas = len(nqs)

    # groups[i]: measurement number associated with iteration i.
    groups = [0] * wNs + list(chain(*[[i+1]*Ns for i in range(Nmeas-1)]))
    # rgroups[i]: measurement number associated with resampling iteration rl[i]. 
    rgroups = [0 if x < wNs else 1 + (x - wNs) // Ns for x in rl]

    gis = sorted(set(groups))
    ris = sorted(set(rgroups))
    
    rnqs = [nqs[i] for i in ris]
    essl = [mean(val for i, val in enumerate(essl) if groups[i] == g)
             for g in gis]
    accl = [mean(val for i, val in enumerate(accl) if rgroups[i] == g)
             for g in ris]
    
    return rnqs, accl, essl

def average_first_N(l, N):
    '''
    l is a list whose Nm first item are to be averaged into a single item,
    and the following ones are to be used as is.
    '''
    # Summarize warm up statistics as average.
    m = np.mean(l[:N])
    # Statistics for other updates are used as is.
    l = [m] + l[N:]
    return l

def process_and_plot(raw_estdata, save = True, show = True, processing = "binning",
                     stats = ["mean", "median"], title = None):
    '''
    Up to 2 plots: one with the root mean squared error, one with the median
    error (in separate graphs).
    '''
    assert processing in ["binning", "averaging", "averaging2", "none"]
    for stat in stats:
        estdata = process(raw_estdata, stat, processing)
        plot_est_evol(estdata, save = save, show = show, stat = stat, 
                      exp_fit = False, title = title)
    return estdata

def process(raw_estdata, stat, how = "binning"):
    assert how in ["binning", "averaging", "averaging2", "none"], how
    if how == "binning":
        try:
            estdata = process_raw_estdata(raw_estdata, stat = stat)
        except ValueError as e:
            print("> Could not bin due to error:")
            print(e)
            estdata = raw_estdata
    if how == "averaging":
        estdata = process_nonadapt_data(raw_estdata, stat = stat)
    if how == "averaging2":
        estdata = process_nonadapt_data(raw_estdata, by_step = True, stat = stat)
    if how == "none":
        estdata = raw_estdata
    return estdata

def plot_est_evol(*args, **kwargs): 
    # expfit could be removed, just fit always. Just keeping for compatibility.
    '''
    Plot (number of queries, error) points for one or more datasets, on a 
    loglog scale. 
    
    Additionally, plot the standard quantum and Heisenberg estimation limits
    in the case of a single dataset (otherwise the different y offsets and the
    many points will make things confusing).
    
    Can also plot the CR lower bounds if intended (as long as given by the 
    'lb_dict' property of the 'estdata' objects).
    '''
    save = kwargs.pop('save', True)
    show = kwargs.pop('show', True)
    ys = ["RMSE", "std"]
    for y in ys:
        id = plot_err_evol(y, *args, **kwargs)
        if id is not None:
            if save:
                safe_save_fig(id + "_est_evol")
            if show:
                plt.show()
        else:
            plt.close()
    
LONG =  {"RMSE": "*avgtype* error (normalized)",
         "std":  "*avgtype* standard deviation (normalized)"}

def plot_err_evol(which, estdatas, stat = "mean", yintercept = "fit", 
                  limits = True, CRbounds = False, plot_fit = False, 
                  iconpath = None, exp_fit = True, lims = None,
                  title = None, plotlims = True): 
    '''
    Plot either the evolution of either the true error, given by the RMSE 
    (which = "RMSE"), or its estimate, given by the standard deviation 
    (which = "std"). Use either mean or median depending on "stat" arg. 
    '''
    # Support single estdata input for compatibility with earlier code.
    if type(estdatas) is not list: estdatas = [estdatas]
    
    label = LONG[which].replace("*avgtype*", stat)
    if stat == "mean" and which=="RMSE":
        label = label.replace("mean", "root mean squared")
        
    ax = get_logplot(ylabel = label.capitalize(), title = title)
    for estdata in estdatas:
        Nq_dict, lb_dict, err_dict, std_dict = estdata.unpack_data()
        if which=="RMSE":
            y_dict = err_dict
        elif which=="std":
            y_dict = std_dict
        if len(y_dict)==0:
            # print(f"> No {stat} data to plot for {which}. [plot_err_evol]")
            return 
        id = plot_error_scatter(Nq_dict, y_dict, ax, iconpath)
    
    if len(estdatas) == 1 and plotlims:
        assert not estdata.is_empty()
        # Plot the SQL and HL, unless there are several datasets.
        plot_limits(Nq_dict, y_dict, ax, yintercept, label)
        
    if CRbounds:
        # Plot Cramér-Rao lower bounds for the estimation error.
        plot_CR_bounds(Nq_dict, lb_dict, ax, plot_fit)
        
    plt.legend(loc="lower left", fontsize=12, framealpha=0.8)
    
    if lims is not None:
        xlim, ylim = lims
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    #print("> I would like to let you know I am plotting a figure."
    #      " [plot_err_vs_Nq]")
    
    return id #ax.get_xlim(), ax.get_ylim()
    
def get_logplot(ylabel, title = None, return_fig = False):
    fig, ax = plt.subplots(1,figsize=(10,6))
    
    #title = ("Scaling of the estimation error in a with the number of "
    #    "queries to A")
    xlabel = "Number of queries"
    
    if title is not None:
        ax.set_title(title, fontsize=16, pad=25)

    ax.set_xlabel(xlabel, fontsize=16, style="italic", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=16, style="italic", labelpad=10)
    
    plt.xscale('log'); plt.yscale('log')
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    ax.spines['right'].set_color('lightgray')
    ax.spines['top'].set_color('lightgray')
    
    ax.grid(which='both')
    
    if return_fig:
        return fig, ax
    return ax

def plot_error_scatter(Nq_dict, err_dict, ax, iconpath = None, Nqmin = 0, 
                       Nqmax = None):
    def getImage(path):
        return OffsetImage(plt.imread(path, format="png"), 
                           zoom=0.5 if iconpath=="jface.png" else 0.05)
    
    # id describes the algorithm(s), for e.g. naming figures.
    id = "_".join(Nq_dict.keys())
    for key in sorted(err_dict.keys(), reverse = True):
        x, y = Nq_dict[key], err_dict[key]
        
        first_i = next(i for i,v in enumerate(x) if v >= Nqmin)
        
        if Nqmax and x[-1] > Nqmax:
            # Without the second condition, we last_i = 0 if no elements <= max.
            last_i = -next(i for i,v in enumerate(reversed(x)) if v <= Nqmax)
        else:
            last_i = len(x)

        x, y = x[first_i:last_i], y[first_i:last_i]
        
        if iconpath is not None:
            # Replace markers with a specific icon, and do not caption it.
            for xi, yi in zip(x, y):
               ab = AnnotationBbox(getImage(iconpath), (xi, yi), frameon=False)
               ax.add_artist(ab)
            continue
        
        marker = MARKER_SHAPES[key]
        size = MARKER_SIZES[key]
        color = MARKER_COLORS[key]
        label = label_from_key(key)
        
        ax.scatter(x, y, s=size, marker=marker, color=color, label=label,
                   edgecolors = 'black', linewidth = 0.75)
    return id

MARKER_SHAPES = {'classical': 's',
                 'canonical': 'o',
                 'LIS': 'd',
                 'EIS': '*', 
                 'QAES': '^',
                 'SQAE #1': 'v',
                 'SQAE #2': 'v',
                 'FQAE': 'p',
                 'IQAE - chernoff': 'P',
                 'mIQAE - chernoff': 'X',
                 'BAE': '8',
                 'aBAE': '8',
                 'adaptive': '8'}
     
MARKER_COLORS = {'classical': 'firebrick',
                 'canonical': 'gray',
                 'LIS': 'darkseagreen',
                 'EIS': 'salmon', 
                 'QAES': 'lightskyblue',
                 'SQAE #1': 'yellow',
                 'SQAE #2': 'yellow',
                 'FQAE': 'orange',
                 'IQAE - chernoff': 'violet',
                 'mIQAE - chernoff': '#DC143C',
                 'BAE': 'navy',
                 'aBAE': '#008080',
                 'adaptive': 'navy'}

MARKER_SIZES = {'classical': 82,
                'canonical': 130,
                'LIS': 110, 
                'EIS':  200, 
                'QAES': 120,
                'SQAE #1': 120,
                'SQAE #2': 120,
                'FQAE': 130,
                'IQAE - chernoff': 180,
                'mIQAE - chernoff': 180,
                'BAE': 80,
                'aBAE': 80,
                'adaptive': 80}

def label_from_key(key):
    '''
    Clean keys up for proper caption.
    '''
    if key == "LIS" or key=="EIS":
        return f"MLAE ({key})" 
    elif key == "SQAE #2" or key == "SQAE #1" or key == "SQAE #0":
        return "SAE"
    elif key == "IQAE - chernoff":
        return "IAE"
    elif key == "mIQAE - chernoff":
        return "mIAE"
    elif key == "canonical":
        return "QAE"
    else:
        # Write IQAE -> IAE, etc. for simplicity
        return key.replace("QAE", "AE", 1)

def plot_CR_bounds(Nq_dict, lb_dict, ax, plot_fit):
    if len(lb_dict)==0:
        print("> I don't have any Cramer Rao bound evaluations to plot! "
              "[plot_CR_bounds]")
        return
    
    for key in sorted(lb_dict.keys(), reverse = True):
        x, y = Nq_dict[key], lb_dict[key]
        color = MARKER_COLORS.get(key, "indianred")
        ax.plot(x, y, linewidth=2, color=color, linestyle="-.",
                label=f"Cramér-Rao ({key})")
        
        m, b = power_fit(x,y, seq=key)
            
        if plot_fit:
            yfit = match_intercept(x, y, m)
            ax.plot(x, yfit, linewidth=1.5, linestyle="dashed",  
                    color="black", label=f"O(Nq^{round(m,2)})")
                
def plot_limits(Nq_dict, err_dict, ax, yintercept, label):
    '''
    Plot the standard quantum and Heisenberg limits overposed with the data, 
    which describe the scaling of y (RMSE) wrt x (number of queries).
    
    These limits are represented by exponential decay in a linear graph / 
    straight lines in a loglog graph, but they only determine the power / slope 
    respectively. This leaves an undetermined parameter: a constant factor (B)
    / y intercept (b):
        
                 y = B*x^m <-> log(y) = m*log(x)+log(B)   (B = e^b)
    
    Thus, we must choose this parameter to fully specify a graphical 
    representation. Two options are provided:
    - yintercept == "fit": fit a log(y)=m*log(x)+b to the data, use 'b' to 
    adjust the limit lines' f(x0). 
    - yintercept == "1st": make the limit lines pass by the first datapoint 
    (x0, y0).
    '''
    bounds = ['sql','hl']
    # Draw SQL and HL as a function of x. Since they produce straight 
    # lines, the list of x coords only needs to cover the graph's width. 
    # We can take the largest Nq from the (Nq, epsilon) points; if there are 
    # several datasets, we pick the one that reaches a larger N_q (e.g. for 
    # LIS vs EIS, they're generally slightly different).    
    keys = list(err_dict.keys())
    xmaxs = [err_dict[key][-1] for key in keys]
    ref_key = keys[np.argmax(xmaxs)]
    if len(Nq_dict[ref_key]) <= 1:
        return
    
    if yintercept == "fit":
        # Determine y intercept using dataset spanning largest x-axis section.
        # print(f"> Fitting parameters for {ref_key} (to be used as reference)...")
        m, b = power_fit(Nq_dict[ref_key], err_dict[ref_key], 
                         f"{ref_key} {label}")
    
    # Do fits for other datasets if they exist, to print the fit parameters.
    for key in keys:
        if key!= ref_key:
            # print(f"> Fitting parameters for {key}...")
            power_fit(Nq_dict[key], err_dict[key], f"{key} {label}")
    
    for bound in bounds:
        power = -0.5 if bound=="sql" else -1
        xrge = Nq_dict[key]
        xs = logspace(xrge[0], xrge[-1], 1000)
        
        if yintercept == "fit":
            y0 = xs[0]**m*np.exp(b)
        if yintercept=="1st":
            y0 = err_dict[key][0]
        ys = match_intercept(xs, y0, power)
        
        ax.plot(xs, ys, 
                linewidth=3 if bound=="sql" else 2, 
                linestyle=":" if bound=="sql" else "--", 
                color="cadetblue" if bound=="sql" else"indianred", 
                label="Standard quantum limit" if bound=="sql" 
                    else "Heisenberg limit")
    
def power_fit(xs, ys, label = "", seq=None):
    '''
    Fit the parameters 'm' and 'b' in: 
                         y = x^m*exp(b) = B*x^m, B := e^b 
    Or equivalently:
                               log(y)=m*log(x)+b
    '''
    cf = opt.curve_fit(lambda x, m, b: m*x+b,  
                                np.log(xs),  np.log(ys))
    m, b = cf[0] 
    
    label = "RMSE" if len(label) == 0 else label
    label = label.capitalize()
    print(f"> {label} = O(Nq^{round(m,2)})", end = ";")

    print(f" offset = {round(b, 2)}.")
    if seq is not None:
        m_pred = -0.75 if seq=="LIS" else -1
        print(" This should be compared with the theoretical prediction "
              f"CR=O(Nq^{m_pred}) for {seq} [power_fit]. ")
    return m, b

def match_intercept(xs, y0, m):
    # Match coordinate of a y=x^m*B function with the (xs,ys) data at y 
    # intercept by adjusting B (preserve only scaling/slope).
    b_ = np.log(y0) - m*np.log(xs[0])
    # Convert back to normal scale for plotting. The scale can be converted 
    # into loglog after if desired.
    ys_ = [np.exp(m*np.log(x) + b_) for x in xs]
    return ys_  

def plot_graph(xs, ys, startat0 = None, title="", xlabel="", ylabel=""):
    fig, ax = plt.subplots(1,figsize=(10,6))
    ax.plot(xs, ys, linewidth=1, color="black")
        
    if startat0=="x" or startat0=="both":
        ax.set_xlim(left=0)
    if startat0=="y"or startat0=="both":
        ax.set_ylim(bottom=0)
    if startat0 is None:
        ax.set_xlim(left=min(xs))
        ax.set_ylim(bottom=min(ys))
    
    ax.set_xlim(right=max(xs))
    
    ax.set_title(title, fontsize=18, pad=25)
    ax.set_xlabel(xlabel, fontsize=16, style="italic", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=16, style="italic", labelpad=10)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    if startat0=="x" or startat0=="both":
        # Avoid the ugly whitespace around the yaxis.
        fill = ax.fill_between([0]+xs, [ys[0]]+ys, alpha = 0.5)
    else:
        fill = ax.fill_between(xs, ys, y2=ax.get_ylim()[0], alpha = 0.5)
    fill.set_facecolors('darkgray')
    fill.set_edgecolors('darkgray')
    
    ax.spines['right'].set_color('lightgray')
    ax.spines['top'].set_color('lightgray')
    
    ax.xaxis.set_major_locator(MaxNLocator())
    ax.yaxis.set_major_locator(MaxNLocator())
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax.grid()
    ax.grid(which='minor', alpha=0.2, linestyle='--')
    ax.grid(which='major', alpha=0.8)
    plt.show()
    safe_save_fig("graph")

def barplot_from_data(ddict, ax, title = None, label = None):
    '''
    Plot a bar plot with x ticks and labels given by the keys in 'ddict',
    and bar heights given by the corresponding normalized values.
    '''
    keys = list(ddict.keys())
    vals = list(ddict.values())

    Z = sum(vals)
    # Normalize the relative frequencies for the histogram.
    for key in keys:
        ddict[key] = ddict[key]/Z
        
    # If the keys are strings, they're assumed to be binary numbers.
    if isinstance(keys[0], str):
        # Decimal keys to be used for calculations. For labeling keep binary.
        dkeys = [int(key, 2) for key in keys]
        # Adapt the width to the x span of the graph, or else overthin bars.
        width = (max(dkeys)-1)/10
    else:
        width = 0.1
        dkeys = keys
        
    ax.bar(keys, ddict.values(), width = width, color = 'lightgray', label=label)
        
    ax.set_xlim((min(dkeys)-width, max(dkeys)+width))
    ax.set_xticks(dkeys)
    ax.set_xticklabels(keys)
    
    if title is not None:
        plt.title(title, fontsize=14, pad=15)
        
def plot_warn(f):
    def wrapper(*args, **kwargs):
        Ngraphs = len(args[1])
        if Ngraphs <= 10: 
            return f(*args, **kwargs)
        ans = ""; 
        while ans!="Y" and ans!="N":
            ans = input("\n> This is going to plot over 10 graphs. "
                        f"More specifically, {Ngraphs} graphs. "
                        "Are you sure you want that?"\
                        f" [{f.__qualname__ }]\n(Y/N)\n")
        if ans=="Y":
            return f(*args, **kwargs)
    return wrapper

def sqe_evol_from_file(filename):
    estdata = data_from_file(filename).estdata
    process_and_plot(estdata)
    


def process_nonadapt_data(raw_estdata, stat, by_step = False, every = 2):
    '''
    by_step: whether the data is ordered already by step (each element is a list
    of errors for multiple runs for a fixed step/Nq) or not (each element is a
    list of errors foor multiple steps for a fixed run).
    '''
    # Same x values across all runs. 
    keys = list(raw_estdata.Nq_dict.keys())
    estdata = deepcopy(raw_estdata)
    
    for key in keys:
        sqe_list = raw_estdata.err_dict[key]
        err_per_step = estimation_errors(sqe_list, stat = stat, 
                                         by_step = by_step)
        estdata.err_dict[key] = thin_list(err_per_step, 1, 5) # err_per_step
        estdata.Nq_dict[key] = thin_list(estdata.Nq_dict[key], 1, 5)
    
    return estdata

