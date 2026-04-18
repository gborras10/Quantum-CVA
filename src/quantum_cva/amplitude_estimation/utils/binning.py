'''
Binning to treat adaptive QAE data.
'''
from scipy import interpolate as interp, optimize as opt
import numpy as np
from pandas import cut, DataFrame

from quantum_cva.amplitude_estimation.utils.mydataclasses import EstimationData
from quantum_cva.amplitude_estimation.utils.files import data_from_file
    
strats = ["y_mean", "y_median", "slope_mean", "slope_median", "fit", "spline"]
STRAT_USED = {s: False for s in strats}
def bin_and_average(xs, ys, fixed_point = None, nbins = 15, ypower = 0.5, 
                    add_after = None, full_output = False, return_err = False,
                    strategy = "y_mean", logdomain = False, silent = True):
    '''
    Split (x, y) data points into groups depending on the x coordinate,
    and calculate the average xs and ys**ypower for each group.
    
    - If strategy=="y_mean" or "y_median", the mean/median ys among each group 
    are used directly.
    
    - If strategy=="slope_mean" or "slope_median", the mean/median slopes among 
    each group are calculated; along with the fixed point, this defines a 
    function, which we can then apply to the average x. That way, averaging 
    over points belonging to the ideal line will produce points still contained 
    in it.
    '''
    
    # Use slightly enlarged xrange so endpoints are included in the bins.
    # xrange = min(xs) - 1, max(xs) + 1
    xrange = min(xs), max(xs)
    
    if strategy not in ["slope_mean", "slope_median", "fit", "spline"]:
         df = DataFrame({"x": xs, "y": ys})
    else:
        if fixed_point is None:
            # Use K% first points to define fixed point, necessary for the 
            # slope strategy.
            K = 1
            xs, ys = np.array(xs), np.array(ys)
            N = len(xs)
            imax = int(N*K/100)
            si = np.argsort(xs)
            ii = si[:imax]

            fxs = xs[ii]
            fys = ys[ii]
            fixed_point = np.mean(fxs), np.mean(fys**0.5)

        # y coordinates are MSEs, while fixed_point[1] is MSE**0.5.
        fixed_point_sq = (fixed_point[0], fixed_point[1]**2)
            
        slopes = [calculate_slope(fixed_point_sq, (x, y)) for x,y in zip(xs,ys)]
        df = DataFrame({"x": xs, "y": ys, "slope": slopes})
    
    scale = "log"
    if not STRAT_USED[strategy] and not silent:
        print(f"> Binning ({strategy}) on [{xrange[0]},{xrange[1]}]. "
              f"Number of bins: {nbins} (evenly spaced on a {scale} scale)."
              " [bin_and_average]")
        STRAT_USED[strategy] = True
        if fixed_point is not None:
            print(f"> Using fixed point {fixed_point}.")
    
    binned_indices, bins = bin_by_values(df, xrange, nbins = nbins+1, 
                                         scale = scale)

    grouped_points = group_points(xs, ys, binned_indices)
    
    if logdomain:
        df['x'] = np.log(df['x'])
        df['y'] = np.log(df['y'])
    
    grouped = df.groupby(['bin'], observed = True)
    if strategy=="y_mean" or strategy=="slope_mean":
        df = grouped.mean().dropna()
    
    if strategy=="y_median" or strategy=="slope_median":
        df = grouped.median().dropna()

    if logdomain:
        df['x'] = np.exp(df['x'])
        df['y'] = np.exp(df['y'])

    xs = df['x'].values

    
    if strategy in ["y_mean", "y_median"]:
        ys = df['y'].values**ypower
    if strategy in ["slope_mean", "slope_median"]:
        ys = eval_power_function(df['x'], df['slope'], fixed_point) 
        ys = np.array(ys.tolist())**0.5
    if strategy=="fit":
        ys = list(map(lambda arg: arg**0.5, ys))
        f = lambda x, slope: eval_power_function(x, slope, fixed_point) 
        slope, _ = opt.curve_fit(f, xs, ys)
        
        # Take care to evenly space evaluations in logspace. For other methods,
        # this is done by construction due to calculating points not a function.
        xrange = list(map(np.log, xrange))
        binwidth = (xrange[1]-xrange[0])/nbins
        xrange[0] += binwidth/2
        xrange[1] -= binwidth/2
        
        xs = np.logspace(*xrange, nbins, base = np.e)
        ys = [eval_power_function(x, slope, fixed_point) for x in xs]
    if strategy=="spline":
        ys = list(map(lambda arg: arg**0.5, ys))
        xrange = list(map(np.log, xrange))
        
        which = 3
        if which==0:
            sp = interp.interp1d(xs, ys, kind = "cubic")
        if which==1:
            sp = interp.InterpolatedUnivariateSpline(xs,ys)
        if which==2:
            # The factor is trying to get ~5 knots, same as nbins I was using.
            sp = interp.UnivariateSpline(xs,ys,s=len(xs)*2.4e-6)
        if which==3:
            # Choose the middle knots to match bin boundaries. Exclude the 
            # endpoints (do [1:-1]) because they're added automatically, see:
            # print("knots", list(map(np.log10,sp.get_knots())))
            t = np.logspace(*xrange, nbins, base = np.e)[1:-1]
             # Remove duplicates or error 'x must be increasing if s>0'.
            xs = np.array(xs)
            ys = np.array(ys)
            _, unique_indices = np.unique(xs, return_index=True)
            xs = xs[unique_indices]
            ys = ys[unique_indices]

            
            sp = interp.LSQUnivariateSpline(xs,ys,t)
            
        binwidth = (xrange[1]-xrange[0])/nbins
        xrange[0] += binwidth/2
        xrange[1] -= binwidth/2
        xs = np.logspace(*xrange, nbins, base = np.e)
        ys = sp(list(xs))

    if add_after is not None:
        # Add some fixed point(s).
        for point in add_after:
            xs = np.insert(xs, 0, point[0])
            ys = np.insert(ys, 0, point[1])
    
    if full_output:
        return xs, ys, bins, grouped_points
    elif return_err:
        assert strategy == "y_mean", logdomain == False
        std_df = grouped.std()
        # Standard error of the mean.
        # std_df = std_df / grouped.count().pow(0.5)
        dx = std_df['x'].values
        # Error propagation. 
        dy = std_df['y'].values * ypower * (grouped.mean()['y'].values ** (ypower - 1))
        return xs, ys, dx, dy
    else:
        return xs, ys
    
def bin_by_values(df, xrange, nbins, by="x", scale="log"):
    if scale == "log":
        # Make the bins evenly spaced in logspace.
        bins = np.logspace(np.log(xrange[0]), np.log(xrange[1]), 
                              nbins, base = np.e)
    if scale == "linear":
        bins = np.linspace(xrange[0], xrange[1], nbins)
        
    df["bin"] = cut(df["x"], bins=bins, include_lowest = True)
    # A dictionary {bin interval: [indices]}.
    # Observed irrelevant (not Categoricals), just to silence deprecation warning.
    binned_indices = df.groupby("bin", observed = True).groups
    return binned_indices, bins

def uniform_points(xrange, npoints, log = False):
    # Generate evenly spaced points in xrange, in a linear or on a log scale.
    unif = [np.random.uniform(0,1) for i in range(npoints)]
    if log:
        # Make the samples be uniformly distributed in logspace.
        xmin, xmax = np.log(xrange)
        xwidth = xmax - xmin
        xs = [np.exp(xwidth*x + xmin) for x in unif]
    else:
        # Make the samples be uniformly distributed in the traditional sense.
        xmin, xmax = xrange
        xwidth = xmax-xmin
        xs = [xwidth*x + xmin for x in unif]
    return xs

def generate_points(npoints, xrange, mean, std_fun, 
                    logspace = True, noise = True):
    xs = uniform_points(xrange, npoints, logspace)
      
    if noise:
        samples = [np.random.normal(mean, std_fun(x)) for x in xs]
        sqerrs = [(x - mean)**2 for x in samples] 
    else:
        sqerrs = [std_fun(x)**2 for x in xs]
    add = "" if noise else "out"
    print(f"> Generated squared error evolution simulation with{add} noise.")
    
    # Sort for ease of use.
    sorted_pairs = sorted(zip(xs,sqerrs))
    # zip(*l) where l = [(x0,y0), (x1,y1)] combines same-index elements   
    # among all (tuple) inputs, producing a [(x0, x1), (y0, y1)] generator.
    xs, sqerrs = [list(tuple) for tuple in zip(*sorted_pairs)] 
    
    return xs, sqerrs

def group_points(xs, ys, binned_indices):
    groups = []
    for key in binned_indices.keys():
        idx_list = binned_indices[key]
        group_xs = [xs[i] for i in idx_list]
        group_ys = [ys[i] for i in idx_list]
        group_pts = list(zip(group_xs, group_ys))
        groups.append(group_pts)
        
    return groups

def eval_power_function(x, power, fixed_point):
    x0, y0 = fixed_point
    const = x0*y0
    y = const*x**power
    return y

def power_function(power, fixed_point):
    x0, y0 = fixed_point
    const = x0*y0
    f = lambda x: const*x**power
    return f

def affine_function(slope, fixed_point, log = True):
    x0, y0 = to_log(fixed_point) if log else fixed_point
    f = lambda x: slope*(x - x0) + y0
    if log:
        f = logfun(f)
    return f

def to_log(point):
    # Convert a point into a logscale representation.
    x, y = point
    x, y = np.log(x), np.log(y)
    return (x, y)
        
def logfun(f):
    # Convert a function into acting on log scale f(log(x))=...
    flog = lambda x: np.exp(f(np.log(x)))
    return flog

def logscale(f):
    def wrapper(*args):
        return f(*np.log(args))
    return wrapper

@logscale
def calculate_slope(reference, point):
    x0, y0 = reference
    x, y = point
    slope = (y0-y)/(x0-x)
    return slope

def row_log_slope(row, reference):
    # Calculate the slope associated with a dataframe row.
    point = (row["x"], row["y"])
    slope = calculate_slope(reference, point)
    return slope

def sqe_evol_from_file(filename, preprocessed, label):
    '''
    Plots the RMSE evolution from a file. Processes the data by binning if 
    not preprocessed.
    '''
    execdata = data_from_file(filename)
    estdata = execdata.estdata
    
    if not preprocessed: 
        estdata = process_raw_estdata(estdata, label)
        
    plot_err_vs_Nq(estdata, exp_fit = False)

def process_raw_estdata(raw_estdata, stat, label = None):
    '''
    Read raw data from EstimationData object, process it, then save processed 
    data to other EstimationData object.
    '''
    if label is None:
        keys = list(raw_estdata.Nq_dict.keys())
        estdatas = []
        for key in keys:
            estdata_i = process_raw_estdata(raw_estdata, stat, label = key)
            estdatas.append(estdata_i)
        # Join everything in single EstimationData object to plot in same grpah.
        estdata = EstimationData.join(estdatas)
        return estdata
        
    
    nqs = raw_estdata.Nq_dict[label]
    sqes = raw_estdata.err_dict[label]
    
    # Binning strategies need identifiers because they concern quantities other
    # than y, such as slopes.
    strat = "y_" + stat
    gxs, gys = bin_and_average(nqs, sqes, strategy = strat)
    if label in list(raw_estdata.std_dict.keys()):
        # Plot also std.
        stds = raw_estdata.std_dict[label]
        gxs, gy2s = bin_and_average(nqs, stds, ypower = 1, strategy = strat)
    else:
        gy2s = None

    estdata = EstimationData()
    estdata.add_data(label, nqs = gxs, lbs = None, errs = gys, stds = gy2s)
    return estdata

def test_plot_err_vs_Nq():
    '''
    Test the 'plot_err_vs_Nq' function from the 'utils.plotting' using the data
    generated in this script.
    '''
    
    npoints = 10
    power = -0.5
    fixed_point = (100, 1e-2) 
    mean = 0.7
    xrange = (fixed_point[0], 10**5)
    f = power_function(power, fixed_point)
    xs, ys = generate_points(npoints, xrange, mean, f, noise = False)
    # Perturb the 1st point (bring it lower wrt line relative to the others) 
    # to assess effect. If yintercept == "1st", the line will fit weird.
    ys[0] /= 1.5
    
    estdata = EstimationData()
    estdata.add_data("*test*", nqs = xs, lbs = None, errs = ys)
    plot_err_evol("RMSE", estdata, yintercept = "fit")