import pymc, seaborn as sns
import gzip
import commons
import sys
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import pickle
import random
import time
from pymc import Exponential, deterministic, Poisson, Uniform, Normal, observed, MCMC, Matplot, stochastic
from argparse import ArgumentParser
from operator import itemgetter

import PyGammaCombo
tstr = PyGammaCombo.TString

import numpy as np

def parse_dv(args, bins):
    arglen = 4 if bins else 3
    names = args[0:len(args):arglen]
    lower_bounds = list(map(float, args[1:len(args):arglen]))
    upper_bounds = list(map(float, args[2:len(args):arglen]))
    if bins:
        bin_number = list(map(int, args[3:len(args):arglen]))
        edges = {}
        for i, name in enumerate(names):
            edges[name] = np.linspace(lower_bounds[i], upper_bounds[i], bin_number[i] + 1)
        return names, lower_bounds, upper_bounds, edges
    else:
        return names, lower_bounds, upper_bounds, None

def load_default_variables(cmb, bins):
    gc_parameters = PyGammaCombo.ParametersGammaCombo()
    parameter_names = list(cmb.getParameterNames())
    variables = {}

    for parameter_name in parameter_names:
        min = gc_parameters.var(tstr(parameter_name)).scan.min
        max = gc_parameters.var(tstr(parameter_name)).scan.max
        variables[parameter_name] = {"min": min,
                                     "max": max,
                                     "edges": np.linspace(min, max, bins + 1)}
    return variables



ap = ArgumentParser("Using Monte-Carlo Markov-Chains for estimating LHC fits.")

# Default constants
ap.add_argument('-c', action='store', required=True, type=int, help='Number of the combination')
ap.add_argument('-n', action='store', required=True, type=int, help='Number of events')
ap.add_argument('-b', action='store', default=5000, type=int, help='Number of burnout')
ap.add_argument('-bins', action='store_true', help='If set, then output as histrograms')
ap.add_argument('-plot', action='store_true', help='Plot bars and graphs')
ap.add_argument('-u', action='store_true', help='Switch combination function to uniform')
args = vars(ap.parse_args())
print(' '.join(sys.argv))

# Constants
N = args['n']
combination = args['c']
is_mcmc = not args['u']
bins = args['bins']
bins_amount = 400
plot_graph = args['plot']
burnout = args['b']

sns.set_style("whitegrid")
matplotlib.rcParams.update({'font.size': 10})
pgc_utils = PyGammaCombo.gammacombo_utils
cout = PyGammaCombo.gammacombo_utils.getCout()
extract = PyGammaCombo.gammacombo_utils.extractFromRooArgSet
toRooRealVar = PyGammaCombo.gammacombo_utils.toRooRealVar

if __name__ == "__main__":
    start_time = time.time()
    gce = PyGammaCombo.gammacombo_utils.getGammaComboEngine("")
    cmb = gce.getCombiner(combination)
    cmb.combine()
    pdf = cmb.getPdf()
    parameters = cmb.getParameters()

    weights=None

    variables = load_default_variables(cmb, bins_amount)

    if is_mcmc:
        # Declare vars
        var_dict = {}
        for name in variables:
            var_dict[name] = Uniform(name,
                                     doc="{}".format(name),
                                     lower=variables[name]["min"],
                                     upper=variables[name]["max"])
        # Dynamically create PyMC sampling function
        stochastic_args = ','.join(["{}=var_dict['{}']".format(k, k) for k in var_dict.keys()])
        exec("@stochastic\n"
                 "def combi({}, value=0):\n"
                 "\tfor p in variables:\n"
                 "\t\tparameters.setRealValue(p, var_dict[p])\n"
                 "\treturn max(pdf.getLogVal(), -300)\n".format(stochastic_args))
        # Define and start sampling
        mcmc = MCMC([combi] + list(var_dict.values()),
                    db='pickle',
                    dbmode='w',
                    dbname='mcmc-{}_combiner.pickle'.format(combination))
        mcmc.sample(iter=N, burn=burnout, thin=1)
        # Output
        data = {v: mcmc.trace(v)[:] for v in variables}
        if bins:
            with  gzip.open('output/bins.dat.gz', 'w') as file:
                data_to_save = {}
                for name in variables:
                    data_to_save[name] = np.histogram(data[name], bins=variables[name]['edges'])
                pickle.dump({'data': data_to_save}, file, protocol=2)
        else:
            with gzip.open('output/raw.dat.gz', 'w') as file:
                pickle.dump({'data': data}, file, protocol=2)
    else: #not MCMC
        var_dict = {}
        weights = []
        data = {v: [] for v in variables}
        for i in range(N):
            for name in variables:
                rval = np.random.uniform(variables[name]['min'], variables[name]['max'])
                data[name].append(rval)
                parameters.setRealValue(name, rval)
            weights.append(pdf.getVal())
        if bins:
            with gzip.open('output/bins.dat.gz', 'w') as file:
                data_to_save = {}
                for name in variables:
                    data_to_save[name] = np.histogram(data[name], bins=variables[name]['edges'], weights=weights)
                pickle.dump({'data': data_to_save}, file, protocol=2)
        else:
            with gzip.open('output/raw.dat.gz', 'w') as file:
                pickle.dump({'data': data, 'weights': weights}, file, protocol=2)
    if plot_graph:
        for name in variables:
            commons.plot(np.array(data[name]), combination, name, weights=weights, bins=bins_amount)

    print("Total time: {}".format(time.time() - start_time))