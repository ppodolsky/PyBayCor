import pymc, seaborn as sns
import gzip
import commons
from pymc import Exponential, deterministic, Poisson, Uniform, Normal, observed, MCMC, Matplot, stochastic
from argparse import ArgumentParser
from operator import itemgetter

import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import pickle
import random

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


ap = ArgumentParser("Using Monte-Carlo Markov-Chains for estimating LHC fits.")

# Default constants
ap.add_argument('-n', action='store', default=1000000, type=int, help='Number of events')
ap.add_argument('-b', action='store', default=5000, type=int, help='Number of burnout')
ap.add_argument('-c', action='store', required=True, type=int, help='Number of the combination')
ap.add_argument('-bins', action='store_true', help='If set, then output as histrograms')
ap.add_argument('-i', action='store_true', help='Print information about combiner')
ap.add_argument('-u', action='store_true', help='Switch combination function to uniform')
ap.add_argument('-vars', nargs='+', required=True, help='Variables')
args = vars(ap.parse_args())

# Constants
N = args['n']
combination = args['c']
print_information = args['i']
is_mcmc = not args['u']
bins = args['bins']
burnout = args['b']
desired_variables, lower_bounds, upper_bounds, edges = parse_dv(args['vars'], bins)

sns.set_style("whitegrid")
matplotlib.rcParams.update({'font.size': 10})
pgc_utils = PyGammaCombo.gammacombo_utils
cout = PyGammaCombo.gammacombo_utils.getCout()
extract = PyGammaCombo.gammacombo_utils.extractFromRooArgSet
toRooRealVar = PyGammaCombo.gammacombo_utils.toRooRealVar

if __name__ == "__main__":
    gce = PyGammaCombo.gammacombo_utils.getGammaComboEngine("")
    cmb = gce.getCombiner(combination)
    cmb.combine()
    parameters = cmb.getParameters()
    parameter_names = list(cmb.getParameterNames())
    pdf = cmb.getPdf()
    if print_information:
        print("Required parameters for combination:")
        for p in parameter_names:
            param = extract(parameters, p)
            param.Print()

    if is_mcmc:
        # Declare vars
        var_dict = {}
        for i, p in enumerate(desired_variables):
            var_dict[p] = Uniform(p, doc="{}".format(p), lower=lower_bounds[i], upper=upper_bounds[i])

        # Dynamically create PyMC sampling function
        stochastic_args = ','.join(["{}=var_dict['{}']".format(k, k) for k in var_dict.keys()])
        exec("@stochastic\n"
                 "def combi({}, value=0):\n"
                 "\tfor p in desired_variables:\n"
                 "\t\tparameters.setRealValue(p, var_dict[p])\n"
                 "\treturn pdf.getLogVal()\n".format(stochastic_args))
        # Define and start sampling
        mcmc = MCMC([combi] + list(var_dict.values()),
                    db='pickle',
                    dbmode='w',
                    dbname='mcmc-{}_combiner.pickle'.format(combination))
        mcmc.sample(iter=N, burn=burnout, thin=1)

        # Output
        data = {v: mcmc.trace(v)[:] for v in desired_variables}
        if bins:
            with  gzip.open('output/bins.dat.gz', 'w') as file:
                data_to_save = {}
                for v in data:
                    data_to_save[v] = np.histogram(data[v], bins=edges[v])
                pickle.dump({'data': data_to_save}, file, protocol=2)
        else:
            with gzip.open('output/raw.dat.gz', 'w') as file:
                pickle.dump({'data': data}, file, protocol=2)
        for v in desired_variables:
            commons.plot(data[v], combination, v)

    else: #not MCMC
        var_dict = {}
        weights = []
        data = {v: [] for v in desired_variables}
        for i in range(N):
            for i, p in enumerate(desired_variables):
                rval = np.random.uniform(lower_bounds[i], upper_bounds[i])
                data[p].append(rval)
                parameters.setRealValue(p, rval)
            weights.append(pdf.getVal())
        if bins:
            with gzip.open('output/bins.dat.gz', 'w') as file:
                data_to_save = {}
                for v in data:
                    data_to_save[v] = np.histogram(data[v], bins=edges[v], weights=weights)
                    hist, bin_edges = data_to_save[v]
                    plt.bar(bin_edges[:-1], hist, width=(max(bin_edges) - min(bin_edges))/len(bin_edges))
                    plt.xlim(min(bin_edges), max(bin_edges))
                    plt.savefig('output/{}_{}.png'.format(combination, v))
                    plt.clf()
                pickle.dump({'data': data_to_save}, file, protocol=2)
        else:
            with gzip.open('output/raw.dat.gz', 'w') as file:
                pickle.dump({'data': data, 'weights': weights}, file, protocol=2)