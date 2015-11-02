import numpy as np
import commons
import pymc
import seaborn as sns
from pymc import Exponential, deterministic, Poisson, Uniform, Normal, observed, MCMC, Matplot, stochastic
from itertools import combinations
from ggsz_native import plot, plot_2d_hist

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker

import PyGammaCombo
tstr = PyGammaCombo.TString

# Constants
sns.set_style("whitegrid")
matplotlib.rcParams.update({'font.size': 10})
N = 1000000 # Sample size (N/10 used for burning out)
number_of_bins = 90
load_last_model = False

if __name__ == "__main__":
    for name, lower_limit, upper_limit in [('q1', 0, 180), ('q4', -180, 0)]:
        print("Calculations for {}...".format(name))
        # Vars
        gamma = Uniform("gamma", doc="$\gamma$", lower=lower_limit, upper=upper_limit)
        deltaB = Uniform("deltaB", doc="$\delta_B$", lower=lower_limit, upper=upper_limit)
        rB = Uniform("rB", doc='$r_B$', lower=0, upper=0.5)
        var_list = [gamma, deltaB, rB]
        parameters = PyGammaCombo.ParametersAbs()
        g = parameters.newParameter(tstr("g"))
        g.scan = parameters.range(commons.degToRad(lower_limit), commons.degToRad(upper_limit))
        r_dk = parameters.newParameter(tstr("r_dk"))
        d_dk = parameters.newParameter(tstr("d_dk"))
        d_dk.scan = parameters.range(commons.degToRad(lower_limit), commons.degToRad(upper_limit))

        #Averaged experiments
        @stochastic
        def ggsz(gamma=gamma, deltaB=deltaB, rB=rB, value=0):
            g.startvalue = commons.degToRad(gamma)
            r_dk.startvalue = rB
            d_dk.startvalue = commons.degToRad(deltaB)
            GGSZ_pdf = PyGammaCombo.PDF_GGSZ(pars=parameters)
            pdf = GGSZ_pdf.getPdf()
            return pdf.getLogVal()

        # Model
        if load_last_model:
            mcmc = pymc.database.pickle.load('mcmc-{}_gc.pickle'.format(name))
        else:
            mcmc = MCMC([ggsz, gamma, deltaB, rB],
                        db='pickle',
                        dbmode='w',
                        dbname='mcmc-{}_gc.pickle'.format(name))
            mcmc.sample(iter = N, burn = min(5000, int(N/10)), thin = 1)

        for v in var_list:
            plot(mcmc.trace(v.__name__)[:], v.__doc__, "{}_{}_gc.png".format(name, v.__name__))

        for x,y in combinations(var_list, 2):
            plot_2d_hist(mcmc.trace(x.__name__)[:], mcmc.trace(y.__name__)[:],
                         "{}_{}-{}_hist_gc.png".format(name, x.__name__, y.__name__))