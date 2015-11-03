import numpy as np
import math
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
number_of_bins = 100
load_last_model = False
pgc_utils = PyGammaCombo.gammacombo_utils
cout = PyGammaCombo.gammacombo_utils.getCout()

if __name__ == "__main__":
    for name, lower_limit, upper_limit in [('q1', 0, 180)]:
        print("Calculations for {}...".format(name))
        # Vars
        gamma = Uniform("gamma", doc="$\gamma$", lower=lower_limit, upper=upper_limit)
        deltaB = Uniform("deltaB", doc="$\delta_B$", lower=lower_limit, upper=upper_limit)
        rB = Uniform("rB", doc='$r_B$', lower=0.02, upper=0.2)
        var_list = [gamma, deltaB, rB]
        gce = PyGammaCombo.gammacombo_utils.getGammaComboEngine("")
        cmb = gce.getCombiner(26)
        cmb.combine()
        parameters = cmb.getParameters()
        pdf = cmb.getPdf()

        #Averaged experiments
        @stochastic
        def ggsz(gamma=gamma, deltaB=deltaB, rB=rB, value=0):
            parameters.setRealValue("g", commons.degToRad(gamma))
            parameters.setRealValue("r_dk", rB)
            parameters.setRealValue("d_dk", commons.degToRad(deltaB))
            return pdf.getLogVal()

        # Model
        if load_last_model:
            mcmc = pymc.database.pickle.load('mcmc-{}_combiner.pickle'.format(name))
        else:
            mcmc = MCMC([ggsz, gamma, deltaB, rB],
                        db='pickle',
                        dbmode='w',
                        dbname='mcmc-{}_combiner.pickle'.format(name))
            mcmc.sample(iter = N, burn = min(5000, int(N/10)), thin = 1)

        for v in var_list:
            plot(mcmc.trace(v.__name__)[:], v.__doc__, "pics/{}_{}_combiner.png".format(name, v.__name__))

        for x,y in combinations(var_list, 2):
            plot_2d_hist(mcmc.trace(x.__name__)[:], mcmc.trace(y.__name__)[:],
                         "pics/{}_{}-{}_hist_combiner.png".format(name, x.__name__, y.__name__))