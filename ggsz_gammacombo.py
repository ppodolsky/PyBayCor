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
    for name, lower_limit, upper_limit in [('q1', 0, 180)]:
        print("Calculations for {}...".format(name))
        # Vars
        gamma = Uniform("gamma", doc="$\gamma$", lower=lower_limit, upper=upper_limit)
        deltaB = Uniform("deltaB", doc="$\delta_B$", lower=lower_limit, upper=upper_limit)
        rB = Uniform("rB", doc='$r_B$', lower=0.02, upper=0.2)
        var_list = [gamma, deltaB, rB]
        parameters = PyGammaCombo.ParametersAbs()
        g = parameters.newParameter(tstr("g"))
        g.unit = tstr('Rad')
        g.scan = parameters.range(commons.degToRad(lower_limit), commons.degToRad(upper_limit))
        g.phys = parameters.range(-7, 7)
        g.force = parameters.range(commons.degToRad(0), commons.degToRad(90))
        g.bboos = parameters.range(commons.degToRad(0), commons.degToRad(180))

        r_dk = parameters.newParameter(tstr("r_dk"))
        r_dk.scan = parameters.range(0.02, 0.2)
        r_dk.phys = parameters.range(0, 1e-4)
        r_dk.force = parameters.range(0.02, 0.16)
        r_dk.bboos = parameters.range(0.01, 0.22)


        d_dk = parameters.newParameter(tstr("d_dk"))
        d_dk.unit = tstr('Rad')
        d_dk.scan = parameters.range(commons.degToRad(lower_limit), commons.degToRad(upper_limit))
        g.phys = parameters.range(-7, 7)
        g.force = parameters.range(commons.degToRad(0), commons.degToRad(90))
        g.bboos = parameters.range(commons.degToRad(-180), commons.degToRad(180))

        #Averaged experiments
        @stochastic
        def ggsz(gamma=gamma, deltaB=deltaB, rB=rB, value=0):
            g.startvalue = commons.degToRad(gamma)
            r_dk.startvalue = rB
            d_dk.startvalue = commons.degToRad(deltaB)
            GGSZ_pdf = PyGammaCombo.PDF_GGSZ(PyGammaCombo.Utils.lumi3fb,
                                             PyGammaCombo.Utils.lumi3fb,
                                             PyGammaCombo.Utils.lumi3fb,
                                             pars=parameters)
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
            plot(mcmc.trace(v.__name__)[:], v.__doc__, "pics/{}_{}_gc.png".format(name, v.__name__))

        for x,y in combinations(var_list, 2):
            plot_2d_hist(mcmc.trace(x.__name__)[:], mcmc.trace(y.__name__)[:],
                         "pics/{}_{}-{}_hist_gc.png".format(name, x.__name__, y.__name__))