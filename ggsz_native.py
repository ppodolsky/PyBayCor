import numpy as np
import commons
import pymc
import seaborn as sns
from pymc import Exponential, deterministic, Poisson, Uniform, Normal, observed, MCMC, Matplot, stochastic
from itertools import combinations

import matplotlib
import matplotlib.ticker

# Constants
sns.set_style("whitegrid")
matplotlib.rcParams.update({'font.size': 10})
N = 1000000 # Sample size (N/10 used for burning out)
load_last_model = False
experiment = commons.lumi3fb

if __name__ == "__main__":
    for name, lower_limit, upper_limit in [('q1', 0, 180)]:
        print("Calculations for {}...".format(name))
        # Vars
        gamma = Uniform("gamma", doc="$\gamma$", lower=lower_limit, upper=upper_limit)
        deltaB = Uniform("deltaB", doc="$\delta_B$", lower=lower_limit, upper=upper_limit)
        rB = Uniform("rB", doc='$r_B$', lower=0.02, upper=0.2)
        var_list = [gamma, deltaB, rB]

        @deterministic
        def x_plus_gen(gamma=gamma, deltaB=deltaB, rB=rB):
            return commons.x_plus(gamma, deltaB, rB)
        @deterministic
        def y_plus_gen(gamma=gamma, deltaB=deltaB, rB=rB):
            return commons.y_plus(gamma, deltaB, rB)
        @deterministic
        def x_minus_gen(gamma=gamma, deltaB=deltaB, rB=rB):
            return commons.x_minus(gamma, deltaB, rB)
        @deterministic
        def y_minus_gen(gamma=gamma, deltaB=deltaB, rB=rB):
            return commons.y_minus(gamma, deltaB, rB)

        #Averaged experiments
        @stochastic
        def vars(x_plus_gen=x_plus_gen, y_plus_gen=y_plus_gen, x_minus_gen=x_minus_gen, y_minus_gen=y_minus_gen, value=0):
            return pymc.mv_normal_like([x_plus_gen, y_plus_gen, x_minus_gen, y_minus_gen], experiment['y_exp'], experiment['y_covar_inv'])

        # Model
        if load_last_model:
            mcmc = pymc.database.pickle.load('mcmc-{}.pickle'.format(name))
        else:
            mcmc = MCMC([vars, gamma, deltaB, rB],
                        db='pickle',
                        dbmode='w',
                        dbname='mcmc-{}.pickle'.format(name))
            mcmc.sample(iter = N, burn = min(5000, int(N/10)), thin = 1)

        for v in var_list:
            commons.plot(mcmc.trace(v.__name__)[:], v.__doc__, "pics/{}_{}.png".format(name, v.__name__))

        for x,y in combinations(var_list, 2):
           commons.plot_2d_hist(mcmc.trace(x.__name__)[:], mcmc.trace(y.__name__)[:],
                         "pics/{}_{}-{}_hist.png".format(name, x.__name__, y.__name__))