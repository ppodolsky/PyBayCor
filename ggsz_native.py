import numpy as np
import commons
import pymc
import seaborn as sns
from pymc import Exponential, deterministic, Poisson, Uniform, Normal, observed, MCMC, Matplot, stochastic
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker

# Constants
sns.set_style("whitegrid")
matplotlib.rcParams.update({'font.size': 10})
N = 1000000 # Sample size (N/10 used for burning out)
number_of_bins = 90
load_last_model = False
experiment = commons.lumi3fb

# Output fuctions
def plot(trace, var_name, fname):
    hpd9999 = pymc.utils.hpd(trace , 1.-0.9999)
    hpd95 = pymc.utils.hpd(trace , 1.-0.95)
    hpd683 = pymc.utils.hpd(trace , 1.-0.683)
    trace_mean = trace.mean()

    plt.autoscale(tight=True)
    plt.tight_layout()

    ax = plt.subplot(211)
    plt.xticks(rotation=70)
    plt.hist(trace[(hpd9999[0] < trace) & (trace < hpd9999[1])], histtype='stepfilled', bins=number_of_bins, alpha=0.75, color="#A60628", normed=True)
    plt.axvline(hpd95[0], color='black', alpha=0.55, linestyle='dashed', linewidth=1)
    plt.axvline(hpd95[1], color='black', alpha=0.55, linestyle='dashed', linewidth=1)
    plt.axvline(hpd683[0], color='black', alpha=0.95, linestyle='dashed', linewidth=1)
    plt.axvline(hpd683[1], color='black', alpha=0.95, linestyle='dashed', linewidth=1)
    plt.axvline(trace_mean, color='black', alpha=0.95, linestyle='dashed', linewidth=2)
    ticks = hpd9999.tolist() + hpd95.tolist() + hpd683.tolist() + [trace_mean]
    ax.set_xticks(ticks)
    ax.set_xticklabels(list("%.4f" % tick for tick in ticks))
    ax.set_yticks([])

    plt.title("Posterior distributions of the variable {}".format(var_name))
    plt.xlabel("{} value".format(var_name))
    plt.tight_layout()

    ax = plt.subplot(212)
    ax.set_xticks([])
    plt.plot(trace)
    plt.axhline(hpd95[0], color='black', alpha=0.55, linestyle='dashed', linewidth=1)
    plt.axhline(hpd95[1], color='black', alpha=0.55, linestyle='dashed', linewidth=1)
    plt.axhline(hpd683[0], color='black', alpha=0.95, linestyle='dashed', linewidth=1)
    plt.axhline(hpd683[1], color='black', alpha=0.95, linestyle='dashed', linewidth=1)
    plt.axhline(trace_mean, color='black', alpha=0.95, linestyle='dashed', linewidth=2)
    plt.title("Values of the trace for {}".format(var_name))
    plt.tight_layout()
    plt.savefig(fname)
    plt.clf()

def plot_2d_hist(x, y, fname):
    sns.jointplot(x, y, kind="hex", stat_func=None)
    plt.savefig(fname)
    plt.clf()

def print_hpd(var, lvl):
    print("{} {}% HPD: {}".format(var.__name__, int(lvl*100), pymc.utils.hpd(var.trace[:] , 1.-lvl)))

if __name__ == "__main__":
    for name, lower_limit, upper_limit in [('q1', 0, 180), ('q4', -180, 0)]:
        print("Calculations for {}...".format(name))
        # Vars
        gamma = Uniform("gamma", doc="$\gamma$", lower=lower_limit, upper=upper_limit)
        deltaB = Uniform("deltaB", doc="$\delta_B$", lower=lower_limit, upper=upper_limit)
        rB = Uniform("rB", doc='$r_B$', lower=0, upper=0.5)
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
            plot(mcmc.trace(v.__name__)[:], v.__doc__, "pics/{}_{}.png".format(name, v.__name__))

        for x,y in combinations(var_list, 2):
            plot_2d_hist(mcmc.trace(x.__name__)[:], mcmc.trace(y.__name__)[:],
                         "pics/{}_{}-{}_hist.png".format(name, x.__name__, y.__name__))