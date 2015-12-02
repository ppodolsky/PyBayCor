import pymc, seaborn as sns
import sys
import gzip
import commons
from pymc import Exponential, deterministic, Poisson, Uniform, Normal, observed, MCMC, Matplot, stochastic
from itertools import combinations
from ggsz_native import plot, plot_2d_hist

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker

sys.path.append('/home/pygammacombo')

import PyGammaCombo
tstr = PyGammaCombo.TString

# Constants
sns.set_style("whitegrid")
matplotlib.rcParams.update({'font.size': 10})
if len(sys.argv) == 2: # Set sample size (N/10 used for burning out)
    N = int(sys.argv[1])
else:
    N = 1000000
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
        cmb = gce.getCombiner(113)
        cmb.combine()
        parameters = cmb.getParameters()
        pdf = cmb.getPdf()

        #Averaged experiments
        @stochastic
        def combi(gamma=gamma, deltaB=deltaB, rB=rB, value=0):
            parameters.setRealValue("g", commons.degToRad(gamma))
            parameters.setRealValue("r_dk", rB)
            parameters.setRealValue("d_dk", commons.degToRad(deltaB))
            return pdf.getLogVal()

        mcmc = MCMC([combi, gamma, deltaB, rB],
                    db='pickle',
                    dbmode='w',
                    dbname='mcmc-{}_combiner.pickle'.format(name))
        mcmc.sample(iter = N, burn = min(5000, int(N/10)), thin = 1)

        for v in var_list:
            with gzip.open("/output/{}.dat.gz".format(v.__name__), 'wb') as file:
                file.write(mcmc.trace(v.__name__)[:].tostring())