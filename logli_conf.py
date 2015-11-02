import sympy as sp
import numpy as np
import math
import scipy.optimize as opt
import commons
from sympy.utilities.lambdify import lambdify, implemented_function

from scipy.stats import chi2, norm

solutions = {1: (67.45671972804791, 130.5332959537717, 0.0940606150464772),
             4: (-112.54293893364066, -49.46663201608742, 0.09406008329558219)}

ll_hat1 = -0.5 * commons.chi2(solutions[1]) #Values of ll function in globally optimal points
ll_hat2 = -0.5 * commons.chi2(solutions[4])

#Technical stuff
variable_to_others = {"gamma": ("deltaB", "rB"), "deltaB": ("gamma", 'rB'), "rB": ("gamma", "deltaB")}
variable_index = {"gamma": 0, "deltaB": 1, "rB": 2}
variable_ref = {"gamma" : commons.gamma, "deltaB": commons.deltaB, "rB": commons.rB}


def find_CI(quartile, variable, critical, search = "left", grid_left = None, grid_right = None, precision = 0.001):
    '''
    We take {regions} points in our domain and test every point for its likelihood value,
    then we look at points where values of ll ratio criteria cross critical value of chi^2 distribution
    It means, we should choose {regions} in such way that at least one point lays in the confidence interval
    at the first step.
    '''
    regions = 20
    solution = solutions[quartile]
    ll_hat = -0.5 * commons.chi2(solution)
    critical_val = chi2.ppf(critical, df=1)
    other1 = variable_to_others[variable][0]
    other2 = variable_to_others[variable][1]
    variable_bounds = (grid_left if not grid_left is None else commons.bounds[quartile][variable_index[variable]][0],
                       grid_right if not grid_right is None else commons.bounds[quartile][variable_index[variable]][1])
    grid = np.linspace(start = variable_bounds[0], stop = variable_bounds[1], num = regions)
    other_bounds = (commons.bounds[quartile][variable_index[other1]], commons.bounds[quartile][variable_index[other2]])
    ll_prev_criteria = None
    #print("Critical value is {}".format(critical_val))
    for i, v in enumerate(grid):
        #print("Check {} = {}".format(variable, v))
        ys = sp.Matrix([variable_ref[other1], variable_ref[other2]])
        pll_func_sym = commons.chi2_sym.subs(variable_ref[variable], v)
        pll_func = lambda args: lambdify(ys, pll_func_sym, 'numpy')(*args)[0, 0]
        pll_func_jacob = lambda args: lambdify(ys, pll_func_sym.jacobian(ys), 'numpy')(*args)
        result = opt.minimize(pll_func, [solution[variable_index[other1]], solution[variable_index[other2]]],
                          method='TNC',
                          jac=pll_func_jacob,
                          bounds=other_bounds)
        pll_val = -.5 * result.fun
        ll_criteria = 2 * (ll_hat - pll_val)
        #print("\tProfile LL ratio value = {}".format(ll_criteria))
        if not ll_prev_criteria is None:
            l = grid[i - 1]
            r = grid[i]
            if ll_prev_criteria > critical_val and ll_criteria < critical_val and search == "left":
                #print("Critical point lays between {} and {}\n".format(l, r))
                if (r - l) < precision:
                    return l, r
                else:
                    return find_CI(quartile, variable, critical,
                                   search = search,
                                   grid_left = l,
                                   grid_right = r,
                                   precision = precision)
            elif ll_prev_criteria < critical and ll_criteria > critical and search == "right":
                #print("Critical point lays between {} and {}\n".format(l, r))
                if (r - l) < precision:
                    return l, r
                else:
                    return find_CI(quartile, variable, critical,
                                   search = search,
                                   grid_left = l,
                                   grid_right = r,
                                   precision = precision)
        ll_prev_criteria = ll_criteria


for qu in [1, 4]:
    for vv in ['gamma', 'deltaB']:
        for point in ['left', 'right']:
            r = find_CI(quartile=qu, variable = vv, critical=.95, search=point, precision=0.001)
            print("{} {} {} {}".format(qu, vv, point, (r[0] + r[1])/2.0))


for qu in [1, 4]:
    for vv in ['rB']:
        for point in ['left', 'right']:
            r = find_CI(quartile=qu, variable = vv, critical=.95, search=point, precision=0.00001)
            print("{} {} {} {}".format(qu, vv, point, (r[0] + r[1])/2.0))






