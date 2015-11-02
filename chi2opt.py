import sympy as sp
import numpy as np
import scipy.optimize as opt
from commons import *
from sympy.utilities.lambdify import lambdify, implemented_function

#Symbolic solutions

#print("Symbolic: ")
#print(sp.solvers.solve(chi2_sym.jacobian(ys), ys))

#Numerical solutions

print("Numerical: ")

#Calculate using Newton
result = opt.minimize(chi2, [0, 0, 0.01], method='TNC', bounds=bounds[1])
print("Newton's method")
print(result.fun)
print(result.x.tolist())

#Calculate using jacobian
print("Newton's with Jacobian matrix")
jacobian = lambda x: lambdify(ys, chi2_sym.jacobian(ys), 'numpy')(*x)
hessian = lambda x: sp.hessian(ys, sp.hessian(chi2_sym, ys), 'numpy')(*x)
result = opt.minimize(chi2, [0, 0, 0.01],
                      method='TNC',
                      jac=jacobian,
                      #hess=hessian,
                      bounds=bounds[1])
print(result.fun)
print(result.x.tolist())



