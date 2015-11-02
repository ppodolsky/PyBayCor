import math
import numpy as np
import sympy as sp

from sympy.utilities.lambdify import lambdify

def degToRad(degs):
    return degs*(math.pi/180)
def cos(x):
    return math.cos(degToRad(x))
def sin(x):
    return math.sin(degToRad(x))

def x_plus(gamma, deltaB, rB):
    return rB*cos(deltaB + gamma)
def y_plus(gamma, deltaB, rB):
    return rB*sin(deltaB + gamma)
def x_minus(gamma, deltaB, rB):
    return rB*cos(deltaB - gamma)
def y_minus(gamma, deltaB, rB):
    return rB*sin(deltaB - gamma)


gamma = sp.Symbol('gamma')
deltaB = sp.Symbol('deltaB')
rB = sp.Symbol('rB')

y_theo = np.array([x_plus, y_plus, x_minus, y_minus]).T

y_corr = np.mat([[1.000, 0.093, 0.000,-0.000],
                      [0.093, 1.000,-0.000, 0.000],
                      [0.000,-0.000, 1.000,-0.132],
                      [-0.000,0.000,-0.132, 1.000]])
y_corr_inv = np.linalg.inv(y_corr)

y_exp = np.array([-0.085, -0.027, 0.044, 0.090])
y_var = np.array([(0.023**2 + 0.004**2), (0.023**2 + 0.010**2), (0.023**2 + 0.005**2), (0.026**2 + 0.014**2)])
y_std = np.sqrt(y_var)

y_covar = np.mat(np.diag(y_std))*y_corr*np.mat(np.diag(y_std))
y_covar_inv = np.linalg.inv(y_covar)

y_theo_sym = sp.Matrix([rB*sp.cos((deltaB + gamma)*(math.pi/180)),
                        rB*sp.sin((deltaB + gamma)*(math.pi/180)),
                        rB*sp.cos((deltaB - gamma)*(math.pi/180)),
                        rB*sp.sin((deltaB - gamma)*(math.pi/180))])
chi2_sym = (y_theo_sym - y_exp).T*y_covar_inv*(y_theo_sym - y_exp)
chi2 = lambda args: lambdify(ys, chi2_sym, 'numpy')(*args)[0, 0]
ys = sp.Matrix([gamma, deltaB, rB])

bounds = {1: ((0,180), (0,180), (0.0001, 0.5)),
          4: ((-180,0),(-180,0),(0.0001, 0.5))}
