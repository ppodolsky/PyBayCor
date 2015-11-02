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

# gamma = 70, deltaB = 120, rB = 0.1
direct = {'y_corr': np.mat([[1.000, 0.000, 0.000, 0.000],
                      [0.000, 1.000, 0.000, 0.000],
                      [0.000, 0.000, 1.000, 0.000],
                      [0.000, 0.000, 0.000, 1.000]]),
              'y_exp': np.array([-0.09848, -0.01736, 0.06427, 0.0766]),
              'y_var': np.array([0.00000001, 0.00000001, 0.00000001, 0.00000001])}

direct['y_corr_inv'] = np.linalg.inv(direct["y_corr"])
direct['y_std'] = np.sqrt(direct["y_var"])
direct['y_covar'] = np.mat(np.diag(direct['y_std']))*direct["y_corr"]*np.mat(np.diag(direct['y_std']))
direct['y_covar_inv'] = np.linalg.inv(direct['y_covar'])

summer2015 = {'y_corr': np.mat([[1.000, 0.093, 0.000,-0.000],
                      [0.093, 1.000,-0.000, 0.000],
                      [0.000,-0.000, 1.000,-0.132],
                      [-0.000,0.000,-0.132, 1.000]]),
              'y_exp': np.array([-0.085, -0.027, 0.044, 0.090]),
              'y_var': np.array([(0.023**2 + 0.004**2), (0.023**2 + 0.010**2), (0.023**2 + 0.005**2), (0.026**2 + 0.014**2)])}

summer2015['y_corr_inv'] = np.linalg.inv(summer2015["y_corr"])
summer2015['y_std'] = np.sqrt(summer2015["y_var"])
summer2015['y_covar'] = np.mat(np.diag(summer2015['y_std']))*summer2015["y_corr"]*np.mat(np.diag(summer2015['y_std']))
summer2015['y_covar_inv'] = np.linalg.inv(summer2015['y_covar'])

lumi1fb = {'y_stat_corr': np.mat([
        [ 1.000, 0.170,-0.000,-0.000],
        [ 0.170, 1.000,-0.000,-0.000],
        [-0.000,-0.000, 1.000,-0.110],
        [-0.000,-0.000,-0.110, 1.000]]),
            'y_sys_corr': np.mat([
        [ 1.000, 0.360,-0.000,-0.000],
        [ 0.360, 1.000,-0.000,-0.000],
        [-0.000,-0.000, 1.000,-0.050],
        [-0.000,-0.000,-0.050, 1.000]]),
              'y_exp': np.array([-0.103, -0.009, 0.000, 0.027]),
              'y_stat_err': np.array([0.045, 0.037, 0.043, 0.052]),
              'y_sys_err': np.array([(0.018**2 + 0.014**2)**0.5,
                                     (0.008**2 + 0.030**2)**0.5,
                                     (0.015**2 + 0.006**2)**0.5,
                                     (0.008**2 + 0.023**2)**0.5])}

lumi1fb['y_covar'] = np.mat(np.diag(lumi1fb['y_stat_err']))*lumi1fb["y_stat_corr"]*np.mat(np.diag(lumi1fb['y_stat_err'])) + \
                     np.mat(np.diag(lumi1fb['y_sys_err']))*lumi1fb["y_sys_corr"]*np.mat(np.diag(lumi1fb['y_sys_err']))
lumi1fb['y_covar_inv'] = np.linalg.inv(lumi1fb['y_covar'])

lumi3fb = {
            'y_stat_corr': np.mat([
                [ 1.000, 0.106,-0.136,-0.186],
                [ 0.106, 1.000,-0.031,-0.074],
                [-0.136,-0.031, 1.000,-0.053],
                [-0.186,-0.074,-0.053, 1.000]]),
            'y_sys_corr': np.mat([
                [ 1.000, 0.000,-0.000,-0.000],
                [ 0.000, 1.000,-0.000,-0.000],
                [-0.000,-0.000, 1.000,-0.000],
                [-0.000,-0.000,-0.000, 1.000]]),
            'y_exp': np.array([-8.85e-2, -0.12e-2, 3.46e-2, 7.91e-2]),
            'y_stat_err': np.array([3.12e-2, 3.65e-2, 2.89e-2/5, 3.83e-2]),
            'y_sys_err': np.array([0,0,0,0])}

lumi3fb['y_covar'] = np.mat(np.diag(lumi3fb['y_stat_err']))*lumi3fb["y_stat_corr"]*np.mat(np.diag(lumi3fb['y_stat_err'])) + \
                     np.mat(np.diag(lumi3fb['y_sys_err']))*lumi3fb["y_sys_corr"]*np.mat(np.diag(lumi3fb['y_sys_err']))
lumi3fb['y_covar_inv'] = np.linalg.inv(lumi3fb['y_covar'])

y_theo_sym = sp.Matrix([rB*sp.cos((deltaB + gamma)*(math.pi/180)),
                        rB*sp.sin((deltaB + gamma)*(math.pi/180)),
                        rB*sp.cos((deltaB - gamma)*(math.pi/180)),
                        rB*sp.sin((deltaB - gamma)*(math.pi/180))])
ys = sp.Matrix([gamma, deltaB, rB])

bounds = {1: ((0,180), (0,180), (0.0001, 0.5)),
          4: ((-180,0),(-180,0),(0.0001, 0.5))}
