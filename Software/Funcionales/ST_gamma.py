import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math

#%%
H0= sym.Symbol('H0')
Rs = sym.Symbol('Rs')
lamb = sym.Symbol('lamb')
R = sym.Symbol('R')

omega_m=0.3
c = sym.Symbol('c')

#omega_m = sym.Symbol('omega_m')
#c = 6*(1-omega_m)*(c2/c1)

#r = R/(c*H0**2)
F = R - lamb * Rs * (1-(1+(R/Rs)**2)**(-1))

#Calculo las derivadas
F_R = sym.simplify(sym.diff(F,R))
F_2R = sym.simplify(sym.diff(F_R,R))

gamma = F_R/(R*F_2R)
print(sym.simplify(gamma))

#%%
if __name__ == '__main__':
    import sys
    from numpy import linspace
    from scipy.integrate import odeint
    from scipy.optimize import fsolve
    from matplotlib import pyplot as plt

    y0 = [0, 5] #condiciones iniciales
    time = linspace(0., 10., 1000)
    #parametros
    F_lon = 10.
    mass = 1000.

    def Ricci(H, H_N):
        return 6*(2H**2+H+H_N)
    def H_lambda(N):
        return np.sqrt((1+N))
    def constraint(a, v):
        return (F_lon - F_r(a, v)) / mass - a

    def integral(y, _):
        v = y[1]
        a, _, ier, mesg = fsolve(constraint, 0, args=[v, ], full_output=True)
        if ier != 1:
            print ("I coudn't solve the algebraic constraint, error:\n\n", mesg)
            sys.stdout.flush()
        return [v, a]

    dydt = odeint(integral, y0, time)
    plt.plot(dydt[1])
    pass
