import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math

#%%

#c1=1
#c2=0.5
#c=1
#H0=73.48
H0= sym.Symbol('H0')
c1 = sym.Symbol('c1')
c2 = sym.Symbol('c2')
R = sym.Symbol('R')

omega_m=0.3
c = sym.Symbol('c')

#omega_m = sym.Symbol('omega_m')
#c = 6*(1-omega_m)*(c2/c1)

#r = R/(c*H0**2)
F = R - ((c1*R)/((c2*R/(c*H0**2))+1))

#Calculo las derivadas
F_R = sym.simplify(sym.diff(F,R))
F_2R = sym.simplify(sym.diff(F_R,R))

gamma_2 = F_R/(R*F_2R)
print(sym.simplify(gamma_2))

gamma_1=((R*F_R/F)-1)**(-2)
print(sym.simplify(gamma_1/gamma_2))

aux=-(F_R**2/F*F_2R)
print(sym.simplify(aux))

#%%
#func = lambdify(N, HS_tay,'numpy') # returns a numpy-ready function

#    N_dato = -np.log(1+z)
#    numpy_array_of_results = func(N_dato)
#    return numpy_array_of_results
r = sym.Symbol('r')
R = r*(c*H0**2)

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
