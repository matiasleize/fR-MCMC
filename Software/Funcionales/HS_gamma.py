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

print(F_R)

gamma = F_R/(R*F_2R)
print(sym.simplify(gamma))



#func = lambdify(N, HS_tay,'numpy') # returns a numpy-ready function

#    N_dato = -np.log(1+z)
#    numpy_array_of_results = func(N_dato)
#    return numpy_array_of_results
r = sym.Symbol('r')
R = r*(c*H0**2)

#%%
if __name__ == '__main__':
    pass
