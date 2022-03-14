import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math

#%%
beta = sym.Symbol('beta')

Re = sym.Symbol('Re')
r = sym.Symbol('r') #R/Re
e = sym.Symbol('e')
R = Re*r

#beta=1
F = R - beta * Re * (1-e**(-R/Re))
#Calculo las derivadas
F_R = sym.diff(F,r)/Re
F_2R = sym.diff(F_R,r)/Re

gamma = F_R/(R*F_2R)
print(gamma)
print(sym.simplify(gamma))

math.e

math.e**2
np.e**2
np.exp(2)
