import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math

#%%
n = sym.Symbol('n')
#n=1 #Descomentar para obtener el caso n=1

lamb = sym.Symbol('lamb')

n=1
c1 = sym.Symbol('c1')
c2 = sym.Symbol('c2')
lamb=c1/(c2**(0.5))

Rs = sym.Symbol('Rs')
r = sym.Symbol('r') #R/Rs
R = Rs*r

F = (R - lamb * Rs * (1-(1+(R/Rs)**2)**(-n)))
F = Rs*(r - lamb * (1-(1+r**2)**(-n)))

#Calculo las derivadas
F_R = sym.simplify(sym.diff(F,r))/Rs
F_2R = sym.simplify(sym.diff(F_R,r))/Rs

gamma = F_R/(R*F_2R)
print(sym.simplify(gamma))
#
