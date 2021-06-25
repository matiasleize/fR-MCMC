import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math

#%%
n = sym.Symbol('n')
#n=1 #Descomentar para obtener el caso n=1
n=2 #Descomentar para obtener el caso n=2

c1 = sym.Symbol('c1')
c2 = sym.Symbol('c2')

r = sym.Symbol('r') #r = R/R0
R0 = sym.Symbol('R0')
R = R0*r

F = R - ((c1*R0*r**n)/((c2*r**n)+1))
#Calculo las derivadas
F_R = sym.simplify(sym.diff(F,r))/R0
F_2R = sym.simplify(sym.diff(F_R,r))/R0

gamma = F_R/(R*F_2R)
print(sym.simplify(gamma))
