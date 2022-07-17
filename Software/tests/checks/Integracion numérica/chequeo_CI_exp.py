import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math
from scipy.constants import c as c_luz #meters/seconds
c_luz_km = c_luz/1000;

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/utils/')
from cambio_parametros import params_fisicos_to_modelo_HS
#%%

z0 = 30
omega_m = 0.24
b = .3
H0 = 73.48
n=1
model='E'
R = sym.Symbol('R')
Lamb = 3 * (1-omega_m)
#%%
if model=='HS':
    c1,c2 = params_fisicos_to_modelo_HS(omega_m,b,n)
    R_HS = 2 * Lamb * c2/c1
    R_0 = R_HS #No confundir con R0 que es R en la CI!

    #Calculo F. Ambas F dan las mismas CI para z=z0 :)
    #F = R - ((c1*R)/((c2*R/R_HS)+1))
    F = R - 2 * Lamb * (1 - 1/ (1 + (R/(b*Lamb)**n)) )
elif model=='ST':
    #lamb = 2 / b
    R_ST = Lamb * b
    R_0 = R_ST #No confundir con R0 que es R en la CI!

    #Calculo F.
    F = R - 2 * Lamb * (1 - 1/ (1 + (R/(b*Lamb)**2) ))

elif model=='EXP':
    R_E = Lamb * b
    R_0 = R_E #No confundir con R0 que es R en la CI!

    #Calculo F.
    F = R - 2 * Lamb * (1 - math.e**(-R/(b*Lamb)))
#%%
#Calculo las derivadas de F
#F_R = sym.simplify(sym.diff(F,R))
#F_2R = sym.simplify(sym.diff(F_R,R))

F_R = sym.diff(F,R)
F_2R = sym.diff(F_R,R)

F_R
z = sym.Symbol('z')
H = (omega_m*(1+z)**3 + (1-omega_m))**(0.5)

#H_z = ((1+z)**2 * 3 * omega_m)/(2*(1+omega_m*(-1+(1+z)**3))**(0.5)) #Derivada de H(z)
H_z = sym.simplify(sym.diff(H,z)) #Derivada de H(z)
H_2z = sym.simplify(sym.diff(H_z,z))

Ricci = (12*H**2 + 6*H_z*(-H*(1+z)))
Ricci_t=sym.simplify(sym.diff(Ricci,z)*(-H*(1+z)))

Ricci_ci=sym.lambdify(z,Ricci,'numpy')
Ricci_t_ci=sym.lambdify(z,Ricci_t,'numpy')
H_ci=sym.lambdify(z,H)
H_z_ci=sym.lambdify(z,H_z,'numpy')
F_ci=sym.lambdify(R,F,'numpy')
F_R_ci=sym.lambdify(R,F_R,'numpy')
F_2R_ci=sym.lambdify(R,F_2R,'numpy')
R0=Ricci_ci(z0)
R0
Ricci_t_ci(z0)
H_ci(z0) #Chequie que de lo esperado x Basilakos
H_z_ci(z0) #Chequie que de lo esperado x Basilakos
F_ci(R0) # debe ser simil a R0-2*Lamb
F_R_ci(R0) # debe ser simil a 1
F_2R_ci(R0) # debe ser simil a 0

x0 = Ricci_t_ci(z0)*F_2R_ci(R0) / (H_ci(z0)*F_R_ci(R0))
y0 = F_ci(R0) / (6*(H_ci(z0)**2)*F_R_ci(R0))
#x0=0
#y0=1.5
v0 = R0 / (6*H_ci(z0)**2)
w0 = 1+x0+y0-v0
r0 = R0/R_0
w0
