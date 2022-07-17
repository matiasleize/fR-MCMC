import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #meters/seconds
c_luz_km = c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/utils/')

from condiciones_iniciales import condiciones_iniciales, z_condicion_inicial
from cambio_parametros import params_fisicos_to_modelo_HS
from int import integrador
from matplotlib import pyplot as plt

b = 0.1
omega_m = 0.1
H0 = 73.48
eps = 10**(-41)
mstep = 8*10**(-8)
params_fisicos = [omega_m,b,H0]

#Integramos con RK45
#zs_ode, H_HS = integrador(params_fisicos, verbose=True, model='HS')
#_, H_ST = integrador(params_fisicos, verbose=True, model='ST')
#_, H_EXP = integrador(params_fisicos, epsilon=eps, verbose=True, model='EXP',max_step=10**(-6))

#Integramos con LSODA
#zs_ode, H_HS_1 = integrador(params_fisicos, verbose=True, model='HS',method='LSODA')
#_, H_ST_1 = integrador(params_fisicos, verbose=True, model='ST',method='LSODA')
zs_ode, H_EXP_1 = integrador(params_fisicos, epsilon=eps,
                verbose=True, model='EXP',method='LSODA'
                ,max_step=mstep)
plt.plot(zs_ode,H_EXP_1)
#%%
def porcentual_error(vec_ref,vec_compare):
    error = 100 * (1 - (vec_compare/vec_ref))
    return error

#error_HS = porcentual_error(H_HS,H_HS_1)
#error_ST = porcentual_error(H_ST,H_ST_1)
error_EXP = porcentual_error(H_EXP,H_EXP_1)

%matplotlib qt5
plt.figure()
plt.title('Error en integracion comparando distintos métodos')
plt.xlabel('z (redshift)')
plt.ylabel('Error porcentual')
#plt.plot(zs_ode,error_HS,label='HS')
#plt.plot(zs_ode,error_ST,label='ST')
plt.plot(zs_ode,error_EXP,label='Exp')
plt.legend(loc = 'best')
plt.grid(True)
