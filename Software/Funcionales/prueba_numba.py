import time
import numpy as np
from numba import jit
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_condiciones_iniciales import condiciones_iniciales, z_condicion_inicial
from funciones_cambio_parametros import params_fisicos_to_modelo_HS
#%%
@jit
def dX_dz(z, variables, params_modelo, model='HS'):
    E = variables[0]
    tildeR = variables[1]

    [omega_m,beta] = params_modelo
    omega_l = 1 - omega_m

    s0 = omega_l * tildeR/E - 2 * E
    s1 = (np.exp(beta*tildeR)/(beta**2)) * (omega_m * np.exp(-3*z)/(E**2)-1+beta*np.exp(-beta*tildeR) + omega_l*(1-(1+beta*tildeR)*np.exp(-beta*tildeR))/(E**2))

    return [s0,s1]

def integrador(params_fisicos,epsilon=10**(-10), n=1, cantidad_zs=int(10**5), max_step=10**(-5),
                z_inicial=3, z_final=0, sistema_ec=dX_dz, verbose=False,
                model='EXP',method='RK45'):
    '''
    Integración numérica del sistema de ecuaciones diferenciales entre
    z_inicial y z_final, dadas las condiciones iniciales de las variables
    (x,y,v,w,r) y los parámetros 'con sentido físico' del modelo f(R).

    Parameters:
        cantidad_zs: int
            cantidad de puntos (redshifts) en los que se evalúa la
            integración nummérica. Es necesario que la cantidad de puntos en el
            área de interés $(z \in [0,3])$.
        max_step: float
            paso de la integración numérica. Cuando los parámetros que aparecen
            en el sistema de ecuaciones se vuelven muy grandes (creo que esto implica
            que el sistema a resolver se vuelva más inestable) es necesario que el paso
            de integración sea pequeño.
        verbose: Bool
            if True, imprime el tiempo que tarda el proceso de integración.

    Output: list
        Un array de Numpy de redshifts z y un array de H(z).
    '''
    [omega_m, b, H0] = params_fisicos
    if model=='EXP':
        eps = epsilon
        z_ci = z_condicion_inicial(params_fisicos,eps)
        #print(z_ci)
        if (np.isnan(z_ci)==True or z_ci<=0):
            zs_LCDM = np.linspace(z_final,z_inicial,cantidad_zs)
            Hs_LCDM = H0 * np.sqrt(omega_m * (1+zs_LCDM)**3 + (1-omega_m)) #O bien un Taylor

            t2 = time.time()
            if verbose == True:
                print('Duración {} minutos y {} segundos'.format(int((t2-t1)/60),
                int((t2-t1) - 60*int((t2-t1)/60))))
            return zs_LCDM, Hs_LCDM

        else:
            beta = 2/b
            params_modelo = [omega_m,beta]
            cond_iniciales = condiciones_iniciales(omega_m, b, z0=z_ci, model='EXP')
            #Integramos el sistema
            zs_int = np.linspace(z_ci,z_final,cantidad_zs)

            x_ci = -np.log(1 + z_ci)
            x_final = -np.log(1 + z_final)
            xs_int = -np.log(1 + zs_int)

            sol = solve_ivp(sistema_ec, (x_ci,x_final),
                cond_iniciales, t_eval=xs_int, args=(params_modelo,model),
                 max_step=max_step,method=method)

            xs_ode = sol.t[::-1]
            zs_ode = np.exp(-xs_ode)-1
            Hs_ode = H0 * sol.y[0][::-1]

            ## La parte LCDM
            zs_LCDM = np.linspace(z_ci,z_inicial,cantidad_zs)
            Hs_LCDM = H0 * np.sqrt(omega_m * (1+zs_LCDM)**3 + (1-omega_m))

            zs_aux = np.concatenate((zs_ode,zs_LCDM),axis = None)
            Hs_aux = np.concatenate((Hs_ode,Hs_LCDM),axis = None)

            f = interp1d(zs_aux,Hs_aux)

            zs_final = np.linspace(z_final,z_inicial,cantidad_zs)
            Hs_final = f(zs_final)

            return zs_final,Hs_final
#%%
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    b = 1
    omega_m = 0.24
    H0 = 73.48

    params_fisicos = [omega_m,b,H0]
    #%timeit integrador(params_fisicos, verbose=True, model='HS')
    zs_ode, H_EXP = integrador(params_fisicos, epsilon=10**(-10), verbose=True, model='EXP')

    #%matplotlib qt5
    plt.figure()
    plt.title('Integrador $f(R)$')
    plt.xlabel('z (redshift)')
    plt.ylabel('H(z) ((km/seg)/Mpc)')
    plt.plot(zs_ode,H_HS,label='HS')
    plt.plot(zs_ode,H_ST,label='ST')
    plt.plot(zs_ode,H_EXP,label='Exp')
    plt.legend(loc = 'best')
    plt.grid(True)
#%% BORRAR
    from matplotlib import pyplot as plt
    H0 = 73.48
    omega_m = 0.1

    eps = 10**(-10)
    b = (37/9)/(-np.log10(eps))
    print(b)
    max_step = 2*10**(-4)

    params_fisicos = [omega_m,b,H0]
    zs_ode, H_EXP = integrador(params_fisicos, epsilon=eps, verbose=True, model='EXP',max_step=max_step)
