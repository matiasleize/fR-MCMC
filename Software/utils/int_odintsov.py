'''
(Spanish documentation)
Integra el modelo de Hu-Sawicki y el modelo exponencial utilizando en ambos casos 
el sistema de ecuaciones de Odintsov. Las CI en este caso se calculan de la misma manera
para ambos modelos, que es la que corresponde con la opción (model='EXP') en 
el script solve_sys.py. Integra ST ademas de HS y EXP!

Tarea: Ver cuanto tarda integrar HS con este sistema en comparacion con De la Cruz y cuanta
diferencia hay.

'''

import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #meters/seconds
c_luz_km = c_luz/1000

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_datos_global = os.path.dirname(path_git)
os.chdir(path_git); os.sys.path.append('./Software/utils/')

from initial_conditions_1 import condiciones_iniciales, z_condicion_inicial
from LambdaCDM import H_LCDM
from taylor import Taylor_HS, Taylor_ST
#%%

def dX_dz(x, variables, params_modelo, model='HS'):
    '''
    Sistema de ecuaciones a resolver por la función Integrador.

    Parameters:
        x: list?
            Variable independiente definida como x = ln(1+z) o con un -?
        variables: list?
            Varibles dependientes del sistema de ecuaciones. La cantidad
            de variables es igual a la cantidad de ecuaciones del sistema
            a integrar.
        params_modelo: list
            lista de n parámetros, donde los primeros n-1 elementos son los
            parametros del sistema, mientras que el útimo argumento especifica el modelo
            en cuestión, matemáticamente dado por la función \Gamma.
        model: string
            Modelo que estamos integrando.

    Returns: list
        Set de ecuaciones diferenciales para las variables (E, tildeR).
    '''
    E = variables[0]
    tildeR = variables[1]

    [omega_m,beta] = params_modelo
    omega_l = 1 - omega_m

    s0 = omega_l * tildeR/E - 2 * E
    if model == 'EXP':
        s1 = (np.exp(beta*tildeR)/(beta**2)) * (omega_m * np.exp(-3*x)/(E**2)-1+beta*np.exp(-beta*tildeR) + omega_l*(1-(1+beta*tildeR)*np.exp(-beta*tildeR))/(E**2))
    elif model == 'HS':
        alpha = 1 + beta * tildeR
        s1 = alpha**3/(2*beta**2) * (omega_m * np.exp(-3*x)/(E**2)-1+beta*alpha**(-2) + omega_l*(1-alpha**(-1)-tildeR*beta*alpha**(-2))/(E**2))
    elif model == 'ST':
        gamma = 1 + (beta * tildeR)**2
        s1 = gamma**3/(8*beta**2) * (omega_m * np.exp(-3*x)/(E**2)-1+2*beta*gamma**(-2) + omega_l*(1-gamma**(-1)-tildeR*2*beta*gamma**(-2))/(E**2))
    return [s0,s1]


def integrador(params_fisicos,epsilon=10**(-10), cantidad_zs=int(10**5),
                z_inicial=10, z_final=0,
                sistema_ec=dX_dz, verbose=False, eval_data=False, z_data = None,
                model='HS',method='RK45', rtol=1e-11, atol=1e-16,):
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

    t1 = time.time()
    [omega_m, b, H0] = params_fisicos
    if model=='EXP':
        z_ci = z_condicion_inicial(params_fisicos, epsilon)
    elif (model=='HS' or model=='ST'):
        z_ci = z_inicial

    beta = 2/b
    params_modelo = [omega_m,beta]
    cond_iniciales = condiciones_iniciales(params_fisicos, 
                    zi = z_ci, model='Odintsov')

    #Integramos el sistema
    zs_int = np.linspace(z_ci,z_final,cantidad_zs)

    x_ci = -np.log(1 + z_ci)
    x_final = -np.log(1 + z_final)
    xs_int = -np.log(1 + zs_int)

    sol = solve_ivp(sistema_ec, (x_ci,x_final),
        cond_iniciales, t_eval=xs_int, args=(params_modelo,model),
        rtol=rtol, atol=atol, method=method)

    xs_ode = sol.t[::-1]
    zs_ode = np.exp(-xs_ode)-1
    Hs_ode = H0 * sol.y[0][::-1]

    if model=='EXP':
        ## La parte LCDM
        zs_LCDM = np.linspace(z_ci,z_inicial,cantidad_zs)
        Hs_LCDM = H0 * np.sqrt(omega_m * (1+zs_LCDM)**3 + (1-omega_m))

        zs_aux = np.concatenate((zs_ode,zs_LCDM),axis = None)
        Hs_aux = np.concatenate((Hs_ode,Hs_LCDM),axis = None)

        f = interp1d(zs_aux,Hs_aux)

        if eval_data == False:
            zs_final = np.linspace(z_final,z_inicial,cantidad_zs)
            Hs_final = f(zs_final)

        else:
            zs_final = z_data
            Hs_final = f(zs_final)

    elif (model=='HS' or model=='ST'):
        if eval_data == False:
            zs_final = zs_ode
            Hs_final = Hs_ode

        else:
            f = interp1d(zs_ode,Hs_ode)
            zs_final = z_data
            Hs_final = f(z_data)

    t2 = time.time()
    if verbose == True:
        print('Duration: {} minutes and {} seconds'.format(int((t2-t1)/60),
        int((t2-t1) - 60*int((t2-t1)/60))))
    return zs_final,Hs_final


def Hubble_th(params_fisicos, b_crit=0.15, all_analytic=False,
                    eval_data=False, z_data=None, epsilon=10**(-10), n=1,
                    cantidad_zs=int(10**5),
                    z_min=0, z_max=10, sistema_ec=dX_dz,
                    verbose=False, model='HS', method='RK45',
                    rtol=1e-11, atol=1e-16):

    [omega_m,b,H0] = params_fisicos
    if model=='LCDM':
        zs_modelo = np.linspace(z_min,z_max,cantidad_zs)
        Hs_modelo = H_LCDM(zs_modelo, omega_m, H0)
        return zs_modelo, Hs_modelo

    elif model=='EXP': #b critico para el modelo exponencial
        log_eps_inv = -np.log10(epsilon)
        b_crit = (4 + omega_m/(1-omega_m)) / log_eps_inv
        #method = 'LSODA' (Deprecated)
    else:
        pass

    if (b <= b_crit) or (all_analytic==True): #Aproximacion analitica
        if eval_data == False:
            zs_modelo = np.linspace(z_min,z_max,cantidad_zs)
        else:
            zs_modelo = z_data

        if (model=='HS') and (n==1):
            Hs_modelo = Taylor_HS(zs_modelo, omega_m, b, H0)
        elif (model=='HS') and (n==2):
            Hs_modelo = Taylor_ST(zs_modelo, omega_m, b, H0)
        #elif (model=='ST') and (n==1):
        #    Hs_modelo = Taylor_ST(zs_modelo, omega_m, b, H0)
        elif model=='EXP': #Devuelvo LCDM
            Hs_modelo = H_LCDM(zs_modelo, omega_m, H0)

    else: #Integro
        if eval_data == False:
            zs_modelo, Hs_modelo = integrador(params_fisicos, epsilon=epsilon,
                                    cantidad_zs=cantidad_zs,
                                    z_inicial=z_max, z_final=z_min, sistema_ec=sistema_ec,
                                    verbose=verbose, model=model,
                                     method=method,rtol=rtol, atol=atol)
        else:
            zs_modelo, Hs_modelo = integrador(params_fisicos, epsilon=epsilon,
                                    cantidad_zs=cantidad_zs,
                                    z_inicial=z_max, z_final=z_min, sistema_ec=sistema_ec,
                                    verbose=verbose, eval_data=True, z_data = z_data,
                                    model=model, method=method,rtol=rtol, atol=atol)
    return zs_modelo, Hs_modelo


#%%
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    #Esto correrlo con b grande para que ande!
    b = 0.5
    omega_m = 0.3
    H0 = 73.48

    params_fisicos = [omega_m,b,H0]
    zs_ode, Hs_HS = integrador(params_fisicos, verbose=True, model='HS',z_inicial=10)
    _, Hs_EXP = integrador(params_fisicos, epsilon=10**(-10),
                verbose=True, model='EXP',z_inicial=10)


    #%matplotlib qt5
    plt.figure()
    plt.title('Integrador $f(R)$')
    plt.xlabel('z (redshift)')
    plt.ylabel('H(z) ((km/seg)/Mpc)')
    plt.plot(zs_ode,Hs_HS,'.',label='HS')
    plt.plot(zs_ode,Hs_EXP,'.',label='Exp')
    plt.legend(loc = 'best')
    plt.grid(True)
