'''
(Spanish documentation)
Integra el modelo de Hu-Sawicki con el modelo de De la Cruz et al. y el modelo exponencial
utilizando el sistema de ecuaciones de Odintsov. Las CI son distintas para ambos modelos
(ver script int_sist_1.py). 

Tarea: Ver cuanto tarda integrar HS con este sistema en comparacion con Odintsov y cuanta
diferencia hay.

El int_2 (que esta en la carpeta test) solo cambia en como se calcula eta (alfa en el codigo).

Tarea: Probar si es mas rapido eta_1 o eta_2.
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
os.chdir(path_git)
os.sys.path.append('./Software/utils/')

from initial_conditions_1 import condiciones_iniciales, z_condicion_inicial
from change_of_parameters import params_fisicos_to_modelo_HS
from LambdaCDM import H_LCDM
from taylor import Taylor_HS
#%%

def dX_dz(z, variables, params_fisicos, model='HS'):
    '''
    Sistema de ecuaciones a resolver pot la función Integrador.
    Parameters:
        params_fisicos: list
            #lista de n parámetros, donde los primeros n-1 elementos son los
            #parametros del sistema, mientras que el útimo argumento especifica el modelo
            #en cuestión, matemáticamente dado por la función \Gamma.
        model: string
            Modelo que estamos integrando.
    Returns: list
        Set de ecuaciones diferenciales para las variables dinámicas.
    '''
    
    [omega_m, b, _] = params_fisicos

    if model == 'EXP':
        E = variables[0]
        tildeR = variables[1]

        beta = 2/b
        omega_l = 1 - omega_m

        s0 = omega_l * tildeR/E - 2 * E
        s1 = (np.exp(beta*tildeR)/(beta**2)) * (omega_m * np.exp(-3*z)/(E**2)-1+beta*np.exp(-beta*tildeR) + omega_l*(1-(1+beta*tildeR)*np.exp(-beta*tildeR))/(E**2))

        return [s0,s1]

    elif model == 'HS': #Para el modelo de Hu-Sawicki
        x = variables[0]
        y = variables[1]
        v = variables[2]
        w = variables[3]
        r = variables[4]

        #Calculamos los parámetros del modelo
        B, D = params_fisicos_to_modelo_HS(omega_m, b) # (c1,c2) = (B,D) from De la Cruz et al.

        gamma = lambda r,c1,c2: -(c1 - (c2*r + 1)**2)*(c2*r + 1)/(2*c1*c2*r)
        G = gamma(r,B,D) #Va como r^3/r = r^2

        s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
        s1 = (- (v*x*G - x*y + 4*y - 2*y*v)) / (z+1)
        s2 = (-v * (x*G + 4 - 2*v)) / (z+1)
        s3 = (w * (-1 + x + 2*v)) / (z+1)
        s4 = (-(x * r * G)) / (1+z)

        return [s0,s1,s2,s3,s4]

    else:
        print('Choose a valid model!')
        pass

def integrador(params_fisicos, epsilon=10**(-10), cantidad_zs=int(10**5),
                z_inicial=10, z_final=0,
                sistema_ec=dX_dz, verbose=False, eval_data=False, z_data = None,
                model='HS',method='RK45', rtol=1e-11, atol=1e-16):
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
        z_ci = z_condicion_inicial(params_fisicos,epsilon)
        beta = 2/b
        cond_iniciales = condiciones_iniciales(params_fisicos, zi=z_ci, model='EXP')

        #Integramos el sistema
        zs_int = np.linspace(z_ci,z_final,cantidad_zs)

        x_ci = -np.log(1 + z_ci)
        x_final = -np.log(1 + z_final)
        xs_int = -np.log(1 + zs_int)

        sol = solve_ivp(sistema_ec, (x_ci,x_final),
            cond_iniciales, t_eval=xs_int, args=(params_fisicos, model),
            rtol=rtol, atol=atol, method=method)

        xs_ode = sol.t[::-1]
        zs_ode = np.exp(-xs_ode)-1
        Hs_ode = H0 * sol.y[0][::-1]

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

    elif model=='HS':
        #Calculo las condiciones cond_iniciales, eta
        # y los parametros de la ecuación
        cond_iniciales = condiciones_iniciales(params_fisicos, zi=z_inicial)

        h = H0/100
        R_HS = (omega_m * h**2)/(0.13*8315**2) #Mpc**(-2)
        eta = c_luz_km * np.sqrt(R_HS/6) #(km/seg)/Mpc

        if eval_data == False:
            zs_int = np.linspace(z_inicial, z_final, cantidad_zs)

            sol = solve_ivp(sistema_ec, (z_inicial,z_final),
                cond_iniciales, t_eval=zs_int, args=(params_fisicos, model),
                rtol=rtol, atol=atol, method=method)

            if (len(sol.t)!=cantidad_zs):
                print('Está integrando mal!')
            if np.all(zs_int==sol.t)==False:
                print('Hay algo raro!')
        else:
            sol = solve_ivp(sistema_ec, (z_inicial,z_final),
                cond_iniciales, t_eval=z_data.reverse(), args=(params_fisicos,model),
                rtol=rtol, atol=atol, method=method)

            if (len(sol.t)!=len(z_data)):
                print('Está integrando mal!')
            if np.all(z_data==sol.t)==False:
                print('Hay algo raro!')

        #Calculamos el Hubble
        zs_final = sol.t[::-1]

        v=sol.y[2][::-1]
        r=sol.y[4][::-1]
        Hs_final = eta * np.sqrt(r/v)

    t2 = time.time()

    if verbose == True:
        print('Duration: {} minutes and {} seconds'.format(int((t2-t1)/60),
                int((t2-t1) - 60*int((t2-t1)/60))))

    return zs_final, Hs_final
    if model=='HS':
        [omega_m, b, H0] = params_fisicos
        #Calculo las condiciones cond_iniciales, eta
        # y los parametros de la ecuacion
        cond_iniciales = condiciones_iniciales(omega_m, b, z0=z_inicial, n=n)
        alfa = H0*np.sqrt((1-omega_m)*b/2)
        c1, c2 = params_fisicos_to_modelo_HS(omega_m, b, n=n)

        params_modelo = [c1,c2,n]

    #Integramos el sistema
    zs_int = np.linspace(z_inicial,z_final,cantidad_zs)
    sol = solve_ivp(sistema_ec, (z_inicial,z_final),
        cond_iniciales, t_eval=zs_int, args=(params_modelo,model),
        max_step=max_step)

    if (len(sol.t)!=cantidad_zs):
        print('Esta integrando mal!')
    if np.all(zs_int==sol.t)==False:
        print('Hay algo raro!')



def Hubble_th_1(params_fisicos, b_crit=0.15, all_analytic=False,
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

    #%% Hu-Sawicki (n=1)
    params_fisicos = [0.3, 0.1, 73] # [omega_m, b, H0]
    zs_ode, H_HS = integrador(params_fisicos, verbose=True, model='HS')
    _, H_HS_1 = Hubble_th_1(params_fisicos, verbose=True, model='HS')
    #%% Exponencial
    params_fisicos = [0.3, 2, 73] # [omega_m, b, H0]
    zs_ode, H_EXP = integrador(params_fisicos, verbose=True, model='EXP')
    _, H_EXP_1 = Hubble_th_1(params_fisicos, verbose=True, model='EXP')
    #%% Graficamos todos los datos juntos
    #%matplotlib qt5
    plt.figure()
    plt.title('Integrador $f(R)$')
    plt.xlabel('z (redshift)')
    plt.ylabel('H(z) ((km/seg)/Mpc)')
    plt.plot(zs_ode,H_HS,'.',label='HS')
    plt.plot(zs_ode,H_EXP,'.',label='Exp')
    plt.plot(zs_ode,H_LCDM(zs_ode,0.3,73),'.',label='LCDM') #H_LCDM(zs_ode,omega_m,H0)
    plt.legend(loc = 'best')
    plt.grid(True)
    plt.show()
