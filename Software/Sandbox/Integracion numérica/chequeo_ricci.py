import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_condiciones_iniciales import condiciones_iniciales
from funciones_cambio_parametros import params_fisicos_to_modelo_HS, params_fisicos_to_modelo_ST
#%%

def dX_dz(z, variables, params_modelo, model='HS'):
    '''
    Sistema de ecuaciones a resolver pot la función Integrador.

    Parameters:
        params_modelo: list
            lista de n parámetros, donde los primeros n-1 elementos son los
            parametros del sistema, mientras que el útimo argumento especifica el modelo
            en cuestión, matemáticamente dado por la función \Gamma.
        model: string
            Modelo que estamos integrando.

    Returns: list
        Set de ecuaciones diferenciales para las variables (x,y,v,w,r).
    '''

    x = variables[0]
    y = variables[1]
    v = variables[2]
    w = variables[3]
    r = variables[4]

    #[p1,p2,n,model]=imput

    if model == 'ST': #Para el modelo de Starobinsky
        [lamb,N] = params_modelo
        if N==1:
            gamma = lambda r,lamb: -(r**2 + 1)*(2*lamb*r - (r**2 + 1)**2)/(2*lamb*r*(3*r**2 - 1))
            G = gamma(r,lamb) #Va como r^6/r^3 = r^3
        else:
            gamma = lambda r,lamb, N: (r**2 + 1)**(N + 2)*(-lamb*N*r*(r**2 + 1)**(-N - 1) + 1/2)/(lamb*N*r*(2*N*r**2 + r**2 - 1))
            G = gamma(r,lamb,N)
            #print('Guarda que estas poniendo n!=1')
            pass


    elif model == 'HS': #Para el modelo de Hu-Sawicki
        [B,D,N] = params_modelo
        if N==1:
            gamma = lambda r,c1,c2: -(c1 - (c2*r + 1)**2)*(c2*r + 1)/(2*c1*c2*r)
            G = gamma(r,B,D) #Va como r^3/r = r^2
        elif N==2:
            gamma = lambda r,c1,c2: (-2*c1*c2*r**3 - 2*c1*r + c2**3*r**6 + 3*c2**2*r**4 + 3*c2*r**2 + 1)/(2*c1*r*(3*c2*r**2 - 1))
            G = gamma(r,B,D) #Va como r^6/r^3 = r^3
        else:
            gamma = lambda r,c1,c2,n: r**(-n)*(-c1*c2*n*r**(2*n) - c1*n*r**n + c2**3*r**(3*n + 1) + 3*c2**2*r**(2*n + 1) + 3*c2*r**(n + 1) + r)/(c1*n*(c2*n*r**n + c2*r**n - n + 1))
            G = gamma(r,B,D,N)
            #print('Guarda que estas poniendo n!=1')
    else:
        print('Elegir un modelo válido!')
        pass

    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
    s1 = (- (v*x*G - x*y + 4*y - 2*y*v)) / (z+1)
    s2 = (-v * (x*G + 4 - 2*v)) / (z+1)
    s3 = (w * (-1 + x + 2*v)) / (z+1)
    s4 = (-(x * r * G)) / (1+z)
    return [s0,s1,s2,s3,s4]


def integrador_r(params_fisicos, n=1, cantidad_zs=int(10**5), max_step=10**(-5),
                z_inicial=30, z_final=0, sistema_ec=dX_dz, verbose=False,
                model='HS'):
    #Para HS n=1 con max_step 0.003 alcanza.
    t1 = time.time()

    if model=='HS':
        [omega_m, b, H0] = params_fisicos
        #Calculo las condiciones cond_iniciales, eta
        # y los parametros de la ecuación
        cond_iniciales = condiciones_iniciales(omega_m, b, z0=z_inicial, n=n)
        eta = (c_luz_km/(8315*100)) * np.sqrt(omega_m/0.13)
        c1, c2 = params_fisicos_to_modelo_HS(omega_m, b, n=n)

        params_modelo = [c1,c2,n]

    elif model=='ST':
        #OJO Rs depende de H0, pero no se usa :) No como HS!! Cambiarlo en la tesis!!!
        [omega_m, b, H0] = params_fisicos
        lamb = 2/b

        cond_iniciales = condiciones_iniciales(omega_m, b, model='ST')
        eta = np.sqrt(3 * (1 - omega_m) * b)

        params_modelo=[lamb, n]

    #Integramos el sistema
    zs_int = np.linspace(z_inicial,z_final,cantidad_zs)
    sol = solve_ivp(sistema_ec, (z_inicial,z_final),
        cond_iniciales, t_eval=zs_int, args=(params_modelo,model),
        max_step=max_step)

    if (len(sol.t)!=cantidad_zs):
        print('Está integrando mal!')
    if np.all(zs_int==sol.t)==False:
        print('Hay algo raro!')

    #Calculamos el Hubble
    zs = sol.t[::-1]
    r=sol.y[4][::-1]

    t2 = time.time()

    if verbose == True:
        print('Duración {} minutos y {} segundos'.format(int((t2-t1)/60),
              int((t2-t1) - 60*int((t2-t1)/60))))

    return zs, r



#%%
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    z_final = 0
    z_inicial = 30
    cantidad_zs = int(10**5)
    max_step = 10**(-4)

    H0 = 69.076
    b = 0.122
    omega_m = 0.262
    params_fisicos = [omega_m,b,H0]

    zs, r = integrador_r(params_fisicos, n=1, cantidad_zs=cantidad_zs,
                max_step=max_step, z_inicial=z_inicial, z_final=0, sistema_ec=dX_dz,
                verbose=True,model='HS')

    h=H0/100
    R_HS=(omega_m*h**2)/(0.13*8315**2)
    R = R_HS * r
    print(r[0],R_HS,R[0])

    #%matplotlib qt5
    #plt.close()
    plt.figure()
    plt.grid(True)
    plt.title('Escalar de Ricci por integración numérica')
    plt.xlabel('z(redshift)')
    plt.ylabel('R (Mpc^{-2})')
    plt.plot(zs, r, label='Ricci')
    plt.legend(loc='best')
