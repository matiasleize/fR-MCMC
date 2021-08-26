import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import c as c_luz #metros/segundos
from scipy.integrate import simps,trapz,cumtrapz
c_luz_km = c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_cambio_parametros import params_fisicos_to_modelo_HS, params_fisicos_to_modelo_ST


import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000;

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_cambio_parametros import params_fisicos_to_modelo_HS
from funciones_LambdaCDM import H_LCDM
from funciones_taylor import Taylor_HS, Taylor_ST
#%%
def condiciones_iniciales(omega_m, b, z0=30, n=1, model='HS'):
    '''
    Calculo las condiciones iniciales para el sistema de ecuaciones diferenciales
    para el modelo de Hu-Sawicki y el de Starobinsky n=1
    OBSERVACION IMPORTANTE: Lamb, R_HS estan reescalados por un factor H0**2 y
    H reescalado por un facor H0. Esto es para librarnos de la dependencia de
    las condiciones iniciales con H0. Ademas, como el output son adminensionales
    podemos tomar c=1 (lo chequeamos en el papel).
    '''
    R = sym.Symbol('R')
    Lamb = 3 * (1-omega_m)

    if model=='HS':
        c1,c2 = params_fisicos_to_modelo_HS(omega_m,b,n)
    #    R_HS = 2 * Lamb * c2/c1
        alfa=3*(1-omega_m)*b
        R_0 = alfa #No confundir con R0 que es R en la CI!

        #Calculo F. Ambas F dan las mismas CI para z=z0 :)
        #F = R - ((c1*R)/((c2*R/R_HS)+1))
        F = R - 2 * Lamb * (1 - 1/ (1 + (R/(b*Lamb)**n)) )
    elif model=='ST':
        #lamb = 2 / b
        R_ST = Lamb * b
        R_0 = R_ST #No confundir con R0 que es R en la CI!

        #Calculo F.
        F = R - 2 * Lamb * (1 - 1/ (1 + (R/(b*Lamb)**2) ))

    #Calculo las derivadas de F
    F_R = sym.simplify(sym.diff(F,R))
    F_2R = sym.simplify(sym.diff(F_R,R))

    z = sym.Symbol('z')
    H = (omega_m*(1+z)**3 + (1-omega_m))**(0.5)
    #H_z = ((1+z)**3 *3 * omega_m)/(2*(1+omega_m*(-1+(1+z)**3))**(0.5))
    H_z = sym.simplify(sym.diff(H,z))
    H_2z = sym.simplify(sym.diff(H_z,z))

    Ricci = (12*H**2 + 6*H_z*(-H*(1+z)))
    Ricci_t=sym.simplify(sym.diff(Ricci,z)*(-H*(1+z)))

    Ricci_ci=sym.lambdify(z,Ricci)
    Ricci_t_ci=sym.lambdify(z,Ricci_t)
    H_ci=sym.lambdify(z,H)
    H_z_ci=sym.lambdify(z,H_z)
    F_ci=sym.lambdify(R,F)
    F_R_ci=sym.lambdify(R,F_R)
    F_2R_ci=sym.lambdify(R,F_2R)

#alpha
#    alfa=exp(z0)-1
    R0=Ricci_ci(z0)
    Ricci_t_ci(z0)
    H_ci(z0) #Chequie que de lo esperado x Basilakos
    H_z_ci(z0) #Chequie que de lo esperado x Basilakos
    F_ci(R0) # debe ser simil a R0-2*Lamb
    F_R_ci(R0) # debe ser simil a 1
    F_2R_ci(R0) # debe ser simil a 0

    x0 = Ricci_t_ci(z0)*F_2R_ci(R0) / (H_ci(z0)*F_R_ci(R0))
    y0 = F_ci(R0) / (6*(H_ci(z0)**2)*F_R_ci(R0))
    v0 = R0 / (6*H_ci(z0)**2)
    w0 = 1+x0+y0-v0
    r0 = R0/R_0

    return[x0,y0,v0,w0,r0]



def integrals(ys, xs):
    x_range = []
    y_range = []
    results = []
    for i in range(len(xs)):
        x_range.append(xs[i])
        y_range.append(ys[i])
        integral = trapz(y_range, x_range) #Da un error relativo de 10**(-7)
            #integral = trapz(y_range, x_range) #Da un error relativo de 10**(-15)
        results.append(integral)
    return np.array(results)

def dX_dz(z, variables, params_modelo, model='HS'):
    '''
    Sistema de ecuaciones a resolver por la funcion integrador
    Parameters:
        params_modelo: list
            lista de n parametros, donde los primeros n-1 elementos son los
            parametros del sistema, mientras que el ultimo argumento especifica el modelo
            en cuestion, matematicamente dado por la funcion \Gamma.
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
            (r**2 + 1)**(N + 2)*(-lamb*N*r*(r**2 + 1)**(-N - 1) + 1/2)/(lamb*N*r*(2*N*r**2 + r**2 - 1))
            print('Guarda que estas poniendo n!=1')
            pass


    elif model == 'HS': #Para el modelo de Hu-Sawicki
        [B,D,N] = params_modelo
        if N==1:
            gamma = lambda r,c1,c2: -(c1 - (r + 1)**2)*(r + 1)/(2*c1*r)
            G = gamma(r,B,D) #Va como r^3/r = r^2
        elif N==2:
            gamma = lambda r,c1,c2: (-2*c1*c2*r**3 - 2*c1*r + c2**3*r**6 + 3*c2**2*r**4 + 3*c2*r**2 + 1)/(2*c1*r*(3*c2*r**2 - 1))
            G = gamma(r,B,D) #Va como r^6/r^3 = r^3
        else:
            gamma = lambda r,c1,c2,n: r**(-n)*(-c1*c2*n*r**(2*n) - c1*n*r**n + c2**3*r**(3*n + 1) + 3*c2**2*r**(2*n + 1) + 3*c2*r**(n + 1) + r)/(c1*n*(c2*n*r**n + c2*r**n - n + 1))
            G = gamma(r,B,D,N)
            #print('Guarda que estas poniendo n!=1')
    else:
        print('Elegir un modelo valido!')
        pass



    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
    s1 = (- (v*x*G - x*y + 4*y - 2*y*v)) / (z+1)
    s2 = (-v * (x*G + 4 - 2*v)) / (z+1)
    s3 = (w * (-1 + x + 2*v)) / (z+1)
    s4 = (-(x * r * G)) / (1+z)
    return [s0,s1,s2,s3,s4]


def integrador(params_fisicos, n=1, cantidad_zs=int(10**5), max_step=10**(-5),
                z_inicial=30, z_final=0, sistema_ec=dX_dz, verbose=False,
                model='HS'):
    #Para HS n=1 con max_step 0.003 alcanza.
    '''
    Integracion numerica del sistema de ecuaciones diferenciales entre
    z_inicial y z_final, dadas las condiciones iniciales de las variables
    (x,y,v,w,r) y los parametros 'con sentido fisico' del modelo f(R).

    Parameters:
        cantidad_zs: int
            cantidad de puntos (redshifts) en los que se evalua la
            integracion numerica. Es necesario que la cantidad de puntos en el
            area de interes $(z \in [0,3])$.
        mas_step: int
            paso de la integracion numerica. Cuando los parametros que aparecen
            en el sistema de ecuaciones se vuelven muy grandes (creo que esto implica
            que el sistema a resolver se vuelva mas inestable) es necesario que el paso
            de integracion sea pequenio
        verbose: Bool
            if True, imprime el tiempo que tarda el proceso de integracion.

    Output: list
        Un array de Numpy de redshifts z y un array de H(z).
    '''

    t1 = time.time()

    if model=='HS':
        [omega_m, b, H0] = params_fisicos
        #Calculo las condiciones cond_iniciales, eta
        # y los parametros de la ecuacion
        cond_iniciales = condiciones_iniciales(omega_m, b, z0=z_inicial, n=n)
        alfa = H0*np.sqrt((1-omega_m)*b/2)
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
        print('Esta integrando mal!')
    if np.all(zs_int==sol.t)==False:
        print('Hay algo raro!')

    #Calculamos el Hubble
    zs = sol.t[::-1]
    v=sol.y[2][::-1]
    r=sol.y[4][::-1]
# metodo de la raiz2
    Hs = alfa*np.sqrt(r/v)
# metodo de la integral
#    integrando2=np.interp(zs2,zs,integrando)
#    zs3=zs2[::-1]
#    integrando3=integrando2[::-1]
#    F=integrals(integrando3,zs3)
#    Hs2=H0*np.sqrt(omega_m*(1+z_inicial)**3+1-omega_m)*(1+zs3)**2/(1+z_inicial)**2*np.exp(-F)
#    Hs3=Hs2[::-1]


    t2 = time.time()

    if verbose == True:
        print('Duracion {} minutos y {} segundos'.format(int((t2-t1)/60),
              int((t2-t1) - 60*int((t2-t1)/60))))

#    return zs, Hs,zs2, Hs3
    return zs,Hs

def Hubble_teorico_2(params_fisicos, b_crit=0.15, all_analytic=False,
                    eval_data=False, z_data=None, epsilon=10**(-10), n=1,
                    cantidad_zs=int(10**5), max_step=10**(-4),
                    z_min=0, z_max=10, sistema_ec=dX_dz,
                    verbose=False, model='HS', method='LSODA'):

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

        if model=='HS':
            Hs_modelo = Taylor_HS(zs_modelo, omega_m, b, H0)
        elif model=='ST':
            Hs_modelo = Taylor_ST(zs_modelo, omega_m, b, H0)
        elif model=='EXP': #Devuelvo LCDM
            Hs_modelo = H_LCDM(zs_modelo, omega_m, H0)

    else: #Integro
        if eval_data == False:
            zs_modelo, Hs_modelo = integrador(params_fisicos,
                                    cantidad_zs=cantidad_zs, max_step=max_step,
                                    z_inicial=z_max, z_final=z_min, sistema_ec=sistema_ec,
                                    verbose=verbose, model=model)
        else:
            zs_modelo, Hs_modelo = integrador(params_fisicos,
                                    cantidad_zs=cantidad_zs, max_step=max_step,
                                    z_inicial=z_max, z_final=z_min, sistema_ec=sistema_ec,
                                    verbose=verbose, eval_data=True, z_data = z_data,
                                    model=model)
    return zs_modelo, Hs_modelo

#%%
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    #Esto correrlo con b grande para que ande!
    b = 1
    omega_m = 0.3
    H0 = 73.48

    params_fisicos = [omega_m,b,H0]
    zs_ode, Hs_HS = integrador(params_fisicos, verbose=True, model='HS',z_inicial=10)
    #_, Hs_EXP = integrador(params_fisicos, epsilon=10**(-10),
    #            verbose=True, model='EXP',z_inicial=10)


    #%% La otra funcion:
    b = 0.1
    omega_m = 0.1
    H0 = 73.48
    params_fisicos = [omega_m,b,H0]
    zs_ode, Hs_HS = Hubble_teorico(params_fisicos, verbose=True, model='HS')
    zs_ode, Hs_HS
    #%%
    #%matplotlib qt5
    plt.figure()
    plt.title('Integrador $f(R)$')
    plt.xlabel('z (redshift)')
    plt.ylabel('H(z) ((km/seg)/Mpc)')
    plt.plot(zs_ode,Hs_HS,'.',label='HS')
    plt.legend(loc = 'best')
    plt.grid(True)
