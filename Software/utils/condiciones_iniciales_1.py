'''
Initial conditions for the sctipt "int_sis_1.py"
'''

import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000;

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_datos_global = os.path.dirname(path_git)
os.chdir(path_git)
os.sys.path.append('./Software/utils/')
from cambio_parametros import params_fisicos_to_modelo_HS

def z_condicion_inicial(params_fisicos,eps=10**(-10)):
    [omega_m,b,_] = params_fisicos
    beta = 2/b
    omega_l = 1 - omega_m
    z0 = (2 * omega_l*(-np.log(eps)-2*beta)/(beta*omega_m))**(1/3) - 1
    return z0

def condiciones_iniciales(omega_m, b, z0=30, n=1, model='HS',CI_aprox=True):
    '''
    Calculo las condiciones iniciales para el sistema de ecuaciones diferenciales
    para el modelo de Hu-Sawicki y el de Starobinsky n=1
    OBSERVACION IMPORTANTE: Lamb, R_HS z=están reescalados por un factor H0**2 y
    H reescalado por un facor H0. Esto es para librarnos de la dependencia de
    las condiciones iniciales con H0. Además, como el output son adminensionales
    podemos tomar c=1 (lo chequeamos en el papel).
    '''
    R = sym.Symbol('R')
    Lamb = 3 * (1-omega_m)


    if model=='EXP':
        #Defino el z inicial
        omega_l = 1-omega_m
        beta = 2/b
        #eps = 10**(-9)
        #x0 = -(1/3) * np.log((-omega_l/omega_m)*((np.log(eps)/beta)+2))
        #z0 = np.exp(-x0)-1


        z = sym.Symbol('z')
        H = (omega_m*(1+z)**3 + omega_l)**(0.5)

        tildeR = 2 + (omega_m/(2*(1 - omega_m))) * (1+z)**3

        tildeR_ci=sym.lambdify(z,tildeR)
        H_ci=sym.lambdify(z,H)

        tildeR0=tildeR_ci(z0)
        E0 = H_ci(z0) #Esta ya noramlizado por H0!

        return[E0,tildeR0]

    elif model=='HS':
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

    #Calculo las derivadas de F
    F_R = sym.diff(F,R) #saué el sym.simplify para que ande el modelo exp!
    F_2R = sym.diff(F_R,R)

    z = sym.Symbol('z')
    H = (omega_m*(1+z)**3 + (1-omega_m))**(0.5)
    #H_z = ((1+z)**3 *3 * omega_m)/(2*(1+omega_m*(-1+(1+z)**3))**(0.5))
    H_z = sym.simplify(sym.diff(H,z))

    Ricci = (12*H**2 + 6*H_z*(-H*(1+z)))
    Ricci_t=sym.simplify(sym.diff(Ricci,z)*(-H*(1+z)))

    Ricci_ci=sym.lambdify(z,Ricci)
    Ricci_t_ci=sym.lambdify(z,Ricci_t)
    H_ci=sym.lambdify(z,H)
    H_z_ci=sym.lambdify(z,H_z)
    F_ci=sym.lambdify(R,F)
    F_R_ci=sym.lambdify(R,F_R)
    F_2R_ci=sym.lambdify(R,F_2R)

    R0=Ricci_ci(z0)
    #Ricci_t_ci(z0)
    #H_ci(z0) #Chequie que de lo esperado x Basilakos
    #H_z_ci(z0) #Chequie que de lo esperado x Basilakos
    #F_ci(R0) # debe ser simil a R0-2*Lamb
    #F_R_ci(R0) # debe ser simil a 1
    #F_2R_ci(R0) # debe ser simil a 0


    if CI_aprox == True:
        #Hibrid initial conditions
        x0 = Ricci_t_ci(z0)*F_2R_ci(R0) / (H_ci(z0)*F_R_ci(R0))
        y0 = F_ci(R0) / (6*(H_ci(z0)**2)*F_R_ci(R0))
        v0 = R0 / (6*H_ci(z0)**2)
        w0 = 1+x0+y0-v0
        r0 = R0/R_0
    else:
        #LCDM initial conditions
        x0 = 0
        y0 = (R0  - 2 * Lamb) / (6*H_ci(z0)**2)
        v0 = R0 / (6*H_ci(z0)**2)
        w0 = 1+x0+y0-v0
        r0 = R0/R_0
    return[x0,y0,v0,w0,r0]

#%%
if __name__ == '__main__':
    omega_m = 0.2
    b = 0.6
    params_fisicos = [omega_m,b,_]
    print(z_condicion_inicial(params_fisicos,eps=10**(-10)))
    #%%
    H0 = 73.48
    z0 = 30
    cond_iniciales=condiciones_iniciales(omega_m,b,z0=z0,model='HS')
    print(cond_iniciales)
    cond_iniciales=condiciones_iniciales(omega_m,b,z0=z0,model='ST')
    print(cond_iniciales)
    #%%
    bs = np.arange(0.2,1.1,0.1)
    omegas = np.arange(0.2,0.51,0.01)
    output = np.zeros((len(bs),len(omegas)))
    bs
    omegas
    for i, b in enumerate(bs):
        for j, omega in enumerate(omegas):
            cond_iniciales=condiciones_iniciales(omega_m=omega,b=b,z0=3,
                            model='EXP')
            output[i,j] = 2 * cond_iniciales[1]/b #lo convierto en r para comparar
    np.savetxt('2darray.csv', output, delimiter=',', fmt='%1.2f')
    output
    #%%
    cond_iniciales=condiciones_iniciales(omega_m,b,z0=z0,model='HS')
    cond_iniciales_1=condiciones_iniciales(omega_m,b,z0=z0,model='HS',CI_aprox=False)
    print(cond_iniciales)
    print(cond_iniciales_1)
