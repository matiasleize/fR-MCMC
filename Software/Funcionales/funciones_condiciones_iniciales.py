import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np

def z_condicion_inicial(params_fisicos, eps = 10**(-10)):
    [omega_m,b,_] = params_fisicos
    beta = 2/b
    omega_l = 1 - omega_m
    z0 = (2 * omega_l*(-np.log(eps)-2*beta)/(beta*omega_m))**(1/3) - 1
    return z0

def condiciones_iniciales(omega_m, z0):
    '''
    (ACTUALIZAR, ya no importa H0, en vez de H aparece E que es lo que usa Odinstov
    en su sistema!)
    Calculo las condiciones iniciales para el sistema de ecuaciones diferenciales
    para el modelo de Hu-Sawicki y el de Starobinsky n=1
    OBSERVACION IMPORTANTE: Lamb, R_HS z=están reescalados por un factor H0**2 y
    H reescalado por un facor H0. Esto es para librarnos de la dependencia de
    las condiciones iniciales con H0. Además, como el output son adminensionales
    podemos tomar c=1 (lo chequeamos en el papel).
    '''
    #Defino el z inicial
    omega_l = 1-omega_m

    z = sym.Symbol('z')
    E = (omega_m*(1+z)**3 + omega_l)**(0.5)

    tildeR = 2 + (omega_m/(2*(1 - omega_m))) * (1+z)**3

    tildeR_ci = sym.lambdify(z,tildeR)
    E_ci = sym.lambdify(z,E)

    tildeR0 = tildeR_ci(z0)
    E0 = E_ci(z0) #Esta ya noramlizado por H0!

    return[E0, tildeR0]

#%%
if __name__ == '__main__':
    omega_m = 0.3
    b = 0.5
    params_fisicos = [omega_m,b,_]
    print(z_condicion_inicial(params_fisicos,eps=10**(-10)))
    #%%
    z0 = 50
    cond_iniciales=condiciones_iniciales(omega_m,z0)
    print(cond_iniciales)
