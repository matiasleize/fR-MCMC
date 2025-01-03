'''
Initial conditions for the sctipt "solve_sys.py"
'''

import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math
from scipy.constants import c as c_light #meters/seconds
c_light_km = c_light/1000;

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_global = os.path.dirname(path_git)
#os.chdir(path_git)
#os.sys.path.append('./fr_mcmc/utils/')
#from change_of_parameters import physical_to_model_params_HS

def redshift_initial_condition(physical_params, eps=10**(-10)):
    '''
    z_i for the Odintsov system of equations
    x0 = -(1/3) * np.log((-omega_l/omega_m)*((np.log(eps)/beta)+2))
    z0 = np.exp(-x0)-1    
    '''
 
    [omega_m, b, _] = physical_params
    beta = 2/b
    omega_l = 1 - omega_m
    
    #Initial z
    zi = (2 * omega_l*(-np.log(eps)-2*beta)/(beta*omega_m))**(1/3) - 1
    return zi

def calculate_initial_conditions(physical_params, zi = 30, model = 'HS', CI_aprox=False):
    '''
    Initial conditions for the differential equations
    of the Hu-Sawicki and Starobinsky model (n=1).
    Important: Lamb, R_HS z are rescaled by a factor H0**2 and
    H is rescaled by a factor H0. In this way the initial conditions are 
    independent from H0. Besides, since this functions return admientional values,
    we can take c=1 (we checked this analitically).
    '''

    [omega_m, b, _] = physical_params

    z = sym.Symbol('z')
    E = (omega_m*(1+z)**3 + (1-omega_m))**(0.5)

    if (model=='EXP' or model=='Odintsov'):
        omega_l = 1-omega_m

        tildeR = 2 + (omega_m/(2*(1 - omega_m))) * (1+z)**3

        tildeR_ci = sym.lambdify(z,tildeR)
        E_ci = sym.lambdify(z,E)

        tildeR_i = tildeR_ci(zi)
        E0 = E_ci(zi) #Already normalized by H0!

        return[E0, tildeR_i]

    elif (model=='HS' or model=='ST'):
        R = sym.Symbol('R')
        Lamb = 3 * (1-omega_m) #/c_light_km**2
        
        #c1,c2 = physical_to_model_params_HS(omega_m,b)
        #R_HS = 2 * Lamb * c2/c1
        #R_HS = 6 * c_light_km**2 * omega_m / (7800 * (8315)**2)

        R_0 = Lamb #This is not the same of R_i, which is R on the IC!

        #F calculation
        F = R - 2 * Lamb * (1 - 1/ (1 + (R/(b*Lamb))) )
   
        #Derivatives of F
        F_R = sym.diff(F,R)
        F_2R = sym.diff(F_R,R)

        E_z = sym.simplify(sym.diff(E,z))

        #Since there is an independency of H0 in the final results, we define H=E in order
        # to have the correct units on the final results.
        H = E
        H_z = E_z

        Ricci = (12*H**2 + 6*H_z*(-H*(1+z)))#/c_light_km**2
        Ricci_t = sym.simplify(sym.diff(Ricci,z)*(-H*(1+z)))

        Ricci_ci = sym.lambdify(z,Ricci)
        Ricci_t_ci = sym.lambdify(z,Ricci_t)
        H_ci = sym.lambdify(z,H)
        H_z_ci = sym.lambdify(z,H_z)
        F_ci = sym.lambdify(R,F)
        F_R_ci = sym.lambdify(R,F_R)
        F_2R_ci = sym.lambdify(R,F_2R)

        R_i = Ricci_ci(zi)
        #H_ci(zi) #Same as Basilakos
        #H_z_ci(zi) #Same as Basilakos

        if CI_aprox == True: #Hibrid initial conditions
            xi = Ricci_t_ci(zi) * F_2R_ci(R_i) / (H_ci(zi) * F_R_ci(R_i))
            yi = F_ci(R_i) / (6 * (H_ci(zi)**2) * F_R_ci(R_i))
            vi = R_i / (6 * H_ci(zi)**2)
            wi = 1 + xi + yi - vi
            ri = R_i / R_0

        else: #LCDM initial conditions
            xi = 0
            yi = (R_i  - 2 * Lamb) / (6 * H_ci(zi)**2)
            vi = R_i / (6 * H_ci(zi)**2)
            wi = 1 + xi + yi - vi
            ri = R_i / R_0

        return[xi,yi,vi,wi,ri]

#%%
if __name__ == '__main__':
    omega_m = 0.2
    b = 0.6
    physical_params = [omega_m, b, 0]
    print(redshift_initial_condition(physical_params, eps=10**(-10)))
    #%%
    H0 = 73.48
    zi = 30
    initial_cond = calculate_initial_conditions(physical_params, zi=zi, model='HS')
    print(initial_cond)
    #%%
    bs = np.arange(0.2,1.1,0.1)
    omegas = np.arange(0.2,0.51,0.01)
    output = np.zeros((len(bs),len(omegas)))
    for i, b in enumerate(bs):
        for j, omega in enumerate(omegas):
            physical_params = [omega_m,b,0]
            initial_cond=calculate_initial_conditions(physical_params,zi=3,
                            model='EXP')
            output[i,j] = 2 * initial_cond[1]/b #lo convierto en r para comparar
    #np.savetxt('2darray.csv', output, delimiter=',', fmt='%1.2f')
    print(output)
    #%%
    initial_cond_hibrid = calculate_initial_conditions(physical_params, zi=zi, model='HS', CI_aprox=True)
    initial_cond_LCDM = calculate_initial_conditions(physical_params, zi=zi, model='HS', CI_aprox=False)
    print(initial_cond_LCDM)
    print(initial_cond_hibrid)

