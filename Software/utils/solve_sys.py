'''
Integration of the ODE for the different cosmological models. For the Hu-Sawicki model
we use De la Cruz et al. ODE. Besides, for the Exponential model we use the Odintsov ODE.
Note that the IC are different for the two models. 

int_2 (located at test folder) only changes how is the calculation of eta (alfa on the code).

TODO: Check if it is faster to use eta_1 or eta_2.

TODO: Check the times of integrations in comparison of HS in comparison with the one of Odintsov
and evaluate this difference.
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

from initial_conditions import condiciones_iniciales, z_condicion_inicial
from change_of_parameters import params_fisicos_to_modelo_HS
from LambdaCDM import H_LCDM
from taylor import Taylor_HS
#%%

def dX_dz(z, variables, params_fisicos, model='HS'):
    '''
    System of equations to solve.
    Parameters:
        params_fisicos: list
            # list of  n parameters, where the first n-1 elements are the model parameters,
            # while the last one specify the cosmological model. 
            # Mathematically, this information is contained in the function Gamma.
        model: string
            Cosmological model that is integrated.
    Returns: list
        Set of ODE for the dynamical variables.
    '''
    
    [omega_m, b, _] = params_fisicos

    if model == 'EXP': #For the Exponential model
        E = variables[0]
        tildeR = variables[1]

        beta = 2/b
        omega_l = 1 - omega_m

        s0 = omega_l * tildeR/E - 2 * E
        s1 = (np.exp(beta*tildeR)/(beta**2)) * (omega_m * np.exp(-3*z)/(E**2)-1+beta*np.exp(-beta*tildeR) + omega_l*(1-(1+beta*tildeR)*np.exp(-beta*tildeR))/(E**2))

        return [s0,s1]

    elif model == 'HS': #For the Hu-Sawicki model
        x = variables[0]
        y = variables[1]
        v = variables[2]
        w = variables[3]
        r = variables[4]

        #Calculate the model parameters
        B, D = params_fisicos_to_modelo_HS(omega_m, b) # (c1,c2) = (B,D) from De la Cruz et al.

        gamma = lambda r,c1,c2: -(c1 - (c2*r + 1)**2)*(c2*r + 1)/(2*c1*c2*r)
        G = gamma(r,B,D) #Goes like r^3/r = r^2

        s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
        s1 = (- (v*x*G - x*y + 4*y - 2*y*v)) / (z+1)
        s2 = (-v * (x*G + 4 - 2*v)) / (z+1)
        s3 = (w * (-1 + x + 2*v)) / (z+1)
        s4 = (-(x * r * G)) / (1+z)

        return [s0,s1,s2,s3,s4]

    else:
        print('Choose a valid model!')
        pass

def integrator(params_fisicos, epsilon=10**(-10), cantidad_zs=int(10**5),
                z_inicial=10, z_final=0,
                sistema_ec=dX_dz, verbose=False, eval_data=False, z_data = None,
                model='HS',method='RK45', rtol=1e-11, atol=1e-16):
    '''
    Numerical integration of the system of differential equations between
    z_inicial and z_final, given the initial conditions of the variables
    (x,y,v,w,r) and the 'physically meaningful' parameters of the f(R) model.
    Parameters: 
        cantidad_zs: int
            number of points (in redshift) in which the numerical integration
            is evaluated.
        verbose: Bool
            if True, prints the time of integration.
        TODO: complete the list of input parameters
    Output: list
        Un array de Numpy de redshifts z y un array de H(z).
    '''

    t1 = time.time()
    [omega_m, b, H0] = params_fisicos

    if model=='EXP':
        z_ci = z_condicion_inicial(params_fisicos,epsilon)
        beta = 2/b
        cond_iniciales = condiciones_iniciales(params_fisicos, zi=z_ci, model='EXP')

        #Integrate the system
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

        # LCDM part
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
        # Calculate the IC, eta
        # and the parameters of the ODE.
        cond_iniciales = condiciones_iniciales(params_fisicos, zi=z_inicial)

        h = H0/100
        R_HS = (omega_m * h**2)/(0.13*8315**2) #Mpc**(-2)
        eta = c_luz_km * np.sqrt(R_HS/6) #(km/seg)/Mpc

        if eval_data == False:
            zs_int = np.linspace(z_inicial, z_final, cantidad_zs)

            sol = solve_ivp(sistema_ec, (z_inicial,z_final),
                cond_iniciales, t_eval=zs_int, args=(params_fisicos, model),
                rtol=rtol, atol=atol, method=method)

            assert len(sol.t)==cantidad_zs, 'Something is wrong with the integration!'
            assert np.all(zs_int==sol.t), 'Not all the values of z coincide with the ones that were required!'

        else:
            sol = solve_ivp(sistema_ec, (z_inicial,z_final),
                cond_iniciales, t_eval=z_data.reverse(), args=(params_fisicos,model),
                rtol=rtol, atol=atol, method=method)

            assert len(sol.t)==cantidad_zs, 'Something is wrong with the integration!'
            assert np.all(zs_int==sol.t), 'Not all the values of z coincide with the ones that were required!'

        # Calculate the Hubble parameter
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
        # Calculate the IC, eta
        # and the parameters of the ODE.
        cond_iniciales = condiciones_iniciales(omega_m, b, z0=z_inicial, n=n)
        alfa = H0*np.sqrt((1-omega_m)*b/2)
        c1, c2 = params_fisicos_to_modelo_HS(omega_m, b, n=n)

        params_modelo = [c1,c2,n]

    # Integrate the system
    zs_int = np.linspace(z_inicial,z_final,cantidad_zs)
    sol = solve_ivp(sistema_ec, (z_inicial,z_final),
        cond_iniciales, t_eval=zs_int, args=(params_modelo,model),
        max_step=max_step)
    
    assert len(sol.t)==cantidad_zs, 'Something is wrong with the integration!'
    assert np.all(zs_int==sol.t), 'Not all the values of z coincide with the ones that were required!'



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

    elif model=='EXP': #b critical for the Exponential model
        log_eps_inv = -np.log10(epsilon)
        b_crit = (4 + omega_m/(1-omega_m)) / log_eps_inv
    else:
        pass

    if (b <= b_crit) or (all_analytic==True): #Analytic approximation
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
        elif model=='EXP': #Return LCDM
            Hs_modelo = H_LCDM(zs_modelo, omega_m, H0)

    else: #Integrate
        if eval_data == False:
            zs_modelo, Hs_modelo = integrator(params_fisicos, epsilon=epsilon,
                                    cantidad_zs=cantidad_zs,
                                    z_inicial=z_max, z_final=z_min, sistema_ec=sistema_ec,
                                    verbose=verbose, model=model,
                                     method=method,rtol=rtol, atol=atol)
        else:
            zs_modelo, Hs_modelo = integrator(params_fisicos, epsilon=epsilon,
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
    zs_ode, H_HS = integrator(params_fisicos, verbose=True, model='HS')
    _, H_HS_1 = Hubble_th_1(params_fisicos, verbose=True, model='HS')
    #%% Exponential
    params_fisicos = [0.3, 2, 73] # [omega_m, b, H0]
    zs_ode, H_EXP = integrator(params_fisicos, verbose=True, model='EXP')
    _, H_EXP_1 = Hubble_th_1(params_fisicos, verbose=True, model='EXP')
    #%% Plot all models together
    #%matplotlib qt5
    plt.figure()
    plt.title('Integrator $f(R)$')
    plt.xlabel('z (redshift)')
    plt.ylabel('H(z) ((km/seg)/Mpc)')
    plt.plot(zs_ode,H_HS,'.',label='HS')
    plt.plot(zs_ode,H_EXP,'.',label='Exp')
    plt.plot(zs_ode,H_LCDM(zs_ode,0.3,73),'.',label='LCDM') #H_LCDM(zs_ode,omega_m,H0)
    plt.legend(loc = 'best')
    plt.grid(True)
    plt.show()
