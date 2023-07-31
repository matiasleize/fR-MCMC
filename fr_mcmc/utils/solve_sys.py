'''
Integration of the ODE for the different cosmological models. For the Hu-Sawicki model
we use De la Cruz et al. ODE. Besides, for the Exponential model we use the Odintsov ODE.
Note that the initial conditions are different for the two models. 

TODO: Check the times of integrations of HS using De la Cruz ODE in comparison with the one of Odintsov
and evaluate the difference.

TODO: Implement Starobinsky model integration.
'''
import os
import time

import git
import numpy as np
from scipy.constants import c as c_light  # units of m/s
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from change_of_parameters import physical_to_model_params_HS
from initial_conditions import calculate_initial_conditions, redshift_initial_condition

c_light_km = c_light / 1000 # units of km/s
path_git = git.Repo(".", search_parent_directories=True).working_tree_dir
path_global = os.path.dirname(path_git)
os.chdir(path_git)
os.sys.path.append("./fr_mcmc/utils/")
from LambdaCDM import H_LCDM
from taylor import Taylor_HS
#%%

def get_odes(z, variables, physical_params, model='HS'):
    '''
    Returns the system of ODEs for the given cosmological model.


    Parameters:
    -----------
    z : float
        Redshift value.
    variables : list
        List of values for the dynamical variables.
    physical_params : list
        List of model parameters, where the first n-1 elements are the model parameters,
        while the last one specifies the cosmological model. Mathematically, this information is contained in
        the function Gamma.
    model : str, optional
        Cosmological model that is being integrated. Defaults to 'HS'.

    Returns:
    -----------
    list
        Set of ODEs for the dynamical variables.
    '''    
    omega_m, b, _ = physical_params

    if model == 'EXP': # For the Exponential model
        E, tildeR = variables

        beta = 2/b
        omega_l = 1 - omega_m

        s0 = omega_l * tildeR/E - 2 * E
        s1 = (np.exp(beta*tildeR)/(beta**2)) * (omega_m * np.exp(-3*z)/(E**2)-1+beta*np.exp(-beta*tildeR) + omega_l*(1-(1+beta*tildeR)*np.exp(-beta*tildeR))/(E**2))

        return [s0, s1]

    elif model == 'HS': # For the Hu-Sawicki model
        x, y, v, w, r = variables

        # Calculate the model parameters
        B, D = physical_to_model_params_HS(omega_m, b) # (c1,c2) = (B,D) from De la Cruz et al.

        def gamma(r, c1, c2):
            return -(c1 - (c2*r + 1)**2)*(c2*r + 1)/(2*c1*c2*r)

        G = gamma(r, B, D) # Goes like r^3/r = r^2

        s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
        s1 = (- (v*x*G - x*y + 4*y - 2*y*v)) / (z+1)
        s2 = (-v*(x*G + 4 - 2*v)) / (z+1)
        s3 = (w*(-1 + x + 2*v)) / (z+1)
        s4 = -(x*r*G) / (1+z)

        return [s0, s1, s2, s3, s4]

    else:
        raise ValueError('Invalid model specified. Choose from "EXP" or "HS".')

def integrator(physical_params, epsilon=10**(-10), num_z_points=int(10**5),
                initial_z=10, final_z=0,
                system_equations=get_odes, verbose=False,
                model='HS',method='RK45', rtol=1e-11, atol=1e-16):
    '''
    Numerical integration of the system of differential equations between
    initial_z and final_z, given the initial conditions of the variables
    and the 'physically meaningful' parameters of the f(R) model.

    Parameters: 
    -----------
    physical_params: list
        A list containing the 'physically meaningful' parameters of the f(R) model,
        the list must have three elements, the first is the density parameter of matter,
        the second is the parameter 'b' of the f(R) model, and the third is the Hubble
        constant in units of km/s/Mpc.

    initial_z: float
        The initial redshift value.

    final_z: float
        The final redshift value.

    system_equations: function
        A function that returns the system of differential equations.

    model: str, default='HS'
        The f(R) model to use, this parameter should be one of the following:
        'HS', 'EXP'.

    epsilon: float, default=10**(-10)
        A small float number used to find the initial condition in the EXP model.

    num_z_points: int, default=int(10**5)
        The number of points (in redshift) in which the numerical integration
        is evaluated.

    verbose: bool, default=False
        If True, prints the time of integration.

    method: str, default='RK45'
        Integration method, the method must be compatible with the method
        argument of scipy.integrate.solve_ivp.

    rtol: float, default=1e-11
        Relative tolerance, the tolerance must be compatible with the rtol
        argument of scipy.integrate.solve_ivp.

    atol: float, default=1e-16
        Absolute tolerance, the tolerance must be compatible with the atol
        argument of scipy.integrate.solve_ivp.

    Returns: 
    --------
    tuple
        An array of Numpy of redshifts z and an array of H(z).
    '''


    t1 = time.time()
    omega_m, b, H0 = physical_params

    if model=='EXP':
        z_ci = redshift_initial_condition(physical_params,epsilon)
        beta = 2/b
        initial_cond = calculate_initial_conditions(physical_params, zi=z_ci, model='EXP')

        #Integrate the system
        zs_int = np.linspace(z_ci,final_z,num_z_points)

        x_ci = -np.log1p(z_ci)
        x_final = -np.log1p(final_z)
        xs_int = -np.log1p(zs_int)

        sol = solve_ivp(system_equations, (x_ci,x_final),
            initial_cond, t_eval=xs_int, args=(physical_params, model),
            rtol=rtol, atol=atol, method=method)

        xs_ode = sol.t[::-1]
        zs_ode = np.exp(-xs_ode)-1
        Hs_ode = H0 * sol.y[0][::-1]

        # LCDM part
        zs_LCDM = np.linspace(z_ci,initial_z,num_z_points)
        Hs_LCDM = H0 * np.sqrt(omega_m * (1+zs_LCDM)**3 + (1-omega_m))

        zs_aux = np.hstack(zs_ode,zs_LCDM)
        Hs_aux = np.hstack(Hs_ode,Hs_LCDM)

        f = interp1d(zs_aux,Hs_aux)

        zs_final = np.linspace(final_z,initial_z,num_z_points)
        Hs_final = f(zs_final)

    elif model=='HS':
        # Calculate the IC, eta and the parameters of the ODE.
        initial_cond = calculate_initial_conditions(physical_params, zi=initial_z)

        h = H0/100
        R_HS = (omega_m * h**2)/(0.13*8315**2) # units of Mpc**(-2)
        eta = c_light_km * np.sqrt(R_HS/6) # units of (km/seg)/Mpc

        zs_int = np.linspace(initial_z, final_z, num_z_points)

        sol = solve_ivp(system_equations, (initial_z,final_z),
            initial_cond, t_eval=zs_int, args=(physical_params, model),
            rtol=rtol, atol=atol, method=method)

        assert len(sol.t)==num_z_points, 'Something is wrong with the integration!'
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


def Hubble_th(physical_params, *args, b_crit=0.15, all_analytic=False,
                epsilon=10**(-10), n=1, num_z_points=int(10**5),
                z_min=0, z_max=10, model='HS', **kwargs):

    '''
    Calculates the Hubble parameter as a function of redshift for different cosmological models,
    given physical parameters such as the matter density, curvature, and Hubble constant.

    Args:
        physical_params: A tuple of three physical parameters in the order (matter density, curvature, Hubble constant).
        model: A string that specifies the cosmological model to use. Valid options are 'LCDM' (Lambda-CDM),
            'EXP' (exponential model), 'HS' (Hu-Sawicki model), and 'ST' (Starobinsky model).
        b_crit: A critical value for the distortion parameter use in HS and ST models.
        all_analytic: A boolean flag that specifies whether to use an analytic approximation for the Hubble parameter
            or numerical integration.
        epsilon: A tune parameter that is used to calculate b_crit in the exponential model.
        n: An integer that specifies which of the two possible Taylor series approximations to use for the Hubble
            parameter in the power-law growth models (HS or ST).
        num_z_points: An integer that specifies the number of redshifts at which to compute the Hubble parameter.
        z_min: The minimum redshift value to consider.
        z_max: The maximum redshift value to consider.

    Returns:
        A tuple of two NumPy arrays containing the redshifts and the corresponding Hubble parameters.
    '''
    
    omega_m, b, H0 = physical_params
    
    if model == 'LCDM':
        zs = np.linspace(z_min, z_max, num_z_points)
        Hs = H_LCDM(zs, omega_m, H0)

    elif model == 'EXP':
        log_eps_inv = -np.log10(epsilon)
        b_crit = (4 + omega_m/(1-omega_m)) / log_eps_inv
        if (b <= b_crit) or (all_analytic==True): # Analytic approximation
            zs = np.linspace(z_min, z_max, num_z_points)
            Hs = H_LCDM(zs, omega_m, H0)
        else:
            zs, Hs = integrator([omega_m, b, H0], *args, initial_z=z_max, final_z=z_min, **kwargs)

    else: # HS or ST
        if (b <= b_crit) or (all_analytic==True): # Analytic approximation
            zs = np.linspace(z_min, z_max, num_z_points)
            Hs = Taylor_HS(zs, omega_m, b, H0) if n == 1 else Taylor_ST(zs, omega_m, b, H0)
        else:
            zs, Hs = integrator([omega_m, b, H0], initial_z=z_max, final_z=z_min, **kwargs)     
    
    return zs, Hs   

#%%   
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    def plot_hubble_diagram(model_name, physical_params,hubble_th=True):
        """
        Plots the Hubble diagram for a given cosmological model and physical parameters.

        Args:
            model_name (str): Name of the cosmological model to use.
            physical_params (tuple): A tuple of three physical parameters in the order (matter density, curvature, Hubble constant).
            hubble_th (bool, optional): Whether to use the Hubble function obtained from theory or numerical integration. Default is True.

        Returns:
            None. The plot is displayed in the console.

        """

        # Integrate (or evaluate) Hubble function        
        redshifts, hubble_values = Hubble_th(physical_params, model=model_name) if hubble_th else \
                                   integrator(physical_params, model=model_name)
        # Plot Hubble function
        plt.plot(redshifts, hubble_values, '.', label=model_name)


    # Set physical parameters
    omega_m = 0.3
    b = 0.1
    H_0 = 73
    physical_params_hs = np.array([omega_m, b, H_0])
    physical_params_exp = np.array([omega_m, 10, H_0])

    # Plot Hubble diagrams for different models
    plt.figure()
    for model_name, physical_params in [('HS', physical_params_hs), ('EXP', physical_params_exp)]:
        plot_hubble_diagram(model_name, physical_params)
    
    #Plot LCDM Hubble parameter
    redshift_LCDM = np.linspace(0,10,int(10**5))
    plt.plot(redshift_LCDM, H_LCDM(redshift_LCDM,omega_m,H_0),'k--',label=r'$\rm \Lambda CDM$') 
    
    # Format plot
    plt.title('Hubble parameter for $f(R)$ models')
    plt.xlabel('z')
    plt.ylabel(r'H(z) $\rm [(km/seg)/Mpc]$')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()