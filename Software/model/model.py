'''
Run the MCMC analyses and the calculation of the physical parameters \Omega_{m} and H_{0}.
For now it only runs for 4 parameters on HS and EXP models and for 3 parameters on the LCDM model.

Parameters order on this file: Mabs,omega_m,b,H_0,n
'''

import numpy as np; #np.random.seed(42)
import emcee
import yaml
from scipy.optimize import minimize

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir

os.chdir(path_git); os.sys.path.append('./Software/')
from Funcionales.funciones_sampleo import MCMC_sampler
from Funcionales.funciones_data import leer_data_pantheon, leer_data_cronometros, leer_data_BAO, leer_data_AGN
from Funcionales.funciones_alternativos import log_likelihood
from Funcionales.funciones_parametros_derivados import parametros_derivados
from config import cfg # Still is not implemented in this script

os.chdir(path_git + '/Software/model/')
import click #Allows to run with different configuration files
@click.command()
@click.argument("config_path")
def run(config_path='./config.yml'):
    # Read in yaml file
    with open(config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    #print(config)

    model = config['MODEL']
    params_fijos = config['FIXED_PARAMS'] #Fixed parameters
    index = config['LOG_LIKELIHOOD_INDEX']
    num_params = int(str(index)[0])
    all_analytic = config['ALL_ANALYTIC']
    
    bnds = config['BOUNDS']
    [M_min, M_max] = config['M_PRIOR']
    [omega_m_min, omega_m_max] = config['OMEGA_M_PRIOR']
    if model != 'LCDM':
        [b_min, b_max] = config['B_PRIOR']
    [H0_min, H0_max] = config['H0_PRIOR']

    #%% Import cosmological data
    path_data = path_git + '/Software/source/'
    datasets = []

    # Supernovae type IA
    if config['USE_SN'] == True:
        os.chdir(path_data + 'Pantheon/')
        ds_SN = leer_data_pantheon('lcparam_full_long_zhel.txt')
        datasets.append('_SN')
    else:
        ds_SN = None

    # Cosmic Chronometers
    if config['USE_CC'] == True:
        os.chdir(path_data + 'CC/')
        ds_CC = leer_data_cronometros('datos_cronometros.txt')
        datasets.append('_CC')
    else:
        ds_CC = None

    # BAO
    if config['USE_BAO'] == True:    
        os.chdir(path_data + 'BAO/')
        ds_BAO = []
        archivos_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                        'datos_BAO_dv.txt','datos_BAO_H.txt']
        for i in range(5):
            aux = leer_data_BAO(archivos_BAO[i])
            ds_BAO.append(aux)
        datasets.append('_BAO')
    else:
        ds_BAO = None


    # AGN
    if config['USE_AGN'] == True:
        os.chdir(path_data + 'AGN/')
        ds_AGN = leer_data_AGN('table3.dat')
        datasets.append('_AGN')
    else:
        ds_AGN = None

    # Riess H0
    if config['USE_H0'] == True:
        H0_Riess = config['USE_H0']
        datasets.append('_H0')
    else:
        H0_Riess = False

    datasets = str(''.join(datasets))

    #%% Define the log-likelihood distribution
    ll = lambda theta: log_likelihood(theta, params_fijos, 
                                        index=index,
                                        dataset_SN = ds_SN,
                                        dataset_CC = ds_CC,
                                        dataset_BAO = ds_BAO,
                                        dataset_AGN = ds_AGN,
                                        H0_Riess = H0_Riess,
                                        model = model,
                                        all_analytic = all_analytic
                                        )

    nll = lambda theta: -ll(theta) #negative log likelihood

    # Define the prior distribution
    def log_prior(theta):
        if index == 4:
            M, omega_m, b, H0 = theta
            if (M_min < M < M_max and omega_m_min < omega_m < omega_m_max and b_min < b < b_max and H0_min < H0 < H0_max):
                return 0.0
        elif index == 31:
            omega_m, b, H0 = theta
            if (omega_m_min < omega_m < omega_m_max and b_min < b < b_max and H0_min < H0 < H0_max):
                return 0.0
        elif index == 32:
            M, omega_m, H0 = theta
            if (M_min < M < M_max and omega_m_min < omega_m < omega_m_max and H0_min < H0 < H0_max):
                return 0.0
        elif index == 33:
            M, omega_m, b = theta
            if (M_min < M < M_max and omega_m_min < omega_m < omega_m_max and b_min < b < b_max):
                return 0.0
        elif index == 21:
            omega_m, b = theta
            if (omega_m_min < omega_m < omega_m_max and b_min < b < b_max):
                return 0.0
        elif index == 22:
            omega_m, H0 = theta
            if (omega_m_min < omega_m < omega_m_max and H0_min < H0 < H0_max):
                return 0.0
        elif index == 1:
            omega_m = theta
            if omega_m_min < omega_m < omega_m_max:
                return 0.0
        return -np.inf
    
    # Define the posterior distribution
    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp): #Maybe this condition is not necessary..
            return -np.inf
        return lp + ll(theta)

    filename_mv = 'valores_medios_' + model + datasets + '_' + str(num_params) + 'params' + '_borrarr'

    # If exist, import mean values of the free parameters. If not, calculate, save and load calculation.
    os.chdir(path_git+'/Software/Resultados_simulaciones/')
    if (os.path.exists(filename_mv + '.npz') == True):
        with np.load(filename_mv + '.npz') as data:
            sol = data['sol']
    else:
        print('No estan calculados los valores medios de los parametros')
        initial = np.array(config['GUEST'])
        soln = minimize(nll, initial, options = {'eps': 0.01}, bounds = bnds)

        os.chdir(path_git + '/Software/Resultados_simulaciones')
        np.savez(filename_mv, sol=soln.x)
        with np.load(filename_mv + '.npz') as data:
            sol = data['sol']
    print(sol)


    #%% Define initial values of each chain using the minimun 
    # values of the chisquare.
    pos = sol * (1 +  0.01 * np.random.randn(config['NUM_WALKERS'], num_params))
    
    filename = 'sample_' + model + datasets + '_' + str(num_params) + 'params' + '_borrarr' 
    filename_h5 = filename + '.h5'
    
    MCMC_sampler(log_probability,pos,
                filename = filename_h5,
                witness_file = 'witness_' + str(config['WITNESS_NUM']) + '.txt',
                witness_freq = config['WITNESS_FREQ'],
                max_samples = config['MAX_SAMPLES'])

    #%% If it corresponds, derive physical parameters
    #Fill in here:
    if model != 'LCDM':
        root_directory=path_datos_global+'/Resultados_cadenas/'
        os.chdir(root_directory)
        reader = emcee.backends.HDFBackend(filename_h5)
        nwalkers, ndim = reader.shape #Number of walkers and parameters

        #%% Define burnin and thin using harcoding: thin=1 and burnin 0.2*num_steps
        
        samples = reader.get_chain()
        burnin= int(0.2*len(samples[:,0])) #Burnin del 20%
        thin = 1
        #%%
        samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
        print(len(samples)) #Number of effective steps
        print('Tiempo estimado:{} min'.format(len(samples)/60))
        new_samples = parametros_derivados(reader,discard=burnin,thin=thin,model=model)
        np.savez(filename+'_deriv', new_samples=new_samples)
        #%% Print the output
        with np.load(filename+'_deriv.npz') as data:
            ns = data['new_samples']


if __name__ == "__main__":
    run()