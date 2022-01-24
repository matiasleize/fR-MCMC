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
path_datos_global = os.path.dirname(path_git)

# Obs: Para importar paquetes la sintaxis de cambio de path es esta
os.chdir(path_git); os.sys.path.append('./Software/')
from utils.sampleo import MCMC_sampler
from utils.data import leer_data_pantheon, leer_data_cronometros, leer_data_BAO, leer_data_AGN
from utils.alternativos import log_likelihood
from utils.parametros_derivados import parametros_derivados
from config import cfg as config
os.chdir(path_git); os.sys.path.append('./Software/plotting/')
import analysis

os.chdir(path_git + '/Software/mcmc/')
def run():
    output_dir = config.OUTPUT_DIR
    model = config.MODEL
    params_fijos = config.FIXED_PARAMS # Fixed parameters
    index = config.LOG_LIKELIHOOD_INDEX
    num_params = int(str(index)[0])
    all_analytic = config.ALL_ANALYTIC

    witness_file = 'witness_' + str(config.WITNESS_NUM) + '.txt'
    
    bnds = config.BOUNDS
    [M_min, M_max] = config.M_PRIOR
    [omega_m_min, omega_m_max] = config.OMEGA_M_PRIOR
    if model != 'LCDM':
        [b_min, b_max] = config.B_PRIOR
    [H0_min, H0_max] = config.H0_PRIOR

    #%% Import cosmological data
    path_data = path_git + '/Software/source/'
    datasets = []

    # Supernovae type IA
    if config.USE_SN == True:
        os.chdir(path_data + 'Pantheon/')
        ds_SN = leer_data_pantheon('lcparam_full_long_zhel.txt')
        datasets.append('_SN')
    else:
        ds_SN = None

    # Cosmic Chronometers
    if config.USE_CC == True:
        os.chdir(path_data + 'CC/')
        ds_CC = leer_data_cronometros('datos_cronometros.txt')
        datasets.append('_CC')
    else:
        ds_CC = None

    # BAO
    if config.USE_BAO == True:    
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
    if config.USE_AGN == True:
        os.chdir(path_data + 'AGN/')
        ds_AGN = leer_data_AGN('table3.dat')
        datasets.append('_AGN')
    else:
        ds_AGN = None

    # Riess H0
    if config.USE_H0 == True:
        H0_Riess = config.USE_H0
        datasets.append('_H0')
    else:
        H0_Riess = False

    datasets = str(''.join(datasets))

    # Define the log-likelihood distribution
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

    nll = lambda theta: -ll(theta) # negative log likelihood

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
        elif index == 23:
            M, omega_m = theta
            if (M_min < M < M_max and omega_m_min < omega_m < omega_m_max):
                return 0.0

        elif index == 1:
            omega_m = theta
            if omega_m_min < omega_m < omega_m_max:
                return 0.0
        return -np.inf
    
    # Define the posterior distribution
    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp): # Maybe this condition is not necessary..
            return -np.inf
        return lp + ll(theta)


    filename = 'sample_' + model + datasets + '_' + str(num_params) + 'params'
    output_directory = path_datos_global + output_dir + filename

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    filename_ml = 'maximun_likelihood' + '_' + model + datasets + '_' + str(num_params) + 'params'

    
    # If exist, import mean values of the free parameters. If not, calculate, save and load calculation.
    os.chdir(output_directory)
    if (os.path.exists(filename_ml + '.npz') == True):
        with np.load(filename_ml + '.npz') as data:
            sol = data['sol']
    else:
        print('Calculating maximum likelihood parameters ..')
        initial = np.array(config.GUEST)
        soln = minimize(nll, initial, options = {'eps': 0.01}, bounds = bnds)

        np.savez(filename_ml, sol=soln.x)
        with np.load(filename_ml + '.npz') as data:
            sol = data['sol']
    print('Maximun likelihood corresponds to the parameters: {}'.format(sol))


    # Define initial values of each chain using the minimun 
    # values of the chisquare.
    pos = sol * (1 +  0.01 * np.random.randn(config.NUM_WALKERS, num_params))

    
    filename_h5 = filename + '.h5'

    MCMC_sampler(log_probability,pos, 
                filename = filename_h5,
                witness_file = witness_file,
                witness_freq = config.WITNESS_FREQ,
                max_samples = config.MAX_SAMPLES,
                save_path = output_directory)

    # If it corresponds, derive physical parameters
    if model != 'LCDM':
        os.chdir(output_directory)
 
        textfile_witness = open(witness_file,'a')
        textfile_witness.write('\n Initializing derivation of parameters..')
        textfile_witness.close()

        reader = emcee.backends.HDFBackend(filename_h5)
        nwalkers, ndim = reader.shape #Number of walkers and parameters

        # Hardcode definition of burnin and thin
        samples = reader.get_chain()
        burnin= int(0.2*len(samples[:,0])) # Burnin 20% 
        thin = 1

        samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

        textfile_witness = open(witness_file,'a')
        textfile_witness.write('\n Number of effective steps: {}'.format(len(samples))) 
        textfile_witness.write(('\n Estimated time: {} min'.format(len(samples)/60)))
        textfile_witness.close()

        new_samples = parametros_derivados(reader,discard=burnin,thin=thin,model=model)
        np.savez(filename+'_deriv', new_samples=new_samples)

        textfile_witness = open(witness_file,'a')
        textfile_witness.write('\n Done!')
        textfile_witness.close()

        # Print the output
        with np.load(filename+'_deriv.npz') as data:
            ns = data['new_samples']
        

    # Plot the results
    analysis.run(filename)


if __name__ == "__main__":
    run()
