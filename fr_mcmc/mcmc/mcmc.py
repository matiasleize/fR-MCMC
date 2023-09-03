'''
Run MCMC analyses and calculations of the physical parameters of the models.

Parameter order in this file: Mabs,omega_m,b,H_0,n
'''

import numpy as np; #np.random.seed(42)
import emcee
import yaml
from scipy.optimize import minimize

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_global = os.path.dirname(path_git)

# Obs: To import packages this is the sintaxis to change paths:
os.chdir(path_git); os.sys.path.append('./fr_mcmc/')
from utils.sampling import MCMC_sampler
from utils.derived_parameters import derived_parameters
from utils.data import read_data_pantheon_plus_shoes, read_data_pantheon_plus, read_data_pantheon,\
                       read_data_chronometers, read_data_BAO, read_data_AGN
from utils.chi_square import log_likelihood
from utils.priors import log_prior # Must be the last import before config to avoid issues!
from config import cfg as config
os.chdir(path_git); os.sys.path.append('./fr_mcmc/plotting/')
import analysis

os.chdir(path_git + '/fr_mcmc/mcmc/')
def run():
    output_dir = config.OUTPUT_DIR
    model = config.MODEL
    fixed_params = config.FIXED_PARAMS # Fixed parameters
    index = config.LOG_LIKELIHOOD_INDEX
    num_params = int(str(index)[0])
    all_analytic = config.ALL_ANALYTIC
    bnds = config.BOUNDS

    witness_file = 'witness_' + str(config.WITNESS_NUM) + '.txt'

    #%% Import cosmological data
    path_data = path_git + '/fr_mcmc/source/'
    datasets = []

    # Pantheon Plus + Shoes
    if config.USE_PPLUS_SHOES == True:
        os.chdir(path_data + 'Pantheon_plus_shoes/')

        ds_SN_plus_shoes = read_data_pantheon_plus_shoes('Pantheon+SH0ES.dat',
                                    'Pantheon+SH0ES_STAT+SYS.cov')
        datasets.append('_PPS')
    else:
        ds_SN_plus_shoes = None

    # Pantheon Plus
    if config.USE_PPLUS == True:
        os.chdir(path_data + 'Pantheon_plus_shoes/')
        ds_SN_plus = read_data_pantheon_plus('Pantheon+SH0ES.dat',
                                'covmat_pantheon_plus_only.npz')        
        datasets.append('_PP')
    else:
        ds_SN_plus = None

    # Supernovae type IA
    if config.USE_SN == True:
        os.chdir(path_data + 'Pantheon/')
        ds_SN = read_data_pantheon('lcparam_full_long_zhel.txt')
        datasets.append('_SN')
    else:
        ds_SN = None

    # Cosmic Chronometers
    if config.USE_CC == True:
        os.chdir(path_data + 'CC/')
        ds_CC = read_data_chronometers('chronometers_data.txt')

        #ds_CC = read_data_chronometers('/home/matias/Documents/Repos/fR-MCMC/notebooks/CC_from_LCDM_8.txt')
        #ds_CC = read_data_chronometers('/home/matias/Documents/Repos/fR-MCMC/notebooks/CC_from_HS_8.txt')
        #ds_CC = read_data_chronometers('/home/matias/Documents/Repos/fR-MCMC/notebooks/CC_from_LCDM_5.txt')
        #ds_CC = read_data_chronometers('/home/matias/Documents/Repos/fR-MCMC/notebooks/CC_from_HS_5.txt')

        datasets.append('_CC')
    else:
        ds_CC = None

    # BAO
    if config.USE_BAO == True:    
        os.chdir(path_data + 'BAO/')
        ds_BAO = []
        files_BAO = ['BAO_data_da.txt','BAO_data_dh.txt','BAO_data_dm.txt',
                        'BAO_data_dv.txt','BAO_data_H.txt']
        for i in range(5):
            aux = read_data_BAO(files_BAO[i])
            ds_BAO.append(aux)
        datasets.append('_BAO')
    else:
        ds_BAO = None

    # AGN
    if config.USE_AGN == True:
        os.chdir(path_data + 'AGN/')
        ds_AGN = read_data_AGN('table3.dat')
        datasets.append('_AGN')
    else:
        ds_AGN = None

    # Riess H0
    if config.USE_H0 == True:
        H0_Riess = config.USE_H0
        datasets.append('_H0')
    else:
        H0_Riess = False

    #Related to priors
    if config.OMEGA_M_ASTRO_PRIOR == True: #Omega_m gaussian prior
        datasets.append('_PROA')
    if config.M_ABS_CM_PRIOR == True: #M_abs Camarena & Marra prior
        datasets.append('_PRCM')

    # Define the log-likelihood distribution
    ll = lambda theta: log_likelihood(theta, fixed_params, 
                                        index=index,
                                        dataset_SN_plus_shoes = ds_SN_plus_shoes,
                                        dataset_SN_plus = ds_SN_plus,
                                        dataset_SN = ds_SN,
                                        dataset_CC = ds_CC,
                                        dataset_BAO = ds_BAO,
                                        dataset_AGN = ds_AGN,
                                        H0_Riess = H0_Riess,
                                        model = model,
                                        all_analytic = all_analytic
                                        )
    nll = lambda theta: -ll(theta) # negative log likelihood

    #Define the prior distribution
    lp = lambda theta: log_prior(theta, index) 
    # Define the posterior distribution
    def log_probability(theta):
        if not np.isfinite(lp(theta)): # Maybe this condition is not necessary..
            return -np.inf
        return lp(theta) + ll(theta)

    datasets = str(''.join(datasets))

    filename = 'sample_' + model + datasets + '_' + str(num_params) + 'params'
    output_directory = path_global + output_dir + filename

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

        new_samples = derived_parameters(reader,discard=burnin,thin=thin,model=model)
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
