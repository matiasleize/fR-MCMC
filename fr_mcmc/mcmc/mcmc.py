'''
Run MCMC analyses and calculations of the physical parameters of the models.

Parameter order in this file: Mabs,omega_m,b,H_0,n
'''

import os
# One thread per process for the linear algebra (BLAS). With several threads per process, and
# with several processes (N_PROCESSES), the cores are oversubscribed and it is much slower.
# Must be set before importing numpy.
for _var in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(_var, '1')

import multiprocessing
import numpy as np; #np.random.seed(42)
import emcee
from scipy.optimize import minimize

import git

# Get the root directory of the Git repository
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_global = os.path.dirname(path_git)

# Add necessary paths to sys.path
os.sys.path.extend([
    os.path.join(path_git, 'fr_mcmc'),
    os.path.join(path_git, 'fr_mcmc', 'plotting')
])
# Get the root directory of the Git repository
from utils.sampling import MCMC_sampler
from utils.data import (
    read_data_pantheon_plus_shoes, read_data_pantheon_plus, read_data_pantheon,
    read_data_chronometers, read_data_BAO, read_data_DESI, read_data_BAO_full, read_data_AGN
)
from utils.QSO import read_data_QSO
from utils.CC_cov import read_data_CC_cov
from utils.likelihood_ext import log_likelihood
from utils.derived_parameters import derived_parameters

from config import cfg as config
import analysis

# Change to the mcmc directory
os.chdir(os.path.join(path_git, 'fr_mcmc', 'mcmc'))

# The pool of processes pickles the function it evaluates, and log_probability (defined inside
# run) can not be pickled. It is stored in this global before creating the pool ('fork' start
# method, the workers inherit it) and evaluated through a top level function.
_log_probability = None

def _log_probability_global(theta):
    return _log_probability(theta)

def run():
    output_dir = config.OUTPUT_DIR
    model = config.MODEL
    fixed_params = config.FIXED_PARAMS # Fixed parameters
    index = config.LOG_LIKELIHOOD_INDEX
    num_params = int(str(index)[0])
    all_analytic = config.ALL_ANALYTIC
    use_c = config.USE_C == True # Compute H(z) in C (utils/solve_sys_c)
    rd_grid = config.RD_GRID == True # r_d from the CLASS grid instead of CLASS at each step
    rd_fixed = config.RD_FIXED # r_d (Mpc) given directly; empty: computed (overrides RD_GRID)
    wb_from_param = config.WB_FROM_PARAM == True # omega_b = bao_param (sampled) for r_d

    witness_file = f'witness_{config.WITNESS_NUM}.txt'
    
    bnds = config.BOUNDS
    if model == 'LCDM':
        [omega_m_min, omega_m_max] = config.OMEGA_M_PRIOR
        [H0_min, H0_max] = config.H0_PRIOR

    elif model in ['HS', 'ST', 'EXP']:
        [omega_m_min, omega_m_max] = config.OMEGA_M_PRIOR
        [b_min, b_max] = config.B_PRIOR
        [H0_min, H0_max] = config.H0_PRIOR


    if config.USE_BAO_LEGACY_1 or config.USE_DESI_DR2 or config.USE_BAO_LEGACY_2:
        [bao_param_min, bao_param_max] = config.BAO_PARAM_PRIOR

    if config.USE_SN or config.USE_PPLUS or config.USE_PPLUS_SHOES:
        [M_min, M_max] = config.M_PRIOR

    #%% Import cosmological data
    path_data = os.path.join(path_git, 'fr_mcmc', 'source')
    
    datasets = []

    # Pantheon Plus + Shoes
    if config.USE_PPLUS_SHOES == True:
        os.chdir(os.path.join(path_data, 'Pantheon_plus_shoes'))

        ds_SN_plus_shoes = read_data_pantheon_plus_shoes('Pantheon+SH0ES.dat',
                                    'Pantheon+SH0ES_STAT+SYS.cov')
        datasets.append('_PPS')
    else:
        ds_SN_plus_shoes = None

    # Pantheon Plus
    if config.USE_PPLUS == True:
        os.chdir(os.path.join(path_data, 'Pantheon_plus_shoes'))

        ds_SN_plus = read_data_pantheon_plus('Pantheon+SH0ES.dat',
                                'Pantheon+SH0ES_STAT+SYS.cov')
        datasets.append('_PP')
    else:
        ds_SN_plus = None

    # Supernovae type IA
    if config.USE_SN == True:
        os.chdir(os.path.join(path_data, 'Pantheon'))

        ds_SN = read_data_pantheon('lcparam_full_long_zhel.txt')
        datasets.append('_SN')
    else:
        ds_SN = None

    # Cosmic Chronometers
    if config.USE_CC_LEGACY == True:
        # CC_legacy (30 puntos, diagonal): la tabla de Leizerovich et al. (2022), PRD 105, 103526.
        # La compilacion nueva de 33 puntos (source/CC/) se usa con la covarianza de Moresco,
        # via CC_cov.read_data_CC_cov; en diagonal no corresponde a nada publicado.
        os.chdir(os.path.join(path_data, 'CC_legacy'))

        ds_CC = read_data_chronometers('chronometers_data.txt')
        datasets.append('_CC_legacy')
    else:
        ds_CC = None

    # Cosmic Chronometers with the covariance of Moresco et al. (2020):
    # la tabla nueva de 33 puntos (source/CC/) mas la covarianza (source/CC_cov/).
    if config.USE_CC == True:
        ds_CC_cov = read_data_CC_cov(os.path.join(path_data, 'CC', 'chronometers_data.txt'),
                                     os.path.join(path_data, 'CC_cov', 'HzTable_MM_BC03.dat'),
                                     os.path.join(path_data, 'CC_cov', 'data_MM20.dat'))
        datasets.append('_CC')
    else:
        ds_CC_cov = None

    # Quasars of Benetti et al. (2025). k_QSO empty: k marginalized analytically.
    if config.USE_QSO == True:
        os.chdir(os.path.join(path_data, 'QSO'))

        ds_QSO = read_data_QSO('qso_benetti2025.txt')
        datasets.append('_QSO')
    else:
        ds_QSO = None

    # BAO
    if config.USE_BAO_LEGACY_1 == True:    
        os.chdir(os.path.join(path_data, 'BAO_legacy_1'))

        ds_BAO = []
        files_BAO = ['BAO_data_da.txt','BAO_data_dh.txt','BAO_data_dm.txt',
                        'BAO_data_dv.txt','BAO_data_H.txt']
        for i in range(5):
            aux = read_data_BAO(files_BAO[i])
            ds_BAO.append(aux)
        datasets.append('_BAO_legacy_1')
    else:
        ds_BAO = None

    # DESI
    if config.USE_DESI_DR2 == True:    
        os.chdir(os.path.join(path_data, 'DESI'))

        ds_DESI = read_data_DESI('DESI_DR2_dm_dh.txt','DESI_DR2_dv.txt') # DR1: 'DESI_data_dm_dh.txt','DESI_data_dv.txt'
        datasets.append('_DESI_DR2')
    else:
        ds_DESI = None

    # BAO full
    if config.USE_BAO_LEGACY_2 == True:    
        os.chdir(os.path.join(path_data, 'BAO_legacy_2'))
        ds_BAO_full = read_data_BAO_full('BAO_full_1.csv','BAO_full_2.csv')
        datasets.append('_BAO_legacy_2')
    else:
        ds_BAO_full = None

    # AGN
    if config.USE_AGN == True:
        os.chdir(os.path.join(path_data, 'AGN'))
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
    #if config.OMEGA_M_ASTRO_PRIOR == True: #Omega_m gaussian prior
    #    datasets.append('_PROA')
    #if config.M_ABS_CM_PRIOR == True: #M_abs Camarena & Marra prior
    #    datasets.append('_PRCM')

    datasets = str(''.join(datasets))

    # Define the log-likelihood distribution
    ll = lambda theta: log_likelihood(theta, fixed_params, 
                                        index=index,
                                        dataset_SN_plus_shoes = ds_SN_plus_shoes,
                                        dataset_SN_plus = ds_SN_plus,
                                        dataset_SN = ds_SN,
                                        dataset_CC = ds_CC,
                                        dataset_CC_cov = ds_CC_cov,
                                        dataset_QSO = ds_QSO,
                                        k_QSO = config.K_QSO,
                                        dataset_BAO = ds_BAO,
                                        dataset_DESI = ds_DESI,
                                        dataset_BAO_full = ds_BAO_full,
                                        dataset_AGN = ds_AGN,
                                        H0_Riess = H0_Riess,
                                        model = model,
                                        all_analytic = all_analytic,
                                        use_c = use_c,
                                        rd_grid = rd_grid,
                                        rd_fixed = rd_fixed,
                                        wb_from_param = wb_from_param
                                        )

    nll = lambda theta: -ll(theta) # negative log likelihood

    # Define the prior distribution
    def log_prior(theta, model):
        if model == 'LCDM':
            if index == 4:
                M, bao_param, omega_m, H0 = theta
                if (M_min < M < M_max and bao_param_min < bao_param < bao_param_max and omega_m_min < omega_m < omega_m_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 31:
                M, omega_m, H0 = theta
                if (M_min < M < M_max and omega_m_min < omega_m < omega_m_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 32:
                bao_param, omega_m, H0 = theta
                if (bao_param_min < bao_param < bao_param_max and omega_m_min < omega_m < omega_m_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 21:
                omega_m, H0 = theta
                if (omega_m_min < omega_m < omega_m_max and H0_min < H0 < H0_max):
                    return 0.0

        elif model in ['HS', 'ST', 'EXP']:
            if index == 5:
                M, bao_param, omega_m, b, H0 = theta
                if (M_min < M < M_max and bao_param_min < bao_param < bao_param_max and omega_m_min < omega_m < omega_m_max and b_min < b < b_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 41:
                M, bao_param, b, H0 = theta
                if (M_min < M < M_max and bao_param_min < bao_param < bao_param_max and b_min < b < b_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 42:
                M, omega_m, b, H0 = theta
                if (M_min < M < M_max and omega_m_min < omega_m < omega_m_max and b_min < b < b_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 43:
                bao_param, omega_m, b, H0 = theta
                if (bao_param_min < bao_param < bao_param_max and omega_m_min < omega_m < omega_m_max and b_min < b < b_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 31:
                M, b, H0 = theta
                if (M_min < M < M_max and b_min < b < b_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 32:
                bao_param, b, H0 = theta
                if (bao_param_min < bao_param < bao_param_max and b_min < b < b_max and H0_min < H0 < H0_max):
                    return 0.0
            elif index == 33:
                omega_m, b, H0 = theta
                if (omega_m_min < omega_m < omega_m_max and b_min < b < b_max and H0_min < H0 < H0_max):
                    return 0.0
        return -np.inf
    
    # Define the posterior distribution
    def log_probability(theta):
        lp = log_prior(theta, model)
        if not np.isfinite(lp): # Maybe this condition is not necessary..
            return -np.inf
        return lp + ll(theta)

    filename = f'sample_{model}{datasets}_{num_params}params'
    output_directory = path_global + output_dir + filename

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    filename_ml = f'maximun_likelihood_{model}{datasets}_{num_params}params'
    
    # If exist, import mean values of the free parameters. If not, calculate, save and load calculation.
    os.chdir(output_directory)
    if (os.path.exists(filename_ml + '.npz') == True):
        with np.load(filename_ml + '.npz') as data:
            sol = data['sol']
    else:
        print('Calculating maximum likelihood parameters ..')
        initial = np.array(config.GUEST)
        # Nelder-Mead (derivative free). L-BFGS-B with finite differences (eps=0.01) stopped
        # at the initial point (ABNORMAL_TERMINATION_IN_LNSRCH).
        soln = minimize(nll, initial, method='Nelder-Mead', bounds = bnds,
                        options = {'xatol': 1e-4, 'fatol': 1e-3, 'maxiter': 5000})
        np.savez(filename_ml, sol=soln.x)
        with np.load(filename_ml + '.npz') as data:
            sol = data['sol']
    print(f'Maximun likelihood corresponds to the parameters: {sol}')

    # Define initial values of each chain using the minimun 
    # values of the chi-squared: 1% scatter, plus a small additive scatter (for parameters
    # close to 0, e.g. b). Walkers outside of the prior are drawn again.
    width = np.array([hi - lo for lo, hi in bnds])
    pos = np.zeros((config.NUM_WALKERS, num_params))
    for i in range(config.NUM_WALKERS):
        for _ in range(10000):
            pos[i] = sol * (1 + 0.01 * np.random.randn(num_params)) + 1e-3 * width * np.random.randn(num_params)
            if np.isfinite(log_prior(pos[i], model)):
                break
        else:
            raise ValueError('Could not place the initial walkers inside the prior')

    filename_h5 = filename + '.h5'

    # Parallel evaluation of the walkers
    n_proc = config.N_PROCESSES or min(os.cpu_count(), config.NUM_WALKERS)
    global _log_probability
    _log_probability = log_probability
    pool = multiprocessing.get_context('fork').Pool(n_proc) if n_proc > 1 else None
    try:
        MCMC_sampler(_log_probability_global if pool else log_probability, pos,
                    filename = filename_h5,
                    witness_file = witness_file,
                    witness_freq = config.WITNESS_FREQ,
                    max_samples = config.MAX_SAMPLES,
                    save_path = output_directory,
                    conv_freq = config.CONV_FREQ or 100,
                    pool = pool,
                    moves = config.MOVES if 'MOVES' in config else None)
    finally:
        if pool:
            pool.close()
            pool.join()

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

        #new_samples = derived_parameters(reader,discard=burnin,thin=thin,model=model)
        #np.savez(filename+'_deriv', new_samples=new_samples)

        textfile_witness = open(witness_file,'a')
        textfile_witness.write('\n Done!')
        textfile_witness.close()

        # Print the output
        #with np.load(filename+'_deriv.npz') as data:
        #    ns = data['new_samples']        

    # Plot the results
    analysis.run(filename)


if __name__ == "__main__":
    run()
