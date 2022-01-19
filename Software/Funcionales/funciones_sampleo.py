import numpy as np
import emcee
import time

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_datos_global = os.path.dirname(path_git)

def MCMC_sampler(log_probability, initial_values,
                filename = "default.h5",
                witness_file = 'witness.txt',
                max_samples = 10000,
                witness_freq = 100,
                tolerance = 0.01):

    nwalkers, ndim = initial_values.shape

    # Set up the backend
    os.chdir(path_datos_global+'/Resultados_cadenas/')
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim) # Don't forget to clear it in case the file already exists
    textfile_witness = open(witness_file,'w+')
    textfile_witness.close()
    #%%
    #Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend)

	#sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend,
	#        moves=[(emcee.moves.DEMove(), 0.4), (emcee.moves.DESnookerMove(), 0.3)
	#        , (emcee.moves.KDEMove(), 0.3)])

    # This will be useful to testing convergence
    old_tau = np.inf
    t1 = time.time()
    # Now we'll sample for up to max_samples steps
    for sample in sampler.sample(initial_values, iterations=max_samples, progress=True):
        # Only check convergence every 'witness_freq' steps
        if sampler.iteration % witness_freq: #'witness_freq' es cada cuanto chequea convergencia
            continue

        os.chdir(path_datos_global+'/Resultados_cadenas/')
        textfile_witness = open(witness_file,'w')
        textfile_witness.write('Número de iteración: {} \t'.format(sampler.iteration))

        t2 = time.time()
        textfile_witness.write('Duración {} minutos y {} segundos'.format(int((t2-t1)/60),
              int((t2-t1) - 60*int((t2-t1)/60))))
        textfile_witness.close()

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration) #100 es el threshold de convergencia
        #También pido que tau se mantenga relativamente constante:
        converged &= np.all((np.abs(old_tau - tau) / tau) < tolerance)
        if converged:
            textfile_witness = open(witness_file,'a')
            textfile_witness.write('Convergió!')
            textfile_witness.close()
            break
        old_tau = tau
