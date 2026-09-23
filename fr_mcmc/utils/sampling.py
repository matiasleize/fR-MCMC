'''
Define the function related with the Markov Chain Monter Carlo (MCMC) process.
'''

import numpy as np
import emcee
import time

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_global = os.path.dirname(path_git)

def MCMC_sampler(log_probability, initial_values,
                filename = "default.h5",
                witness_file = 'witness.txt',
                max_samples = 10000,
                witness_freq = 100,
                tolerance = 0.01,
                save_path = None,
                conv_freq = 100,
                tau_factor = 100,
                pool = None,
                moves = None):
	'''
	log_probability: logarithm of the posterior distribution that will be sampled.

	initial_values: object that contains the initial value of the parameters to sample

	filename: name of the h5 file that contains the chains information.

	witness_file: name of the witness file.

	max_samples: maximun number of sample, if the chains not converge.

	witness_freq: frequency use to print the state of the calculation in the witness file.

	tolerance: tolerance parameter on the convergence method.

	save_path: directory in which the outputs are stored. Change this atribute on the

	configuration file is recommended .

	conv_freq: frequency (in iterations) in which the convergence is checked. The chains are
	converged when N > tau_factor * tau and tau changed less than 'tolerance' since the
	previous check (as in the emcee documentation, which checks every 100 iterations).

	tau_factor: number of autocorrelation times required (default 100).

	pool: optional pool (e.g. multiprocessing.Pool) to evaluate the walkers in parallel.

	moves: proposal of emcee. None or 'stretch': stretch move (emcee default). 'DE':
	differential evolution, 0.8 DEMove + 0.2 DESnookerMove (recommended in the emcee docs
	for posteriors with degeneracies or boundaries). A list of emcee moves is passed as is.
	'''

	nwalkers, ndim = initial_values.shape

	# Set up the backend
	os.chdir(save_path)
	backend = emcee.backends.HDFBackend(filename)
	backend.reset(nwalkers, ndim) # Don't forget to clear it in case the file already exists
	textfile_witness = open(witness_file,'w+')
	textfile_witness.close()
	#%%
	#Initialize the sampler
	if moves is None or moves == 'stretch':
		moves = None
	elif moves == 'DE':
		moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend, pool=pool,
									moves=moves)

	#sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend,
	#        moves=[(emcee.moves.DEMove(), 0.4), (emcee.moves.DESnookerMove(), 0.3)
	#        , (emcee.moves.KDEMove(), 0.3)])

	# This will be useful to testing convergence
	old_tau = np.inf
	tau = np.full(ndim, np.nan)
	converged = False
	t1 = time.time()
	# Now we'll sample for up to max_samples steps
	for sample in sampler.sample(initial_values, iterations=max_samples, progress=True):
		it = sampler.iteration

		if it % witness_freq == 0:
			os.chdir(save_path)
			textfile_witness = open(witness_file,'w')
			textfile_witness.write('Iteration number: {} \t'.format(it))
			t2 = time.time()
			textfile_witness.write('Duration: {} minutes and {} seconds'.format(int((t2-t1)/60),
				  int((t2-t1) - 60*int((t2-t1)/60))))
			textfile_witness.write('\n Last tau: {}'.format(np.round(tau, 1)))
			textfile_witness.close()

		# Only check convergence every 'conv_freq' steps
		if it % conv_freq:
			continue

		# Compute the autocorrelation time so far
		# Using tol=0 means that we'll always get an estimate even
		# if it isn't trustworthy
		tau = sampler.get_autocorr_time(tol=0)

		# Check convergence
		converged = np.all(tau * tau_factor < it) #tau_factor is the convergence threshold
		#Also, ask \tau to stay relatively constant:
		converged &= np.all((np.abs(old_tau - tau) / tau) < tolerance)
		if converged:
			break
		old_tau = tau

	# Final summary: autocorrelation time and effective number of samples
	# (after discarding 20% as burn-in, as in mcmc.py)
	it = sampler.iteration
	tau = sampler.get_autocorr_time(tol=0)
	n_eff = nwalkers * (it - int(0.2 * it)) / tau
	t2 = time.time()
	os.chdir(save_path)
	textfile_witness = open(witness_file,'w')
	textfile_witness.write('Iteration number: {} \t'.format(it))
	textfile_witness.write('Duration: {} minutes and {} seconds'.format(int((t2-t1)/60),
		  int((t2-t1) - 60*int((t2-t1)/60))))
	textfile_witness.write('\n {}'.format('Converged!' if converged else 'NOT converged (max_samples reached)'))
	textfile_witness.write('\n tau: {}'.format(np.round(tau, 1)))
	textfile_witness.write('\n N_eff (after 20% burn-in): {}'.format(np.round(n_eff).astype(int)))
	textfile_witness.write('\n Acceptance fraction: {:.3f}'.format(np.mean(sampler.acceptance_fraction)))
	textfile_witness.close()
	return converged, tau, n_eff
