import numpy as np
import emcee

class Cadenas:
	def __init__(self):
			pass
	#def min_chi2(self,)
#		pass
	def cadenas(self,sol,log_probability,filename,witness='witness.txt',num_pasos=10000,
		num_cadenas=12,tol=0.001,check_tol=20):
#		pass

		pos = sol * (1 +  0.1 * np.random.randn(num_cadenas, len(sol)))
		nwalkers, ndim = pos.shape

		# Set up the backend
		os.chdir(path_datos_global+'/Resultados_cadenas/')
		filename = filename
		backend = emcee.backends.HDFBackend(filename)
		backend.reset(nwalkers, ndim) # Don't forget to clear it in case the file already exists
		textfile_witness = open(witness,'w+')
		textfile_witness.close()

		#Initialize the sampler
		sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend,
		        moves=[(emcee.moves.DEMove(), 0.4), (emcee.moves.DESnookerMove(), 0.3)
		        , (emcee.moves.KDEMove(), 0.3)])
		max_n = num_pasos

		# This will be useful to testing convergence
		old_tau = np.inf
		# Now we'll sample for up to max_n steps
		for sample in sampler.sample(pos, iterations=max_n, progress=True):
		    # Only check convergence every 100 steps
		    if sampler.iteration % check_tol: #cada cuanto chequea convergencia
		        continue
		    os.chdir(path_datos_global+'/Resultados_cadenas/')
		    textfile_witness = open(witness,'w')
		    textfile_witness.write('Iteration number: {} \t'.format(sampler.iteration))
		    textfile_witness.write('Tiempo: {}'.format(time.time()))
		    textfile_witness.close()

		    # Compute the autocorrelation time so far
		    # Using tol=0 means that we'll always get an estimate even
		    # if it isn't trustworthy
			tau = sampler.get_autocorr_time(tol=0)

			# Check convergence
			converged_1 = np.all(tau * 100 < sampler.iteration) #100 es el threshold de convergencia
			#También pido que tau se mantenga relativamente constante:
		    converged_2 = np.all((np.abs(old_tau - tau)/tau) < tol)
		    if (converged_1 and converged_2):
		        textfile_witness = open(witness,'a')
		        textfile_witness.write('Convergió!')
		        textfile_witness.close()
		        break
		    old_tau = tau