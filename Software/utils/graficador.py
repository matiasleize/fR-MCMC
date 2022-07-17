import numpy as np
from matplotlib import pyplot as plt
from getdist import plots, MCSamples
import time
import emcee
import corner
import seaborn as sns
import pandas as pd
from IPython.display import display, Math
import arviz as az
from scipy.stats import scoreatpercentile

class Graficador:
	'''
	Takes the sampler object, the labels of each chain and title of the analysis.
	'''

	def __init__(self,sampler,labels,title):
		self.sampler=sampler
		self.labels=labels
		self.title=title

	def graficar_cadenas(self, num_chains = None):
		'''Plot the chains for each parameter.'''
		samples = self.sampler.get_chain()
		len_chain,nwalkers,ndim=self.sampler.get_chain().shape
		sns.set(style='darkgrid', palette="muted", color_codes=True)
		sns.set_context("paper", font_scale=1.5, rc={"font.size":10,"axes.labelsize":17})
		fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)

		for i in range(ndim):
			ax = axes[i]
			if num_chains != None:
				ax.plot(samples[:, 0:num_chains, i], alpha=0.3)
			else: #Plot all the chains
				ax.plot(samples[:, :, i], alpha=0.3)
			ax.set_xlim(0, len(samples))
			ax.set_ylabel(self.labels[i])
		ax.yaxis.set_label_coords(-0.1, 0.5)
		axes[-1].set_xlabel("Number of steps");
		if not self.title==None:
			fig.suptitle(self.title);

	def graficar_cadenas_derivs(self):
		'''Plot the posprocessed chains for each parameter.'''
		if isinstance(self.sampler, np.ndarray)==True: #Posprocessed chains
			samples = self.sampler
			len_chain,ndim=samples.shape
		sns.set(style='darkgrid', palette="muted", color_codes=True)
		sns.set_context("paper", font_scale=1.5, rc={"font.size":10,"axes.labelsize":17})
		fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)

		for i in range(ndim):
		    ax = axes[i]
		    ax.plot(samples[:, i], alpha=0.3)
		    ax.set_xlim(0, len(samples))
		    ax.set_ylabel(self.labels[i])
		ax.yaxis.set_label_coords(-0.1, 0.5)
		axes[-1].set_xlabel("Number of steps");
		if not self.title==None:
			fig.suptitle(self.title);


	def graficar_contornos(self, discard=0, thin=1, color='b'):
		'''
		Grafica los cornerplots para los parámetros a partir de las cadenas
		de Markov. En la diagonal aparecen las  distribuciones de probabilidad
		proyectadas para cada parámetro, y fuera de la diagonal los contornos
		de confianza 2D.

		FALTA: poder cambiar color del plot y darle un label a los plots
		'''
		if isinstance(self.sampler, np.ndarray)==True: #Posprocessed chains
			flat_samples = self.sampler
		else:
			flat_samples = self.sampler.get_chain(discard=discard, flat=True, thin=thin)
		
		names = [i.replace('$','') for i in self.labels]; 
		ndim = len(self.labels)
		samples1 = MCSamples(samples=flat_samples, names=names, labels=names)
		samples1 = samples1.copy(label=r'Lowest-order with $0.3\sigma$ smoothing',
					settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3,
					'smooth_scale_1D':0.3})

		g = plots.get_subplot_plotter()
		g.triangle_plot(samples1,
						filled=True, params=names,
						#contour_colors=color,
						contour_lws=1,
						legend_labels='')


	def reportar_intervalos(self, discard, thin, save_path, hdi=True):
		'''
		Print parameters values, not only the mode values but also their values
		at one and two sigmas.
		'''
		sns.set(style='darkgrid', palette="muted", color_codes=True)
		sns.set_context("paper", font_scale=1.2, rc={"font.size":10,"axes.labelsize":12})


		if isinstance(self.sampler, np.ndarray)==True: #Posprocessed chains
			samples = self.sampler
			len_chain,ndim=samples.shape
		else:
			samples = self.sampler.get_chain(discard=discard, flat=True, thin=thin)
			len_chain, nwalkers, ndim = self.sampler.get_chain().shape

		textfile_witness = open(save_path + '/intervals.dat','w')
		labels = self.labels
		for i in range(ndim):
			mean = np.mean(samples[:,i])
			one_s = 68
			two_s = 95

			if hdi==True:
				one_sigma = az.hdi(samples,hdi_prob = one_s/100)[i]
				two_sigma = az.hdi(samples,hdi_prob = two_s/100)[i]
			else:
				one_sigma = [scoreatpercentile(samples[:,i], 100-one_s), scoreatpercentile(samples[:,i], one_s)]
				two_sigma = [scoreatpercentile(samples[:,i], 100-two_s), scoreatpercentile(samples[:,i], two_s)]

			q1 = np.diff([one_sigma[0],mean,one_sigma[1]])
			q2 = np.diff([two_sigma[0],mean,two_sigma[1]])
			#print(one_sigma,two_sigma)
			if np.abs(one_sigma[0]) < 10**(-2): #Upper limit interval
				txt = "\mathrm{{{0}}} < {1:.3f}({2:.3f})"
				txt = txt.format(labels[i], mean + q1[1], mean + q2[1])

			else:
				txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}({4:.3f})}}^{{+{2:.3f}({5:.3f})}}"
				txt = txt.format(mean, q1[0], q1[1], labels[i], q2[0], q2[1])
			textfile_witness.write('{} \n'.format(txt))
			#display(Math(txt))


	def graficar_taus_vs_n(self, num_param=None,threshold=100.0):
		'''
		Plot the integrated autocorrelation time with respect to the chain length.
		Obs: Threshold shouldn't be llower than 50, according to Emcee library.
		For more info: https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
		 '''
		labels = self.labels
		sns.set(style='darkgrid', palette="muted", color_codes=True)
		sns.set_context("paper", font_scale=1.5, rc={"font.size":8,"axes.labelsize":17})
		plt.grid(True)
		plt.xlabel("Número de muestras $N$",fontsize=15)
		plt.ylabel(r"$\hat{\tau}$",fontsize=15)
		plt.legend(fontsize=17);
		if num_param==None:
			for j in range(len(self.sampler.get_chain()[0, 0, :])):
				chain = self.sampler.get_chain()[:, :, j].T

				# Compute the estimators for a few different chain lengths
				N = np.exp(np.linspace(np.log(threshold), np.log(chain.shape[1]),
				 	int(chain.shape[1]/threshold))).astype(int)
				#donde chain.shape[1] es el largo de las cadenas

				taus = np.empty(len(N))
				for i, n in enumerate(N):
				    taus[i] = emcee.autocorr.integrated_time(chain[:, :n],quiet=True);
				taus=np.cumsum(taus)

				#plt.axhline(true_tau, color="k", label="truth", zorder=-100)
				#plt.loglog(N, taus, '.-', label="{}".format(labels[j]))
				plt.plot(N, taus, '.-', label="{}".format(labels[j]))
			ylim = plt.gca().get_ylim()
			plt.ylim(ylim)
			plt.plot(N, N / threshold, "--k", label=r"$\tau = N/{}$".format(threshold))
			plt.legend(loc = 'best', fontsize=12)
			plt.show()

		else:
			chain = self.sampler.get_chain()[:, :, num_param].T

			# Compute the estimators for a few different chain lengths
			N = np.exp(np.linspace(np.log(threshold), np.log(chain.shape[1]),
			 	int(chain.shape[1]/threshold))).astype(int)
			#donde chain.shape[1] es el largo de las cadenas

			taus = np.empty(len(N))
			for i, n in enumerate(N):
			    taus[i] = emcee.autocorr.integrated_time(chain[:, :n],quiet=True);
			taus=np.cumsum(taus)

			plt.loglog(N, taus, '.-')
			plt.plot(N, N / threshold, "--k", label=r"$\tau = N/{}$".format(threshold))
			#plt.axhline(true_tau, color="k", label="truth", zorder=-100)
			ylim = plt.gca().get_ylim()
			plt.ylim(ylim)
			plt.xlabel("Número de muestras $N$")
			plt.ylabel(r"$\hat{\tau}$")
			plt.legend(fontsize=17);