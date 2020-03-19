"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
import time
import emcee
import corner


def graficar_taus_vs_n(sampler, num_param=0):
	'''Esta función grafica el tiempo de autocorrelación integraodo
	 en función del largo de la cadena.'''
	chain = sampler.get_chain()[:, :, num_param].T
	#chain.shape[1] es el largo de las cadenas

	# Compute the estimators for a few different chain lengths
	N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 100)).astype(int)

	#len_chain,nwalkers,ndim=sampler.get_chain().shape
	#N = np.linspace(1, len_chain, 100).astype(int)
	taus = np.empty(len(N))
	for i, n in enumerate(N):
	    taus[i] = emcee.autocorr.integrated_time(chain[:, :n],quiet=True);
	taus=np.cumsum(taus)
	plt.loglog(N, taus, '.-')
	plt.plot(N, N / 100.0, "--k", label=r"$\tau = N/100$")

	ylim = plt.gca().get_ylim()
	plt.ylim(ylim)
	plt.xlabel("number of samples, $N$")
	plt.ylabel(r"$\tau$ estimates")
	plt.legend(fontsize=14);

def graficar_cadenas(sampler,labels= ['omega_m','b']):
	'''Esta función grafica las cadenas en función del largo de las mismas.'''
	samples = sampler.get_chain()
	len_chain,nwalkers,ndim=sampler.get_chain().shape
	fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
	for i in range(ndim):
	    ax = axes[i]
	    ax.plot(samples[:, :, i], "k", alpha=0.3)
	    ax.set_xlim(0, len(samples))
	    ax.set_ylabel(labels[i])
	ax.yaxis.set_label_coords(-0.1, 0.5)
	axes[-1].set_xlabel("step number");

def graficar_contornos(sampler,params_truths,discard=20,thin=None,labels= ['omega_m','b']):
	'''Grafica los contornos de confianza.'''
	if thin == None:
		flat_samples = sampler.get_chain(discard, flat=True)
	else:
		flat_samples = sampler.get_chain(discard, flat=True, thin=thin)
	print(flat_samples.shape)
	fig = corner.corner(flat_samples, labels=labels, truths=params_truths);
