"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
import time


def graficar_taus_vs_n(sampler):
'''Grafico de la autocorrelación en función del largo de la cadena. (Hacer)'''
	n = 100 * np.arange(1, index + 1)
	y = autocorr[:index] #Aun no esta definido autocorr, calcularlo a partir de sampler..
	plt.plot(n, n / 100.0, "--k")
	plt.plot(n, y)
	plt.xlim(0, n.max())
	plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
	plt.xlabel("number of steps")
	plt.ylabel(r"mean $\hat{\tau}$");

#%matplotlib qt5
#plt.close()
def graficar_cadenas(sampler,labels= ['omega_m','b']):
	'''Grafico de las cadenas'''
	fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
	samples = sampler.get_chain()
	for i in range(ndim):
	    ax = axes[i]
	    ax.plot(samples[:, :, i], "k", alpha=0.3)
	    ax.set_xlim(0, len(samples))
	    ax.set_ylabel(labels[i])
	    ax.yaxis.set_label_coords(-0.1, 0.5)

	axes[-1].set_xlabel("step number");

def grafiar_contornos(sampler,params_truths=[omega_m_ml,b_ml],discard=20,thin=None,labels= ['omega_m','b']):
	'''Grafico de los contornos de confianza.'''
	if thin == None:
		flat_samples = sampler.get_chain(discard=20, flat=True)
	else:
		flat_samples = sampler.get_chain(discard=20, flat=True, thin=thin)
	print(flat_samples.shape)
	fig = corner.corner(flat_samples, labels=labels, truths=params_truths);

