"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
import time
import emcee
import corner
import seaborn as sns
import pandas as pd
from itertools import cycle
cycol = cycle('bgrcmk')

def graficar_cadenas(sampler,labels= ['omega_m','b','H0'],title=None):
	'''Esta función grafica las cadenas en función del largo de las mismas.'''
	samples = sampler.get_chain()
	len_chain,nwalkers,ndim=sampler.get_chain().shape
	fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
	for i in range(ndim):
		#ax.plot(samples[:, :, i], "k", alpha=0.3)
		ax = axes[i]
		ax.plot(samples[:, :, i],c=next(cycol), alpha=0.3)
		ax.set_xlim(0, len(samples))
		ax.set_ylabel(labels[i])
		ax.yaxis.set_label_coords(-0.1, 0.5)
	axes[-1].set_xlabel("step number");
	if not title==None:
		fig.suptitle(title);

def graficar_cadenas_flat(sampler,labels= ['omega_m','b'],title=None):
	'''Esta función grafica las cadenas en función del largo de las mismas.'''
	samples = sampler.get_chain() #Entra un array!
	len_chain,ndim=sampler.get_chain().shape
	fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
	for i in range(ndim):
		ax = axes[i]
		ax.plot(samples[:, i], "k", alpha=0.3)
		ax.set_xlim(0, len(samples))
		ax.set_ylabel(labels[i])
		ax.yaxis.set_label_coords(-0.1, 0.5)
	axes[-1].set_xlabel("step number");
	if not title==None:
		fig.suptitle(title);

def graficar_contornos(sampler,params_truths,discard=20,
						thin=1,labels= ['omega_m','b'],title=None,
						poster=False,color='b'):
	'''Grafica los contornos de confianza.'''
	if isinstance(sampler,emcee.backends.hdf.HDFBackend)==True:
		flat_samples = sampler.get_chain(discard=discard, flat=True, thin=thin)
	elif isinstance(sampler,np.ndarray)==True:
		flat_samples=sampler
	if poster==True:
		df = pd.DataFrame(flat_samples,columns=labels)
		sns.set(style='darkgrid', palette="muted", color_codes=True)
		sns.set_context("paper", font_scale=1.5, rc={"font.size":10,"axes.labelsize":17})

		g = sns.PairGrid(df, diag_sharey=False, corner=True)
		g.map_diag(sns.kdeplot, lw=2, shade=True,color=color)
		g.map_lower(sns.kdeplot,levels=5,shade=True,shade_lowest=False,color=color)
		g.fig.set_size_inches(8,8)
		if not title==None:
			# Access the Figure
			g.fig.suptitle(title, fontsize=20)
	else:
		fig = corner.corner(flat_samples, labels=labels, truths=params_truths,
			 plot_datapoints=False,quantiles=(0.16, 0.84));
		if not title==None:
			fig.suptitle(title);

def graficar_taus_vs_n(sampler, num_param=0,threshold=100.0):
	'''Esta función grafica el tiempo de autocorrelación integraodo
	 en función del largo de la cadena.'''
	chain = sampler.get_chain()[:, :, num_param].T

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

	ylim = plt.gca().get_ylim()
	plt.ylim(ylim)
	plt.xlabel("number of samples, $N$")
	plt.ylabel(r"$\tau$ estimates")
	plt.legend(fontsize=14);


#%%
if __name__ == '__main__':
	import numpy as np
	import emcee
	import seaborn as sns
	from matplotlib import pyplot as plt
	import corner
	import time
	import pandas as pd

	import os
	import git
	path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
	path_datos_global = os.path.dirname(path_git)
	#%%
	os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
	with np.load('valores_medios_HS_cronom_2params_taylor.npz') as data:
	    sol = data['sol']
	#%%
	os.chdir(path_datos_global+'/Resultados_cadenas')
	filename = "sample_HS_cronom_2params_taylor.h5"
	reader = emcee.backends.HDFBackend(filename)
	# Algunos valores
	nwalkers, ndim = reader.shape
	tau = reader.get_autocorr_time()
	burnin = int(2 * np.max(tau))
	thin = int(0.5 * np.min(tau))
	samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
	print(nwalkers,ndim)
	print(samples.shape[0])#numero de pasos
	print(tau)

	#%%
	burnin=100
	burnin = int(2 * np.max(tau))
	thin = int(0.5 * np.min(tau))
	graficar_contornos(reader,params_truths=sol,discard=burnin,thin=thin,
	                    labels= ['omega_m','b'])
	#%%
	#%matplotlib qt5
	labels= ['$\Omega_{m}$','b']
	title='HS Taylor (H0 fijo)'
	data=reader.get_chain(flat=True)
	df = pd.DataFrame(data,columns=labels)
	#%%
	#sns.set()
	sns.set(style='darkgrid', palette="muted", color_codes=True)
	sns.set_context("paper", font_scale=1.5, rc={"font.size":10,"axes.labelsize":17})
	#font_scale: numeros de los ejes
	#font.size:
	#axes.labelsize: tamano de las labels

	g = sns.PairGrid(df, diag_sharey=False, corner=True)
	#g.map_diag(sns.kdeplot, lw=2, shade=True,color='r')
	g.map_lower(sns.kdeplot,
				shade=True,
				shade_lowest=False,
				levels=6,
				color='r')
	#g.map_lower(sns.kdeplot,
#				levels=[1-np.exp(-0.5),0.9],
	#			levels=[0.01,0.4],
#				color='b')

	# Access the Figure
	#g.add_legend()
	g.fig.set_size_inches(8,8)
	g.fig.suptitle(title, fontsize=20)
	#df['b'].to_numpy()
