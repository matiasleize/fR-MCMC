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
from IPython.display import display, Math
import arviz as az

class Graficador:
	'''Falta: poder agregar $$ al principio y al final
	de cada item de la lista de labels (son necesarios
	para los graficos pero no para reportar_intervalos)'''
	def __init__(self,sampler,labels,title):
		self.sampler=sampler
		self.labels=labels
		self.title=title

	def graficar_cadenas(self):
		'''Esta función grafica las cadenas en función del largo
		de las mismas.'''
		samples = self.sampler.get_chain()
		len_chain,nwalkers,ndim=self.sampler.get_chain().shape
		fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
		for i in range(ndim):
		    ax = axes[i]
		    ax.plot(samples[:, :, i], "k", alpha=0.3)
		    ax.set_xlim(0, len(samples))
		    ax.set_ylabel(self.labels[i])
		ax.yaxis.set_label_coords(-0.1, 0.5)
		axes[-1].set_xlabel("step number");
		if not self.title==None:
			fig.suptitle(self.title);

	def graficar_contornos(self, params_truths, discard=20,
							thin=1, poster=False, color='b'):
		'''Grafica los contornos de confianza.'''
		flat_samples = self.sampler.get_chain(discard=discard, flat=True, thin=thin)
		if poster==True:
			df = pd.DataFrame(flat_samples,columns=self.labels)
			sns.set(style='darkgrid', palette="muted", color_codes=True)
			sns.set_context("paper", font_scale=1.5, rc={"font.size":10,"axes.labelsize":17})

			g = sns.PairGrid(df, diag_sharey=False, corner=True)
			g.map_diag(sns.kdeplot, lw=2, shade=True,color=color)
			g.map_lower(sns.kdeplot,levels=5,shade=True,shade_lowest=False,color=color)
			g.fig.set_size_inches(8,8)
			if not self.title==None:
				# Access the Figure
				g.fig.suptitle(self.title, fontsize=20)
		else:
			#print(flat_samples.shape)
			fig = corner.corner(flat_samples, labels=self.labels, truths=params_truths,
				 plot_datapoints=False,quantiles=(0.16, 0.84));
			if not self.title==None:
				fig.suptitle(self.title);


	def reportar_intervalos(self, params_truths, burnin=20, thin=1,UL_index=None):
		'''Printeo los valores!
		Falta: Printear a dos sigmas
		'''
		samples = self.sampler.get_chain(discard=burnin, flat=True, thin=thin)
		labels = self.labels
		len_chain, nwalkers, ndim = self.sampler.get_chain().shape
		for i in range(ndim):
			mode = params_truths[i] #Correción de mati: Ponemos la moda (el minimo del chi2)
			one_sigma = az.hdi(samples,hdi_prob=0.68)[i]
			two_sigma = az.hdi(samples,hdi_prob=0.95)[i]

			q1 = np.diff([one_sigma[0],mode,one_sigma[1]])
			q2 = np.diff([two_sigma[0],mode,two_sigma[1]])

			txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}({4:.3f})}}^{{+{2:.3f}({5:.3f})}}"
			txt = txt.format(mode, q1[0], q1[1], labels[i], q2[0], q2[1])
			display(Math(txt))


	def graficar_taus_vs_n(self, num_param=0,threshold=100.0):
		'''Esta función grafica el tiempo de autocorrelación integraodo
		 en función del largo de la cadena.
		 OBS: threshold no debería nunca ser menor que 50, según la
		 documentación de la librería emcee.
		 '''
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
		plt.axhline(true_tau, color="k", label="truth", zorder=-100)
		ylim = plt.gca().get_ylim()
		plt.ylim(ylim)
		plt.xlabel("number of samples, $N$")
		plt.ylabel(r"$\tau$ estimates")
		plt.legend(fontsize=14);



#%%
if __name__ == '__main__':
	import sys
	import os

	from pc_path import definir_path
	path_git, path_datos_global = definir_path()

	#%%
	os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
	with np.load('valores_medios_HS_CC+H0_3params.npz') as data:
	    sol = data['sol']
	#%%
	os.chdir(path_datos_global+'/Resultados_cadenas')
	filename = 'sample_HS_CC+H0_3params.h5'
	reader = emcee.backends.HDFBackend(filename)

	# Algunos valores
	tau = reader.get_autocorr_time()
	burnin = int(2 * np.max(tau))
	thin = int(0.5 * np.min(tau))
#%%
	#%matplotlib qt5
	analisis = Graficador(reader,['omega_m','b', 'H0'],'HS CC+H0')
	analisis.graficar_cadenas()
	analisis.graficar_contornos(sol,discard=burnin,thin=thin,poster=False)
	analisis.reportar_intervalos(sol)
#%%
	plt.figure()
	analisis.graficar_taus_vs_n(num_param=0)
#	analisis.graficar_taus_vs_n(num_param=1)
#	analisis.graficar_taus_vs_n(num_param=2)

#%% Forma alternativa de graficar, seguir investigando!
	#%matplotlib qt5
	reader = emcee.backends.HDFBackend(filename)
	samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
	emcee_data = az.from_emcee(reader, var_names=['$\Omega_{m}$','b', '$H_{0}$'])
	emcee_data
	az.plot_pair(emcee_data,
            	kind=['kde'],
				kde_kwargs={"fill_last": False},
            	divergences=True,
				marginals = True,
				point_estimate = 'mode',
            	textsize=18)
