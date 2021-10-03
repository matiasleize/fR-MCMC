import numpy as np
from matplotlib import pyplot as plt
import time
import emcee
import corner
import seaborn as sns
import pandas as pd
from IPython.display import display, Math
import arviz as az
from scipy.stats import scoreatpercentile

#b_w=0.25 #CC+H0 Mejor sin smoothear!
#b_w=0.005 #Nuisance Mejor sin smoothear!
#b_w=0.12 #CC+SN
#b_w=0.15 #CC+SN+BAO
class Graficador:
	'''
	Esta clase genera un objeto "Graficador" que toma el objeto sampler
	de las cadenas generadas por el método MonteCarlo, las etiquetas
	de cada cadena y el título del análisis.

	Falta: poder agregar $$ al principio y al final
	de cada item de la lista de labels (son necesarios
	para los graficos pero no para reportar_intervalos)
	'''

	def __init__(self,sampler,labels,title):
		self.sampler=sampler
		self.labels=labels
		self.title=title

	def graficar_cadenas(self, num_chains = None):
		'''Esta función grafica las cadenas en función del largo
		de las mismas para cada parámetro.'''
		samples = self.sampler.get_chain()
		len_chain,nwalkers,ndim=self.sampler.get_chain().shape
		sns.set(style='darkgrid', palette="muted", color_codes=True)
		sns.set_context("paper", font_scale=1.5, rc={"font.size":10,"axes.labelsize":17})
		fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)

		for i in range(ndim):
			ax = axes[i]
			if num_chains != None:
				ax.plot(samples[:, 0:num_chains, i], alpha=0.3)
			else: #Grafico todas las cadenas
				ax.plot(samples[:, :, i], alpha=0.3)
			ax.set_xlim(0, len(samples))
			ax.set_ylabel(self.labels[i])
		ax.yaxis.set_label_coords(-0.1, 0.5)
		axes[-1].set_xlabel("Número de pasos N");
		if not self.title==None:
			fig.suptitle(self.title);

	def graficar_cadenas_derivs(self):
		'''Esta función grafica las cadenas en función del largo
		de las mismas para cada parámetro.'''
		if isinstance(self.sampler, np.ndarray)==True: #Es una cadenas procesada
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
		axes[-1].set_xlabel("Número de pasos N");
		if not self.title==None:
			fig.suptitle(self.title);


	def graficar_contornos(self, discard,
							thin, poster=False,
							color='b', nuisance_only=False):
		'''
		Grafica los cornerplots para los parámetros a partir de las cadenas
		de Markov. En la diagonal aparecen las  distribuciones de probabilidad
		 proyectadas para cada parámetro, y fuera de la diagonal los contornos
		 de confianza 2D.

		 Poster: If True, hace los graficos sin relleno y solo plotea los
		 		contornos a 98% CL.
				If False, Realiza los contornos de confianza utilizando la
				libreria corner, que es mas rapido pero es más feo.

		 '''
		if isinstance(self.sampler, np.ndarray)==True: #Es una cadenas procesada
			flat_samples = self.sampler
		else:
			flat_samples = self.sampler.get_chain(discard=discard, flat=True, thin=thin)

		params_truths = np.zeros(len(flat_samples[0,:]))
		for i in range(len(params_truths)):
			params_truths[i] = np.mean(flat_samples[:,i])

		if nuisance_only==True:
			flat_samples=flat_samples[:,3:] #Solo para el grafico de nuisance only!

		if poster==True:
			viz_dict = {
	    	#'axes.titlesize':5,
			#'font.size':36,
	    	'axes.labelsize':26,
			'xtick.labelsize':15,
			'ytick.labelsize':15,
			}
			df = pd.DataFrame(flat_samples,columns=self.labels)
			sns.set(style='darkgrid', palette="muted", color_codes=True)
			sns.set_context("paper", font_scale=3, rc=viz_dict)#{"font.size":15,"axes.labelsize":17})

			g = sns.PairGrid(df, diag_sharey=False, corner=True)
			g.map_diag(sns.kdeplot, lw=2, fill=False,color=color)#,bw=b_w)
			g.map_lower(sns.kdeplot,levels=[1-0.95,1-0.68,1],fill=True,color=color)
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


	def reportar_intervalos(self, discard, thin,hdi=True):
		'''
		Imprimer los valores de los parámetros, tanto los valores más
		probables, como las incertezas a uno y dos sigmas.
		'''
		sns.set(style='darkgrid', palette="muted", color_codes=True)
		sns.set_context("paper", font_scale=1.2, rc={"font.size":10,"axes.labelsize":12})


		if isinstance(self.sampler, np.ndarray)==True: #Es una cadenas procesada
			samples = self.sampler
			len_chain,ndim=samples.shape
		else:
			samples = self.sampler.get_chain(discard=discard, flat=True, thin=thin)
			len_chain, nwalkers, ndim = self.sampler.get_chain().shape

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

		#	if np.abs(one_sigma[0]) < 10**(-2): #Reporto intervalo inferior
		#		txt = "\mathrm{{{1}}} < {2:.3f}({3:.3f})"
		#		txt = txt.format(mean, labels[i], mean + q1[1], mean + q2[1])

		#	else:
			txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}({4:.3f})}}^{{+{2:.3f}({5:.3f})}}"
			txt = txt.format(mean, q1[0], q1[1], labels[i], q2[0], q2[1])
			display(Math(txt))


	def graficar_taus_vs_n(self, num_param=None,threshold=100.0):
		'''
		Esta función grafica el tiempo de autocorrelación integrado
		en función del largo de la cadena.
		OBS: threshold no debería nunca ser menor que 50, según la
		documentación de la librería emcee.
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



#%%
if __name__ == '__main__':
	import sys
	import os
	from pc_path import definir_path
	path_git, path_datos_global = definir_path()


	samples = np.random.normal(size=100000)
	one_sigma = [scoreatpercentile(samples, 32), scoreatpercentile(samples, 68)]
	one_sigma
	q1 = np.diff([one_sigma[0],np.mean(samples),one_sigma[1]])
	q1
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
	thin = 100
#%%
	#%matplotlib qt5
	analisis = Graficador(reader,['omega_m','b', 'H0'],'HS CC+H0')
	analisis.graficar_contornos(sol,discard=burnin,thin=thin,poster=True)
#%%
	analisis.graficar_cadenas()
	analisis.reportar_intervalos(sol)
#%%
	analisis.graficar_taus_vs_n(num_param=None)
#	analisis.graficar_taus_vs_n(num_param=1)
#	analisis.graficar_taus_vs_n(num_param=2)

#%% Forma alternativa de graficar, seguir investigando!
	#%matplotlib qt5
	reader = emcee.backends.HDFBackend(filename)
	samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
	emcee_data = az.from_emcee(reader, var_names=['$\Omega_{m}$','b', '$H_{0}$'])
	emcee_data
	az.plot_pair(emcee_data,
            	kind='kde',
				#kde_kwargs={"fill_last": False, 'bw':'scott','levels':[1-0.95, 1-0.68]},
				contour=True,
            	divergences=True,
				marginals = True,
				point_estimate = 'mean',
            	textsize=18)
#%%
	az.plot_posterior(emcee_data)
	plt.show()

	#%%
	labels = ['omega_m','b', 'H0']
	sampler = reader
	flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
	df = pd.DataFrame(flat_samples,columns=labels)
	sns.set(style='darkgrid', palette="muted", color_codes=True)
	sns.set_context("paper", font_scale=1.5, rc={"font.size":10,"axes.labelsize":17})

	g = sns.PairGrid(df, diag_sharey=False, corner=True)
	g.map_diag(sns.kdeplot, lw=2, shade=False)
	g.map_lower(sns.kdeplot,
				levels = [1-0.95],
				#n_levels=[0.68,0.95],
				#shade_lowest=True)
				shade=False
				)
	g.map_lower(sns.kdeplot, levels = [1-0.68], shade=False)
	g.fig.set_size_inches(8,8)
	#%%
	fig,ax=plt.subplots()
	g = sns.PairGrid(df, diag_sharey=False, corner=True)
	g.map_lower(sns.kdeplot,
				shade=True,
				#n_levels=[1-0.95, 1-0.68],
				#n_levels=[0.68,0.95],
				levels=5,
				#normed=True,
				shade_lowest=False
				)

	ax.set_aspect('equal')
	plt.show()
