'''
Este script hace el gráfico de la comparación de los distintos intervalos. Esto lo hacemos para distintos autores y distintos datasets.
'''

import numpy as np
from matplotlib import pyplot as plt 

def interval_plot():
	'''''
	authors: (list) list of strings with the name of the authors.
	mode_values: (list) list of the mode values of the intervals.
	min_sigmas (list): list of the min sigma values.
	max_sigmas (list): list of the max sigma values.
	index (int): parameter that it is evaluated. 0: '\Omega_m', 1: 'b', 2: 'H_0', 3: '\M_abs'(?). 
	'''''


	authors = ["This work", "Farrugia et al.", "D'agostino et al.", "This work", "D'agostino et al."]

	modes = np.array([[0.305, 0.30078, 0.267, 0.27, 0.289], #omega_m
			[0.278, 0.2738*10**(-4), 0.19, 0.587, 0.32], #b
			[67.550, 68.5280, 72.4, 68.899, 69.5]]) #H0
	
	min_sigmas = np.array([[0.010, 0.0047, 0.023, 0.033, 0.025], #omega_m
				[0.278, 0.2739*10**(-4), 0.16, 0.587, 0.25], #b
				[1.004, 0.4851, 1.4, 1.603, 2]]) #H0
	
	max_sigmas = np.array([[0.011, 0.0047, 0.023, 0.029, 0.028], #omega_m
				[0.278, 0.1789*10**(-4), 0.1, 0.587, 0.17], #b
				[1.148, 0.4797, 1.4, 1.802, 2]]) #H0


	fig, ax = plt.subplots(1, 3, figsize=(12,15), sharey=True)
	ax[1].set_title('Hu-Sawicki model', fontsize=18)
	#ax.grid(True)
	for j in range(3):
		ax[j].hlines(y=0.5, xmin=0,xmax=max(modes[j,:]+max_sigmas[j,:])*1.1,color="r",linestyle='-',lw=3)
		ax[j].hlines(y=1.5, xmin=0,xmax=max(modes[j,:]+max_sigmas[j,:])*1.1,color="r",linestyle='-',lw=3)
		ax[j].hlines(y=2.5, xmin=0,xmax=max(modes[j,:]+max_sigmas[j,:])*1.1,color="r",linestyle='-',lw=3)

	ax[0].set_xlabel('$\Omega_m$', fontsize=22)
	#ax[0].set_xticks()

	ax[1].set_xlabel('$b$', fontsize=22)
	#plt.xticks(fontsize=18)
	
	ax[2].set_xlabel('$H_0$', fontsize=22)
	#plt.xticks(fontsize=18)

	ax[0].set_yticks(np.arange(len(authors)))
	ax[0].set_yticklabels(authors,fontsize='large')
		
	ax[0].set_xlim([0.2, max(modes[0,:]+max_sigmas[0,:])*1.1])
	ax[1].set_xlim([0, max(modes[1,:]+max_sigmas[1,:])*1.1])
	ax[2].set_xlim([65, 78])



	ax2 = ax[2].twinx()
	#ax2.set_ylabel('Datasets', fontsize=14)
	ax2.set_yticks(np.arange(len(authors)))
	ax2.set_yticklabels(['CC+SnIa\n+BAO+AGN','CC+SnIa\n+BAO_2\n+RSD+CMB','CC+SnIa\n+H0licow','CC+SnIa','CC+SnIa']
						,fontsize='large')

	for j in range(3):
		for i in range(len(modes[0,:])):
			ax[j].hlines(y=i, xmin=modes[j][i]-min_sigmas[j][i],
						xmax=modes[j][i]+max_sigmas[j][i],
						lw=4)
			ax[j].plot(modes[j][i], i, 'o',markersize=10)
	
interval_plot()
plt.savefig('/home/matias/Desktop/intervals_HS.png')
#plt.show()

from scipy.stats import norm
H0 = norm.rvs(loc=68.5280, scale=0.48, size=1000)
omega_b_h2= norm.rvs(loc=22.4790, scale=0.145, size=1000)*10**(-3)
omega_cdm_h2= norm.rvs(loc=0.1188, scale=0.001, size=1000)

omega_m = (omega_b_h2 + omega_cdm_h2) * (100 / H0)**2
#plt.hist(omega_m)
plt.show()
print(np.mean(omega_m), np.std(omega_m))