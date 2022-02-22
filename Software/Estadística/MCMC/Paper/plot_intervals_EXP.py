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


	authors = ["This work", "Odintsov et al.",  "Odintsov et al.", "This work", "Odintsov et al.","Odintsov et al.","Farrugia et al."]

	modes = np.array([[0.305, 0.2769, 0.2771, 0.298,0.2799, 0.309,0.3025], #omega_m
			[0.786, None, None, 0.934,None, None, 0.1601], #b
			[68.345, 69.22, 70.28, 67.946, 68.15, 68.59, 68.399]],dtype=object) #H0
	
	min_sigmas = np.array([[0.011, 0.0006, 0.0005, 0.011, 0.018, 0.0255, 0.0049], #omega_m
				[0.524, None, None, 0.877, None, None, 0.1565], #b
				[0.783, 0.73, 1.6, 0.787, 1.2, 1.82, 0.481]],dtype=object) #H0
	
	max_sigmas = np.array([[0.010, 0.0003, 0.0004, 0.011, 0.0186, 0.0215,0.0049], #omega_m
				[0.468, None, None, 0.338, None, None, 0.096], #b
				[0.905, 0.66, 1.6, 1.156, 1.21, 1.85, 0.48]],dtype=object) #H0
	print(max_sigmas[0][:])

	fig, ax = plt.subplots(1, 3, figsize=(12,15), sharey=True)
	ax[1].set_title('Exponential model', fontsize=18)
	#ax.grid(True)
	
	for j in [0,2]:
		ax[j].hlines(y=0.5, xmin=0,xmax=max(modes[j][:]+max_sigmas[j][:])*1.1,color="r",linestyle='-',lw=3)
		ax[j].hlines(y=1.5, xmin=0,xmax=max(modes[j][:]+max_sigmas[j][:])*1.1,color="r",linestyle='-',lw=3)
		ax[j].hlines(y=2.5, xmin=0,xmax=max(modes[j][:]+max_sigmas[j][:])*1.1,color="r",linestyle='-',lw=3)
		ax[j].hlines(y=5.5, xmin=0,xmax=max(modes[j][:]+max_sigmas[j][:])*1.1,color="r",linestyle='-',lw=3)
		ax[j].hlines(y=4.5, xmin=0,xmax=max(modes[j][:]+max_sigmas[j][:])*1.1,color="r",linestyle='-',lw=3)

	ax[1].hlines(y=0.5, xmin=0,xmax=1.5,color="r",linestyle='-',lw=3)
	ax[1].hlines(y=1.5, xmin=0,xmax=1.5,color="r",linestyle='-',lw=3)
	ax[1].hlines(y=2.5, xmin=0,xmax=1.5,color="r",linestyle='-',lw=3)
	ax[1].hlines(y=5.5, xmin=0,xmax=1.5,color="r",linestyle='-',lw=3)
	ax[1].hlines(y=4.5, xmin=0,xmax=1.5,color="r",linestyle='-',lw=3)


	ax[0].set_xlabel('$\Omega_m$', fontsize=22)
	#ax[0].set_xticks()

	ax[1].set_xlabel('$b$', fontsize=22)
	#plt.xticks(fontsize=18)
	
	ax[2].set_xlabel('$H_0$', fontsize=22)
	#plt.xticks(fontsize=18)

	ax[0].set_yticks(np.arange(len(authors)))
	ax[0].set_yticklabels(authors,fontsize='large')
		
	ax[0].set_xlim([0.25, max(modes[0][:]+max_sigmas[0][:])*1.1])
	ax[1].set_xlim([0, 1.5])
	ax[2].set_xlim([65, 75])

	data = ['CC+SnIa\n+BAO+AGN','CC+SnIa\n+$H_{BAO}$\n+CMB','CC+SnIa\n+CMB','CC+SnIa\n+BAO', 'CC+SnIa\n+$H_{BAO}$','CC+SnIa','CC+SnIa\n+BAO_2\n+RSD+CMB']

	ax2 = ax[2].twinx()
	#ax2.set_ylabel('Datasets', fontsize=14)
	ax2.set_yticks(np.arange(len(authors)))
	ax2.set_yticklabels(data, fontsize='large')

	for j in range(3):
		for i in range(len(modes[0][:])):
			if modes[j][i] != None:
				ax[j].hlines(y=i, xmin=modes[j][i]-min_sigmas[j][i],
							xmax=modes[j][i]+max_sigmas[j][i],
							lw=4)
				ax[j].plot(modes[j][i], i, 'o',markersize=10)
	
interval_plot()
plt.savefig('/home/matias/Desktop/intervals_EXP.png')
#plt.show()

from scipy.stats import norm
H0 = norm.rvs(loc=68.399, scale=0.48, size=1000)
omega_b_h2= norm.rvs(loc=22.4560, scale=0.145, size=1000)*10**(-3)
omega_cdm_h2= norm.rvs(loc=0.1191, scale=0.001, size=1000)

omega_m = (omega_b_h2 + omega_cdm_h2) * (100 / H0)**2
#plt.hist(omega_m)
plt.show()
print(np.mean(omega_m), np.std(omega_m))