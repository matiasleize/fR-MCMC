import numpy as np
import emcee
from matplotlib import pyplot as plt
import corner
import sys
import os
import time

from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_analisis_cadenas import graficar_cadenas,graficar_contornos,graficar_taus_vs_n
#%%
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/LCDM')

with np.load('valores_medios_LCDM_AGN_4params_nuisance.npz') as data:
    sol = data['sol']
#%%
os.chdir(path_datos_global+'/Resultados_cadenas')
filename = "sample_LCDM_AGN_4params_nuisance.h5"

reader = emcee.backends.HDFBackend(filename)
# Algunos valores
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
print(tau)
#%%
%matplotlib qt5
graficar_cadenas(reader,
                labels = ['omega_m','beta','gamma','delta'])
 #%%
burnin=1500
thin=1
graficar_contornos(reader,params_truths=sol,discard=burnin,thin=thin,
                    labels = ['omega_m','beta','gamma','delta'])
#%%
#Ojo, siempre muestra que convergio, aun cuando no
#plt.figure()
#graficar_taus_vs_n(reader,num_param=0,threshold=1000)
#graficar_taus_vs_n(reader,num_param=1,threshold=1000)
#%% Printeo los valores!
thin=1
from IPython.display import display, Math
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
labels = ['omega_m','beta','gamma','delta']
len_chain,nwalkers,ndim=reader.get_chain().shape
print(len_chain)
for i in range(ndim):
    mcmc = np.percentile(samples[:, i], [16, 50, 84])
    mcmc[1]=sol[i] #Correción de mati: En vez de percentil 50 poner el mu
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

#%%
betas_2_unflitered = samples[:, 1]
gammas_unflitered = samples[:, 2]

burnin = 1500
thin = 15

def filtrar_puntos(sample, burnin=0,thin=1):
    sample_2 = sample[burnin:]
    sample_3 =[]
    for i in range(len(sample_2)):
        if (i%thin==0):
            sample_3.append(sample_2[i])
    return np.array(sample_3)

betas_2 = filtrar_puntos(betas_2_unflitered, burnin=burnin,thin=thin)
gammas = filtrar_puntos(gammas_unflitered, burnin=burnin,thin=thin)
len(betas_2),len(gammas)
#%%
betas = betas_2 + (gammas-1) * (np.log10(4*np.pi) - 2 * np.log10(70))
np.mean(betas)
np.std(betas)
beta_posta = np.random.normal(7.735,0.244,10**7)
plt.close()
plt.figure()
plt.title('LCDM')
plt.xlabel(r'$\beta$')
plt.hist(betas,density=True,bins=round(np.sqrt(len(betas))),label=r'$\beta_{propagacion}$')
plt.hist(beta_posta,density=True,bins=round(np.sqrt(len(beta_posta))),label=r'$\beta_{paper}$')
plt.grid(True)
plt.legend()
plt.savefig( '/home/matias/propagacion_beta_LCDM.png')
#%%

mcmc = np.percentile(betas, [5, 50, 95]) #Hay coincidencia a 2 sigma :/
q = np.diff(mcmc)
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
txt = txt.format(mcmc[1], q[0], q[1], r'\beta')
display(Math(txt))
txt = "\mathrm{{{2}}} = {0:.3f}\pm{{{1:.3f}}}"
txt = txt.format(np.mean(beta_posta), np.std(beta_posta), r'\beta')
display(Math(txt))

#%%
gamma_posta = np.random.normal(0.648,0.007,10**7)
plt.close()
plt.figure()
plt.title('LCDM')
plt.xlabel(r'$\gamma$')
plt.hist(gammas,density=True,bins=round(np.sqrt(len(gammas))),label=r'$\gamma_{cadenas}$')
plt.hist(gamma_posta,density=True,bins=round(np.sqrt(len(gamma_posta))),label=r'$\gamma_{paper}$')
plt.grid(True)
plt.legend()
plt.savefig( '/home/matias/propagacion_gamma_LCDM.png')

#%%
mcmc = np.percentile(gammas, [5, 50, 95]) #Hay coincidencia a 2 sigma :/
q = np.diff(mcmc)
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
txt = txt.format(mcmc[1], q[0], q[1], r'\gamma')
display(Math(txt))
txt = "\mathrm{{{2}}} = {0:.3f}\pm{{{1:.3f}}}"
txt = txt.format(np.mean(gamma_posta), np.std(gamma_posta), r'\gamma')
display(Math(txt))
