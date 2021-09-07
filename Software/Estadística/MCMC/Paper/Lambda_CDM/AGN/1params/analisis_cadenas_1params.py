import numpy as np
from matplotlib import pyplot as plt
import emcee
import sys
import os

from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/Funcionales/Clases')
from funciones_graficador import Graficador

#%% Importo los mínimos del chi2
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_LCDM_AGN_1params.npz') as data:
    sol = data['sol']

#%% Importo las cadenas
os.chdir(path_datos_global+'/Resultados_cadenas')
filename = "sample_LCDM_AGN_1params.h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))

#%%
%matplotlib qt5
samples = reader.get_chain()
label = "omega_m"
plt.figure()
plt.grid()
plt.title('AGN LCDM')
plt.plot(samples[:, :,0], "k", alpha=0.3)
plt.xlim(0, len(samples))
plt.ylabel(label)
plt.xlabel("step number");
plt.savefig( '/home/matias/cadenas_omega_m_LCDM.png')

#%%
sample = reader.get_chain(discard=burnin, flat=True, thin=thin)
plt.close()
plt.figure()
plt.grid(True)
plt.title('Lambda CDM')
plt.xlabel(r'$\Omega_{m}$')
plt.hist(sample,density=True,bins=round(np.sqrt(len(samples))),label=r'$\Omega_m$')
plt.hist(sample)
plt.legend()

#Reporto intevalo
from IPython.display import display, Math
mcmc = np.percentile(samples[:, 0], [16, 50, 84])
mcmc[1]=sol[0] #Correción de mati: En vez de percentil 50 poner el mu
q = np.diff(mcmc)
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
txt = txt.format(mcmc[1], q[0], q[1], label)
display(Math(txt))
