import numpy as np
from matplotlib import pyplot as plt
import emcee
import sys
import os

from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/utils/Clases')
from plotter import Plotter

#%% Importo las cadenas
os.chdir(path_datos_global+'/Resultados_cadenas/')
filename = "sample_LCDM_AGN_4params_nuisance_67.h5"
#filename = "sample_LCDM_AGN_4params_nuisance_72.h5"
reader = emcee.backends.HDFBackend(filename)
# Algunos valores
tau = reader.get_autocorr_time()
samples = reader.get_chain()
burnin= burnin=int(0.2*len(samples[:,0]))
thin=1
samples


#%%
%matplotlib qt5
label = "omega_m"
plt.figure()
plt.grid()
plt.title('AGN LCDM')
plt.plot(samples[:, :,0], "k", alpha=0.3)
plt.xlim(0, len(samples))
plt.ylabel(label)
plt.xlabel("step number");

#%% Distribucion de omega_m
sample = reader.get_chain(discard=burnin, flat=True, thin=thin)
plt.close()
plt.figure()
plt.grid(True)
plt.title('Lambda CDM')
plt.xlabel(r'$\Omega_{m}$')
plt.hist(sample,density=True,bins=round(np.sqrt(len(samples))),label=r'$\Omega_m$')
#plt.hist(sample)
plt.legend()

#%% Reporto intevalo de omega_m
from IPython.display import display, Math
mcmc = np.percentile(samples[:, 0], [16, 50, 84])
#mcmc[1]=sol[0] #Correción de mati: En vez de percentil 50 poner el mu
q = np.diff(mcmc)
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
txt = txt.format(mcmc[1], q[0], q[1], label)
display(Math(txt))
