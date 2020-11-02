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
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_HS_CC+H0_3params_taylor.npz') as data:
    sol = data['sol']
#%%
os.chdir(path_datos_global+'/Resultados_cadenas')
filename = "sample_HS_CC+H0_3params_taylor.h5"

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
                labels = ['omega_m','b', 'H0'])
#%%
burnin=1000
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
graficar_contornos(reader,params_truths=sol,discard=burnin,thin=thin,
                    labels= ['omega_m','b','H0'],
                    title='CC+H0'#,
                    #poster=True,color='b'
                    )

#%% Printeo los valores!
from IPython.display import display, Math
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
labels = ['omega_m','b', 'H0']
len_chain,nwalkers,ndim=reader.get_chain().shape
for i in range(ndim):
    mcmc = np.percentile(samples[:, i], [16, 50, 84])
    mcmc[1]=sol[i] #Correción de mati: En vez de percentil 50 poner el mu
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

#%%
plt.figure()
graficar_taus_vs_n(reader,num_param=2)
graficar_taus_vs_n(reader,num_param=0)
graficar_taus_vs_n(reader,num_param=1)

#%%
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
with np.load('valores_medios_HS_CC+H0_3params_taylor.npz') as data:
    sol = data['sol']

#%% Importo las cadenas
os.chdir(path_datos_global+'/Resultados_cadenas')
#filename = "sample_HS_CC+H0_3params_nunes.h5" #Con prior de b entre 0 y 7
filename = "sample_HS_CC+H0_3params_taylor.h5" #Con prior de b entre -2 y 7
reader = emcee.backends.HDFBackend(filename)

# Algunos valores
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
#%%
%matplotlib qt5
analisis = Graficador(reader, ['omega_m','b', 'H0'], 'CC+H0 (Taylor)')
analisis.graficar_cadenas()
analisis.graficar_contornos(sol, discard=burnin, thin=thin, poster=False)
analisis.reportar_intervalos(sol)
