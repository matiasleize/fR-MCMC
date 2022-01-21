import numpy as np
from matplotlib import pyplot as plt
import emcee
import sys
import os

from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/utils/Clases')
from graficador import Graficador

#%% Importo los mínimos del chi2
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_HS_SN_6params.npz') as data:
    sol = data['sol']

#%% Importo las cadenas
os.chdir(path_datos_global+'/Resultados_cadenas')
filename = "sample_HS_SN_6params.h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores
burnin=10000
thin=200
#tau = reader.get_autocorr_time()
#burnin = int(2 * np.max(tau))
#thin = int(0.5 * np.min(tau))
#%%
%matplotlib qt5
analisis = Graficador(reader,
            ['$M_{abs}$','$\Omega_{m}$','b','$\\alpha$','$\\beta$','$\gamma$'],
            #['$\\alpha$','$\\beta$','$\gamma$'],
             '')

analisis.graficar_contornos(sol, discard=burnin, thin=thin,
                            poster=True, color='k', nuisance_only=False)
#%%
analisis.graficar_cadenas()
analisis.reportar_intervalos(sol)