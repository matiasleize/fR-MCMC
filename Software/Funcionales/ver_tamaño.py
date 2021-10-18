import numpy as np
import emcee
import sys
import os

from pc_path import definir_path
path_git, path_datos_global = definir_path()


#%% Importo las cadenas
os.chdir(path_datos_global+'/Resultados_cadenas/Paper/HS')
#os.chdir(path_datos_global+'/Resultados_cadenas/Paper/EXP')
filename = "sample_HS_CC+SN+BAO+AGN_4params.h5"
reader = emcee.backends.HDFBackend(filename)
samples = reader.get_chain()
num_walkers = samples.shape[1]
print(num_walkers)
