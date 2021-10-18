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
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones')
with np.load('valores_medios_EXP_CC+SN_4params.npz') as data:
    sol = data['sol']

#%% Importo las cadenas
os.chdir(path_datos_global+'/Resultados_cadenas/')
filename = "sample_EXP_CC+SN_4params.h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores
#tau = reader.get_autocorr_time()
#burnin = int(2 * np.max(tau))
#thin = int(0.5 * np.min(tau))
sample = reader.get_chain()
burnin= burnin=int(0.2*len(sample[:,0]))
thin=1
#%%
%matplotlib qt5
analisis = Graficador(reader, ['$M_{abs}$','$\Omega_{m}$','b','$H_{0}$'],'CC+SN (EXP)')
                    #'Supernovas tipo IA + Cronómetros Cósmicos + BAO')
analisis.graficar_contornos(discard=burnin, thin=thin, poster=False,color='r')
#plt.savefig('/home/matias/Desktop/CC+SN')
analisis.graficar_cadenas()
analisis.reportar_intervalos(discard=burnin,thin=thin)
#%% Calculamos el intervalo de beta
%matplotlib qt5
bs = reader.get_chain(discard=burnin,thin=thin,flat=True)[:,2]
max(bs)
betas = 2/bs
#betas = betas[betas<50]
len(betas)
bi=int(np.sqrt(len(betas)))
plt.close()
plt.figure()
plt.grid()
plt.hist(bs,bins='auto',density=True);

#plt.hist(betas,bins=bi*5,density=True);
#plt.hist(betas,bins='auto',density=True);
#plt.xlabel('r$\beta$')
#plt.xlim([0,50])
#%%
len(betas)
betas
min(bs)
max(betas)
import arviz as az
mean = np.mean(betas)
one_s = 68
two_s = 95
one_sigma = az.hdi(betas,hdi_prob = one_s/100)
two_sigma = az.hdi(betas,hdi_prob = two_s/100)
one_sigma
two_sigma
mean
