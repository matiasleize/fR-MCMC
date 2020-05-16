"""
Created on Wed Feb  5 16:26:16 2020

@author: matias
"""


import numpy as np
from numpy.linalg import inv
import numpy.ma as ma
def leer_data_pantheon(archivo_pantheon,min_z = 0,max_z = 3):

    '''Toma la data de Pantheon y extrae la data de los redshifts zcmb y zhel
    su error dz, además de los datos de la magnitud aparente con su error:
    mb y dm. Con los errores de la magnitud aparente construye la
    matriz de correlación asociada. La función devuelve la información
    de los redshifts, la magnitud aparente y la matriz de correlación
    inversa.'''

    # leo la tabla de datos:
    zcmb,zhel,dz,mb,dmb=np.loadtxt(archivo_pantheon
                                   , usecols=(1,2,3,4,5),unpack=True)
    #creamos la matriz diagonal con los errores de mB. ojo! esto depende de alfa y beta:
    Dstat=np.diag(dmb**2.)
    # hay que leer la matriz de los errores sistematicos que es de NxN
    sn=len(zcmb)
    Csys=np.loadtxt('lcparam_full_long_sys.txt',unpack=True)
    Csys=Csys.reshape(sn,sn)
    #armamos la matriz de cov final y la invertimos:
    Ccov=Csys+Dstat
    Cinv=inv(Ccov)

    mask = ma.masked_where((zcmb <= max_z) & ((zcmb >= min_z)) , zcmb).mask
    mask_1 = mask[np.newaxis, :] & mask[:, np.newaxis]

    zhel = zhel[mask]
    mb = mb[mask]
    Cinv = Cinv[mask_1]
    Cinv = Cinv.reshape(len(zhel),len(zhel))
    zcmb = zcmb[mask]

    return zcmb,zhel, Cinv, mb
#%%
if __name__ == '__main__':

    import numpy as np
    import sys
    import os
    from os.path import join as osjoin
    from pc_path import definir_path

    path_git, path_datos_global = definir_path()
    os.chdir(path_git)
    sys.path.append('./Software/Funcionales/')
    from funciones_data import leer_data_pantheon
    #%%
    os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
    zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')

    min_z = 0
    max_z = 3

    mask = ma.masked_where((zcmb <= max_z) & ((zcmb >= min_z)) , zcmb).mask
    mask_1 = mask[np.newaxis, :] & mask[:, np.newaxis]

    zhel = zhel[mask]
    mb = mb[mask]
    Cinv_1 = Cinv[mask_1]
    Cinv_1 = Cinv_1.reshape(len(zhel),len(zhel))
    np.all(Cinv_1==Cinv)
    zcmb = zcmb[mask]










#%%
def leer_data_cronometros(archivo_cronometros):

    '''Toma la data de Pantheon y extrae la data de los redshifts zcmb y zhel
    su error dz, además de los datos de la magnitud aparente con su error:
    mb y dm. Con los errores de la magnitud aparente construye la
    matriz de correlación asociada. La función devuelve la información
    de los redshifts, la magnitud aparente y la matriz de correlación
    inversa.'''

    # leo la tabla de datos:
    z, h, dh = np.loadtxt(archivo_cronometros, usecols=(0,1,2), unpack=True)
    return z, h, dh
