"""
Created on Wed Feb  5 16:26:16 2020

@author: matias
"""


import numpy as np
from numpy.linalg import inv

def leer_data_pantheon(archivo_pantheon):
    
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

    return zcmb,zhel, Cinv, mb



def leer_data_cronometros(archivo_cronometros):
    
    '''Toma la data de Pantheon y extrae la data de los redshifts zcmb y zhel
    su error dz, además de los datos de la magnitud aparente con su error: 
    mb y dm. Con los errores de la magnitud aparente construye la 
    matriz de correlación asociada. La función devuelve la información
    de los redshifts, la magnitud aparente y la matriz de correlación 
    inversa.'''
    
    # leo la tabla de datos:
    z, h ,dh =np.loadtxt(archivo_cronometros, usecols=(0,1,2),unpack=True)
    
    return z,h,dh

