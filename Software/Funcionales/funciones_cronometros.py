"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np

def leer_data_cronometros(archivo):
    
    '''Toma la data de Pantheon y extrae la data de los redshifts zcmb y zhel
    su error dz, además de los datos de la magnitud aparente con su error: 
    mb y dm. Con los errores de la magnitud aparente construye la 
    matriz de correlación asociada. La función devuelve la información
    de los redshifts, la magnitud aparente y la matriz de correlación 
    inversa.'''
    
    # leo la tabla de datos:
    z, h ,dh =np.loadtxt(archivo, usecols=(1,2,3),unpack=True)
    
    return z,h


def chi_2_cronometros(H_data, H_teo, dH):
    chi2 = np.sum((H_data-H_teo)**2/dH**2)
    return chi2
