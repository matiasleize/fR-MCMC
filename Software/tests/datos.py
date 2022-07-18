'''(Spanish documentation)'''
import numpy as np
import numpy.ma as ma
from numpy.linalg import inv
import pandas as pd

class DataFrame:
	def __init__(self,dataset,datafile):
		self.dataset=dataset #Implementar poner varios datasets
		self.datafile=datafile

	def crear_df(self ,min_z = 0 ,max_z = 3):
		if self.dataset=='SN':
			'''Toma la data de Pantheon y extrae la data de los redshifts zcmb y zhel
			su error dz, además de los datos de la magnitud aparente con su error:
			mb y dm. Con los errores de la magnitud aparente construye la
			matriz de correlación asociada. La función devuelve la información
			de los redshifts, la magnitud aparente y la matriz de correlación
			inversa.'''
			'''IMPLEMENTAR min_z y max_z'''
			# leo la tabla de datos:
			zcmb,zhel,dz,mb,dmb=np.loadtxt(self.datafile[0],
			usecols=(1,2,3,4,5),unpack=True)
			#creamos la matriz diagonal con los errores de mB. ojo! esto depende de alfa y beta:
			Dstat=np.diag(dmb**2.)

		    # hay que leer la matriz de los errores sistematicos que es de NxN
			sn=len(zcmb)
			Csys=np.loadtxt(self.datafile[1],unpack=True)
			Csys=Csys.reshape(sn,sn)
			#armamos la matriz de cov final y la invertimos:
			Ccov=Csys+Dstat
			Cinv=inv(Ccov)

			#ARREGLAR
			df_1 = pd.read_csv(self.datafile[0],delimiter='\t',header=None,
				 			 usecols=(1,2,4),names=['zcmb', 'zhel', 'mb'])
			df=(df_1,Cinv)


		elif self.dataset=='SN_nuicence':
			'''IMPLEMENTAR min_z y max_z'''
			# leo la tabla de datos:
			zcmb0,zhel0,dz0,mb0,dmb0=np.loadtxt(self.datafile[0],
										usecols=(1,2,3,4,5),unpack=True)

			#creamos la matriz diagonal con los errores de mB. ojo! esto depende de alfa y beta:
			Dstat=np.diag(dmb0**2.)

			# hay que leer la matriz de los errores sistematicos que es de NxN
			sn=len(zcmb0)
			Csys=np.loadtxt(self.datafile[1],unpack=True)
			Csys=Csys.reshape(sn,sn)
			#armamos la matriz de cov final y la invertimos:
			Ccov=Csys+Dstat
			Cinv=inv(Ccov)

			zcmb_1,hmass,x1,cor=np.loadtxt(self.datafile[2],usecols=(7,13,20,22),
			unpack=True)

			df = pd.DataFrame([zcmb0, zcmb_1, zhel0, Cinv, mb0, x1, cor, hmass])

		elif self.dataset=='CC':
			'''Data de Cronometros'''
			'''IMPLEMENTAR min_z y max_z'''
			# leo la tabla de datos:
			df = pd.read_csv(self.datafile,delimiter='\t',header=None,
				 			 usecols=(0,1,2),names=['z', 'H', 'dH'])


		elif self.dataset=='BAO':
			'''Toma datos de BAO..'''
			'''IMPLEMENTAR min_z y max_z'''

			df = pd.read_csv(self.datafile,delimiter='\t',usecols=(0,1,2,4))

		return df


#%%
if __name__ == '__main__':
	import os
	import git
	path_git = git.Repo('.', search_parent_directories=True).working_tree_dir

	#%% Supernovas Nuisence
	os.chdir(path_git+'/Software/source/Pantheon')
	df_SN_nuisence = DataFrame('SN',['lcparam_full_long_zhel.txt','lcparam_full_long_sys.txt'])
	df_SN_nuisence.crear_df() #OJO, habria que trasponerlo!
	#%% Cronómetros Cósmicos
    os.chdir(path_git+'/Software/source/CC')
	df_CC = DataFrame('CC','chronometers_data_nunes.txt')
	df_CC.crear_df()
	#%% BAO
    os.chdir(path_git+'/Software/source/BAO')
	df_BAO = DataFrame('BAO','BAO_data.txt')
	df_BAO.crear_df()
