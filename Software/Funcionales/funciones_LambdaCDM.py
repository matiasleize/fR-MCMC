"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import time
#import camb #No lo reconoce la compu del df
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps as simps
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000
#%%

def E_LCDM(z, omega_m):
    omega_lambda = 1 - omega_m
    E = np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
    return E

def H_LCDM(z, omega_m, H_0):
    H = H_0 * E_LCDM(z,omega_m)
    return H

def H_LCDM_rad(z, omega_m, H_0):
    omega_r = 4.18343*10**(-5) / (H_0/100)**2
    omega_lambda = 1 - omega_m - omega_r

    if isinstance(z, (np.ndarray, list)):
        H = H_0 * np.sqrt(omega_r * (1 + z)**4 + omega_m * (1 + z)**3 + omega_lambda)
    else:
        H = H_0 * (omega_r * (1 + z)**4 + omega_m * (1 + z)**3 + omega_lambda)**(1/2)

    return H


#%%

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import os
    import git
    path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
    path_datos_global = os.path.dirname(path_git)
    os.chdir(path_git)
    sys.path.append('./Software/Funcionales/')
    from funciones_data import leer_data_BAO,leer_data_AGN

    #%% BAO
    os.chdir(path_git+'/Software/Estadística/Datos/BAO/')
#    os.chdir(path_git+'/Software/Estadística/Datos/BAO/Datos_sin_nuevos')
    dataset_BAO = []
    archivo_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                    'datos_BAO_dv.txt','datos_BAO_H.txt']
    for i in range(5):
        aux = leer_data_BAO(archivo_BAO[i])
        dataset_BAO.append(aux)

    [omega_m,H_0] = [0.311, 66.012] #Usando los datos nuevos
    #[omega_m,H_0] = [0.314, 66.432] #Usando datos nuevos (rd camb)

    #[omega_m,H_0] = [0.302, 62.806] #Sin usar datos nuevos
    #[omega_m,H_0] = [0.301, 66.003] #Sin usar datos nuevos (rd camb)

    theta = [omega_m, H_0]

    (z_data,valores,errores_cuad,wd_fid)=dataset_BAO[i]

    zs = np.linspace(0.01, 3, 10**6)
    H_teo = H_LCDM(zs, omega_m, H_0)
    z_data
    rd = r_drag(omega_m,H_0,wd_fid[0]) #Calculo del rd

    #%matplotlib qt5
    legends = ['Da_rd','Dh_rd','Dm_rd','Dv_rd','H*rd']
    num_datasets=5
    chies_BAO = np.zeros(num_datasets)
    for i in range(5):
        (z_data_BAO, valores_data, errores_data_cuad,wb_fid) = dataset_BAO[i]
        if i==0: #Dato de Da
            #La parte teo del chi2
            rd = r_drag_camb(omega_m,H_0,wb_fid) #Calculo del rd
            distancias_teoricas = Hs_to_Ds(zs,H_teo,z_data_BAO,i) #Bien hecho!
            output_teorico = Ds_to_obs_final(zs, distancias_teoricas, rd, i)
        else: #De lo contrario..
            distancias_teoricas = Hs_to_Ds(zs,H_teo,z_data_BAO,i)
            output_teorico = np.zeros(len(z_data_BAO))
            for j in range(len(z_data_BAO)):
                #La parte teo del chi2
                 rd = r_drag_camb(omega_m,H_0,wb_fid[j]) #Calculo del rd
                 output_teorico[j] = Ds_to_obs_final(zs,distancias_teoricas[j],rd,i) #Bien hecho!
        #Calculo el chi2 para cada tipo de dato (i)
        chies_BAO[i] = chi_2_BAO(output_teorico,valores_data,errores_data_cuad)
    print(np.sum(chies_BAO)/(20-2))
#%%
    plt.figure(2)
    plt.grid()
    plt.errorbar(z_data_BAO,valores_data,np.sqrt(errores_data_cuad),fmt='.r')
    plt.plot(z_data_BAO,output_teorico,'.-b')#los puntos unidos por lineas, no es la forma real
    plt.title(legends[i])
    plt.xlabel('z (redshift)')
    plt.ylabel(legends[i])
    os.chdir(path_git+'/Software/Estadística/Datos/BAO/Imagen')
    #os.chdir(path_git+'/Software/Estadística/Datos/BAO/Datos_sin_nuevos/Imagen')
    plt.savefig('{}'.format(legends[i]))
    plt.close()
    #Hay algo mal, los chi2 no coinciden con los grafs. Revisar los errores_data_cuad.

#%%
    #Data AGN
    os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')
    data_agn = leer_data_AGN('table3.dat')
    H_0 = 73.48
    params_fijos = H_0
    omegas = np.linspace(0.1,1,50)
    output = np.zeros(len(omegas))
    for i,omega in enumerate(omegas):
        theta = omega
        output[i] = params_to_chi2_AGN(theta, params_fijos, data_agn)
    #%matplotlib qt5
    plt.plot(omegas,output/(len(data_agn[0])-1))
    plt.title('Chi2 reducido vs omega_m')
    index = np.where(output==min(output))[0][0]
    omegas[index]
    min(output/(len(data_agn[0])-1))

    #%%
    #Defino los parámetros que voy a utilizar
    omega_m = 0.3
    #H_0 = params_fijos
    gamma = 0.648
    beta = 7.735
    egamma = 0.007
    ebeta = 0.244
    #Importo los datos
    data_agn = leer_data_AGN('table3.dat')
    z_data, logFuv, eFuv, logFx, eFx  = data_agn

    zs = np.linspace(0,30,100000)
    Es = E_LCDM(zs, omega_m)
    INT = cumtrapz(Es**(-1), zs, initial=0)
    DlH0 = (c_luz_km * (1 + zs)) * INT
    output = interp1d(zs,DlH0)
    DlH0_teo = np.log10(output(z_data))
    #-np.log10(4*np.pi)/2
    DlH0_obs =  np.log10(3.24)-25+(logFx - gamma * logFuv - beta) / (2*gamma - 2)
    df_dgamma = (1/(2*(gamma-1))**2) * (-logFx + beta + logFuv)
    eDlH0_cuad = (eFx**2 + gamma**2 * eFuv**2 + ebeta**2) / (2*gamma - 2)**2 + (df_dgamma)**2 * egamma**2 #El cuadrado de los errores
    chi2_AGN = chi_2_AGN(DlH0_teo, DlH0_obs, eDlH0_cuad)

    #%matplotlib qt5
    plt.figure()
    plt.xlabel('z (redshift)')
    plt.ylabel(r'$\log(d_{L}H_{0})$')
    plt.errorbar(z_data,DlH0_obs,np.sqrt(eDlH0_cuad),marker='.',linestyle='')
    plt.plot(z_data,DlH0_obs,'.r')
    plt.plot(z_data,DlH0_teo)
    print(chi2_AGN/(len(data_agn[0])-1))

    N = len(data_agn[0]) #Número de datos
    P = 1 #Número de parámetros
    np.sqrt(2*(N-P))/(N-P) #Fórmula del error en el chi2 reducido

    #%%
    H_0 = 70
    omega_m = 0.83
    gamma = 0.6
    beta = 8.3
    delta = 0.63
    theta = [omega_m,beta,gamma,delta]

    #params_to_chi2_AGN_nuisance(theta, _, data_agn)/(len(z_data)-4)

    data_agn = leer_data_AGN('table3.dat')
    z_data_1, logFuv_1, eFuv_1, logFx_1, eFx_1  = data_agn

    zmin = 0
    zmax = 100
    mask = (z_data_1 > zmin) & (z_data_1 < zmax)

    z_data = z_data_1[mask]
    logFuv = logFuv_1[mask]
    logFx = logFx_1[mask]
    eFx = eFx_1[mask]
    eFuv = eFuv_1[mask]

    zs_modelo = np.linspace(0,30,10**6)
    Dl_teo = -np.log10(H_0) + zs_2_logDlH0(zs_modelo,omega_m,z_data)
    Dl_teo_cm = Dl_teo - np.log10(3.24) + 25
    psi = beta + gamma * logFuv + 2 * (gamma-1) * (Dl_teo_cm + 0.5 * np.log10(4*np.pi))

    #si_2 = eFx**2 + (gamma * eFuv)**2 + np.exp(2*np.log(delta)) #El cuadrado de los errores
    si_2 = eFx**2 + (gamma * eFuv)**2 + delta**2 #El cuadrado de los errores

    chi2_AGN = np.sum( ((logFx-psi)**2/si_2) + np.log(2*np.pi*si_2)) # menos en el paper

    print(chi2_AGN)
    print(chi2_AGN/(len(z_data)-3))

    plt.figure()
    plt.xlabel('z (redshift)')
    plt.ylabel(r'$Fx$')
    plt.errorbar(z_data,psi,np.sqrt(si_2),marker='.',linestyle='')
    plt.plot(z_data,logFx,'.r')
