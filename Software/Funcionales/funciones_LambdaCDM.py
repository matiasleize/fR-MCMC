"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import time
import camb
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

#Supernovas

def magn_aparente_teorica(z, H, zcmb, zhel):
    '''Calculo del modelo de la distancia teórica a partir del H(z) del modelo.

    Imput:
        z (array): vector de redshifts z
        H (array): vector de H(z)
        zcmb, zcdm (array): vectores con redshifts que voy a usar
        para evaluar la distancia luminosa teórica.

    Output:
        muth (array): magnitud aparente teórica
    '''

    d_c =  c_luz_km * cumtrapz(H**(-1), z, initial=0)
    dc_int = interp1d(z, d_c) #Interpolamos
    d_L = (1 + zhel) * dc_int(zcmb) #Obs, Caro multiplica por Zhel, con Zobs da un poquin mejor
    #Magnitud aparente teorica
    muth = 25.0 + 5.0 * np.log10(d_L)
    return muth

def chi_2_supernovas(muth, muobs, C_invertida):
    '''Dado el resultado teórico muth y los datos de la
    magnitud aparente y absoluta observada, con su matriz de correlación
    invertida asociada, se realiza el cálculo del estadítico chi cuadrado.'''

    deltamu = muth - muobs #vector fila
    transp = np.transpose(deltamu) #vector columna
    aux = np.dot(C_invertida,transp) #vector columna
    chi2 = np.dot(deltamu,aux) #escalar
    return chi2


def params_to_chi2(theta, params_fijos, zcmb, zhel, Cinv, mb,
                    cantidad_zs=int(10**5), fix_M=False,
                     fix_H0=False, fix_M_H0=False):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas.'''

    if fix_M == True:
        [omega_m, H_0] = theta
        Mabs = params_fijos
    elif fix_H0 == True:
        [Mabs, omega_m] = theta
        H_0 = params_fijos
    elif fix_M_H0 == True:
        omega_m = theta
        [Mabs, H_0] = params_fijos
    else:
        [Mabs, omega_m, H_0] = theta

    z = np.linspace(0, 3, cantidad_zs)
    H = H_LCDM(z, omega_m, H_0)

    muth = magn_aparente_teorica(z, H, zcmb, zhel)
    muobs =  mb - Mabs

    chi = chi_2_supernovas(muth, muobs, Cinv)
    return chi


# Cronómetros
def chi_2_cronometros(H_teo, H_data, dH):
    chi2 = np.sum(((H_data - H_teo) / dH)**2)
    return chi2

def params_to_chi2_cronometros(theta, z_data, H_data,
                                dH, cantidad_zs=int(10**6)):
    '''Dados los parámetros libres del modelo (omega y H0), devuelve un chi2
     para los datos de los cronómetros cósmicos'''

    [omega_m, H_0] = theta
    z = np.linspace(0, 3, cantidad_zs)
    H = H_LCDM(z, omega_m, H_0)
    H_int = interp1d(z, H)
    H_teo = H_int(z_data)
    chi = chi_2_cronometros(H_teo, H_data, dH)
    return chi

# BAO
def zdrag(omega_m,H_0,wb=0.0225):
    '''
    omega_b = 0.05 #(o algo asi, no lo usamos directamente)
    wb = 0.0222383 #Planck
    wb = 0.0225 #BBN
    '''
    h = H_0/100
    b1 = 0.313*(omega_m*h**2)**(-0.419)*(1+0.607*(omega_m*h**2)**(0.6748))
    b2 = 0.238*(omega_m*h**2)**0.223
    zd = (1291*(omega_m*h**2)**0.251) * (1+b1*wb**b2) /(1+0.659*(omega_m*h**2)**0.828)
    #zd =1060.31
    return zd

def H_LCDM_rad(z, omega_m, H_0):
    omega_r = 4.18343*10**(-5) / (H_0/100)**2
    omega_lambda = 1 - omega_m - omega_r
    H = H_0 * np.sqrt(omega_r * (1 + z)**4 + omega_m * (1 + z)**3 + omega_lambda)
    return H

def r_drag(omega_m,H_0,wb = 0.0225, int_z=True): #wb x default tomo el de BBN.
    #Calculo del rd:
    h = H_0/100
    zd = zdrag(omega_m,H_0)
    #R_bar = 31500 * wb * (2.726/2.7)**(-4)
    R_bar = wb * 10**5 / 2.473

    #Integral logaritmica
    zs_int_log = np.logspace(np.log10(zd),13,int(10**5))
    H_int_log = H_LCDM_rad(zs_int_log,omega_m,H_0)

    integrando_log = c_luz_km / (H_int_log * np.sqrt(3*(1 + R_bar*(1+zs_int_log)**(-1))))

    rd_log = simps(integrando_log,zs_int_log)
    return rd_log

def r_drag_camb(omega_m,H_0,wb = 0.0225):
    pars = camb.CAMBparams()
    h = (H_0/100)
    pars.set_cosmology(H0=H_0, ombh2=wb, omch2=omega_m*h**2-wb)
    results = camb.get_background(pars)
    rd = results.get_derived_params()['rdrag']
    #print('Derived parameter dictionary: %s'%results.get_derived_params()['rdrag'])
    return rd


def Hs_to_Ds(zs, Hs, z_data, index):
    if index == 4: #H
        aux = Hs
    else:
        INT = cumtrapz(Hs**(-1), zs, initial=0)
        DA = (c_luz_km/(1 + zs)) * INT
        if index == 0: #DA
            aux = DA
        elif index == 1: #DH
            DH = c_luz_km * (Hs**(-1))
            aux = DH
        elif index == 2: #DM
            #aux = (1+zs) * DA
            DM = c_luz_km * INT
            aux = DM
        elif index == 3: #DV
            #aux = (((1 +zs) * DA)**2 * c_luz_km * zs * (Hs**(-1))) ** (1/3)
            DV = c_luz_km * (INT**2 * zs * (Hs**(-1))) ** (1/3)
            aux = DV
    output = interp1d(zs,aux)
    return output(z_data)

def Ds_to_obs_final(zs, Dist, rd, index):
    if index == 4: #H
        output = Dist*rd
    else: #Todas las otras distancias
        output = Dist/rd
    return output



def chi_2_BAO(teo, data, errores_cuad):
    chi2 = np.sum((data-teo)**2/errores_cuad)
    return chi2

def params_to_chi2_BAO(theta, params_fijos, dataset_BAO,
                        cantidad_zs=int(10**6),num_datasets=5):
    '''Dados los parámetros libres del modelo (omega, b y H0) y
    los que quedan params_fijos (n), devuelve un chi2 para los datos
    de BAO'''

    [omega_m, H_0] = theta
    zs = np.linspace(0.01, 3, cantidad_zs)
    H_teo = H_LCDM(zs, omega_m, H_0)

    chies_BAO = np.zeros(num_datasets)
    for i in range(5): #Para cada tipo de dato
        (z_data_BAO, valores_data, errores_data_cuad,wb_fid) = dataset_BAO[i]
        if i==0: #Dato de Da
            #rd = r_drag_camb(omega_m,H_0,wb_fid) #Calculo del rd
            rd = r_drag(omega_m,H_0,wb_fid) #Calculo del rd
            distancias_teoricas = Hs_to_Ds(zs,H_teo,z_data_BAO,i)
            output_teorico = Ds_to_obs_final(zs, distancias_teoricas, rd, i)
        else: #De lo contrario..
            distancias_teoricas = Hs_to_Ds(zs,H_teo,z_data_BAO,i)
            output_teorico = np.zeros(len(z_data_BAO))
            for j in range(len(z_data_BAO)): #Para cada dato de una especie
                 #rd = r_drag_camb(omega_m,H_0,wb_fid[j]) #Calculo del rd
                 rd = r_drag(omega_m,H_0,wb_fid[j]) #Calculo del rd
                 output_teorico[j] = Ds_to_obs_final(zs,distancias_teoricas[j],rd,i)
        #Calculo el chi2 para cada tipo de dato (i)
        chies_BAO[i] = chi_2_BAO(output_teorico,valores_data,errores_data_cuad)

    if np.isnan(sum(chies_BAO))==True:
        print('Hay errores!')
        print(omega_m,H_0,rd)

    #return chies_BAO
    return np.sum(chies_BAO)

#%% AGN
def zs_2_logDlH0(zs,omega_m,z_data):
    Es = E_LCDM(zs, omega_m)
    INT = cumtrapz(Es**(-1), zs, initial=0)
    DlH0 = (c_luz_km * (1 + zs)) * INT #km/seg
    output = interp1d(zs,DlH0)
    return np.log10(output(z_data)) #log(km/seg)

def zs_2_logDl(zs,omega_m,H0,z_data):
    Hs = H_LCDM(zs, omega_m, H0)
    INT = cumtrapz(Hs**(-1), zs, initial=0)
    Dl = (c_luz_km * (1 + zs)) * INT
    output = interp1d(zs,Dl)
    return np.log10(output(z_data))

def chi_2_AGN(teo, data, errores_cuad):
    chi2 = np.sum((data-teo)**2/errores_cuad)
    return chi2


def params_to_chi2_AGN(theta, params_fijos, dataset_AGN, cantidad_zs=10**5):
    '''Dados los parámetros del modelo devuelvo el estadítico chi2 para
    los datos de AGN.'''
    #Defino los parámetros que voy a utilizar
    omega_m = theta
    #H_0 = params_fijos
    beta = 7.735
    ebeta = 0.244
    gamma = 0.648
    egamma = 0.007

    #Importo los datos
    z_data, logFuv, eFuv, logFx, eFx  = dataset_AGN

    zs_modelo = np.linspace(0,30,cantidad_zs)
    DlH0_teo = zs_2_logDlH0(zs_modelo,omega_m,z_data)
    DlH0_obs =  np.log10(3.24)-25+(logFx - gamma * logFuv - beta) / (2*gamma - 2)

    df_dgamma = (1/(2*(gamma-1))**2) * (-logFx+beta+logFuv)
    eDlH0_cuad = (eFx**2 + gamma**2 * eFuv**2 + ebeta**2)/ (2*gamma - 2)**2 + (df_dgamma)**2 * egamma**2 #El cuadrado de los errores

    chi2_AGN = chi_2_AGN(DlH0_teo, DlH0_obs, eDlH0_cuad)

    return chi2_AGN

def chi_2_AGN_nuisance(teo, data, errores_cuad):
    chi2 = np.sum( ((data-teo)**2/errores_cuad) - np.log(errores_cuad)) #o menos en el paper
    return chi2

def params_to_chi2_AGN_nuisance(theta, params_fijos, dataset_AGN, cantidad_zs=10**6):
    '''
    Dados los parámetros del modelo devuelvo el estadítico chi2 para
    los datos de AGN.
    '''
    #Defino los parámetros que voy a utilizar
    if len(theta) == 4:
        [omega_m, beta, gamma, delta] = theta #Este beta es distinto al otro!
        H_0 = params_fijos
    elif len(theta) == 3:
        [omega_m, gamma, delta] = theta #Este beta es distinto al otro!
        [H_0, beta] = params_fijos

    #Importo los datos
#    z_data, logFuv, eFuv, logFx, eFx  = dataset_AGN
    z_data_1, logFuv_1, eFuv_1, logFx_1, eFx_1  = dataset_AGN

    zmin = 0
    zmax = 200
    mask = (z_data_1 > zmin) & (z_data_1 < zmax)

    z_data = z_data_1[mask]
    logFuv = logFuv_1[mask]
    logFx = logFx_1[mask]
    eFx = eFx_1[mask]
    eFuv = eFuv_1[mask]

    zs_modelo = np.linspace(0,30,cantidad_zs)
    Dl_teo = -np.log10(H_0) + zs_2_logDlH0(zs_modelo,omega_m,z_data) #Mpc
    Dl_teo_cm = Dl_teo - np.log10(3.24) + 25
    psi = beta + gamma * logFuv + 2 * (gamma-1) * (Dl_teo_cm + 0.5 * np.log10(4*np.pi))

    si_2 = eFx**2 + gamma**2 * eFuv**2 + np.exp(2*np.log(delta)) #El cuadrado de los errores

    chi2_AGN = chi_2_AGN_nuisance(psi, logFx, si_2)

    return chi2_AGN



#%% AGN + BAO
def Hs_to_Dl(zs, Hs, z_data):
    INT = cumtrapz(Hs**(-1), zs, initial=0)
    DL = c_luz_km*(1 + zs) * INT

    output = interp1d(zs,DL)
    return output(z_data)

def params_to_chi2_BAO_AGN(theta, params_fijos, dataset, z_data,
                            Theta_data,dTheta,Sobs,dSobs,alpha,
                            num_datasets=5, cantidad_zs=10**5,
                            l_fijo=True):
    pass
#%%

if __name__ == '__main__':
    from scipy.constants import c as c_luz #metros/segundos

    from matplotlib import pyplot as plt
    import sys
    import os
    from os.path import join as osjoin
    from pc_path import definir_path
    path_git, path_datos_global = definir_path()
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

    chi2_AGN = np.sum( ((logFx-psi)**2/si_2) - np.log(si_2)) # menos en el paper

    print(chi2_AGN)
    print(chi2_AGN/(len(z_data)-3))

    plt.figure()
    plt.xlabel('z (redshift)')
    plt.ylabel(r'$Fx$')
    plt.errorbar(z_data,psi,np.sqrt(si_2),marker='.',linestyle='')
    plt.plot(z_data,logFx,'.r')
