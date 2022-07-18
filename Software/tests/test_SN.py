
import numpy as np

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_datos_global = os.path.dirname(path_git)

os.chdir(path_git)
os.sys.path.append('./Software/utils/')

from LambdaCDM import H_LCDM

from scipy.integrate import cumtrapz as cumtrapz
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #meters/seconds
c_luz_km = c_luz/1000; #kilometers/seconds
#Parameters order: Mabs,omega_m,b,H_0,n

def testeo_supernovae(theta, params_fijos, zcmb, zhel, Cinv,
                      mb0, x1, color, hmass, cantidad_zs=int(10**5),
                      model='HS',lcdm=False):
    '''
    DEPRECATED
    Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovae. 1 parámetro fijo y 4 variables.
    '''

    if lcdm == True:
        if len(theta)==5:
            [Mabs,omega_m,alpha,beta,gamma] = theta
            [H_0,n] = params_fijos
        if len(theta)==4:
            [Mabs,alpha,beta,gamma] = theta
            [omega_m,H_0,n] = params_fijos

        zs = np.linspace(0,3,cantidad_zs)
        H_modelo = H_LCDM(zs, omega_m, H_0)

    else:
        if len(theta)==7:
            [Mabs,omega_m,b,H_0,alpha,beta,gamma] = theta
            n = params_fijos
        elif len(theta)==6:
            [Mabs,omega_m,b,alpha,beta,gamma] = theta
            [H_0,n] = params_fijos
        elif len(theta)==4:
            [Mabs,alpha,beta,gamma] = theta
            [omega_m,b,H_0,n] = params_fijos

        if (0 <= b < 0.2):
            zs = np.linspace(0,3,cantidad_zs)
            if model=='HS':
                H_modelo = Taylor_HS(zs, omega_m, b, H_0)
            else:
                H_modelo = Taylor_ST(zs, omega_m, b, H_0)

        else:
            params_fisicos = [omega_m,b,H_0]
            zs, H_modelo = integrator(params_fisicos, n, cantidad_zs=cantidad_zs, model=model)

    alpha_0 = 0.154
    beta_0 = 3.02
    gamma_0 = 0.053
    mstep0 = 10.13
    #tau0 = 0.001

    #DeltaM_0=gamma_0*np.power((1.+np.exp((mstep0-hmass)/tau0)),-1)
    #DeltaM=gamma*np.power((1.+np.exp((mstep0-hmass)/tau0)),-1)
    #DeltaM_0 = gamma_0 * np.heaviside(hmass-mstep0, 1)
    #DeltaM = gamma * np.heaviside(hmass-mstep0, 1)

    muobs =  mb0 - Mabs + x1 * (alpha-alpha_0) - color * (beta-beta_0) + np.heaviside(hmass-mstep0, 1) * (gamma-gamma_0)

    muth = magn_aparente_teorica(zs,H_modelo,zcmb,zhel)

    deltamu = muobs - muth
    transp = np.transpose(deltamu)
    aux = np.dot(Cinv,deltamu)
    chi2 = np.dot(transp,aux)

    return chi2
