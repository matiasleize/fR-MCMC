import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math

#%%
def Taylor_ST(z,omega_m,b,H0):
    '''
    Parametrization of the Hubble parameter H(z) given by Basilakos et al. for the Starobinsky model.
    The number of e-folds is not the same as the scale factor (N!=a). They are related according to: N=ln(a)=-ln(1+z)
    '''
    omega_r=0
    N = sym.Symbol('N')
    St_tay = H0*(1+(-1+math.e**(-3*N))*omega_m+(-1+math.e**(-4*N))*omega_r+
    (b**2*math.e**(5*N)*(-1+omega_m+omega_r)**3*(-37*math.e**N*omega_m**2-40*omega_m*omega_r+
    32*math.e**(4*N)*omega_m*(-1+omega_m+omega_r)+
    16*math.e**(3*N)*omega_r*(-1+omega_m+omega_r)+
    32*math.e**(7*N)*(-1+omega_m+omega_r)**2))/(omega_m-
    4*math.e**(3*N)*(-1+omega_m+omega_r))**4+
    b**4*math.e**(11*N)*(-1+omega_m+omega_r)**5*(123*math.e**N*omega_m**6+128*omega_m**5*omega_r-
    82748*math.e**(4*N)*omega_m**5*(-1+omega_m+omega_r)-
    160440*math.e**(3*N)*omega_m**4*omega_r*(-1+omega_m+omega_r)-
    77760*math.e**(2*N)*omega_m**3*omega_r**2*(-1+omega_m+omega_r)-
    44552*math.e**(7*N)*omega_m**4*(-1+omega_m+omega_r)**2-
    277568*math.e**(6*N)*omega_m**3*omega_r*(-1+omega_m+omega_r)**2-
    228096*math.e**(5*N)*omega_m**2*omega_r**2*(-1+omega_m+omega_r)**2+
    289024*math.e**(10*N)*omega_m**3*(-1+omega_m+omega_r)**3+
    310144*math.e**(9*N)*omega_m**2*omega_r*(-1+omega_m+omega_r)**3-
    82944*math.e**(8*N)*omega_m*omega_r**2*(-1+omega_m+omega_r)**3-
    234880*math.e**(13*N)*omega_m**2*(-1+omega_m+omega_r)**4-
    6144*math.e**(12*N)*omega_m*omega_r*(-1+omega_m+omega_r)**4+
    63488*math.e**(16*N)*omega_m*(-1+omega_m+omega_r)**5+
    20480*math.e**(15*N)*omega_r*(-1+omega_m+omega_r)**5+
    20480*math.e**(19*N)*(-1+omega_m+omega_r)**6)/(omega_m-
    4*math.e**(3*N)*(-1+omega_m+omega_r))**10)**(0.5)

    func = lambdify(N, St_tay,'numpy') # returns a numpy-ready function

    N_dato = -np.log(1+z)
    numpy_array_of_results = func(N_dato)
    return numpy_array_of_results

def Taylor_HS(z,omega_m,b,H0):
    '''
    Parametrization of the Hubble parameter H(z) given by Basilakos et al. for the Starobinsky model.
    The number of e-folds is not the same as the scale factor (N!=a). They are related according to: N=ln(a)=-ln(1+z)
    '''
    omega_r=0
    N = sym.Symbol('N')
    Hs_tay=N
    Hs_tay = (H0**2*(1/((omega_m-4*math.e**(3*N)*(omega_m+omega_r-1))**8) * (b**2)*(math.e**(5*N))*((omega_m+omega_r-1)**3) *(37*math.e**(N)*omega_m**6-4656*math.e**(4*N)*omega_m**5*(omega_m+omega_r-1)-
    7452*math.e**(7*N)*omega_m**4*(omega_m+omega_r-1)**2-
    8692*math.e**(3*N)*omega_m**4*omega_r*(omega_m+omega_r-1)-
    4032*math.e**(2*N)*omega_m**3*omega_r**2*(omega_m+omega_r-1)+
    25408*math.e**(10*N)*omega_m**3*(omega_m+omega_r-1)**3-
    25728*math.e**(6*N)*omega_m**3*omega_r*(omega_m+omega_r-1)**2-
    17856*math.e**(5*N)*omega_m**2*omega_r**2*(omega_m+omega_r-1)**2-
    22848*math.e**(13*N)*omega_m**2*(omega_m+omega_r-1)**4+
    22016*math.e**(9*N)*omega_m**2*omega_r*(omega_m+omega_r-1)**3-
    9216*math.e**(8*N)*omega_m*omega_r**2*(omega_m+omega_r-1)**3+
    9216*math.e**(16*N)*omega_m*(omega_m+omega_r-1)**5-
    2048*math.e**(12*N)*omega_m*omega_r*(omega_m+omega_r-1)**4+
    1024*math.e**(19*N)*(omega_m+omega_r-1)**6+
    3072*math.e**(15*N)*omega_r*(omega_m+omega_r-1)**5+40*omega_m**5*omega_r)+
    ((2*b*math.e**(2*N)*(omega_m+omega_r-1)**2*(-6*math.e**(N)*omega_m**2+3*math.e**(4*N)*omega_m*(omega_m+omega_r-1)+12*math.e**(7*N)*(omega_m+omega_r-1)**2+4*math.e**(3*N)*omega_r*(omega_m+omega_r-1)-7*omega_m*omega_r))/(4*math.e**(3*N)*(omega_m+omega_r-1)-omega_m)**3)+
    (math.e**(-3*N)-1)*omega_m + (math.e**(-4*N)-1)*omega_r+1))**(0.5)

    func = lambdify(N, Hs_tay,'numpy') # returns a numpy-ready function

    N_dato = -np.log(1+z)
    numpy_array_of_results = func(N_dato)
    return numpy_array_of_results
    
#%%
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    #%matplotlib qt5

    import os
    import git
    path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
    path_global = os.path.dirname(path_git)
    os.chdir(path_git)
    sys.path.append('./fr_mcmc/utils/')
    from LambdaCDM import H_LCDM

    omega_m = 0.24
    b = 0.2
    H0 = 73.48
    zs = np.linspace(0,1,10000);
    H_LCDM = H_LCDM(zs,omega_m,H0)

    # Hu-Sawicki
    H_HS_taylor = Taylor_HS(zs,omega_m,b,H0)

    # Starobinsky
    H_ST_taylor = Taylor_ST(zs,omega_m,b,H0)

    plt.close()
    plt.figure()
    plt.grid(True)
    plt.xlabel('z (redshift)')
    plt.ylabel('H(z)')
    plt.plot(zs,H_LCDM/H0,label='LCDM')
    plt.plot(zs,H_HS_taylor/H0,'-.',label='HS')
    plt.plot(zs,H_ST_taylor/H0,'-.',label='ST')
    plt.legend(loc='best')
    plt.show()
