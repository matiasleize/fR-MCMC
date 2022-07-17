'''
Change of parameters used in the numeric integrarion
'''

from scipy.constants import c as c_luz # meters/seconds
c_luz_km = c_luz/1000

# Parameters order: omega_m, b, H_0, n

def params_fisicos_to_modelo_HS(omega_m, b):
    '''
    Convert physical parameters (omega_m, b)
    into Hu-Sawicki model parameters c1 y c2. This transformation doesn't depend on H0!
    '''

    aux = c_luz_km**2 * omega_m / (7800 * (8315)**2 * (1-omega_m)) #B en la tesis
    c_1 =  2/b
    c_2 = 2*aux/b

    return c_1, c_2

#%%
if __name__ == '__main__':
    ## Hu-Sawicki
    omega_m_true = 0.24
    b_true = 2
    H_0=73.48

    c1,c2 = params_fisicos_to_modelo_HS(omega_m_true, b_true,n=1)
    print(c1,c2)

    #c1_true = 1
    #c2_true = 1/19
    print(1/19)

    #%%
    aux = c_luz_km**2 * omega_m_true / (7800 * (8315)**2 * (1-omega_m_true)) 
    print(aux)