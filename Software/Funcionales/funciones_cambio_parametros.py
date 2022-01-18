'''
Change of parameters used in the numeric integrarion
'''

from scipy.constants import c as c_luz # meters/seconds
c_luz_km = c_luz/1000

# Parameters order: omega_m, b, H_0, n

def params_fisicos_to_modelo_HS(omega_m, b, n=1):
    '''
    Convert physical parameters (omega_m, b)
    into Hu-Sawicki model parameters c1 y c2. This transformation doesn't depend on H0!
    '''

    aux = c_luz_km**2 * omega_m / (7800 * (8315)**2 * (1-omega_m)) #B en la tesis
    if n==1:
        c_1 =  2/b
        c_2 = 2*aux/b

    else:
        c_2 =  (2*aux/b) ** n
        c_1 =  c_2/aux
    return c_1, c_2

def params_fisicos_to_modelo_ST(omega_m, b, H0):
    '''
    Convert physical parameters (omega_m, b, H0)
    into Starobinsky model parameters (Lambda, Rs). This transformation doesn't depend on n!
    '''

    Lambda = 3 * (H0/c_luz_km)**2 * (1-omega_m) #Cosmological constant
    lamb = 2/b
    Rs = Lambda * b
    return lamb, Rs


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

    #%% aux es B en la tesis
    aux = c_luz_km**2 * omega_m_true / (7800 * (8315)**2 * (1-omega_m_true)) 
    print(aux)

    ## Starobinsky
    lamb,Rs = params_fisicos_to_modelo_ST(omega_m_true, b_true, H_0)
    print(lamb,Rs)
