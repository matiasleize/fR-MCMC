"""
Functions related to BAO data.
"""
import numpy as np
from numba import jit
from scipy.interpolate import interp1d
from scipy.integrate import simpson as simpson
from scipy.integrate import quad as quad

from scipy.constants import c as c_light #meters/seconds
c_light_km = c_light / 1000 # units of km/s

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_global = os.path.dirname(path_git)
os.chdir(path_git)
os.sys.path.append('./fr_mcmc/utils/')
from LambdaCDM import H_LCDM_rad

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

#from change_of_parameters import F_H#, omega_luisa_to_CDM



#Parameters order: Omega_m,b,H_0,n

def zdrag(Omega_m,H_0,wb=0.0225):
    '''
    wb = 0.0222383 #Planck
    wb = 0.0225 #BBN
    '''
    h = H_0/100
    b1 = 0.313*(Omega_m*h**2)**(-0.419)*(1+0.607*(Omega_m*h**2)**(0.6748))
    b2 = 0.238*(Omega_m*h**2)**0.223
    zd = (1291*(Omega_m*h**2)**0.251) * (1+b1*wb**b2) /(1+0.659*(Omega_m*h**2)**0.828)
    #zd =1060.31
    return zd

def r_drag_viejo(Omega_m,H_0,wb = 0.0225, int_z=True): #wb x default tomo el de BBN.
    #rd calculation:
    h = H_0/100
    zd = zdrag(Omega_m,H_0)
    #R_bar = 31500 * wb * (2.726/2.7)**(-4)
    R_bar = wb * 10**5 / 2.473

    #Logarithmic integration
    zs_int_log = np.logspace(np.log10(zd),13,int(10**5))
    H_int_log = H_LCDM_rad(zs_int_log,Omega_m,H_0)

    integrando_log = c_light_km / (H_int_log * np.sqrt(3*(1 + R_bar*(1+zs_int_log)**(-1))))

    rd_log = simpson(integrando_log,zs_int_log)
    return rd_log

@jit
def integrand(z, Om_m_0, H_0, wb):
    R_bar = wb * 10**5 / 2.473

    Om_r = 4.18343*10**(-5) / (H_0/100)**2
    Om_Lambda = 1 - Om_m_0 - Om_r
    H = H_0 * ((Om_r * (1 + z)**4 + Om_m_0 * (1 + z)**3 + Om_Lambda) ** (1/2))
    return c_light_km/(H * (3*(1 + R_bar*(1+z)**(-1)))**(1/2))


def r_drag(Omega_m,H_0,wb = 0.0225, int_z=True): #wb of BBN as default.
    #rd calculation:
    h = H_0/100
    #zd = zdrag(Omega_m, H_0)
    #R_bar = 31500 * wb * (2.726/2.7)**(-4)
    #R_bar = wb * 10**5 / 2.473

    #zd calculation:
    zd = zdrag(Omega_m, H_0)
    # zd = 1000

    R_bar = wb * 10**5 / 2.473

    rd_log, _ = quad(lambda z: integrand(z, Omega_m, H_0, wb), zd, np.inf)

    return rd_log

# r_d computed with CLASS at each call. It requires the Python wrapper of vanilla CLASS
# (classy, https://github.com/lesgourg/class_public), not a modified version. If the environment
# variable CLASSY_PATH is set, classy is imported from there; otherwise from
# ~/Documents/PhD/code/class_public-3.3.4/classy_py310 if it exists, or from the environment.
_class_instance = None
_CLASSY_PATH = os.environ.get('CLASSY_PATH',
                              os.path.expanduser('~/Documents/PhD/code/class_public-3.3.4/classy_py310'))

def r_drag_class(Omega_m, H_0, wb=0.02218, m_ncdm=0.06):
    '''
    Sound horizon at the drag epoch r_d (Mpc) computed with CLASS: LCDM early universe,
    one massive neutrino (m_ncdm, eV) and N_eff = 3.044. Omega_m includes the neutrinos.
    Takes ~70 ms per call (the cost is the recombination in the thermodynamics module).
    '''
    global _class_instance
    if _class_instance is None:
        if os.path.isdir(_CLASSY_PATH) and _CLASSY_PATH not in os.sys.path:
            os.sys.path.insert(0, _CLASSY_PATH)
        import classy
        if 'class_public' not in classy.__file__:
            import warnings
            warnings.warn('classy imported from {}: it may not be vanilla CLASS'.format(classy.__file__))
        _class_instance = classy.Class()
    cosmo = _class_instance
    h = H_0/100
    cosmo.set({'h': h, 'omega_b': wb, 'omega_m': Omega_m*h**2,
               'N_ur': 2.0308, 'N_ncdm': 1, 'm_ncdm': m_ncdm, 'output': ''})
    try:
        cosmo.compute(['thermodynamics'])
        rd = cosmo.rs_drag()
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()
    return rd

# r_d interpolated from a grid computed with vanilla CLASS (same settings as r_drag_class).
# See fr_mcmc/source/rd_grid/make_rd_grid.py and notebooks/check_rd_grid.ipynb.
_rd_grid = None

def _load_rd_grid():
    global _rd_grid
    if _rd_grid is None:
        from scipy import ndimage
        data = np.load(os.path.join(path_git, 'fr_mcmc', 'source', 'rd_grid', 'rd_grid_class.npz'))
        # the omega_m axis is uniform in ln(omega_m)
        axes = [np.log(data['omega_m']), data['omega_b'], data['h']]
        assert all(np.allclose(np.diff(a), a[1] - a[0], rtol=1e-8, atol=0) for a in axes), 'Non-uniform grid'
        _rd_grid = {
            # cubic B-spline coefficients of ln(r_d)
            'coeffs': ndimage.spline_filter(np.log(data['rd']), order=3),
            'x0': np.array([a[0] for a in axes]),
            'dx': np.array([a[1] - a[0] for a in axes]),
            'valid': [data['valid_omega_m'], data['valid_omega_b'], data['valid_h']],
        }
    return _rd_grid

def r_drag_grid(Omega_m, H_0, wb=0.02218):
    '''
    Sound horizon at the drag epoch r_d (Mpc), interpolated (cubic spline in ln r_d) from a
    grid in (ln omega_m, omega_b, h), omega_m = Omega_m h^2, computed with vanilla CLASS: LCDM early
    universe, m_ncdm = 0.06 eV, N_eff = 3.044. Omega_m includes the neutrinos.
    Raises ValueError outside of the valid range of the grid.
    '''
    from scipy import ndimage
    grid = _load_rd_grid()
    h = H_0/100
    x = np.array([Omega_m*h**2, wb, h])
    for xi, (lo, hi), name in zip(x, grid['valid'], ['omega_m', 'omega_b', 'h']):
        # tolerance for round-off (e.g. omega_m -> Omega_m -> Omega_m h^2 at the edges)
        if not (lo*(1 - 1e-12) <= xi <= hi*(1 + 1e-12)):
            raise ValueError('{}={:.5f} outside of the r_d grid [{}, {}]'.format(name, xi, lo, hi))
    idx = ((np.array([np.log(x[0]), x[1], x[2]]) - grid['x0'])/grid['dx']).reshape(3, 1)
    return float(np.exp(ndimage.map_coordinates(grid['coeffs'], idx, order=3, mode='mirror', prefilter=False)[0]))

def Hs_to_Ds(Hs_interpol, int_inv_Hs_interpol, z_data, index):
    if index == 4: #H
        output = Hs_interpol(z_data)

    elif index == 1: #DH
        output = c_light_km * (Hs_interpol(z_data))**(-1)

    else:
        INT = int_inv_Hs_interpol(z_data)

        if index == 0: #DA
            output = (c_light_km/(1 + z_data)) * INT

        elif index == 2: #DM
            #output = (1 + z_data) * DA
            output = c_light_km * INT

        elif index == 3: #DV
            #output = (((1 +z_data) * DA)**2 * c_light_km * z_data * (Hs**(-1))) ** (1/3)
            output = c_light_km * (INT**2 * z_data * (Hs_interpol(z_data)**(-1))) ** (1/3)

    return output

def Ds_to_obs_final(Dist, rd, index):
    if index == 4: #H
        output = Dist*rd
    else: #Every distances
        output = Dist/rd
    return output

#%%
if __name__ == '__main__':
    '''TODO: update the example with these functions'''
    import os
    import git
    path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
    path_global = os.path.dirname(path_git)
    os.chdir(path_git)
    sys.path.append('./fr_mcmc/utils/')
    from data import read_data_BAO
    #%% BAO
    os.chdir(path_git+'/fr_mcmc/source/BAO_legacy_1')
    dataset_BAO = []
    file_BAO = ['BAO_data_da.txt','BAO_data_dh.txt','BAO_data_dm.txt',
                    'BAO_data_dv.txt','BAO_data_H.txt']
    for i in range(5):
        aux = read_data_BAO(file_BAO[i])
        dataset_BAO.append(aux)

    [Omega_m,b,H_0] = [0.28,1,66.012]
    theta = [Omega_m,b,H_0]
    params_to_chi2_BAO(theta,1, dataset_BAO,model='EXP')
    r_drag(Omega_m,H_0,wb = 0.0225)
