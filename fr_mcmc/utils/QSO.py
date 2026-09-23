"""
Quasar (QSO) likelihood of Benetti et al. (2025), arXiv:2506.21477.

Independent alternative to the AGN likelihood of chi_square.py (which is not modified).
The data (fr_mcmc/source/QSO/qso_benetti2025.txt, from
https://github.com/QSO-Cosmology/QSO-for-Cosmology) are the distance moduli of the 2014
quasars of Lusso et al. (2020) that remain after the z > 0.7 cut for the sources with
photometric UV and a 3-sigma clipping. They are obtained from the X-UV relation with
gamma = 0.591 and beta = -31.475, calibrated against Pantheon with a cosmographic fit, and
their errors already include the intrinsic dispersion, delta_DM = 5 delta / (2 |gamma - 1|)
with delta = 0.209. The covariance matrix released with the data is zero, so the errors are
uncorrelated.

The model is mu_obs = mu_th + k, with k a calibration nuisance parameter. By default k is
marginalized analytically with a flat prior, so the likelihood depends only on the shape of
H(z)/H_0 (H_0 drops out).

Usage: log_likelihood() of this module has the same arguments as chi_square.log_likelihood
plus dataset_QSO (and optionally k_QSO):
    from QSO import read_data_QSO, log_likelihood
    dataset_QSO = read_data_QSO('qso_benetti2025.txt')
    log_likelihood(theta, fixed_params, index=42, model='HS', use_c=True,
                   dataset_SN=..., dataset_CC=..., dataset_QSO=dataset_QSO)
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumulative_trapezoid

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')

import chi_square
from chi_square import all_parameters
from LambdaCDM import H_LCDM
from supernovae import aparent_magnitude_th


def read_data_QSO(file_QSO):
    '''
    Read the QSO distance moduli (Pantheon-like format: name zcmb zhel dz mb dmb ...).
    Return the arrays sorted in redshift: z, mu, error of mu.
    '''
    z, mu, emu = np.loadtxt(file_QSO, usecols=(1, 4, 5), unpack=True)
    inds = z.argsort()
    return z[inds], mu[inds], emu[inds]


def chi2_QSO(int_inv_Hs_interp, dataset_QSO, k=None):
    '''
    Chi square of the QSO data.

    int_inv_Hs_interp: interpolation of int_0^z dz'/H(z') (as in chi_square.params_to_chi2).
    dataset_QSO: output of read_data_QSO.
    k (float or None): calibration nuisance parameter. If None, it is marginalized
        analytically with a flat prior: chi2 = A - B^2/C, dropping the constant ln(C/2pi).
    '''
    z, mu, emu = dataset_QSO
    muth = aparent_magnitude_th(int_inv_Hs_interp, z, z)
    w = emu**(-2)
    delta = mu - muth
    if k is not None:
        return np.sum((delta - k)**2 * w)
    A = np.sum(delta**2 * w)
    B = np.sum(delta * w)
    C = np.sum(w)
    return A - B**2 / C


def params_to_chi2_QSO(theta, fixed_params, dataset_QSO, index=0, model='HS', n=1,
                       num_z_points=int(10**5), all_analytic=False, use_c=False, k=None):
    '''
    Given the free parameters of the model, return the chi square of the QSO data.
    Same conventions for theta, fixed_params and index as chi_square.params_to_chi2
    (r_d and Mabs are ignored). The H(z) computation is the same as there.
    '''
    if model == 'LCDM':
        [_, _, omega_m, H_0] = all_parameters(theta, fixed_params, model, index)
        zs_model = np.linspace(0, 10, num_z_points)
        Hs_model = H_LCDM(zs_model, omega_m, H_0)
    else:
        [_, _, omega_m, b, H_0] = all_parameters(theta, fixed_params, model, index)
        Hubble = chi_square.Hubble_th_c if use_c else chi_square.Hubble_th
        try:
            zs_model, Hs_model = Hubble([omega_m, b, H_0], n=n, model=model,
                                        z_min=0, z_max=10, num_z_points=num_z_points,
                                        all_analytic=all_analytic)
        except Exception as e:
            # If integration fails, reject the step
            return np.inf

    int_inv_Hs = cumulative_trapezoid(Hs_model**(-1), zs_model, initial=0)
    int_inv_Hs_interp = interp1d(zs_model, int_inv_Hs)
    return chi2_QSO(int_inv_Hs_interp, dataset_QSO, k=k)


def _last_call_cache(func):
    '''
    Keep the last result of the Hubble integrator, so that chi_square.params_to_chi2 and
    params_to_chi2_QSO do not integrate twice the same H(z) for the same step.
    '''
    cache = {}
    def wrapper(physical_params, **kwargs):
        key = (tuple(physical_params), tuple(sorted(kwargs.items())))
        if cache.get('key') != key:
            cache['key'] = key
            cache['value'] = func(physical_params, **kwargs)
        return cache['value']
    wrapper.__wrapped__ = func
    return wrapper

# Applied to the names that chi_square looks up at each call (the file is not modified).
# Only the results are cached, so exceptions (rejected steps) are re-raised every time.
if not hasattr(chi_square.Hubble_th, '__wrapped__'):
    chi_square.Hubble_th = _last_call_cache(chi_square.Hubble_th)
if not hasattr(chi_square.Hubble_th_c, '__wrapped__'):
    chi_square.Hubble_th_c = _last_call_cache(chi_square.Hubble_th_c)


def log_likelihood(*args, dataset_QSO=None, k_QSO=None, **kargs):
    '''
    chi_square.log_likelihood plus the QSO term. Takes the same arguments as
    chi_square.log_likelihood, plus dataset_QSO (output of read_data_QSO) and k_QSO
    (None: k marginalized analytically).
    '''
    ll = chi_square.log_likelihood(*args, **kargs)
    if dataset_QSO is None or not np.isfinite(ll):
        return ll

    theta, fixed_params = args[:2]
    qso_kargs = {key: kargs[key] for key in ['index', 'model', 'n', 'num_z_points',
                                             'all_analytic', 'use_c'] if key in kargs}
    return ll - 0.5 * params_to_chi2_QSO(theta, fixed_params, dataset_QSO, k=k_QSO,
                                         **qso_kargs)

#%%
if __name__ == '__main__':
    os.chdir(path_git + '/fr_mcmc/source/QSO')
    dataset_QSO = read_data_QSO('qso_benetti2025.txt')
    print('N QSO:', len(dataset_QSO[0]))

    for omega_m in [0.2, 0.3, 0.4, 0.5]:
        print('LCDM Om = {}: chi2 = {:.2f}'.format(omega_m,
              params_to_chi2_QSO([omega_m, 70], [-19.3, 147], dataset_QSO, index=21,
                                 model='LCDM')))
