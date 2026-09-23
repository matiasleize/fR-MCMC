"""
Log likelihood of chi_square.py extended with the likelihoods that live in separate modules
(chi_square.py is not modified):
- dataset_QSO: quasars of Benetti et al. (2025) (QSO.py), k marginalized unless k_QSO is given.
- dataset_CC_cov: cosmic chronometers with the covariance of Moresco et al. (2020) (CC_cov.py).

H(z) is computed once per step: importing QSO keeps the last result of the integrators used
by chi_square, and the extra terms reuse it.

Usage: same arguments as chi_square.log_likelihood plus dataset_QSO, dataset_CC_cov, k_QSO.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumulative_trapezoid

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')

import chi_square
import QSO  # also applies the cache of the Hubble integrators
from chi_square import all_parameters
from LambdaCDM import H_LCDM
from CC_cov import chi2_CC_cov


def Hubble_model(theta, fixed_params, index=0, model='HS', n=1, num_z_points=int(10**5),
                 all_analytic=False, use_c=False):
    '''
    H(z) with the same parameters and settings as chi_square.params_to_chi2.
    Raise the exception of the integrator if it fails.
    '''
    if model == 'LCDM':
        [_, _, omega_m, H_0] = all_parameters(theta, fixed_params, model, index)
        zs_model = np.linspace(0, 10, num_z_points)
        return zs_model, H_LCDM(zs_model, omega_m, H_0)
    [_, _, omega_m, b, H_0] = all_parameters(theta, fixed_params, model, index)
    Hubble = chi_square.Hubble_th_c if use_c else chi_square.Hubble_th
    return Hubble([omega_m, b, H_0], n=n, model=model, z_min=0, z_max=10,
                  num_z_points=num_z_points, all_analytic=all_analytic)


def log_likelihood(*args, dataset_QSO=None, dataset_CC_cov=None, k_QSO=None, **kargs):
    '''
    chi_square.log_likelihood plus the QSO and CC (with covariance) terms.
    '''
    if (dataset_QSO is not None or dataset_CC_cov is not None) and kargs.get('use_ml', False):
        raise NotImplementedError('QSO and CC_cov do not support use_ml=True')

    ll = chi_square.log_likelihood(*args, **kargs)
    if (dataset_QSO is None and dataset_CC_cov is None) or not np.isfinite(ll):
        return ll

    theta, fixed_params = args[:2]
    h_kargs = {key: kargs[key] for key in ['index', 'model', 'n', 'num_z_points',
                                           'all_analytic', 'use_c'] if key in kargs}
    try:
        zs_model, Hs_model = Hubble_model(theta, fixed_params, **h_kargs)
    except Exception as e:
        # If integration fails, reject the step
        return -np.inf

    chi2 = 0
    if dataset_QSO is not None:
        int_inv_Hs = cumulative_trapezoid(Hs_model**(-1), zs_model, initial=0)
        chi2 += QSO.chi2_QSO(interp1d(zs_model, int_inv_Hs), dataset_QSO, k=k_QSO)
    if dataset_CC_cov is not None:
        chi2 += chi2_CC_cov(interp1d(zs_model, Hs_model), dataset_CC_cov)
    return ll - 0.5 * chi2
