"""
Cosmic chronometers with the covariance matrix of Moresco et al. (2020), ApJ 898, 82.

Independent alternative to read_data_chronometers + the diagonal CC chi square of
chi_square.py (which are not modified). It uses the same 30 H(z) points of
fr_mcmc/source/CC/chronometers_data.txt:
- the 15 BC03 measurements of Moresco et al. (2012, 2015, 2016) are replaced by the values of
  fr_mcmc/source/CC_cov/HzTable_MM_BC03.dat (same measurements, with the statistical +
  metallicity error only), and get the covariance of the example of
  https://gitlab.com/mmoresco/CCcovariance (CC_covariance.ipynb):
      Cov = diag(err^2) + Cov_IMF + Cov_SPS (without the outlier model, 'mod_ooo'),
  where Cov_X[i,j] = H_i f_X(z_i) H_j f_X(z_j), with f_X the relative systematic error of
  data_MM20.dat interpolated in z (np.interp, constant beyond z = 1.475);
- the other 15 points keep their diagonal error, uncorrelated with the rest.
"""
import numpy as np
from numpy.linalg import inv


def read_data_CC_cov(file_CC, file_MM, file_MM_syst, tol_z=1e-4):
    '''
    file_CC: current CC table (z, H, error), e.g. CC/chronometers_data.txt.
    file_MM: HzTable_MM_BC03.dat (z, H, error of Moresco et al. BC03 measurements).
    file_MM_syst: data_MM20.dat (relative systematic errors in % vs z).
    tol_z: tolerance to identify the Moresco points inside file_CC.

    Return z, H, Cinv (sorted in z) and the boolean mask of the Moresco points.
    '''
    z, H, eH = np.loadtxt(file_CC, usecols=(0, 1, 2), unpack=True)
    z_MM, H_MM, eH_MM = np.genfromtxt(file_MM, comments='#', usecols=(0, 1, 2), unpack=True,
                                      delimiter=',')
    z_mod, imf, _, _, spsooo = np.genfromtxt(file_MM_syst, comments='#', usecols=(0, 1, 2, 3, 4),
                                             unpack=True)

    is_MM = np.zeros(len(z), dtype=bool)
    for zi, Hi, ei in zip(z_MM, H_MM, eH_MM):
        match = np.where(np.abs(z - zi) < tol_z)[0]
        if len(match) != 1:
            raise ValueError('Moresco point z = {} found {} times in {}'.format(zi, len(match), file_CC))
        j = match[0]
        z[j], H[j], eH[j] = zi, Hi, ei
        is_MM[j] = True

    inds = z.argsort()
    z, H, eH, is_MM = z[inds], H[inds], eH[inds], is_MM[inds]

    Cov = np.diag(eH**2)
    for f in [imf, spsooo]:
        s = np.where(is_MM, H * np.interp(z, z_mod, f) / 100, 0)
        Cov += np.outer(s, s)
    return z, H, inv(Cov), is_MM


def chi2_CC_cov(Hs_interp, dataset_CC_cov):
    '''
    Chi square of the CC data with covariance.
    Hs_interp: interpolation of H(z). dataset_CC_cov: output of read_data_CC_cov.
    '''
    z, H, Cinv, _ = dataset_CC_cov
    delta = Hs_interp(z) - H
    return delta @ Cinv @ delta
