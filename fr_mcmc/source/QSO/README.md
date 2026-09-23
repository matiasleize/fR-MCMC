# QSO distance moduli (Benetti et al. 2025)

`qso_benetti2025.txt`: copy of `MCMC/Cobaya/qso_data/lcparam_full_long_zhel.txt` from
https://github.com/QSO-Cosmology/QSO-for-Cosmology (commit 6500a76, 2025-09-04).

- 2014 quasars of Lusso et al. (2020), A&A 642, A150: z > 0.7 cut for the sources with
  photometric UV and 3-sigma clipping.
- Pantheon-like format; the columns used are `zcmb` (= `zhel`), `mb` (distance modulus)
  and `dmb` (its error, including the intrinsic dispersion). The released covariance
  matrix is zero, so it is not copied.
- X-UV relation fixed to gamma = 0.591 +- 0.011, beta = -31.475 +- 0.008,
  delta = 0.209 +- 0.004 (cosmographic calibration with Pantheon).
- The likelihood (`fr_mcmc/utils/QSO.py`) adds a calibration offset k, marginalized
  analytically by default.

If you use them, cite Benetti et al. (2025), Phys. Dark Univ. 49, 101983 (arXiv:2506.21477)
and Lusso et al. (2020).
