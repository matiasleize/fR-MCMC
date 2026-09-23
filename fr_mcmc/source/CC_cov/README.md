# Cosmic chronometers covariance (Moresco et al. 2020)

Copied from https://gitlab.com/mmoresco/CCcovariance (commit 8814133, 2021-03-19), `data/`:

- `HzTable_MM_BC03.dat`: the 15 BC03 H(z) measurements of Moresco et al. (2012, 2015, 2016),
  with the statistical + metallicity error (diagonal part of the covariance).
- `data_MM20.dat`: relative systematic errors (%) vs z: IMF, stellar library, SPS model and
  SPS model without the outlier (`mod_ooo`).

`fr_mcmc/utils/CC_cov.py` builds the covariance of the 33 points of `../CC/chronometers_data.txt`
(see `../CC/README.md` for the table and its references): the 15 Moresco points get
Cov = diag(err^2) + Cov_IMF + Cov_SPS(ooo), exactly as `cov_mat = cov_mat_spsooo + cov_mat_imf +
cov_mat_diag` in the example `examples/CC_covariance.ipynb` of the repo; the other 18 keep their
diagonal error and are uncorrelated, so the matrix is block diagonal.

The covariance is defined **only** for the Moresco measurements: the two files above cover those
15 points, and `examples/CC_covariance_components.ipynb` decomposes the covariance "for the data
in [1], [2], [3]", i.e. Moresco et al. 2012, 2015 and 2016. There is no published SPS/IMF budget
for the other 18 points.

Validated against the reference fit of `examples/CC_fit.ipynb`, which uses those 15 points alone:
we get H0 = 66.1 (+3.9/-4.0) without systematics and 65.9 +- 5.6 with the full covariance, against
66.2 (+3.8/-3.9) and 66 (+5.5/-5.6) published there.

Cite Moresco et al. (2020), ApJ 898, 82, and the original measurements.
