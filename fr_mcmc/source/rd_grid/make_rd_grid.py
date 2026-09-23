'''
Grid of the sound horizon at the drag epoch r_d (Mpc) computed with vanilla CLASS, as a
function of (omega_m, omega_b, h), with omega_m = Omega_m h^2 (including massive neutrinos)
and omega_b = Omega_b h^2. Used by BAO.r_drag_grid (see notebooks/check_rd_grid.ipynb).
The omega_m axis is uniform in ln(omega_m): r_d has a large curvature at small omega_m
(omega_cdm -> 0), where a linear spacing gives interpolation errors ~1e-2 %.

Settings: LCDM, one massive neutrino with m_ncdm = 0.06 eV, N_ur = 2.0308 (N_eff = 3.044),
default CLASS precision (HyRec). The grid has two extra nodes on each side of the valid
range, so that the cubic spline is accurate up to the edges of the valid range.

Usage (needs the Python wrapper of vanilla CLASS):
    python make_rd_grid.py
classy is taken from the environment; CLASSY_PATH=/path/to/classy overrides it.
'''
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

CLASSY_PATH = os.environ.get('CLASSY_PATH')
if CLASSY_PATH:
    sys.path.insert(0, CLASSY_PATH)
import classy  # noqa: E402

M_NCDM = 0.06
N_UR = 2.0308
MARGIN = 2  # extra nodes on each side of the valid range

# Valid ranges and spacing of each axis (omega_m: spacing in ln(omega_m))
AXES = {
    'omega_m': (0.035, 0.600, 0.005),  # upper edge: Omega_m <= 0.6 and h <= 1
    'omega_b': (0.019, 0.026, 0.00025),
    'h': (0.50, 1.00, 0.1),
}
LOG_AXES = ['omega_m']


def make_axis(lo, hi, step, log=False):
    if log:
        n = int(np.ceil(np.log(hi / lo) / step))
        return np.exp(np.log(lo) + step * np.arange(-MARGIN, n + MARGIN + 1))
    n = int(round((hi - lo) / step))
    return lo + step * np.arange(-MARGIN, n + MARGIN + 1)


def rd_class(params):
    omega_m, omega_b, h = params
    cosmo = classy.Class()
    cosmo.set({'h': h, 'omega_b': omega_b, 'omega_m': omega_m,
               'N_ur': N_UR, 'N_ncdm': 1, 'm_ncdm': M_NCDM, 'output': ''})
    try:
        cosmo.compute(['thermodynamics'])
        return cosmo.rs_drag()
    except Exception:
        return np.nan
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


if __name__ == '__main__':
    axes = {k: make_axis(*v, log=(k in LOG_AXES)) for k, v in AXES.items()}
    grid = np.array(np.meshgrid(axes['omega_m'], axes['omega_b'], axes['h'], indexing='ij'))
    points = grid.reshape(3, -1).T
    print('classy:', classy.__file__)
    print('Grid shape:', grid.shape[1:], ' points:', len(points))

    t = time.time()
    with Pool(int(os.environ.get('NPROC', 20))) as pool:
        rd = np.array(pool.map(rd_class, points, chunksize=64))
    print('Done in {:.1f} min'.format((time.time() - t) / 60))
    rd = rd.reshape(grid.shape[1:])
    assert np.all(np.isfinite(rd)), 'CLASS failed at some grid points'

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rd_grid_class.npz')
    np.savez(out, omega_m=axes['omega_m'], omega_b=axes['omega_b'], h=axes['h'], rd=rd,
             # valid range: interior nodes (the log axis can end slightly above AXES['omega_m'][1])
             **{'valid_' + k: np.array([axes[k][MARGIN], axes[k][-MARGIN - 1]]) for k in axes},
             log_omega_m=True, m_ncdm=M_NCDM, N_ur=N_UR,
             class_version=classy.Class().version() if hasattr(classy.Class(), 'version') else 'unknown',
             classy_path=classy.__file__)
    print('Saved', out)
