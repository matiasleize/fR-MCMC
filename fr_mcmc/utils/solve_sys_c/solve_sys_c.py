'''
Python wrapper (ctypes) of the C implementation of solve_sys.py (fr_ode.c).

Hubble_th is called in the same way as solve_sys.Hubble_th. The shared library
is compiled automatically the first time (or whenever fr_ode.c/fr_ode.h change).
It can also be compiled by hand with "make" inside this folder.
'''
import ctypes
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_DIR) not in sys.path:
    sys.path.append(os.path.dirname(_DIR))
import constants  # noqa: E402  (radiation: constants.Omega_r, read at each call)
_SOURCES = [os.path.join(_DIR, 'fr_ode.c'), os.path.join(_DIR, 'fr_ode.h')]
_LIB_PATH = os.path.join(_DIR, 'libfr_ode.so')

_MODELS = {'LCDM': 0, 'HS': 1, 'ST': 2, 'EXP': 3}
_METHODS = {None: -1, 'RK45': 0, 'ROS23': 1}
_BRANCHES = {0: 'LCDM', 1: 'Taylor', 2: 'numeric'}

# Default redshift of the initial conditions of HS (same as solve_sys.Z_IC_HS)
Z_IC_HS = 30
_ERRORS = {
    1: 'Required step size is less than spacing between numbers.',
    2: 'Maximum number of steps exceeded.',
    3: 'Non-finite value in the solution.',
    4: 'Invalid arguments.',
    5: 'Memory allocation failed.',
}


def build(force=False):
    '''
    Compile fr_ode.c into libfr_ode.so if it does not exist or is outdated.
    '''
    if (not force and os.path.exists(_LIB_PATH) and
            os.path.getmtime(_LIB_PATH) >= max(os.path.getmtime(s) for s in _SOURCES)):
        return _LIB_PATH

    cc = os.environ.get('CC', 'cc')
    # Compile to a temporary file and rename it, so that concurrent imports are safe.
    fd, tmp_path = tempfile.mkstemp(suffix='.so', dir=_DIR)
    os.close(fd)
    cmd = [cc, '-O3', '-std=c99', '-shared', '-fPIC', '-o', tmp_path, _SOURCES[0], '-lm']
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.replace(tmp_path, _LIB_PATH)
    except (OSError, subprocess.CalledProcessError) as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        stderr = getattr(e, 'stderr', b'') or b''
        raise RuntimeError('Could not compile {}:\n{}\n{}'.format(
            _SOURCES[0], ' '.join(cmd), stderr.decode(errors='replace')))
    return _LIB_PATH


def _load_library():
    lib = ctypes.CDLL(build())
    array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    lib.fr_hubble.restype = ctypes.c_int
    lib.fr_hubble.argtypes = [
        ctypes.c_int,                                   # model
        ctypes.c_double, ctypes.c_double, ctypes.c_double,  # omega_m, b, H0
        ctypes.c_double,                                # omega_r
        ctypes.c_int,                                   # n
        ctypes.c_double, ctypes.c_double, ctypes.c_int, # z_min, z_max, num_z_points
        ctypes.c_double,                                # z_ic
        ctypes.c_double, ctypes.c_int, ctypes.c_int,    # b_crit, all_analytic, force_numeric
        ctypes.c_double,                                # epsilon
        ctypes.c_int, ctypes.c_double, ctypes.c_double, # method, rtol, atol
        array, array,                                   # zs, Hs
        ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_int),  # nfev, branch
    ]
    return lib


_lib = _load_library()


def _call(physical_params, model, n, z_min, z_max, num_z_points, b_crit, all_analytic,
          force_numeric, epsilon, method, rtol, atol, verbose, z_ic=None):
    if model not in _MODELS:
        raise ValueError('Invalid model specified. Choose from "LCDM", "EXP", "HS" or "ST".')
    if method not in _METHODS:
        raise ValueError('Invalid method {!r}. Choose from "RK45" or "ROS23".'.format(method))
    if (model in ('HS', 'ST')) and n != 1 and not force_numeric:
        raise ValueError('Not a valid Taylor!')

    # Default tolerances. EXP uses ROS23 (order 2), which needs a smaller rtol to reach
    # the accuracy of RK45/Radau with rtol=1e-8 (error in H(z) ~2e-8 instead of ~5e-7).
    if rtol is None:
        rtol = 1e-10 if model == 'EXP' else 1e-8
    if atol is None:
        atol = 1e-12 if model == 'EXP' else 1e-10
    # Redshift of the initial conditions of HS/ST (same default as solve_sys.integrator)
    if z_ic is None:
        z_ic = Z_IC_HS if model == 'HS' else z_max

    omega_m, b, H0 = physical_params
    omega_r = constants.Omega_r(H0)  # photons + massless neutrinos (0 if constants.RADIATION is False)
    num_z_points = int(num_z_points)
    zs = np.empty(num_z_points)
    Hs = np.empty(num_z_points)
    nfev = ctypes.c_long(0)
    branch = ctypes.c_int(-1)

    t1 = time.time()
    status = _lib.fr_hubble(_MODELS[model], float(omega_m), float(b), float(H0), float(omega_r), int(n),
                            float(z_min), float(z_max), num_z_points, float(z_ic),
                            float(b_crit), int(bool(all_analytic)), int(bool(force_numeric)),
                            float(epsilon), _METHODS[method], float(rtol), float(atol),
                            zs, Hs, ctypes.byref(nfev), ctypes.byref(branch))
    t2 = time.time()

    if status != 0:
        raise RuntimeError('Integration failed for model {} with params {}: {}'.format(
            model, list(physical_params), _ERRORS.get(status, 'error {}'.format(status))))
    if verbose:
        print('Duration: {:.3e} seconds ({} branch, {} function evaluations)'.format(
            t2 - t1, _BRANCHES[branch.value], nfev.value))
    return zs, Hs


def Hubble_th(physical_params, b_crit=0.15, all_analytic=False,
              epsilon=10**(-10), n=1, num_z_points=int(10**5),
              z_min=0, z_max=10, model='HS',
              method=None, rtol=None, atol=None, verbose=False, z_ic=None):
    '''
    C version of solve_sys.Hubble_th. Calculates the Hubble parameter as a function
    of redshift for LCDM and the f(R) models HS, ST and EXP.

    Args:
        physical_params: (omega_m, b, H0).
        b_crit: below this value of b the Taylor expansion is used (HS and ST).
        all_analytic: if True, use the analytic approximation for every b.
        epsilon: tune parameter of the initial redshift and b_crit in the EXP model.
        n: must be 1 for HS and ST (as in solve_sys.Hubble_th).
        num_z_points: number of redshifts at which H(z) is computed.
        z_min, z_max: redshift range.
        model: 'LCDM', 'HS', 'ST' or 'EXP'.
        method: 'RK45' (Dormand-Prince, same algorithm as scipy) or 'ROS23' (stiff
            Rosenbrock method, as MATLAB's ode23s). Default: 'RK45' for HS and ST,
            'ROS23' for EXP (the EXP system is stiff and RK45 fails).
        rtol, atol: relative and absolute tolerances of the integration. Default (None):
            rtol=1e-8, atol=1e-10 for HS and ST; rtol=1e-10, atol=1e-12 for EXP.
        verbose: if True, print the time of computation.
        z_ic: HS and ST: redshift of the (LCDM) initial conditions (>= z_max). Default:
            Z_IC_HS for HS and z_max for ST (as solve_sys.integrator).

    Returns:
        A tuple of two NumPy arrays: the redshifts (ascending) and H(z).

    Raises:
        RuntimeError if the numerical integration fails.
    '''
    return _call(physical_params, model, n, z_min, z_max, num_z_points, b_crit, all_analytic,
                 False, epsilon, method, rtol, atol, verbose, z_ic)


def integrator(physical_params, epsilon=10**(-10), num_z_points=int(10**5),
               initial_z=10, final_z=0, verbose=False, model='HS',
               method=None, rtol=None, atol=None, n=1, z_ic=None):
    '''
    C version of solve_sys.integrator: always integrates the ODE (HS, ST or EXP),
    whatever the value of b. Returns the redshifts (ascending) and H(z).
    '''
    if model not in ('HS', 'ST', 'EXP'):
        raise ValueError('Invalid model specified. Choose from "EXP", "HS" or "ST".')
    return _call(physical_params, model, n, final_z, initial_z, num_z_points, 0.0, False,
                 True, epsilon, method, rtol, atol, verbose, z_ic)
