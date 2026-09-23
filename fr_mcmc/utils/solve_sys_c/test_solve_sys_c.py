'''
Comparison of the C implementation (solve_sys_c) against the Python one (solve_sys.py).
Run from anywhere inside the repository:  python test_solve_sys_c.py
'''
import os
import sys
import time
import warnings

import numpy as np
from scipy.integrate import solve_ivp

_UTILS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_UTILS)

from solve_sys import get_odes, integrator as integrator_py, Hubble_th as Hubble_th_py  # noqa: E402
from LambdaCDM import H_LCDM  # noqa: E402
from taylor import Taylor_HS, Taylor_ST  # noqa: E402
from initial_conditions import calculate_initial_conditions, redshift_initial_condition  # noqa: E402
from solve_sys_c import Hubble_th as Hubble_th_c, integrator as integrator_c  # noqa: E402

warnings.filterwarnings('ignore', category=RuntimeWarning)
failures = []


def check(name, value, tol):
    ok = value < tol
    print('{:<60s} {:.2e}  {}'.format(name, value, 'OK' if ok else 'FAIL (tol {:.0e})'.format(tol)))
    if not ok:
        failures.append(name)


def rel(a, b):
    return np.max(np.abs(a / b - 1))


def exp_reference(physical_params, zs, epsilon=1e-10, rtol=1e-11, atol=1e-13):
    '''EXP model integrated with scipy (Radau), LCDM above the initial redshift.'''
    omega_m, b, H0 = physical_params
    z_ci = redshift_initial_condition(physical_params, epsilon)
    ic = calculate_initial_conditions(physical_params, zi=z_ci, model='EXP')
    mask = zs < z_ci
    Ns = -np.log1p(zs[mask][::-1])
    sol = solve_ivp(get_odes, (-np.log1p(z_ci), Ns[-1]), ic, t_eval=Ns,
                    args=(physical_params, 'EXP'), method='Radau', rtol=rtol, atol=atol)
    Hs = H_LCDM(zs, omega_m, H0)
    Hs[mask] = H0 * sol.y[0][::-1]
    return Hs


N = int(10**5)
z_grid = np.linspace(0, 10, N)

print('== Analytic branches')
for om, H0 in [(0.3, 70), (0.25, 73.5)]:
    zs, Hs = Hubble_th_c([om, 0.5, H0], model='LCDM', num_z_points=N)
    check('LCDM om={} H0={}'.format(om, H0), rel(Hs, H_LCDM(z_grid, om, H0)), 1e-14)
    for b in [0.01, 0.1, 0.15]:
        zs, Hs = Hubble_th_c([om, b, H0], model='HS', num_z_points=N)
        check('Taylor HS om={} b={}'.format(om, b), rel(Hs, Taylor_HS(z_grid, om, b, H0)), 1e-12)
        zs, Hs = Hubble_th_c([om, b, H0], model='ST', num_z_points=N)
        check('Taylor ST om={} b={}'.format(om, b), rel(Hs, Taylor_ST(z_grid, om, b, H0)), 1e-12)
    zs, Hs = Hubble_th_c([om, 0.15, H0], model='EXP', num_z_points=N)
    check('EXP below b_crit (LCDM) om={}'.format(om), rel(Hs, H_LCDM(z_grid, om, H0)), 1e-14)

print('== HS/ST, same algorithm and default tolerances as Python (RK45, rtol=1e-8, atol=1e-10)')
for model in ['HS', 'ST']:
    for om, b, H0 in [(0.3, 0.2, 70), (0.3, 1.0, 70), (0.25, 2.0, 73), (0.35, 4.0, 67)]:
        p = [om, b, H0]
        z_py, H_py = integrator_py(p, model=model, initial_z=10, final_z=0)
        z_c, H_c = integrator_c(p, model=model, initial_z=10, final_z=0)
        check('{} {} redshifts'.format(model, p), np.max(np.abs(z_c - z_py)), 1e-12)
        check('{} {} C vs Python (same method)'.format(model, p), rel(H_c, H_py), 1e-10)

print('== HS/ST, loose tolerances (rtol=1e-3): error of Python and C vs a precise solution')
# RK45 with rtol=1e-3 is at its stability limit for ST and the step-size sequence is
# sensitive to round-off, so C and Python are compared against a precise solution.
for model in ['HS', 'ST']:
    for om, b, H0 in [(0.3, 0.2, 70), (0.3, 1.0, 70)]:
        p = [om, b, H0]
        _, H_ref = integrator_py(p, model=model, initial_z=10, final_z=0, method='DOP853',
                                 rtol=1e-12, atol=1e-14)
        _, H_py = integrator_py(p, model=model, initial_z=10, final_z=0, rtol=1e-3, atol=1e-6)
        _, H_c = integrator_c(p, model=model, rtol=1e-3, atol=1e-6)
        check('{} {} Python error (rtol=1e-3)'.format(model, p), rel(H_py, H_ref), 5e-3)
        check('{} {} C error (rtol=1e-3)'.format(model, p), rel(H_c, H_ref), 5e-3)

print('== HS/ST, tight tolerances: C RK45 and ROS23 vs scipy DOP853')
for model in ['HS', 'ST']:
    for om, b, H0 in [(0.3, 0.2, 70), (0.3, 1.0, 70), (0.25, 2.0, 73)]:
        p = [om, b, H0]
        _, H_ref = integrator_py(p, model=model, initial_z=10, final_z=0, method='DOP853',
                                 rtol=1e-12, atol=1e-14)
        _, H_rk = integrator_c(p, model=model, method='RK45', rtol=1e-10, atol=1e-12)
        _, H_ros = integrator_c(p, model=model, method='ROS23', rtol=1e-8, atol=1e-10)
        check('{} {} C RK45(1e-10) vs DOP853'.format(model, p), rel(H_rk, H_ref), 1e-8)
        check('{} {} C ROS23(1e-8) vs DOP853'.format(model, p), rel(H_ros, H_ref), 1e-6)

print('== EXP (stiff): C ROS23 vs scipy Radau (fixed EXP branch)')
for om, b, H0 in [(0.3, 0.5, 70), (0.3, 1.0, 70), (0.25, 2.0, 73), (0.35, 5.0, 67), (0.3, 30., 70)]:
    p = [om, b, H0]
    H_ref = exp_reference(p, z_grid)
    _, H_c = Hubble_th_c(p, model='EXP', num_z_points=N, rtol=1e-8, atol=1e-10)
    check('EXP {} C ROS23(1e-8) vs Radau(1e-11)'.format(p), rel(H_c, H_ref), 1e-6)
    _, H_c = Hubble_th_c(p, model='EXP', num_z_points=N, rtol=1e-3, atol=1e-6)
    check('EXP {} C ROS23(1e-3) vs Radau(1e-11)'.format(p), rel(H_c, H_ref), 5e-3)
    _, H_py = Hubble_th_py(p, model='EXP', num_z_points=N)
    check('EXP {} Python Radau(default tol) vs Radau(1e-11)'.format(p), rel(H_py, H_ref), 1e-7)

print('== Timing (num_z_points=1e5, default tolerances)')
for model, p in [('HS', [0.3, 1.0, 70]), ('ST', [0.3, 1.0, 70]), ('EXP', [0.3, 1.0, 70])]:
    reps = 20
    t = time.time()
    for _ in range(reps):
        Hubble_th_c(p, model=model)
    t_c = (time.time() - t) / reps
    t = time.time()
    Hubble_th_py(p, model=model)
    t_py = time.time() - t
    print('{:<4s} C: {:.2e} s   Python: {:.2e} s   speed-up: x{:.0f}'.format(model, t_c, t_py, t_py / t_c))

print()
print('ALL TESTS PASSED' if not failures else 'FAILED: {}'.format(failures))
sys.exit(1 if failures else 0)
