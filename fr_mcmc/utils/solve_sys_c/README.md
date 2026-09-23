## solve_sys_c
C implementation of `solve_sys.py` (H(z) for LCDM, HS, ST and EXP) with a Python wrapper (ctypes).

- `fr_ode.c`, `fr_ode.h`: ODE systems, initial conditions, Taylor expansions and integrators.
- `solve_sys_c.py`: wrapper. `Hubble_th` has the same call as `solve_sys.Hubble_th`.
- `test_solve_sys_c.py`: comparison against the Python implementation.

The library (`libfr_ode.so`) is compiled automatically on import (needs `cc`/`gcc`), or by hand with `make`.

Usage from the chi square: `params_to_chi2(..., use_c=True)`, or `USE_C: True` in the config file.

```python
from solve_sys_c import Hubble_th
zs, Hs = Hubble_th([omega_m, b, H0], model='HS', z_min=0, z_max=10, num_z_points=int(1e5))
```

### Radiation
The background includes radiation (photons + massless neutrinos) in LCDM, HS, ST and EXP:
`Omega_r = constants.OMEGA_R_H2 / h^2`, with `OMEGA_R_H2 = 2.47e-5 (1 + 0.2271 N_ur)`, `N_ur = 2.0308`
(N_eff = 3.044 with one massive neutrino, as in `BAO.r_drag_class`; the massive one is in Omega_m).
`omega_Lambda = 1 - omega_m - omega_r`. Set `constants.RADIATION = False` to neglect it (runs before
2026-09-22): the results are then identical to the previous version.

### Integrators
- `method='RK45'` (default for HS and ST): Dormand-Prince 5(4) with the same step control and dense output as scipy's `RK45`.
- `method='ROS23'` (default for EXP): Rosenbrock 2(3) of Shampine & Reichelt (MATLAB's `ode23s`), L-stable. The EXP system is stiff and RK45 does not converge.

### Differences with `solve_sys.Hubble_th`
Default tolerances: `rtol=1e-8`, `atol=1e-10` in both, except EXP in C (`rtol=1e-10`, `atol=1e-12`).
- HS, ST, LCDM and Taylor: same results (|Delta chi2| < 1e-8).
- EXP: Python uses scipy's `Radau` (order 5) and C uses `ROS23` (order 2). The error in H(z) is ~1e-9
  in Python and ~2e-8 in C (~30-50 ms per call for b > b_crit). With `rtol=1e-8` C takes ~8 ms but the
  error grows to ~5e-7 (|Delta chi2| ~ 1e-2).
- `num_z_points` is also used in the numerical branch.
- Integration failures raise `RuntimeError`.
