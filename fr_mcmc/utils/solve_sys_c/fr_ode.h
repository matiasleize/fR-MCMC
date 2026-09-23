/*
 * C implementation of the Hubble parameter H(z) for LCDM and the f(R) models
 * Hu-Sawicki (HS), Starobinsky (ST) and Exponential (EXP).
 *
 * It mirrors fr_mcmc/utils/solve_sys.py (Hubble_th / integrator / get_odes),
 * fr_mcmc/utils/initial_conditions.py and fr_mcmc/utils/taylor.py.
 */
#ifndef FR_ODE_H
#define FR_ODE_H

/* Models */
#define FR_MODEL_LCDM 0
#define FR_MODEL_HS   1
#define FR_MODEL_ST   2
#define FR_MODEL_EXP  3

/* Integration methods */
#define FR_METHOD_DEFAULT -1 /* RK45 for HS/ST, ROS23 for EXP */
#define FR_METHOD_RK45     0 /* Dormand-Prince 5(4), same step control as scipy's RK45 */
#define FR_METHOD_ROS23    1 /* Rosenbrock 2(3) (Shampine & Reichelt, MATLAB's ode23s), L-stable */

/* Branch that was used to compute H(z) */
#define FR_BRANCH_LCDM    0
#define FR_BRANCH_TAYLOR  1
#define FR_BRANCH_NUMERIC 2

/* Return codes */
#define FR_OK             0
#define FR_ERR_STEP       1 /* step size became too small */
#define FR_ERR_MAXSTEPS   2 /* maximum number of steps exceeded */
#define FR_ERR_NONFINITE  3 /* non-finite value in the solution */
#define FR_ERR_ARGS       4 /* invalid arguments */
#define FR_ERR_MEM        5 /* memory allocation failed */

/* omega_r: radiation density parameter today (photons + massless neutrinos, 0 to neglect it) */
int fr_hubble(int model, double omega_m, double b, double H0, double omega_r, int n,
              double z_min, double z_max, int num_z_points, double z_ic,
              double b_crit, int all_analytic, int force_numeric, double epsilon,
              int method, double rtol, double atol,
              double *zs, double *Hs, long *nfev, int *branch);

#endif
