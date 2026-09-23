/*
 * C implementation of the Hubble parameter H(z) for LCDM and the f(R) models
 * Hu-Sawicki (HS), Starobinsky (ST) and Exponential (EXP).
 *
 * Equations are the same as in fr_mcmc/utils/solve_sys.py:
 *   - HS and ST: De la Cruz et al. system in the variables (x, y, v, w, r),
 *     integrated in z from z_max to z_min with LCDM initial conditions at z_max.
 *     H = eta * sqrt(r/v), with eta = H0 * sqrt((1 - omega_m - omega_r)/2).
 *   - Radiation: omega_r is the density parameter today of the radiation (photons + massless
 *     neutrinos, 0 to neglect it); omega_Lambda = 1 - omega_m - omega_r.
 *   - EXP: Odintsov et al. system in (E, tildeR), integrated in N = ln(a) from the
 *     redshift where exp(-beta*tildeR) = epsilon down to z_min. Above that redshift
 *     H is the LCDM one.
 *   - LCDM and Basilakos et al. Taylor expansions (HS and ST, n=1) are analytic.
 *
 * Build: cc -O3 -shared -fPIC -o libfr_ode.so fr_ode.c -lm
 */
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "fr_ode.h"

#define MAXDIM 5
#define MAX_STEPS 1000000L

typedef struct {
    int model;
    int n;
    double omega_m;
    double b;
    double omega_r;
} fr_params;

typedef void (*rhs_fn)(double t, const double *y, double *dy, const fr_params *p);

/* ------------------------------------------------------------------------- */
/* Systems of ODEs (get_odes in solve_sys.py)                                 */
/* ------------------------------------------------------------------------- */

/* Gamma = f_R / (R f_RR) in terms of r = R/Lambda */
static double gamma_fR(double r, double b, int n, int model)
{
    if (model == FR_MODEL_HS) {
        if (n == 1)
            return (r + b) * ((r + b) * (r + b) - 2 * b) / (4 * b * r);
        double rn = pow(r, n), bn = pow(b, n), rbn = pow(r * b, n);
        double N = (rn + bn) * (r * (rn + bn) * (rn + bn) - 2 * n * rbn);
        double D = 2 * n * rbn * ((n + 1) * rn + (1 - n) * bn);
        return N / D;
    }
    /* FR_MODEL_ST */
    double b2 = b * b, r2 = r * r;
    if (n == 1)
        return (r2 + b2) * ((r2 + b2) * (r2 + b2) - 4 * r * b2) / (4 * r * b2 * (3 * r2 - b2));
    double b2n = pow(b, 2 * n);
    double N = (b2 + r2) * (pow(b2 + r2, n + 1) - 4 * n * r * b2n);
    double D = 4 * r * n * b2n * ((2 * n + 1) * r2 - b2);
    return N / D;
}

/* HS and ST: independent variable z, variables (x, y, v, w, r) */
static void rhs_hs_st(double z, const double *s, double *ds, const fr_params *p)
{
    double x = s[0], y = s[1], v = s[2], w = s[3], r = s[4];

    /* ST: f_RR = 0 at (2n+1) r^2 = b^2 (Gamma diverges) and f_RR < 0 below (unstable
     * model). Return NaN so that the integration stops instead of taking tiny steps. */
    if (p->model == FR_MODEL_ST && (2 * p->n + 1) * r * r <= p->b * p->b) {
        for (int i = 0; i < 5; i++)
            ds[i] = NAN;
        return;
    }
    double G = gamma_fR(r, p->b, p->n, p->model);

    ds[0] = (-w + x * x + (1 + v) * x - 2 * v + 4 * y) / (z + 1);
    ds[1] = (-(v * x * G - x * y + 4 * y - 2 * y * v)) / (z + 1);
    ds[2] = (-v * (x * G + 4 - 2 * v)) / (z + 1);
    ds[3] = (w * (-1 + x + 2 * v)) / (z + 1);
    ds[4] = -(x * r * G) / (1 + z);
}

/* EXP: independent variable N = ln(a) = -ln(1+z), variables (E, tildeR) */
static void rhs_exp(double N, const double *s, double *ds, const fr_params *p)
{
    double omega_m = p->omega_m, omega_r = p->omega_r;
    double E = s[0], tildeR = s[1];
    double beta = 2 / p->b;
    double omega_l = 1 - omega_m - omega_r;
    double e_m = exp(-beta * tildeR);

    ds[0] = omega_l * tildeR / E - 2 * E;
    ds[1] = (exp(beta * tildeR) / (beta * beta)) *
            ((omega_m * exp(-3 * N) + omega_r * exp(-4 * N)) / (E * E) - 1 + beta * e_m +
             omega_l * (1 - (1 + beta * tildeR) * e_m) / (E * E));
}

/* ------------------------------------------------------------------------- */
/* Analytic expressions                                                       */
/* ------------------------------------------------------------------------- */

static double E_LCDM(double z, double omega_m, double omega_r)
{
    double opz = 1 + z;
    return sqrt(omega_r * opz * opz * opz * opz + omega_m * opz * opz * opz + (1 - omega_m - omega_r));
}

/* e^{kN} with N = -ln(1+z): ap[k] = (1+z)^(-k), am[k] = (1+z)^k */
#define EK(k) ((k) >= 0 ? ap[(k)] : am[-(k)])

/* Powers of a = 1/(1+z) up to 19, of (1+z) up to 4, and of x up to 6 */
#define TAYLOR_POWERS                                          \
    double ap[20], am[5], om_[7], s_[7];                       \
    ap[0] = 1.0;                                               \
    for (int k = 1; k < 20; k++)                               \
        ap[k] = ap[k - 1] / opz;                               \
    am[0] = 1.0;                                               \
    for (int k = 1; k < 5; k++)                                \
        am[k] = am[k - 1] * opz;                               \
    om_[0] = s_[0] = 1.0;                                      \
    for (int k = 1; k < 7; k++) {                              \
        om_[k] = om_[k - 1] * om;                              \
        s_[k] = s_[k - 1] * s;                                 \
    }

/* Taylor_HS in taylor.py (Basilakos et al.) */
static double taylor_HS(double z, double omega_m, double b, double H0, double omega_r)
{
    const double om = omega_m, orr = omega_r, opz = 1 + z;
    const double s = om + orr - 1;
    TAYLOR_POWERS
    const double den = om - 4 * EK(3) * s;

    double poly2 = 37 * EK(1) * om_[6] - 4656 * EK(4) * om_[5] * s
                 - 7452 * EK(7) * om_[4] * s * s
                 - 8692 * EK(3) * om_[4] * orr * s
                 - 4032 * EK(2) * om_[3] * orr * orr * s
                 + 25408 * EK(10) * om_[3] * s_[3]
                 - 25728 * EK(6) * om_[3] * orr * s * s
                 - 17856 * EK(5) * om * om * orr * orr * s * s
                 - 22848 * EK(13) * om * om * s_[4]
                 + 22016 * EK(9) * om * om * orr * s_[3]
                 - 9216 * EK(8) * om * orr * orr * s_[3]
                 + 9216 * EK(16) * om * s_[5]
                 - 2048 * EK(12) * om * orr * s_[4]
                 + 1024 * EK(19) * s_[6]
                 + 3072 * EK(15) * orr * s_[5]
                 + 40 * om_[5] * orr;
    double term2 = (1 / pow(den, 8)) * b * b * EK(5) * s_[3] * poly2;

    double poly1 = -6 * EK(1) * om * om + 3 * EK(4) * om * s + 12 * EK(7) * s * s
                 + 4 * EK(3) * orr * s - 7 * om * orr;
    double term1 = (2 * b * EK(2) * s * s * poly1) / pow(4 * EK(3) * s - om, 3);

    double term0 = (EK(-3) - 1) * om + (EK(-4) - 1) * orr + 1;

    return sqrt(H0 * H0 * (term2 + term1 + term0));
}

/* Taylor_ST in taylor.py (Basilakos et al.) */
static double taylor_ST(double z, double omega_m, double b, double H0, double omega_r)
{
    const double om = omega_m, orr = omega_r, opz = 1 + z;
    const double s = -1 + om + orr;
    TAYLOR_POWERS
    const double den = om - 4 * EK(3) * s;

    double poly2 = -37 * EK(1) * om * om - 40 * om * orr + 32 * EK(4) * om * s
                 + 16 * EK(3) * orr * s + 32 * EK(7) * s * s;
    double term2 = (b * b * EK(5) * s_[3] * poly2) / pow(den, 4);

    double poly4 = 123 * EK(1) * om_[6] + 128 * om_[5] * orr
                 - 82748 * EK(4) * om_[5] * s
                 - 160440 * EK(3) * om_[4] * orr * s
                 - 77760 * EK(2) * om_[3] * orr * orr * s
                 - 44552 * EK(7) * om_[4] * s * s
                 - 277568 * EK(6) * om_[3] * orr * s * s
                 - 228096 * EK(5) * om * om * orr * orr * s * s
                 + 289024 * EK(10) * om_[3] * s_[3]
                 + 310144 * EK(9) * om * om * orr * s_[3]
                 - 82944 * EK(8) * om * orr * orr * s_[3]
                 - 234880 * EK(13) * om * om * s_[4]
                 - 6144 * EK(12) * om * orr * s_[4]
                 + 63488 * EK(16) * om * s_[5]
                 + 20480 * EK(15) * orr * s_[5]
                 + 20480 * EK(19) * s_[6];
    double term4 = pow(b, 4) * EK(11) * s_[5] * poly4 / pow(den, 10);

    double term0 = 1 + (-1 + EK(-3)) * om + (-1 + EK(-4)) * orr;

    return H0 * sqrt(term0 + term2 + term4);
}

#undef EK
#undef TAYLOR_POWERS

/* ------------------------------------------------------------------------- */
/* Linear algebra and norms                                                   */
/* ------------------------------------------------------------------------- */

/* RMS norm, as scipy.integrate._ivp.common.norm */
static double rms_norm(const double *v, int dim)
{
    double s = 0;
    for (int i = 0; i < dim; i++)
        s += v[i] * v[i];
    return sqrt(s / dim);
}

/* LU decomposition with partial pivoting of a dim x dim matrix (row major) */
static int lu_factor(double *a, int *piv, int dim)
{
    for (int k = 0; k < dim; k++) {
        int p = k;
        for (int i = k + 1; i < dim; i++)
            if (fabs(a[i * dim + k]) > fabs(a[p * dim + k]))
                p = i;
        if (a[p * dim + k] == 0.0 || !isfinite(a[p * dim + k]))
            return -1;
        piv[k] = p;
        if (p != k)
            for (int j = 0; j < dim; j++) {
                double t = a[k * dim + j];
                a[k * dim + j] = a[p * dim + j];
                a[p * dim + j] = t;
            }
        for (int i = k + 1; i < dim; i++) {
            a[i * dim + k] /= a[k * dim + k];
            for (int j = k + 1; j < dim; j++)
                a[i * dim + j] -= a[i * dim + k] * a[k * dim + j];
        }
    }
    return 0;
}

static void lu_solve(const double *lu, const int *piv, double *x, int dim)
{
    for (int k = 0; k < dim; k++) {
        double t = x[k];
        x[k] = x[piv[k]];
        x[piv[k]] = t;
    }
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < i; j++)
            x[i] -= lu[i * dim + j] * x[j];
    for (int i = dim - 1; i >= 0; i--) {
        for (int j = i + 1; j < dim; j++)
            x[i] -= lu[i * dim + j] * x[j];
        x[i] /= lu[i * dim + i];
    }
}

static int all_finite(const double *v, int dim)
{
    for (int i = 0; i < dim; i++)
        if (!isfinite(v[i]))
            return 0;
    return 1;
}

/* ------------------------------------------------------------------------- */
/* Initial step (scipy.integrate._ivp.common.select_initial_step)             */
/* ------------------------------------------------------------------------- */

static double select_initial_step(rhs_fn f, const fr_params *p, int dim, double t0,
                                  const double *y0, double t_bound, const double *f0,
                                  double direction, int order, double rtol, double atol,
                                  long *nfev)
{
    double interval_length = fabs(t_bound - t0);
    if (interval_length == 0.0)
        return 0.0;

    double scale[MAXDIM], tmp[MAXDIM], y1[MAXDIM], f1[MAXDIM];
    for (int i = 0; i < dim; i++)
        scale[i] = atol + fabs(y0[i]) * rtol;

    for (int i = 0; i < dim; i++)
        tmp[i] = y0[i] / scale[i];
    double d0 = rms_norm(tmp, dim);
    for (int i = 0; i < dim; i++)
        tmp[i] = f0[i] / scale[i];
    double d1 = rms_norm(tmp, dim);

    double h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1;
    h0 = fmin(h0, interval_length);

    for (int i = 0; i < dim; i++)
        y1[i] = y0[i] + h0 * direction * f0[i];
    f(t0 + h0 * direction, y1, f1, p);
    (*nfev)++;
    for (int i = 0; i < dim; i++)
        tmp[i] = (f1[i] - f0[i]) / scale[i];
    double d2 = rms_norm(tmp, dim) / h0;

    double h1;
    if (d1 <= 1e-15 && d2 <= 1e-15)
        h1 = fmax(1e-6, h0 * 1e-3);
    else
        h1 = pow(0.01 / fmax(d1, d2), 1.0 / (order + 1));

    return fmin(fmin(100 * h0, h1), interval_length);
}

/* ------------------------------------------------------------------------- */
/* RK45: Dormand-Prince 5(4) with the same step control and dense output as   */
/* scipy.integrate.RK45.                                                       */
/* ------------------------------------------------------------------------- */

static const double RK_C[6] = {0.0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0};
static const double RK_A[6][5] = {
    {0, 0, 0, 0, 0},
    {1.0 / 5, 0, 0, 0, 0},
    {3.0 / 40, 9.0 / 40, 0, 0, 0},
    {44.0 / 45, -56.0 / 15, 32.0 / 9, 0, 0},
    {19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729, 0},
    {9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656}};
static const double RK_B[6] = {35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84};
static const double RK_E[7] = {-71.0 / 57600, 0, 71.0 / 16695, -71.0 / 1920,
                               17253.0 / 339200, -22.0 / 525, 1.0 / 40};
static const double RK_P[7][4] = {
    {1, -8048581381.0 / 2820520608, 8663915743.0 / 2820520608, -12715105075.0 / 11282082432},
    {0, 0, 0, 0},
    {0, 131558114200.0 / 32700410799, -68118460800.0 / 10900136933, 87487479700.0 / 32700410799},
    {0, -1754552775.0 / 470086768, 14199869525.0 / 1410260304, -10690763975.0 / 1880347072},
    {0, 127303824393.0 / 49829197408, -318862633887.0 / 49829197408, 701980252875.0 / 199316789632},
    {0, -282668133.0 / 205662961, 2019193451.0 / 616988883, -1453857185.0 / 822651844},
    {0, 40617522.0 / 29380423, -110615467.0 / 29380423, 69997945.0 / 29380423}};

#define SAFETY 0.9
#define MIN_FACTOR 0.2

/*
 * Integrate from t0 to t1 and evaluate the solution in t_eval (n_eval points,
 * ordered in the direction of integration and inside [t0, t1]).
 * y_eval has n_eval * dim elements (row i is the solution at t_eval[i]).
 */
static int solve_rk45(rhs_fn f, const fr_params *p, int dim, double t0, const double *y0,
                      double t1, const double *t_eval, int n_eval, double *y_eval,
                      double rtol, double atol, long *nfev)
{
    const double MAX_FACTOR = 10.0;
    const double error_exponent = -1.0 / 5; /* -1/(error_estimator_order + 1) */
    double direction = (t1 > t0) ? 1.0 : -1.0;
    double y[MAXDIM], y_new[MAXDIM], ytmp[MAXDIM], K[7][MAXDIM], Q[MAXDIM][4];
    double t = t0;
    int ie = 0;
    long steps = 0;

    memcpy(y, y0, dim * sizeof(double));
    f(t, y, K[0], p);
    (*nfev)++;

    while (ie < n_eval && t_eval[ie] == t0) {
        memcpy(y_eval + (size_t)ie * dim, y, dim * sizeof(double));
        ie++;
    }

    double h_abs = select_initial_step(f, p, dim, t0, y, t1, K[0], direction, 4, rtol, atol, nfev);

    while (direction * (t1 - t) > 0) {
        double min_step = 10 * fabs(nextafter(t, direction * INFINITY) - t);
        if (h_abs < min_step)
            h_abs = min_step;

        int accepted = 0, rejected = 0;
        double h = 0, t_new = t;
        while (!accepted) {
            if (h_abs < min_step)
                return FR_ERR_STEP;

            h = h_abs * direction;
            t_new = t + h;
            if (direction * (t_new - t1) > 0)
                t_new = t1;
            h = t_new - t;
            h_abs = fabs(h);

            /* Stages (K[0] holds f(t, y)) */
            for (int s = 1; s < 6; s++) {
                for (int i = 0; i < dim; i++) {
                    double acc = 0;
                    for (int j = 0; j < s; j++)
                        acc += K[j][i] * RK_A[s][j];
                    ytmp[i] = y[i] + acc * h;
                }
                f(t + RK_C[s] * h, ytmp, K[s], p);
            }
            for (int i = 0; i < dim; i++) {
                double acc = 0;
                for (int j = 0; j < 6; j++)
                    acc += K[j][i] * RK_B[j];
                y_new[i] = y[i] + h * acc;
            }
            f(t + h, y_new, K[6], p);
            *nfev += 6;

            double err[MAXDIM];
            for (int i = 0; i < dim; i++) {
                double acc = 0;
                for (int j = 0; j < 7; j++)
                    acc += K[j][i] * RK_E[j];
                double scale = atol + fmax(fabs(y[i]), fabs(y_new[i])) * rtol;
                err[i] = acc * h / scale;
            }
            double error_norm = rms_norm(err, dim);

            if (error_norm < 1) {
                double factor;
                if (error_norm == 0)
                    factor = MAX_FACTOR;
                else
                    factor = fmin(MAX_FACTOR, SAFETY * pow(error_norm, error_exponent));
                if (rejected)
                    factor = fmin(1, factor);
                h_abs *= factor;
                accepted = 1;
            } else {
                /* fmax ignores NaN, as max(MIN_FACTOR, nan) in scipy */
                h_abs *= fmax(MIN_FACTOR, SAFETY * pow(error_norm, error_exponent));
                rejected = 1;
            }
        }

        if (!all_finite(y_new, dim))
            return FR_ERR_NONFINITE;

        /* Dense output in (t, t_new] */
        if (ie < n_eval && direction * (t_eval[ie] - t_new) <= 0) {
            for (int i = 0; i < dim; i++)
                for (int k = 0; k < 4; k++) {
                    double acc = 0;
                    for (int j = 0; j < 7; j++)
                        acc += K[j][i] * RK_P[j][k];
                    Q[i][k] = acc;
                }
            while (ie < n_eval && direction * (t_eval[ie] - t_new) <= 0) {
                double x = (t_eval[ie] - t) / h;
                double pw[4] = {x, x * x, x * x * x, x * x * x * x};
                double *out = y_eval + (size_t)ie * dim;
                for (int i = 0; i < dim; i++)
                    out[i] = y[i] + h * (Q[i][0] * pw[0] + Q[i][1] * pw[1] +
                                         Q[i][2] * pw[2] + Q[i][3] * pw[3]);
                ie++;
            }
        }

        t = t_new;
        memcpy(y, y_new, dim * sizeof(double));
        memcpy(K[0], K[6], dim * sizeof(double));

        if (++steps > MAX_STEPS)
            return FR_ERR_MAXSTEPS;
    }
    return (ie == n_eval) ? FR_OK : FR_ERR_ARGS;
}

/* ------------------------------------------------------------------------- */
/* ROS23: Rosenbrock 2(3) method of Shampine & Reichelt (MATLAB ode23s).       */
/* L-stable, for stiff problems (the EXP system is stiff).                    */
/* Jacobian and df/dt by central finite differences.                          */
/* ------------------------------------------------------------------------- */

static void jacobian_fd(rhs_fn f, const fr_params *p, int dim, double t, const double *y,
                        double *J, double *dfdt, long *nfev)
{
    const double eps3 = cbrt(DBL_EPSILON);
    double yp[MAXDIM], fp[MAXDIM], fm[MAXDIM];

    memcpy(yp, y, dim * sizeof(double));
    for (int j = 0; j < dim; j++) {
        double del = eps3 * fmax(fabs(y[j]), 1.0);
        yp[j] = y[j] + del;
        f(t, yp, fp, p);
        yp[j] = y[j] - del;
        f(t, yp, fm, p);
        yp[j] = y[j];
        for (int i = 0; i < dim; i++)
            J[i * dim + j] = (fp[i] - fm[i]) / (2 * del);
    }
    double delt = eps3 * fmax(fabs(t), 1.0);
    f(t + delt, y, fp, p);
    f(t - delt, y, fm, p);
    for (int i = 0; i < dim; i++)
        dfdt[i] = (fp[i] - fm[i]) / (2 * delt);
    *nfev += 2 * dim + 2;
}

static int solve_ros23(rhs_fn f, const fr_params *p, int dim, double t0, const double *y0,
                       double t1, const double *t_eval, int n_eval, double *y_eval,
                       double rtol, double atol, long *nfev)
{
    const double d = 1.0 / (2.0 + sqrt(2.0));
    const double e32 = 6.0 + sqrt(2.0);
    const double MAX_FACTOR = 5.0;
    const double error_exponent = -1.0 / 3;
    double direction = (t1 > t0) ? 1.0 : -1.0;
    double y[MAXDIM], y_new[MAXDIM], ytmp[MAXDIM];
    double F0[MAXDIM], F1[MAXDIM], F2[MAXDIM], k1[MAXDIM], k2[MAXDIM], k3[MAXDIM];
    double J[MAXDIM * MAXDIM], W[MAXDIM * MAXDIM], dfdt[MAXDIM], T[MAXDIM];
    int piv[MAXDIM];
    double t = t0;
    int ie = 0;
    long steps = 0;

    memcpy(y, y0, dim * sizeof(double));
    f(t, y, F0, p);
    (*nfev)++;

    while (ie < n_eval && t_eval[ie] == t0) {
        memcpy(y_eval + (size_t)ie * dim, y, dim * sizeof(double));
        ie++;
    }

    double h_abs = select_initial_step(f, p, dim, t0, y, t1, F0, direction, 2, rtol, atol, nfev);

    while (direction * (t1 - t) > 0) {
        double min_step = 10 * fabs(nextafter(t, direction * INFINITY) - t);
        if (h_abs < min_step)
            h_abs = min_step;

        jacobian_fd(f, p, dim, t, y, J, dfdt, nfev);

        int accepted = 0, rejected = 0;
        double h = 0, t_new = t;
        while (!accepted) {
            if (h_abs < min_step)
                return FR_ERR_STEP;

            h = h_abs * direction;
            t_new = t + h;
            if (direction * (t_new - t1) > 0)
                t_new = t1;
            h = t_new - t;
            h_abs = fabs(h);

            for (int i = 0; i < dim * dim; i++)
                W[i] = -h * d * J[i];
            for (int i = 0; i < dim; i++)
                W[i * dim + i] += 1.0;

            double error_norm;
            if (lu_factor(W, piv, dim) != 0) {
                error_norm = NAN;
            } else {
                for (int i = 0; i < dim; i++) {
                    T[i] = h * d * dfdt[i];
                    k1[i] = F0[i] + T[i];
                }
                lu_solve(W, piv, k1, dim);

                for (int i = 0; i < dim; i++)
                    ytmp[i] = y[i] + 0.5 * h * k1[i];
                f(t + 0.5 * h, ytmp, F1, p);
                for (int i = 0; i < dim; i++)
                    k2[i] = F1[i] - k1[i];
                lu_solve(W, piv, k2, dim);
                for (int i = 0; i < dim; i++) {
                    k2[i] += k1[i];
                    y_new[i] = y[i] + h * k2[i];
                }
                f(t_new, y_new, F2, p);
                for (int i = 0; i < dim; i++)
                    k3[i] = F2[i] - e32 * (k2[i] - F1[i]) - 2 * (k1[i] - F0[i]) + T[i];
                lu_solve(W, piv, k3, dim);
                *nfev += 2;

                double err[MAXDIM];
                for (int i = 0; i < dim; i++) {
                    double scale = atol + fmax(fabs(y[i]), fabs(y_new[i])) * rtol;
                    err[i] = (h / 6) * (k1[i] - 2 * k2[i] + k3[i]) / scale;
                }
                error_norm = rms_norm(err, dim);
            }

            if (error_norm < 1) {
                double factor;
                if (error_norm == 0)
                    factor = MAX_FACTOR;
                else
                    factor = fmin(MAX_FACTOR, SAFETY * pow(error_norm, error_exponent));
                if (rejected)
                    factor = fmin(1, factor);
                h_abs *= factor;
                accepted = 1;
            } else {
                h_abs *= fmax(MIN_FACTOR, SAFETY * pow(error_norm, error_exponent));
                rejected = 1;
            }
        }

        if (!all_finite(y_new, dim))
            return FR_ERR_NONFINITE;

        /* Dense output (ntrp23s) in (t, t_new] */
        while (ie < n_eval && direction * (t_eval[ie] - t_new) <= 0) {
            double s = (t_eval[ie] - t) / h;
            double c1 = s * (1 - s) / (1 - 2 * d), c2 = s * (s - 2 * d) / (1 - 2 * d);
            double *out = y_eval + (size_t)ie * dim;
            for (int i = 0; i < dim; i++)
                out[i] = y[i] + h * (c1 * k1[i] + c2 * k2[i]);
            ie++;
        }

        t = t_new;
        memcpy(y, y_new, dim * sizeof(double));
        memcpy(F0, F2, dim * sizeof(double));

        if (++steps > MAX_STEPS)
            return FR_ERR_MAXSTEPS;
    }
    return (ie == n_eval) ? FR_OK : FR_ERR_ARGS;
}

static int solve(int method, rhs_fn f, const fr_params *p, int dim, double t0, const double *y0,
                 double t1, const double *t_eval, int n_eval, double *y_eval,
                 double rtol, double atol, long *nfev)
{
    if (method == FR_METHOD_RK45)
        return solve_rk45(f, p, dim, t0, y0, t1, t_eval, n_eval, y_eval, rtol, atol, nfev);
    if (method == FR_METHOD_ROS23)
        return solve_ros23(f, p, dim, t0, y0, t1, t_eval, n_eval, y_eval, rtol, atol, nfev);
    return FR_ERR_ARGS;
}

/* ------------------------------------------------------------------------- */
/* Initial conditions (initial_conditions.py)                                 */
/* ------------------------------------------------------------------------- */

/* z_i for the Odintsov system: exp(-beta * tildeR_LCDM(z_i)) = eps */
static double redshift_initial_condition(double omega_m, double b, double eps, double omega_r)
{
    double beta = 2 / b;
    double omega_l = 1 - omega_m - omega_r;
    return pow(2 * omega_l * (-log(eps) - 2 * beta) / (beta * omega_m), 1.0 / 3) - 1;
}

/* LCDM initial conditions for HS and ST (CI_aprox=False in initial_conditions.py), in units of H0 */
static void initial_conditions_hs_st(double omega_m, double omega_r, double zi, double *s)
{
    double Lamb = 3 * (1 - omega_m - omega_r);
    double rad = omega_r * pow(1 + zi, 4);
    double E2 = rad + omega_m * pow(1 + zi, 3) + (1 - omega_m - omega_r);
    /* R = 12 H^2 - 6 (1+z) H dH/dz (the radiation is traceless) */
    double R_i = 12 * E2 - 12 * rad - 9 * omega_m * pow(1 + zi, 3);
    s[0] = 0;                                /* x */
    s[1] = (R_i - 2 * Lamb) / (6 * E2);      /* y */
    s[2] = R_i / (6 * E2);                   /* v */
    s[3] = 1 + s[0] + s[1] - s[2] - rad / E2; /* w = Omega_m = 1 + x + y - v - Omega_r */
    s[4] = R_i / Lamb;                       /* r */
}

static void initial_conditions_exp(double omega_m, double omega_r, double zi, double *s)
{
    s[0] = E_LCDM(zi, omega_m, omega_r);                                            /* E */
    s[1] = 2 + (omega_m / (2 * (1 - omega_m - omega_r))) * pow(1 + zi, 3);          /* tildeR */
}

/* ------------------------------------------------------------------------- */
/* Hubble_th                                                                   */
/* ------------------------------------------------------------------------- */

/* numpy.linspace */
static void linspace(double start, double stop, int num, double *out)
{
    if (num == 1) {
        out[0] = start;
        return;
    }
    double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; i++)
        out[i] = i * step + start;
    out[num - 1] = stop;
}

int fr_hubble(int model, double omega_m, double b, double H0, double omega_r, int n,
              double z_min, double z_max, int num_z_points, double z_ic,
              double b_crit, int all_analytic, int force_numeric, double epsilon,
              int method, double rtol, double atol,
              double *zs, double *Hs, long *nfev, int *branch)
{
    const int N = num_z_points;
    fr_params p = {model, n, omega_m, b, omega_r};
    *nfev = 0;

    if (N < 1 || !(z_max > z_min) || z_min <= -1 || !(rtol > 0) || !(atol >= 0))
        return FR_ERR_ARGS;
    if (method == FR_METHOD_DEFAULT)
        method = (model == FR_MODEL_EXP) ? FR_METHOD_ROS23 : FR_METHOD_RK45;

    if (model == FR_MODEL_LCDM) {
        *branch = FR_BRANCH_LCDM;
        linspace(z_min, z_max, N, zs);
        for (int i = 0; i < N; i++)
            Hs[i] = H0 * E_LCDM(zs[i], omega_m, omega_r);
        return FR_OK;
    }

    if (model == FR_MODEL_EXP) {
        double log_eps_inv = -log(epsilon); /* natural log: z_ci(b_crit) = 0 */
        double b_crit_exp = (4 + omega_m / (1 - omega_m - omega_r)) / log_eps_inv;
        double z_ci = redshift_initial_condition(omega_m, b, epsilon, omega_r);

        linspace(z_min, z_max, N, zs);
        if (((b <= b_crit_exp) || all_analytic) && !force_numeric) {
            *branch = FR_BRANCH_LCDM;
            for (int i = 0; i < N; i++)
                Hs[i] = H0 * E_LCDM(zs[i], omega_m, omega_r);
            return FR_OK;
        }
        *branch = FR_BRANCH_NUMERIC;
        if (!(z_ci > z_min)) /* exp(-beta*tildeR) < epsilon everywhere: LCDM */
            z_ci = z_min;

        /* LCDM for z >= z_ci and ODE (in N = -ln(1+z)) for z < z_ci */
        int m = 0;
        while (m < N && zs[m] < z_ci)
            m++;
        for (int i = m; i < N; i++)
            Hs[i] = H0 * E_LCDM(zs[i], omega_m, omega_r);
        if (m == 0)
            return FR_OK;

        double *t_eval = malloc((size_t)m * sizeof(double));
        double *y_eval = malloc((size_t)m * 2 * sizeof(double));
        if (!t_eval || !y_eval) {
            free(t_eval);
            free(y_eval);
            return FR_ERR_MEM;
        }
        for (int k = 0; k < m; k++)
            t_eval[k] = -log1p(zs[m - 1 - k]);

        double s0[2];
        initial_conditions_exp(omega_m, omega_r, z_ci, s0);
        int status = solve(method, rhs_exp, &p, 2, -log1p(z_ci), s0, t_eval[m - 1],
                           t_eval, m, y_eval, rtol, atol, nfev);
        if (status == FR_OK)
            for (int k = 0; k < m; k++)
                Hs[m - 1 - k] = H0 * y_eval[2 * k];
        free(t_eval);
        free(y_eval);
        return status;
    }

    if (model == FR_MODEL_HS || model == FR_MODEL_ST) {
        if (n != 1 && !force_numeric)
            return FR_ERR_ARGS; /* 'Not a valid Taylor!' */

        if (((b <= b_crit) || all_analytic) && !force_numeric) {
            *branch = FR_BRANCH_TAYLOR;
            linspace(z_min, z_max, N, zs);
            for (int i = 0; i < N; i++)
                Hs[i] = (model == FR_MODEL_HS) ? taylor_HS(zs[i], omega_m, b, H0, omega_r)
                                               : taylor_ST(zs[i], omega_m, b, H0, omega_r);
            return FR_OK;
        }
        *branch = FR_BRANCH_NUMERIC;

        /* Integrate from z_max to z_min, evaluating in linspace(z_max, z_min, N) */
        double *t_eval = malloc((size_t)N * sizeof(double));
        double *y_eval = malloc((size_t)N * 5 * sizeof(double));
        if (!t_eval || !y_eval) {
            free(t_eval);
            free(y_eval);
            return FR_ERR_MEM;
        }
        linspace(z_max, z_min, N, t_eval);

        /* The integration starts at z_ic >= z_max (LCDM initial conditions) */
        double zi = (z_ic > z_max) ? z_ic : z_max;
        double s0[5];
        initial_conditions_hs_st(omega_m, omega_r, zi, s0);
        int status = solve(method, rhs_hs_st, &p, 5, zi, s0, z_min,
                           t_eval, N, y_eval, rtol, atol, nfev);
        if (status == FR_OK) {
            double eta = H0 * sqrt((1 - omega_m - omega_r) / 2); /* sqrt(Lambda/6) */
            for (int k = 0; k < N; k++) {
                double v = y_eval[5 * k + 2], r = y_eval[5 * k + 4];
                zs[N - 1 - k] = t_eval[k];
                Hs[N - 1 - k] = eta * sqrt(r / v);
            }
            if (!all_finite(Hs, N))
                status = FR_ERR_NONFINITE;
        }
        free(t_eval);
        free(y_eval);
        return status;
    }

    return FR_ERR_ARGS;
}
