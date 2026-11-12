// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
// C engine of xtrack/_temp/lorentz_reference.py: the same field, force, edge
// terms and Thomas-BMT spin, each particle integrated along z by its own
// adaptive DOP853 (SciPy's tableau and step control). State per particle:
// x, y, ct [m], P_x c, P_y c, E [eV], then spin_x, spin_y, spin_z.
#ifndef XTRACK_LORENTZ_REFERENCE_H
#define XTRACK_LORENTZ_REFERENCE_H
#include <math.h>
#include <stdint.h>
#include "xtrack/_temp/lorentz_reference_src/dop853_coefficients.h"

typedef struct {
    int64_t harmonics;
    const double *wave_number, *re, *im;  // c_m in V/m
    double k, half, q0, mass0, anomaly;   // anomaly NaN: no spin
    int64_t order;                        // 0: exact Bessel
    double ramp;                          // 0: abrupt ends
} LorentzField;

// I0(x) and I1(x)/x by their power series, all terms positive.
static void lorentz_bessel_i(double x, double *i0, double *i1_over_x) {
    const double q = x * x / 4.;
    double term = 1., zero = 1., one = .5, ratio = .5;
    for (int n = 1; n < 200; ++n) {
        term *= q / ((double)n * n);
        ratio *= q / ((double)n * (n + 1));
        zero += term; one += ratio;
        if (term < 1e-17 * zero && ratio < 1e-17 * one) break;
    }
    *i0 = zero; *i1_over_x = one;
}

// Rates of the state at z; returns 0 where the particle cannot move along z.
static int lorentz_rhs(const LorentzField *f, double z, const double *s, int spin,
                       double *rate) {
    const double x = s[0], y = s[1], ct = s[2], px = s[3], py = s[4], E = s[5];
    const double square = E * E - f->mass0 * f->mass0 - px * px - py * py;
    if (!(square > 0.)) return 0;
    const double pz = sqrt(square), r2 = x * x + y * y, r = sqrt(r2);
    double Ez = 0., radial = 0., azimuthal = 0., W = 0., W_z = 0., W_ct = 0.;
    if (fabs(z) <= f->half) {
        for (int64_t m = 0; m < f->harmonics; ++m) {
            const double kz = f->wave_number[m], kappa2 = f->k * f->k - kz * kz;
            const double phase = kz * z - f->k * ct, c = cos(phase), sn = sin(phase);
            const double wr = f->re[m] * c - f->im[m] * sn, wi = f->re[m] * sn + f->im[m] * c;
            double A = 0., B = 0.;
            if (f->order > 0) {
                const double u = -kappa2 * r2 / 4.;
                double power = 1., a = 1., b = -.5;
                for (int64_t n = 0; n <= f->order / 2; ++n) {
                    A += a * power;
                    if (n < f->order / 2) B += b * power;
                    power *= u; a /= (n + 1.) * (n + 1.); b /= (n + 1.) * (n + 2.);
                }
            } else {
                const double argument = sqrt(fabs(kappa2)) * r;
                if (kappa2 == 0. || argument == 0.) { A = 1.; B = -.5; }
                else if (kappa2 > 0.) { A = j0(argument); B = -j1(argument) / argument; }
                else { double one; lorentz_bessel_i(argument, &A, &one); B = -one; }
            }
            Ez += wr * A;
            radial -= kz * wi * B;
            azimuthal -= f->k * wi * B;
            W += wr; W_z -= kz * wi; W_ct += f->k * wi;
        }
    }
    double Ex = radial * x, Ey = radial * y, cBx = -azimuthal * y, cBy = azimuthal * x;
    if (f->ramp > 0. && fabs(z) <= f->half) {
        const double a = fmin(fmax((f->half - fabs(z)) / f->ramp, 0.), 1.);
        const double h = sin(M_PI * a / 2.) * sin(M_PI * a / 2.);
        const double dh = -copysign(1., z) * M_PI / (2. * f->ramp) * sin(M_PI * a);
        Ex = h * Ex - dh * W / 2. * x; Ey = h * Ey - dh * W / 2. * y;
        Ez = h * Ez + dh / 4. * (2. * (x * px + y * py) / pz * W + r2 * (W_ct * E / pz - W_z));
        cBx *= h; cBy *= h;
    }
    const double fx = f->q0 * (Ex - pz * cBy / E), fy = f->q0 * (Ey + pz * cBx / E);
    rate[0] = px / pz; rate[1] = py / pz; rate[2] = E / pz;
    rate[3] = fx * E / pz; rate[4] = fy * E / pz;
    rate[5] = f->q0 * (Ex * px + Ey * py + Ez * pz) / pz;
    if (spin) {
        // Thomas-BMT in eV units, per unit z: dS/dz = Omega x S E / (P_z c).
        const double gamma = E / f->mass0, a = f->anomaly;
        const double beta[3] = {px / E, py / E, pz / E}, cB[3] = {cBx, cBy, 0.};
        const double field[3] = {Ex, Ey, Ez};
        const double along = beta[0] * cB[0] + beta[1] * cB[1];
        const double cross[3] = {beta[1] * field[2] - beta[2] * field[1],
                                 beta[2] * field[0] - beta[0] * field[2],
                                 beta[0] * field[1] - beta[1] * field[0]};
        double omega[3];
        for (int i = 0; i < 3; ++i)
            omega[i] = -f->q0 / f->mass0 * ((a + 1. / gamma) * cB[i]
                - a * gamma / (gamma + 1.) * along * beta[i] - (a + 1. / (gamma + 1.)) * cross[i]);
        const double *S = s + 6;
        rate[6] = (omega[1] * S[2] - omega[2] * S[1]) * E / pz;
        rate[7] = (omega[2] * S[0] - omega[0] * S[2]) * E / pz;
        rate[8] = (omega[0] * S[1] - omega[1] * S[0]) * E / pz;
    }
    return 1;
}

// One particle from z = a to b; returns 1, or 0 if the step control fails or
// the particle cannot move along z.
static int lorentz_dop853(const LorentzField *f, double a, double b, double *s, int spin,
                          double rtol, const double *atol, double max_step) {
    const int n = spin ? 9 : 6;
    double K[LORENTZ_DOP853_STAGES + 1][9], y[9], f0[9];
    if (!lorentz_rhs(f, a, s, spin, f0)) return 0;
    double z = a, h = fmin(max_step, b - a);
    while (z < b) {
        int rejected = 0;
        for (;;) {
            if (h < 10. * fabs(nextafter(z, INFINITY) - z)) return 0;
            // The last step ends exactly on b, as in SciPy.
            const double next = z + h > b ? b : z + h;
            h = next - z;
            for (int i = 0; i < n; ++i) K[0][i] = f0[i];
            for (int stage = 1; stage < LORENTZ_DOP853_STAGES; ++stage) {
                for (int i = 0; i < n; ++i) {
                    double sum = 0.;
                    for (int j = 0; j < stage; ++j) sum += LORENTZ_DOP853_A[stage][j] * K[j][i];
                    y[i] = s[i] + h * sum;
                }
                if (!lorentz_rhs(f, z + LORENTZ_DOP853_C[stage] * h, y, spin, K[stage])) return 0;
            }
            for (int i = 0; i < n; ++i) {
                double sum = 0.;
                for (int j = 0; j < LORENTZ_DOP853_STAGES; ++j) sum += LORENTZ_DOP853_B[j] * K[j][i];
                y[i] = s[i] + h * sum;
            }
            if (!lorentz_rhs(f, z + h, y, spin, K[LORENTZ_DOP853_STAGES])) return 0;
            double err5 = 0., err3 = 0.;
            for (int i = 0; i < n; ++i) {
                const double scale = atol[i] + fmax(fabs(s[i]), fabs(y[i])) * rtol;
                double e5 = 0., e3 = 0.;
                for (int j = 0; j <= LORENTZ_DOP853_STAGES; ++j) {
                    e5 += K[j][i] * LORENTZ_DOP853_E5[j]; e3 += K[j][i] * LORENTZ_DOP853_E3[j];
                }
                err5 += (e5 / scale) * (e5 / scale); err3 += (e3 / scale) * (e3 / scale);
            }
            const double denom = err5 + .01 * err3;
            const double norm = denom > 0. ? h * err5 / sqrt(denom * n) : 0.;
            if (norm < 1.) {
                double factor = norm == 0. ? 10. : fmin(10., .9 * pow(norm, -1. / 8.));
                if (rejected) factor = fmin(1., factor);
                z = next;
                for (int i = 0; i < n; ++i) { s[i] = y[i]; f0[i] = K[LORENTZ_DOP853_STAGES][i]; }
                h = fmin(max_step, h * factor);
                break;
            }
            h *= fmax(.2, .9 * pow(norm, -1. / 8.));
            rejected = 1;
        }
    }
    return 1;
}

// Every particle through the field, from z = -half to half, in place; with a
// ramp, through its three segments. status[p] is 1, or 0 where it failed.
void lorentz_track(int64_t count, double *state, int64_t *status, int64_t harmonics,
                   const double *wave_number, const double *re, const double *im,
                   double k, double half, double q0, double mass0, double anomaly,
                   int64_t order, double ramp, double rtol, const double *atol,
                   double max_step) {
    const LorentzField f = {harmonics, wave_number, re, im, k, half, q0, mass0, anomaly,
                            order, ramp};
    const int spin = !isnan(anomaly), n = spin ? 9 : 6;
    const double planes[4] = {-half, -half + ramp, half - ramp, half};
    for (int64_t p = 0; p < count; ++p) {
        double s[9];
        for (int i = 0; i < n; ++i) s[i] = state[i * count + p];
        int ok = 1;
        if (ramp > 0.)
            for (int segment = 0; segment < 3 && ok; ++segment)
                ok = lorentz_dop853(&f, planes[segment], planes[segment + 1], s, spin, rtol,
                                    atol, fmin(max_step, (planes[segment + 1] - planes[segment]) / 16.));
        else
            ok = lorentz_dop853(&f, -half, half, s, spin, rtol, atol, max_step);
        status[p] = ok;
        for (int i = 0; i < n; ++i) state[i * count + p] = s[i];
    }
}
#endif
