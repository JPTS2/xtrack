// Optional private conservative passive BMT on its own stored Gauss stages.
// No default orbit/tangent path includes or executes this code.
#ifndef XTRACK_CAVITY_NATIVE_SPIN_H
#define XTRACK_CAVITY_NATIVE_SPIN_H
#include "xtrack/beam_elements/elements_src/cavity_native_gauss.h"
#include "xtrack/beam_elements/elements_src/cavity_native_secondary.h"

// Fixed-laboratory-axis rest-spin generator per normalized Z. BMT returns
// its lab-time rate; the actual same-event physical clock is applied ONCE.
// No LL/Thomas-extra force or spin-flip law is part of this rho-zero target.
GPUFUN
int CavityNative_spin_generator(const double q[6], double Q, double anomaly,
        const double profile[10], double aperture, double generator[9]) {
    double omega[3], flow[6];
    int status = CavityNative_bmt(q, Q, anomaly, profile, aperture, omega);
    if (status != CAVITY_NATIVE_OK) return status;
    status = CavityNative_source(q, Q, profile, aperture, flow);
    if (status != CAVITY_NATIVE_OK) return status;
    const double clock = flow[2];
    double result[9] = {0., -clock*omega[2], clock*omega[1],
                       clock*omega[2], 0., -clock*omega[0],
                       -clock*omega[1], clock*omega[0], 0.};
    for (int k=0; k<9; ++k)
        if (!isfinite(result[k])) return CAVITY_NATIVE_NONFINITE;
    for (int k=0; k<9; ++k) generator[k] = result[k];
    return CAVITY_NATIVE_OK;
}

// Passive matrix R maps initial rest spin to final rest spin. Same nominal
// Gauss stages, represented A/C/b and predictor recipe; no parent history.
// R_i=I+h sum_j A_ij G_j R_j; R=I+h sum_i b_i G_i R_i.
// The nominal and spin defining-equation residuals are separate diagnostics.
// Stored dyadic tableau and finite solves have defects: no exact SO(3),
// continuous-history, source, backend or numerical-error admission is claimed.
// Every output is untouched on any nominal/source/spin refusal.
GPUFUN
int CavityNative_gauss_spin(const double q[6], double h, double Q,
        double anomaly, const double profiles[40], double aperture,
        int64_t max_iter, double tol, double spin_tol, double out[6],
        double rotation[9], double* residual, double* spin_residual) {
    if (!isfinite(anomaly) || !isfinite(spin_tol) || spin_tol <= 0.)
        return CAVITY_NATIVE_ARGUMENT;
    double endpoint[6], stages[12], nominal_residual;
    int status = CavityNative_gauss_stages(q, h, Q, profiles, aperture,
        max_iter, tol, endpoint, &nominal_residual, stages);
    if (status != CAVITY_NATIVE_OK) return status;
    double entrance_generator[9], generators[2][9];
    status = CavityNative_spin_generator(q, Q, anomaly, profiles, aperture,
                                         entrance_generator);
    if (status != CAVITY_NATIVE_OK) return status;
    for (int i=0; i<2; ++i) {
        status = CavityNative_spin_generator(stages+i*6, Q, anomaly,
            profiles+(i+1)*10, aperture, generators[i]);
        if (status != CAVITY_NATIVE_OK) return status;
    }
    const double C[2] = {0.21132486540518713, 0.7886751345948129};
    const double A[2][2] = {{0.25, -0.038675134594812866},
                            {0.5386751345948129, 0.25}};
    double spin[2][9], proposed[2][9], products[2][9];
    for (int i=0; i<2; ++i)
        for (int row=0; row<3; ++row)
            for (int col=0; col<3; ++col) {
                const int k = row*3+col;
                spin[i][k] = (row == col ? 1. : 0.)
                    +C[i]*h*entrance_generator[k];
                if (!isfinite(spin[i][k])) return CAVITY_NATIVE_NONFINITE;
            }
    for (int64_t iteration=0; iteration<max_iter; ++iteration) {
        for (int i=0; i<2; ++i)
            for (int row=0; row<3; ++row)
                for (int col=0; col<3; ++col) {
                    double value = 0.;
                    for (int k=0; k<3; ++k)
                        value += generators[i][row*3+k]*spin[i][k*3+col];
                    if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
                    products[i][row*3+col] = value;
                }
        double change = 0.;
        for (int i=0; i<2; ++i)
            for (int row=0; row<3; ++row)
                for (int col=0; col<3; ++col) {
                    const int k = row*3+col;
                    proposed[i][k] = (row == col ? 1. : 0.)
                        +h*(A[i][0]*products[0][k]+A[i][1]*products[1][k]);
                    if (!isfinite(proposed[i][k]))
                        return CAVITY_NATIVE_NONFINITE;
                    change = fmax(change, fabs(proposed[i][k]-spin[i][k]));
                }
        for (int i=0; i<2; ++i)
            for (int k=0; k<9; ++k) spin[i][k] = proposed[i][k];
        if (change < spin_tol) {
            // Reevaluate at reassigned spin stages before accepting the
            // actual full defining residual; an iteration delta is not it.
            for (int i=0; i<2; ++i)
                for (int row=0; row<3; ++row)
                    for (int col=0; col<3; ++col) {
                        double value = 0.;
                        for (int k=0; k<3; ++k)
                            value += generators[i][row*3+k]*spin[i][k*3+col];
                        if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
                        products[i][row*3+col] = value;
                    }
            double actual = 0.;
            for (int i=0; i<2; ++i)
                for (int row=0; row<3; ++row)
                    for (int col=0; col<3; ++col) {
                        const int k = row*3+col;
                        double defect = spin[i][k]-(row == col ? 1. : 0.)
                            -h*(A[i][0]*products[0][k]
                                +A[i][1]*products[1][k]);
                        if (!isfinite(defect)) return CAVITY_NATIVE_NONFINITE;
                        actual = fmax(actual, fabs(defect));
                    }
            if (actual < spin_tol) {
                double result[9];
                for (int row=0; row<3; ++row)
                    for (int col=0; col<3; ++col) {
                        const int k = row*3+col;
                        result[k] = (row == col ? 1. : 0.)
                            +h*((products[0][k]+products[1][k])*0.5);
                        if (!isfinite(result[k]))
                            return CAVITY_NATIVE_NONFINITE;
                    }
                for (int k=0; k<6; ++k) out[k] = endpoint[k];
                for (int k=0; k<9; ++k) rotation[k] = result[k];
                *residual = nominal_residual;
                *spin_residual = actual;
                return CAVITY_NATIVE_OK;
            }
        }
    }
    return CAVITY_NATIVE_ITERATIONS;
}
#endif
