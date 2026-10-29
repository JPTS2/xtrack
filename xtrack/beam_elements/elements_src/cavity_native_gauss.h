// Private stored two-stage Gauss4 word for the full-root cavity source.
// This is diagnostic finite-map execution, not numerical/source admission.
#ifndef XTRACK_CAVITY_NATIVE_GAUSS_H
#define XTRACK_CAVITY_NATIVE_GAUSS_H
#include "xtrack/beam_elements/elements_src/cavity_native_source.h"

// Profiles contain four consecutive uncharged ten-value planes:
// entrance predictor, stored stage0, stored stage1, exit-domain check.
// out is untouched unless every stage, residual and endpoint check passes.
// residual is diagnostic; it is written only on success and is no error bound.
// Only nonzero signed cells are supported; unlike the generic reference helper,
// this private native entry point explicitly refuses a zero-length step.
GPUFUN
int CavityNative_gauss(const double q[6], double h, double Q,
                       const double profiles[40], double aperture,
                       int64_t max_iter, double tol, double out[6],
                       double* residual) {
    if (!isfinite(h) || h == 0. || !isfinite(Q) || !isfinite(aperture)
        || aperture <= 0. || !isfinite(tol) || tol <= 0. || max_iter <= 0)
        return CAVITY_NATIVE_ARGUMENT;
    for (int i=0; i<6; ++i)
        if (!isfinite(q[i])) return CAVITY_NATIVE_NONFINITE;
    for (int i=0; i<40; ++i)
        if (!isfinite(profiles[i])) return CAVITY_NATIVE_NONFINITE;

    // Literal binary64 values represented by verification/support/gauss.py.
    const double C[2] = {0.21132486540518713, 0.7886751345948129};
    const double A[2][2] = {{0.25, -0.038675134594812866},
                            {0.5386751345948129, 0.25}};
    double initial[6], stages[2][6], values[2][6], proposed[2][6];
    int status = CavityNative_source(q, Q, profiles, aperture, initial);
    if (status != CAVITY_NATIVE_OK) return status;
    for (int i=0; i<2; ++i)
        for (int k=0; k<6; ++k) {
            stages[i][k] = q[k]+C[i]*h*initial[k];
            if (!isfinite(stages[i][k])) return CAVITY_NATIVE_NONFINITE;
        }
    for (int64_t iteration=0; iteration<max_iter; ++iteration) {
        for (int i=0; i<2; ++i) {
            status = CavityNative_source(stages[i], Q, profiles+10*(i+1),
                                         aperture, values[i]);
            if (status != CAVITY_NATIVE_OK) return status;
        }
        double change = 0.;
        for (int i=0; i<2; ++i)
            for (int k=0; k<6; ++k) {
                proposed[i][k] = q[k]+h*(A[i][0]*values[0][k]
                                        +A[i][1]*values[1][k]);
                if (!isfinite(proposed[i][k])) return CAVITY_NATIVE_NONFINITE;
                change = fmax(change, fabs(stages[i][k]-proposed[i][k]));
            }
        for (int i=0; i<2; ++i)
            for (int k=0; k<6; ++k) stages[i][k] = proposed[i][k];
        if (change < tol) {
            // Reevaluate F at reassigned stages: an iteration delta alone
            // does not satisfy the reference defining-equation contract.
            for (int i=0; i<2; ++i) {
                status = CavityNative_source(stages[i], Q, profiles+10*(i+1),
                                             aperture, values[i]);
                if (status != CAVITY_NATIVE_OK) return status;
            }
            double actual = 0.;
            for (int i=0; i<2; ++i)
                for (int k=0; k<6; ++k)
                    actual = fmax(actual, fabs(stages[i][k]-q[k]
                        -h*(A[i][0]*values[0][k]+A[i][1]*values[1][k])));
            if (actual < tol) {
                double result[6], final_flow[6];
                for (int k=0; k<6; ++k) {
                    result[k] = q[k]+h*((values[0][k]+values[1][k])*0.5);
                    if (!isfinite(result[k])) return CAVITY_NATIVE_NONFINITE;
                }
                status = CavityNative_source(result, Q, profiles+30,
                                             aperture, final_flow);
                if (status != CAVITY_NATIVE_OK) return status;
                for (int k=0; k<6; ++k) out[k] = result[k];
                *residual = actual;
                return CAVITY_NATIVE_OK;
            }
        }
    }
    return CAVITY_NATIVE_ITERATIONS;
}
#endif
