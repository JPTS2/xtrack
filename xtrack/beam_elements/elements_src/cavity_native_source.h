// Private full-positive-root RF-only M2 source. No public Cavity path uses it.
// Algebra owner: cavity-modelling verification/maps/phase_resolved/
// conservative_source.py:native_rf_source(). Select F6 BEFORE CSE/lowering.
// Profile[0..4]=Re, [5..9]=Im derivatives wrt normalized Z, uncharged,
// normalized to a declared positive source_mass_eV. Runtime Q is actual
// signed charge_e*source_mass_eV/actual_mass_eV, once. Zero static/gauge.
// Diagnostic arithmetic only; no source/error/backend admission or optics.
#ifndef XTRACK_CAVITY_NATIVE_SOURCE_H
#define XTRACK_CAVITY_NATIVE_SOURCE_H
#include <math.h>
#include <stdint.h>

enum CavityNativeStatus {
    CAVITY_NATIVE_OK = 1,
    CAVITY_NATIVE_DOMAIN = -1,
    CAVITY_NATIVE_ITERATIONS = -2,
    CAVITY_NATIVE_NONFINITE = -3,
    CAVITY_NATIVE_ARGUMENT = -4
};

GPUFUN
int CavityNative_inputs(const double q[6], const double profile[10]) {
    for (int i=0; i<6; ++i) if (!isfinite(q[i])) return 0;
    for (int i=0; i<10; ++i) if (!isfinite(profile[i])) return 0;
    return 1;
}

// Transactional uncharged (Ax,Ay,Az,Phi); same-event particle T is retained.
GPUFUN
int CavityNative_potential(const double q[6], const double profile[10],
                           double potential[4]) {
    if (!CavityNative_inputs(q, profile)) return CAVITY_NATIVE_NONFINITE;
    double result[4];
    const double q0 = q[0];
    const double q1 = q[1];
    const double q2 = q[2];
    const double p0 = profile[0];
    const double p1 = profile[1];
    const double p2 = profile[2];
    const double p3 = profile[3];
    const double p4 = profile[4];
    const double p5 = profile[5];
    const double p6 = profile[6];
    const double p7 = profile[7];
    const double p8 = profile[8];
    const double p9 = profile[9];
    const double v0 = sin(q2);
    const double v1 = pow(q0, 2) + pow(q1, 2);
    const double v2 = (1.0/16.0)*v1;
    const double v3 = -p1*v0*v2 + (1.0/2.0)*p1*v0 - p3*v0*v2;
    const double v4 = cos(q2);
    const double v5 = p6*v4;
    const double v6 = p8*v2*v4 + v2*v5 - 1.0/2.0*v5;
    const double v7 = p0*v0;
    const double v8 = (1.0/4.0)*v1;
    const double v9 = pow(v1, 2);
    const double v10 = v0*v9;
    result[0] = q0*v3 + q0*v6;
    result[1] = q1*v3 + q1*v6;
    result[2] = (1.0/4.0)*p0*v0*v1 + (1.0/4.0)*p2*v0*v1 - 1.0/32.0*p2*v10 - 1.0/64.0*p4*v10 - p5*v4*v8 + (1.0/64.0)*p5*v4*v9 + p5*v4 - p7*v4*v8 + (1.0/32.0)*p7*v4*v9 + (1.0/64.0)*p9*v4*v9 - 1.0/64.0*v7*v9 - v7;
    result[3] = 0;
    for (int i=0; i<4; ++i)
        if (!isfinite(result[i])) return CAVITY_NATIVE_NONFINITE;
    for (int i=0; i<4; ++i) potential[i] = result[i];
    return CAVITY_NATIVE_OK;
}

// Canonical layout (X,Y,T,Px,Py,E), negative T--E symplectic block.
// No output is written on failure. Positive mechanical energy/shell and
// normalized radial aperture are checked before evaluating any square root.
GPUFUN
int CavityNative_source(const double q[6], double Q,
                        const double profile[10], double aperture,
                        double flow[6]) {
    if (!isfinite(Q) || !isfinite(aperture) || aperture <= 0.)
        return CAVITY_NATIVE_ARGUMENT;
    double potential[4];
    int status = CavityNative_potential(q, profile, potential);
    if (status != CAVITY_NATIVE_OK) return status;
    double gamma = q[5]-Q*potential[3];
    double ux = q[3]-Q*potential[0];
    double uy = q[4]-Q*potential[1];
    double square = (gamma-1.)*(gamma+1.)-ux*ux-uy*uy;
    double radius = hypot(q[0], q[1]);
    if (!isfinite(square) || !isfinite(radius)) return CAVITY_NATIVE_NONFINITE;
    if (gamma <= 1. || square <= 0. || radius > aperture)
        return CAVITY_NATIVE_DOMAIN;
    double result[6];
    const double q0 = q[0];
    const double q1 = q[1];
    const double q2 = q[2];
    const double q3 = q[3];
    const double q4 = q[4];
    const double q5 = q[5];
    const double p0 = profile[0];
    const double p1 = profile[1];
    const double p2 = profile[2];
    const double p3 = profile[3];
    const double p4 = profile[4];
    const double p5 = profile[5];
    const double p6 = profile[6];
    const double p7 = profile[7];
    const double p8 = profile[8];
    const double p9 = profile[9];
    const double v0 = sin(q2);
    const double v1 = pow(q0, 2) + pow(q1, 2);
    const double v2 = (1.0/16.0)*v1;
    const double v3 = p1*v0;
    const double v4 = v2*v3;
    const double v5 = p3*v0;
    const double v6 = v2*v5;
    const double v7 = (1.0/2.0)*p1*v0 - v4 - v6;
    const double v8 = cos(q2);
    const double v9 = v2*v8;
    const double v10 = -1.0/2.0*p6*v8 + p6*v9 + p8*v9;
    const double v11 = Q*(q0*v10 + q0*v7) - q3;
    const double v12 = -v11;
    const double v13 = Q*(q1*v10 + q1*v7) - q4;
    const double v14 = -v13;
    const double v15 = pow(pow(q5, 2) - pow(v12, 2) - pow(v14, 2) - 1, -1.0/2.0);
    const double v16 = (1.0/2.0)*q0;
    const double v17 = p0*v0;
    const double v18 = p2*v0;
    const double v19 = v16*v8;
    const double v20 = v17*v2;
    const double v21 = (1.0/8.0)*q0;
    const double v22 = v1*v18;
    const double v23 = v0*v2;
    const double v24 = p4*v23;
    const double v25 = q0*v9;
    const double v26 = v21*v8;
    const double v27 = p7*v1;
    const double v28 = -v21*v3 - v21*v5;
    const double v29 = p6*v26 + p8*v26;
    const double v30 = (1.0/2.0)*p1*v0 + v10 - v4 - v6;
    const double v31 = (1.0/2.0)*q1;
    const double v32 = v31*v8;
    const double v33 = (1.0/8.0)*q1;
    const double v34 = q1*v9;
    const double v35 = v33*v8;
    const double v36 = -v3*v33 - v33*v5;
    const double v37 = p6*v35 + p8*v35;
    const double v38 = p0*v8;
    const double v39 = p5*v0;
    const double v40 = pow(v1, 2);
    const double v41 = (1.0/64.0)*v40;
    const double v42 = (1.0/32.0)*v40;
    const double v43 = (1.0/2.0)*p1*v8 - p1*v9 - p3*v9;
    const double v44 = (1.0/2.0)*p6*v0 - p6*v23 - p8*v23;
    result[0] = -v11*v15;
    result[1] = -v13*v15;
    result[2] = q5*v15;
    result[3] = Q*(-p5*v19 + p5*v25 - p7*v19 + p9*v25 - q0*v20 - q0*v24 + v15*(v12*(q0*v28 + q0*v29 + v30) + v14*(q1*v28 + q1*v29)) + v16*v17 + v16*v18 - v21*v22 + v26*v27);
    result[4] = Q*(-p5*v32 + p5*v34 - p7*v32 + p9*v34 - q1*v20 - q1*v24 + v15*(v12*(q0*v36 + q0*v37) + v14*(q1*v36 + q1*v37 + v30)) + v17*v31 + v18*v31 - v22*v33 + v27*v35);
    result[5] = -Q*((1.0/4.0)*p0*v1*v8 + (1.0/4.0)*p2*v1*v8 - p2*v42*v8 - p4*v41*v8 + (1.0/4.0)*p5*v0*v1 + (1.0/4.0)*p7*v0*v1 - p7*v0*v42 - p9*v0*v41 + v15*(v12*(q0*v43 + q0*v44) + v14*(q1*v43 + q1*v44)) - v38*v41 - v38 - v39*v41 - v39);
    for (int i=0; i<6; ++i)
        if (!isfinite(result[i])) return CAVITY_NATIVE_NONFINITE;
    for (int i=0; i<6; ++i) flow[i] = result[i];
    return CAVITY_NATIVE_OK;
}
#endif
