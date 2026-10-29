// Optional private derivatives; the default F-only orbit never includes this.
// Generated arithmetic owner: conservative_source.native_rf_tangent_header().
// Layout is local canonical (X,Y,T,Px,Py,E), negative T--E block.
#ifndef XTRACK_CAVITY_NATIVE_TANGENT_H
#define XTRACK_CAVITY_NATIVE_TANGENT_H
#include "xtrack/beam_elements/elements_src/cavity_native_gauss.h"

// Same-event uncharged potential derivatives. Row-major (Ax,Ay,Az,Phi)
// by (X,Y,T), at fixed physical Z/profile planes. No output on failure.
GPUFUN
int CavityNative_potential_jacobian(const double q[6],
        const double profile[10], double jacobian[12]) {
    if (!CavityNative_inputs(q, profile)) return CAVITY_NATIVE_NONFINITE;
    double result[12];
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
    const double v0 = (1.0/8.0)*q0;
    const double v1 = sin(q2);
    const double v2 = p1*v1;
    const double v3 = p3*v1;
    const double v4 = -v0*v2 - v0*v3;
    const double v5 = cos(q2);
    const double v6 = v0*v5;
    const double v7 = p6*v6 + p8*v6;
    const double v8 = pow(q0, 2) + pow(q1, 2);
    const double v9 = (1.0/16.0)*v8;
    const double v10 = v5*v9;
    const double v11 = (1.0/2.0)*p1*v1 + p6*v10 - 1.0/2.0*p6*v5 + p8*v10 - v2*v9 - v3*v9;
    const double v12 = (1.0/8.0)*q1;
    const double v13 = -v12*v2 - v12*v3;
    const double v14 = v12*v5;
    const double v15 = p6*v14 + p8*v14;
    const double v16 = -p1*v10 + (1.0/2.0)*p1*v5 - p3*v10;
    const double v17 = v1*v9;
    const double v18 = (1.0/2.0)*p6*v1 - p6*v17 - p8*v17;
    const double v19 = (1.0/2.0)*q0*v5;
    const double v20 = q0*v17;
    const double v21 = p2*v1*v8;
    const double v22 = (1.0/2.0)*q1*v5;
    const double v23 = q1*v17;
    const double v24 = p0*v5;
    const double v25 = p5*v1;
    const double v26 = pow(v8, 2);
    const double v27 = (1.0/64.0)*v26;
    const double v28 = (1.0/32.0)*v26;
    result[0] = q0*v4 + q0*v7 + v11;
    result[1] = q0*v13 + q0*v15;
    result[2] = q0*v16 + q0*v18;
    result[3] = q1*v4 + q1*v7;
    result[4] = q1*v13 + q1*v15 + v11;
    result[5] = q1*v16 + q1*v18;
    result[6] = (1.0/2.0)*p0*q0*v1 - p0*v20 + (1.0/2.0)*p2*q0*v1 - p4*v20 + (1.0/16.0)*p5*q0*v5*v8 - p5*v19 + (1.0/8.0)*p7*q0*v5*v8 - p7*v19 + (1.0/16.0)*p9*q0*v5*v8 - v0*v21;
    result[7] = (1.0/2.0)*p0*q1*v1 - p0*v23 + (1.0/2.0)*p2*q1*v1 - p4*v23 + (1.0/16.0)*p5*q1*v5*v8 - p5*v22 + (1.0/8.0)*p7*q1*v5*v8 - p7*v22 + (1.0/16.0)*p9*q1*v5*v8 - v12*v21;
    result[8] = (1.0/4.0)*p0*v5*v8 - p2*v28*v5 + (1.0/4.0)*p2*v5*v8 - p4*v27*v5 + (1.0/4.0)*p5*v1*v8 - p7*v1*v28 + (1.0/4.0)*p7*v1*v8 - p9*v1*v27 - v24*v27 - v24 - v25*v27 - v25;
    result[9] = 0;
    result[10] = 0;
    result[11] = 0;
    for (int i=0; i<12; ++i)
        if (!isfinite(result[i])) return CAVITY_NATIVE_NONFINITE;
    for (int i=0; i<12; ++i) jacobian[i] = result[i];
    return CAVITY_NATIVE_OK;
}

// Registered physical state Jacobian of this same RF-only full-root F6.
// Row-major 6x6. Profile planes/Q fixed under this launch derivative.
GPUFUN
int CavityNative_jacobian(const double q[6], double Q,
        const double profile[10], double aperture, double jacobian[36]) {
    double flow[6];
    int status = CavityNative_source(q, Q, profile, aperture, flow);
    if (status != CAVITY_NATIVE_OK) return status;
    double result[36];
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
    const double v1 = p1*v0;
    const double v2 = (1.0/8.0)*q0;
    const double v3 = p3*v0;
    const double v4 = -v1*v2 - v2*v3;
    const double v5 = cos(q2);
    const double v6 = p6*v5;
    const double v7 = p8*v5;
    const double v8 = v2*v6 + v2*v7;
    const double v9 = q1*v4 + q1*v8;
    const double v10 = pow(q0, 2);
    const double v11 = pow(q1, 2);
    const double v12 = v10 + v11;
    const double v13 = (1.0/16.0)*v12;
    const double v14 = v1*v13;
    const double v15 = v13*v3;
    const double v16 = (1.0/2.0)*p1*v0 - v14 - v15;
    const double v17 = v13*v6 + v13*v7 - 1.0/2.0*v6;
    const double v18 = Q*(q1*v16 + q1*v17) - q4;
    const double v19 = -v18;
    const double v20 = Q*(q0*v16 + q0*v17) - q3;
    const double v21 = -v20;
    const double v22 = (1.0/2.0)*v1 - v14 - v15 + v17;
    const double v23 = q0*v4 + q0*v8 + v22;
    const double v24 = v19*v9 + v21*v23;
    const double v25 = -v24;
    const double v26 = pow(q5, 2);
    const double v27 = -pow(v19, 2) - pow(v21, 2) + v26 - 1;
    const double v28 = 1.0/v27;
    const double v29 = v20*v28;
    const double v30 = pow(v27, -1.0/2.0);
    const double v31 = Q*v30;
    const double v32 = (1.0/8.0)*q1;
    const double v33 = -v1*v32 - v3*v32;
    const double v34 = v32*v6 + v32*v7;
    const double v35 = q0*v33 + q0*v34;
    const double v36 = q1*v33 + q1*v34 + v22;
    const double v37 = v19*v36 + v21*v35;
    const double v38 = -v37;
    const double v39 = p1*v12;
    const double v40 = (1.0/16.0)*v5;
    const double v41 = p3*v12;
    const double v42 = -1.0/2.0*p1*v5 + v39*v40 + v40*v41;
    const double v43 = -v42;
    const double v44 = p6*v12;
    const double v45 = (1.0/16.0)*v0;
    const double v46 = p8*v12;
    const double v47 = -1.0/2.0*p6*v0 + v44*v45 + v45*v46;
    const double v48 = -v47;
    const double v49 = q0*v43 + q0*v48;
    const double v50 = q1*v43 + q1*v48;
    const double v51 = v19*v50 + v21*v49;
    const double v52 = -v51;
    const double v53 = pow(v27, -3.0/2.0);
    const double v54 = v20*v53;
    const double v55 = v18*v28;
    const double v56 = v18*v53;
    const double v57 = q5*v53;
    const double v58 = Q*v57;
    const double v59 = p1 + p3;
    const double v60 = p6 + p8;
    const double v61 = v0*v59 - v5*v60;
    const double v62 = -v61;
    const double v63 = 2*p6*v5 + 2*p8*v5 - 2*v1 - 2*v3 - v61;
    const double v64 = p2*v0;
    const double v65 = (1.0/4.0)*v10;
    const double v66 = p0*v0;
    const double v67 = (1.0/8.0)*v10;
    const double v68 = p4*v0;
    const double v69 = p7*v5;
    const double v70 = p5*v5;
    const double v71 = p9*v5;
    const double v72 = Q*v53;
    const double v73 = v24*v72;
    const double v74 = (1.0/8.0)*v12;
    const double v75 = (1.0/2.0)*p2*v0 - 1.0/2.0*p7*v5 - v13*v66 - v13*v68 + v13*v70 + v13*v71 - v64*v74 + (1.0/2.0)*v66 + v69*v74 - 1.0/2.0*v70;
    const double v76 = p6*v5 + p8*v5 - v1 - v3;
    const double v77 = Q*v23;
    const double v78 = Q*v9;
    const double v79 = v66 - v70;
    const double v80 = q1*v2*(2*p7*v5 + p9*v5 - 2*v64 - v68 - v79) + v30*((1.0/8.0)*q0*v19*v76 + (1.0/8.0)*q1*v21*v76 - v35*v77 - v36*v78);
    const double v81 = v5*v59;
    const double v82 = v0*v60;
    const double v83 = q1*(v81 + v82);
    const double v84 = v42 + v47;
    const double v85 = v12*v5;
    const double v86 = v0*v12;
    const double v87 = (1.0/2.0)*p0*v5 - 1.0/16.0*p0*v85 + (1.0/2.0)*p2*v5 - 1.0/8.0*p2*v85 - 1.0/16.0*p4*v85 + (1.0/2.0)*p5*v0 - 1.0/16.0*p5*v86 + (1.0/2.0)*p7*v0 - 1.0/8.0*p7*v86 - 1.0/16.0*p9*v86;
    const double v88 = q0*v87 + v30*(-v19*v2*v83 + v21*(-v67*v81 - v67*v82 - v84) - v49*v77 - v50*v78);
    const double v89 = v24*v28;
    const double v90 = v37*v72;
    const double v91 = (1.0/4.0)*v11;
    const double v92 = (1.0/8.0)*v11;
    const double v93 = q1*v87 + v30*(-Q*v35*v49 - Q*v36*v50 + v19*(-v81*v92 - v82*v92 - v84) - v2*v21*v83);
    const double v94 = v28*v37;
    const double v95 = v51*v72;
    const double v96 = v0*(-8*p1 + v39 + v41) - v5*(-8*p6 + v44 + v46);
    const double v97 = (1.0/4.0)*v12;
    const double v98 = pow(v12, 2);
    const double v99 = (1.0/32.0)*v98;
    const double v100 = (1.0/64.0)*v98;
    const double v101 = v28*v51;
    result[0] = -v31*(v23 + v25*v29);
    result[1] = -v31*(v29*v38 + v35);
    result[2] = -v31*(v29*v52 + v49);
    result[3] = v30*(-v21*v29 + 1);
    result[4] = -v19*v54;
    result[5] = q5*v54;
    result[6] = -v31*(v25*v55 + v9);
    result[7] = -v31*(v36 + v38*v55);
    result[8] = -v31*(v50 + v52*v55);
    result[9] = -v21*v56;
    result[10] = v30*(-v19*v55 + 1);
    result[11] = q5*v56;
    result[12] = v25*v58;
    result[13] = v38*v58;
    result[14] = v52*v58;
    result[15] = v21*v57;
    result[16] = v19*v57;
    result[17] = v30*(-v26*v28 + 1);
    result[18] = Q*(v25*v73 + v30*(-Q*pow(v23, 2) - Q*pow(v9, 2) + (1.0/8.0)*q0*v21*v63 + (1.0/8.0)*q1*v19*v62) - v64*v65 + v65*v69 - v66*v67 - v67*v68 + v67*v70 + v67*v71 + v75);
    result[19] = Q*(v38*v73 + v80);
    result[20] = Q*(v52*v73 + v88);
    result[21] = v31*(v21*v89 + v23);
    result[22] = v31*(v19*v89 + v9);
    result[23] = -v24*v58;
    result[24] = Q*(v25*v90 + v80);
    result[25] = Q*(v30*(-Q*pow(v35, 2) - Q*pow(v36, 2) + (1.0/8.0)*q0*v21*v62 + (1.0/8.0)*q1*v19*v63) + v38*v90 - v64*v91 - v66*v92 - v68*v92 + v69*v91 + v70*v92 + v71*v92 + v75);
    result[26] = Q*(v52*v90 + v93);
    result[27] = v31*(v21*v94 + v35);
    result[28] = v31*(v19*v94 + v36);
    result[29] = -v37*v58;
    result[30] = -Q*(v25*v95 + v88);
    result[31] = -Q*(v38*v95 + v93);
    result[32] = -Q*(v100*v66 + v100*v68 - v100*v70 - v100*v71 + v30*(-Q*pow(v49, 2) - Q*pow(v50, 2) + (1.0/16.0)*q0*v21*v96 + (1.0/16.0)*q1*v19*v96) + v52*v95 - v64*v97 + v64*v99 - v66*v97 + v69*v97 - v69*v99 + v70*v97 + v79);
    result[33] = -v31*(v101*v21 + v49);
    result[34] = -v31*(v101*v19 + v50);
    result[35] = v51*v58;
    for (int i=0; i<36; ++i)
        if (!isfinite(result[i])) return CAVITY_NATIVE_NONFINITE;
    for (int i=0; i<36; ++i) jacobian[i] = result[i];
    return CAVITY_NATIVE_OK;
}

// Derivative of the mathematical implicit Gauss root at the approximately
// accepted nominal stages, not the stopping-test branch/floating program or
// the exact ODE. Nominal and variational residuals remain distinct diagnostics;
// neither bounds the map/tangent error. All outputs are transactional.
// D_i=I+h sum_j A_ij J_j D_j; M=I+h sum_i b_i J_i D_i.
GPUFUN
int CavityNative_gauss_tangent(const double q[6], double h, double Q,
        const double profiles[40], double aperture, int64_t max_iter,
        double tol, double derivative_tol, double out[6], double matrix[36],
        double* residual, double* derivative_residual) {
    if (!isfinite(derivative_tol) || derivative_tol <= 0.)
        return CAVITY_NATIVE_ARGUMENT;
    double endpoint[6], stages[12], nominal_residual;
    int status = CavityNative_gauss_stages(q, h, Q, profiles, aperture,
        max_iter, tol, endpoint, &nominal_residual, stages);
    if (status != CAVITY_NATIVE_OK) return status;
    double jacobian[2][36];
    for (int i=0; i<2; ++i) {
        status = CavityNative_jacobian(stages+i*6, Q, profiles+(i+1)*10,
                                       aperture, jacobian[i]);
        if (status != CAVITY_NATIVE_OK) return status;
    }
    // Literal stored coefficients, identical to the nominal Gauss word.
    const double A[2][2] = {{0.25, -0.038675134594812866},
                            {0.5386751345948129, 0.25}};
    double variations[2][36], proposed[2][36], products[2][36];
    for (int i=0; i<2; ++i)
        for (int row=0; row<6; ++row)
            for (int col=0; col<6; ++col)
                variations[i][row*6+col] = row == col ? 1. : 0.;
    for (int64_t iteration=0; iteration<max_iter; ++iteration) {
        for (int i=0; i<2; ++i)
            for (int row=0; row<6; ++row)
                for (int col=0; col<6; ++col) {
                    double value = 0.;
                    for (int k=0; k<6; ++k)
                        value += jacobian[i][row*6+k]
                            *variations[i][k*6+col];
                    if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
                    products[i][row*6+col] = value;
                }
        double change = 0.;
        for (int i=0; i<2; ++i)
            for (int row=0; row<6; ++row)
                for (int col=0; col<6; ++col) {
                    const int k = row*6+col;
                    proposed[i][k] = (row == col ? 1. : 0.)
                        +h*(A[i][0]*products[0][k]+A[i][1]*products[1][k]);
                    if (!isfinite(proposed[i][k]))
                        return CAVITY_NATIVE_NONFINITE;
                    change = fmax(change,
                                   fabs(proposed[i][k]-variations[i][k]));
                }
        for (int i=0; i<2; ++i)
            for (int k=0; k<36; ++k) variations[i][k] = proposed[i][k];
        if (change < derivative_tol) {
            // Reevaluate J_j D_j at reassigned variations for the actual
            // defining equation, just as the nominal stage solve does.
            for (int i=0; i<2; ++i)
                for (int row=0; row<6; ++row)
                    for (int col=0; col<6; ++col) {
                        double value = 0.;
                        for (int k=0; k<6; ++k)
                            value += jacobian[i][row*6+k]
                                *variations[i][k*6+col];
                        if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
                        products[i][row*6+col] = value;
                    }
            double actual = 0.;
            for (int i=0; i<2; ++i)
                for (int row=0; row<6; ++row)
                    for (int col=0; col<6; ++col) {
                        const int k = row*6+col;
                        double defect = variations[i][k]
                            -(row == col ? 1. : 0.)
                            -h*(A[i][0]*products[0][k]
                                +A[i][1]*products[1][k]);
                        if (!isfinite(defect)) return CAVITY_NATIVE_NONFINITE;
                        actual = fmax(actual, fabs(defect));
                    }
            if (actual < derivative_tol) {
                double result[36];
                for (int row=0; row<6; ++row)
                    for (int col=0; col<6; ++col) {
                        const int k = row*6+col;
                        result[k] = (row == col ? 1. : 0.)
                            +h*((products[0][k]+products[1][k])*0.5);
                        if (!isfinite(result[k]))
                            return CAVITY_NATIVE_NONFINITE;
                    }
                for (int k=0; k<6; ++k) out[k] = endpoint[k];
                for (int k=0; k<36; ++k) matrix[k] = result[k];
                *residual = nominal_residual;
                *derivative_residual = actual;
                return CAVITY_NATIVE_OK;
            }
        }
    }
    return CAVITY_NATIVE_ITERATIONS;
}
#endif
