// Optional private first-rho derivative of the same implicit Gauss word.
// This is not a nonlinear LL map, stochastic law or response-optics kernel.
#ifndef XTRACK_CAVITY_NATIVE_RESPONSE_H
#define XTRACK_CAVITY_NATIVE_RESPONSE_H
#include "xtrack/beam_elements/elements_src/cavity_native_tangent.h"
#include "xtrack/beam_elements/elements_src/cavity_native_secondary.h"

// Profiles contain four TRUE twelve-value planes: entrance, stage0, stage1,
// exit, each Re0..5/Im0..5. The nominal orbit consumes its Re/Im0..4 subset.
// r=dq/d(rho)|rho=0 in local canonical (X,Y,T,Px,Py,E), at fixed hardware,
// charge/mass and physical planes. Carry rin between cells. Physical rho*r
// must remain a SEPARATE response; adding it to an absolute state can lose it.
// r_i=rin+h sum_j A_ij(J_j r_j+S1_j), using the SAME accepted nominal stages.
//
// work[0]=first-rho changed RF work, work[1]=direct LL reaction work, PER CELL.
// For this Phi=0 target, the defining discrete E equation gives
// rout_E-rin_E=work[0]+work[1], up to floating/solve defects. Schott endpoint
// energy versus its own stage quadrature has a separate finite-word defect;
// no continuum Schott identity or accepted output ledger is asserted here.
// All outputs are transactional. Residuals are diagnostics, not error bounds.
GPUFUN
int CavityNative_gauss_response(const double q[6], const double rin[6],
        double h, double Q, const double profiles[48], double aperture,
        int64_t max_iter, double tol, double response_tol, double out[6],
        double rout[6], double work[2], double* residual,
        double* response_residual) {
    if (!isfinite(response_tol) || response_tol <= 0.)
        return CAVITY_NATIVE_ARGUMENT;
    for (int k=0; k<6; ++k)
        if (!isfinite(rin[k])) return CAVITY_NATIVE_NONFINITE;
    for (int k=0; k<48; ++k)
        if (!isfinite(profiles[k])) return CAVITY_NATIVE_NONFINITE;
    double nominal_profiles[40];
    for (int plane=0; plane<4; ++plane)
        for (int k=0; k<5; ++k) {
            nominal_profiles[plane*10+k] = profiles[plane*12+k];
            nominal_profiles[plane*10+k+5] = profiles[plane*12+k+6];
        }
    double endpoint[6], stages[12], nominal_residual;
    int status = CavityNative_gauss_stages(q, h, Q, nominal_profiles, aperture,
        max_iter, tol, endpoint, &nominal_residual, stages);
    if (status != CAVITY_NATIVE_OK) return status;
    double entrance_jacobian[36], entrance_source[6], entrance_rhs[6];
    status = CavityNative_jacobian(q, Q, nominal_profiles, aperture,
                                    entrance_jacobian);
    if (status != CAVITY_NATIVE_OK) return status;
    status = CavityNative_reaction_coefficient(q, Q, profiles, aperture,
                                               entrance_source);
    if (status != CAVITY_NATIVE_OK) return status;
    for (int row=0; row<6; ++row) {
        double value = entrance_source[row];
        for (int col=0; col<6; ++col)
            value += entrance_jacobian[row*6+col]*rin[col];
        if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
        entrance_rhs[row] = value;
    }
    double jacobian[2][36], forcing[2][6];
    for (int i=0; i<2; ++i) {
        status = CavityNative_jacobian(stages+i*6, Q,
            nominal_profiles+(i+1)*10, aperture, jacobian[i]);
        if (status != CAVITY_NATIVE_OK) return status;
        status = CavityNative_reaction_coefficient(stages+i*6, Q,
            profiles+(i+1)*12, aperture, forcing[i]);
        if (status != CAVITY_NATIVE_OK) return status;
    }
    const double C[2] = {0.21132486540518713, 0.7886751345948129};
    const double A[2][2] = {{0.25, -0.038675134594812866},
                            {0.5386751345948129, 0.25}};
    double response[2][6], proposed[2][6], products[2][6];
    for (int i=0; i<2; ++i)
        for (int k=0; k<6; ++k) {
            response[i][k] = rin[k]+C[i]*h*entrance_rhs[k];
            if (!isfinite(response[i][k])) return CAVITY_NATIVE_NONFINITE;
        }
    for (int64_t iteration=0; iteration<max_iter; ++iteration) {
        for (int i=0; i<2; ++i)
            for (int row=0; row<6; ++row) {
                double value = 0.;
                for (int col=0; col<6; ++col)
                    value += jacobian[i][row*6+col]*response[i][col];
                if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
                products[i][row] = value;
            }
        double change = 0.;
        for (int i=0; i<2; ++i)
            for (int k=0; k<6; ++k) {
                proposed[i][k] = rin[k]+h*(
                    A[i][0]*(products[0][k]+forcing[0][k])
                    +A[i][1]*(products[1][k]+forcing[1][k]));
                if (!isfinite(proposed[i][k])) return CAVITY_NATIVE_NONFINITE;
                change = fmax(change, fabs(proposed[i][k]-response[i][k]));
            }
        for (int i=0; i<2; ++i)
            for (int k=0; k<6; ++k) response[i][k] = proposed[i][k];
        if (change < response_tol) {
            for (int i=0; i<2; ++i)
                for (int row=0; row<6; ++row) {
                    double value = 0.;
                    for (int col=0; col<6; ++col)
                        value += jacobian[i][row*6+col]*response[i][col];
                    if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
                    products[i][row] = value;
                }
            double actual = 0.;
            for (int i=0; i<2; ++i)
                for (int k=0; k<6; ++k) {
                    const double defect = response[i][k]-rin[k]-h*(
                        A[i][0]*(products[0][k]+forcing[0][k])
                        +A[i][1]*(products[1][k]+forcing[1][k]));
                    if (!isfinite(defect)) return CAVITY_NATIVE_NONFINITE;
                    actual = fmax(actual, fabs(defect));
                }
            if (actual < response_tol) {
                double result[6], local_work[2];
                for (int k=0; k<6; ++k) {
                    result[k] = rin[k]+h*0.5*(
                        (products[0][k]+forcing[0][k])
                        +(products[1][k]+forcing[1][k]));
                    if (!isfinite(result[k])) return CAVITY_NATIVE_NONFINITE;
                }
                local_work[0] = h*((products[0][5]+products[1][5])*0.5);
                local_work[1] = h*((forcing[0][5]+forcing[1][5])*0.5);
                for (int k=0; k<2; ++k)
                    if (!isfinite(local_work[k])) return CAVITY_NATIVE_NONFINITE;
                for (int k=0; k<6; ++k) out[k] = endpoint[k];
                for (int k=0; k<6; ++k) rout[k] = result[k];
                for (int k=0; k<2; ++k) work[k] = local_work[k];
                *residual = nominal_residual;
                *response_residual = actual;
                return CAVITY_NATIVE_OK;
            }
        }
    }
    return CAVITY_NATIVE_ITERATIONS;
}
#endif
