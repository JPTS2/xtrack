// Optional private first-rho derivative of the same implicit Gauss word.
// This is not a nonlinear LL map, stochastic law or response-optics kernel.
#ifndef XTRACK_CAVITY_NATIVE_RESPONSE_H
#define XTRACK_CAVITY_NATIVE_RESPONSE_H
#include "xtrack/beam_elements/elements_src/cavity_native_tangent.h"
#include "xtrack/beam_elements/elements_src/cavity_native_secondary.h"
#include "xtrack/beam_elements/elements_src/cavity_native_spin.h"

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

// A DIFFERENT radiative finite word in the Schott-shifted response coordinate.
// Existing particle_response.local_split supplies S1=D0s-ell; set xi=r-s.
// xi'=J*xi+J*s-ell. Initialize xi=physical_rin-s_in ONCE at the global
// entrance, then carry xi_out directly between contiguous cells. Physical
// stage response is xi_i+s_i; physical endpoint response is xi_out+s_out.
// All six components and nonzero interior boundary s values are retained.
// This is not the direct-word derivative above or an endpoint-energy repair.
// Profiles40 are the true Re/Im0..4 values; no material D5 is consumed.
//
// work=(changed_RF_work1, positive_forward_emission1, Schott_boundary1).
// For Phi=0, the transformed defining E equation gives
// physical_rout_E-physical_rin_E=work[0]-work[1]+work[2], up to FP/solve
// defects. This is a property of THIS new word, not a removed direct-word
// Schott quadrature defect or numerical/physical accuracy certificate.
// q, xi, physical response, work and both residual outputs are transactional.
GPUFUN
int CavityNative_gauss_schott_response_stage_data(const double q[6],
        const double xi_in[6],
        double h, double Q, const double profiles[40], double aperture,
        int64_t max_iter, double tol, double response_tol, double out[6],
        double xi_out[6], double physical_response[6], double work[3],
        double* residual, double* response_residual,
        double* accepted_orbit_stages, double* accepted_physical_stages,
        double* accepted_auxiliary_stages) {
    if (!isfinite(response_tol) || response_tol <= 0.)
        return CAVITY_NATIVE_ARGUMENT;
    for (int k=0; k<6; ++k)
        if (!isfinite(xi_in[k])) return CAVITY_NATIVE_NONFINITE;
    double endpoint[6], stages[12], nominal_residual;
    int status = CavityNative_gauss_stages(q, h, Q, profiles, aperture,
        max_iter, tol, endpoint, &nominal_residual, stages);
    if (status != CAVITY_NATIVE_OK) return status;
    double entrance[12], exit[12], entrance_jacobian[36], entrance_rhs[6];
    status = CavityNative_schott_source(q, Q, profiles, aperture, entrance);
    if (status != CAVITY_NATIVE_OK) return status;
    status = CavityNative_schott_source(endpoint, Q, profiles+30, aperture, exit);
    if (status != CAVITY_NATIVE_OK) return status;
    status = CavityNative_jacobian(q, Q, profiles, aperture, entrance_jacobian);
    if (status != CAVITY_NATIVE_OK) return status;
    for (int row=0; row<6; ++row) {
        double value = -entrance[6+row];
        for (int col=0; col<6; ++col)
            value += entrance_jacobian[row*6+col]
                *(xi_in[col]+entrance[col]);
        if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
        entrance_rhs[row] = value;
    }
    double jacobian[2][36], split[2][12];
    for (int i=0; i<2; ++i) {
        status = CavityNative_jacobian(stages+i*6, Q, profiles+(i+1)*10,
                                       aperture, jacobian[i]);
        if (status != CAVITY_NATIVE_OK) return status;
        status = CavityNative_schott_source(stages+i*6, Q, profiles+(i+1)*10,
                                            aperture, split[i]);
        if (status != CAVITY_NATIVE_OK) return status;
    }
    const double C[2] = {0.21132486540518713, 0.7886751345948129};
    const double A[2][2] = {{0.25, -0.038675134594812866},
                            {0.5386751345948129, 0.25}};
    double xi[2][6], proposed[2][6], products[2][6];
    for (int i=0; i<2; ++i)
        for (int k=0; k<6; ++k) {
            xi[i][k] = xi_in[k]+C[i]*h*entrance_rhs[k];
            if (!isfinite(xi[i][k])) return CAVITY_NATIVE_NONFINITE;
        }
    for (int64_t iteration=0; iteration<max_iter; ++iteration) {
        for (int i=0; i<2; ++i)
            for (int row=0; row<6; ++row) {
                double value = 0.;
                for (int col=0; col<6; ++col)
                    value += jacobian[i][row*6+col]*(xi[i][col]+split[i][col]);
                if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
                products[i][row] = value;
            }
        double change = 0.;
        for (int i=0; i<2; ++i)
            for (int k=0; k<6; ++k) {
                proposed[i][k] = xi_in[k]+h*(
                    A[i][0]*(products[0][k]-split[0][6+k])
                    +A[i][1]*(products[1][k]-split[1][6+k]));
                if (!isfinite(proposed[i][k])) return CAVITY_NATIVE_NONFINITE;
                change = fmax(change, fabs(proposed[i][k]-xi[i][k]));
            }
        for (int i=0; i<2; ++i)
            for (int k=0; k<6; ++k) xi[i][k] = proposed[i][k];
        if (change < response_tol) {
            for (int i=0; i<2; ++i)
                for (int row=0; row<6; ++row) {
                    double value = 0.;
                    for (int col=0; col<6; ++col)
                        value += jacobian[i][row*6+col]
                            *(xi[i][col]+split[i][col]);
                    if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
                    products[i][row] = value;
                }
            double actual = 0.;
            for (int i=0; i<2; ++i)
                for (int k=0; k<6; ++k) {
                    const double defect = xi[i][k]-xi_in[k]-h*(
                        A[i][0]*(products[0][k]-split[0][6+k])
                        +A[i][1]*(products[1][k]-split[1][6+k]));
                    if (!isfinite(defect)) return CAVITY_NATIVE_NONFINITE;
                    actual = fmax(actual, fabs(defect));
                }
            if (actual < response_tol) {
                double auxiliary[6], physical[6], local_work[3];
                double physical_stages[12];
                if (accepted_physical_stages != 0)
                    for (int i=0; i<2; ++i)
                        for (int k=0; k<6; ++k) {
                            physical_stages[i*6+k] = xi[i][k]+split[i][k];
                            if (!isfinite(physical_stages[i*6+k]))
                                return CAVITY_NATIVE_NONFINITE;
                        }
                for (int k=0; k<6; ++k) {
                    auxiliary[k] = xi_in[k]+h*0.5*(
                        (products[0][k]-split[0][6+k])
                        +(products[1][k]-split[1][6+k]));
                    physical[k] = auxiliary[k]+exit[k];
                    if (!isfinite(auxiliary[k]) || !isfinite(physical[k]))
                        return CAVITY_NATIVE_NONFINITE;
                }
                local_work[0] = h*((products[0][5]+products[1][5])*0.5);
                local_work[1] = h*((split[0][11]+split[1][11])*0.5);
                local_work[2] = exit[5]-entrance[5];
                for (int k=0; k<3; ++k)
                    if (!isfinite(local_work[k])) return CAVITY_NATIVE_NONFINITE;
                for (int k=0; k<6; ++k) out[k] = endpoint[k];
                for (int k=0; k<6; ++k) xi_out[k] = auxiliary[k];
                for (int k=0; k<6; ++k) physical_response[k] = physical[k];
                for (int k=0; k<3; ++k) work[k] = local_work[k];
                *residual = nominal_residual;
                *response_residual = actual;
                if (accepted_orbit_stages != 0)
                    for (int k=0; k<12; ++k)
                        accepted_orbit_stages[k] = stages[k];
                if (accepted_physical_stages != 0)
                    for (int k=0; k<12; ++k)
                        accepted_physical_stages[k] = physical_stages[k];
                if (accepted_auxiliary_stages != 0)
                    for (int i=0; i<2; ++i)
                        for (int k=0; k<6; ++k)
                            accepted_auxiliary_stages[i*6+k] = xi[i][k];
                return CAVITY_NATIVE_OK;
            }
        }
    }
    return CAVITY_NATIVE_ITERATIONS;
}

// Compatibility stage interface; only the new global spin chart requests
// xi directly, avoiding subtraction of a resolved Schott term from r=xi+s.
GPUFUN
int CavityNative_gauss_schott_response_stages(const double q[6],
        const double xi_in[6], double h, double Q, const double profiles[40],
        double aperture, int64_t max_iter, double tol, double response_tol,
        double out[6], double xi_out[6], double physical_response[6],
        double work[3], double* residual, double* response_residual,
        double* accepted_orbit_stages, double* accepted_physical_stages) {
    return CavityNative_gauss_schott_response_stage_data(q, xi_in, h, Q,
        profiles, aperture, max_iter, tol, response_tol, out, xi_out,
        physical_response, work, residual, response_residual,
        accepted_orbit_stages, accepted_physical_stages, 0);
}

GPUFUN
int CavityNative_gauss_schott_response(const double q[6], const double xi_in[6],
        double h, double Q, const double profiles[40], double aperture,
        int64_t max_iter, double tol, double response_tol, double out[6],
        double xi_out[6], double physical_response[6], double work[3],
        double* residual, double* response_residual) {
    return CavityNative_gauss_schott_response_stages(q, xi_in, h, Q,
        profiles, aperture, max_iter, tol, response_tol, out, xi_out,
        physical_response, work, residual, response_residual, 0, 0);
}

// First-rho spin coefficient on the NEW Schott-response word above. True
// profile48 supplies material D5 for the physical LL Thomas coefficient.
// Two identical nominal solves are deliberate in this private diagnostic;
// finite endpoints and accepted orbit stages must agree bitwise (including
// signed zero). The passive R0 and its accepted stages come from the existing
// passive word, not a newly CSE-lowered reconstruction of source24 G0.
//
// H1_i=skew(Dq(clock*Omega0)_i*(xi_i+s_i)+clock*OmegaExtra1_i).
// P_i=h sum_j A_ij(G0_j P_j+H1_j R0_j), with LOCAL P_in=0.
// Return local U0 and U1 separately; a caller carrying global R0_in/P_in
// composes P_out=U1*R0_in+U0*P_in, R0_out=U0*R0_in. Carry xi globally too.
// No spin Schott boundary rotation occurs in fixed laboratory rest-spin axes.
// No rho multiplication, normalization, LL feedback or stochastic law here.
// Residuals are nominal, auxiliary, passive spin, first-rho spin, in that order.
// All output arrays (including stage buffers above) are transactional.
GPUFUN
int CavityNative_gauss_schott_spin_response(const double q[6],
        const double xi_in[6], double h, double Q, double anomaly,
        const double profiles[48], double aperture, int64_t max_iter,
        double tol, double response_tol, double spin_tol,
        double spin_response_tol, double out[6], double xi_out[6],
        double physical_response[6], double rotation[9],
        double rotation_response[9], double work[3], double residuals[4]) {
    if (!isfinite(spin_response_tol) || spin_response_tol <= 0.)
        return CAVITY_NATIVE_ARGUMENT;
    for (int k=0; k<48; ++k)
        if (!isfinite(profiles[k])) return CAVITY_NATIVE_NONFINITE;
    double nominal_profiles[40];
    for (int plane=0; plane<4; ++plane)
        for (int k=0; k<5; ++k) {
            nominal_profiles[plane*10+k] = profiles[plane*12+k];
            nominal_profiles[plane*10+k+5] = profiles[plane*12+k+6];
        }
    double spin_endpoint[6], mean_endpoint[6], nominal_rotation[9];
    double spin_qstages[12], mean_qstages[12], spin_stages[18];
    double physical_stages[12], auxiliary[6], physical[6], local_work[3];
    double nominal_residual, mean_nominal_residual, auxiliary_residual;
    double passive_residual;
    int status = CavityNative_gauss_spin_stages(q, h, Q, anomaly,
        nominal_profiles, aperture, max_iter, tol, spin_tol, spin_endpoint,
        nominal_rotation, &nominal_residual, &passive_residual,
        spin_qstages, spin_stages);
    if (status != CAVITY_NATIVE_OK) return status;
    status = CavityNative_gauss_schott_response_stages(q, xi_in, h, Q,
        nominal_profiles, aperture, max_iter, tol, response_tol, mean_endpoint,
        auxiliary, physical, local_work, &mean_nominal_residual,
        &auxiliary_residual, mean_qstages, physical_stages);
    if (status != CAVITY_NATIVE_OK) return status;
    for (int k=0; k<6; ++k)
        if (spin_endpoint[k] != mean_endpoint[k]
                || signbit(spin_endpoint[k]) != signbit(mean_endpoint[k]))
            return CAVITY_NATIVE_ARGUMENT;
    for (int k=0; k<12; ++k)
        if (spin_qstages[k] != mean_qstages[k]
                || signbit(spin_qstages[k]) != signbit(mean_qstages[k]))
            return CAVITY_NATIVE_ARGUMENT;
    double entrance_split[12], entrance_packet[24];
    status = CavityNative_schott_source(q, Q, nominal_profiles, aperture,
                                       entrance_split);
    if (status != CAVITY_NATIVE_OK) return status;
    status = CavityNative_spin_response_source(q, Q, anomaly, profiles,
                                               aperture, entrance_packet);
    if (status != CAVITY_NATIVE_OK) return status;
    double entrance_vector[3];
    for (int row=0; row<3; ++row) {
        double value = entrance_packet[21+row];
        for (int k=0; k<6; ++k)
            value += entrance_packet[3+row*6+k]
                *(xi_in[k]+entrance_split[k]);
        if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
        entrance_vector[row] = value;
    }
    double entrance_forcing[9] = {
        0., -entrance_vector[2], entrance_vector[1],
        entrance_vector[2], 0., -entrance_vector[0],
        -entrance_vector[1], entrance_vector[0], 0.};
    double generators[2][9], forcing[2][9];
    for (int i=0; i<2; ++i) {
        status = CavityNative_spin_generator(spin_qstages+i*6, Q, anomaly,
            nominal_profiles+(i+1)*10, aperture, generators[i]);
        if (status != CAVITY_NATIVE_OK) return status;
        double packet[24], vector[3];
        status = CavityNative_spin_response_source(spin_qstages+i*6, Q,
            anomaly, profiles+(i+1)*12, aperture, packet);
        if (status != CAVITY_NATIVE_OK) return status;
        for (int row=0; row<3; ++row) {
            double value = packet[21+row];
            for (int k=0; k<6; ++k)
                value += packet[3+row*6+k]*physical_stages[i*6+k];
            if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
            vector[row] = value;
        }
        double matrix[9] = {0., -vector[2], vector[1],
            vector[2], 0., -vector[0], -vector[1], vector[0], 0.};
        for (int row=0; row<3; ++row)
            for (int col=0; col<3; ++col) {
                double value = 0.;
                for (int k=0; k<3; ++k)
                    value += matrix[row*3+k]*spin_stages[i*9+k*3+col];
                if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
                forcing[i][row*3+col] = value;
            }
    }
    const double C[2] = {0.21132486540518713, 0.7886751345948129};
    const double A[2][2] = {{0.25, -0.038675134594812866},
                            {0.5386751345948129, 0.25}};
    double response[2][9], proposed[2][9], products[2][9];
    for (int i=0; i<2; ++i)
        for (int k=0; k<9; ++k) {
            response[i][k] = C[i]*h*entrance_forcing[k];
            if (!isfinite(response[i][k])) return CAVITY_NATIVE_NONFINITE;
        }
    for (int64_t iteration=0; iteration<max_iter; ++iteration) {
        for (int i=0; i<2; ++i)
            for (int row=0; row<3; ++row)
                for (int col=0; col<3; ++col) {
                    double value = forcing[i][row*3+col];
                    for (int k=0; k<3; ++k)
                        value += generators[i][row*3+k]
                            *response[i][k*3+col];
                    if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
                    products[i][row*3+col] = value;
                }
        double change = 0.;
        for (int i=0; i<2; ++i)
            for (int k=0; k<9; ++k) {
                proposed[i][k] = h*(A[i][0]*products[0][k]
                                    +A[i][1]*products[1][k]);
                if (!isfinite(proposed[i][k])) return CAVITY_NATIVE_NONFINITE;
                change = fmax(change, fabs(proposed[i][k]-response[i][k]));
            }
        for (int i=0; i<2; ++i)
            for (int k=0; k<9; ++k) response[i][k] = proposed[i][k];
        if (change < spin_response_tol) {
            for (int i=0; i<2; ++i)
                for (int row=0; row<3; ++row)
                    for (int col=0; col<3; ++col) {
                        double value = forcing[i][row*3+col];
                        for (int k=0; k<3; ++k)
                            value += generators[i][row*3+k]
                                *response[i][k*3+col];
                        if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
                        products[i][row*3+col] = value;
                    }
            double actual = 0.;
            for (int i=0; i<2; ++i)
                for (int k=0; k<9; ++k) {
                    double defect = response[i][k]-h*(
                        A[i][0]*products[0][k]+A[i][1]*products[1][k]);
                    if (!isfinite(defect)) return CAVITY_NATIVE_NONFINITE;
                    actual = fmax(actual, fabs(defect));
                }
            if (actual < spin_response_tol) {
                double result[9];
                for (int k=0; k<9; ++k) {
                    result[k] = h*((products[0][k]+products[1][k])*0.5);
                    if (!isfinite(result[k])) return CAVITY_NATIVE_NONFINITE;
                }
                for (int k=0; k<6; ++k) out[k] = spin_endpoint[k];
                for (int k=0; k<6; ++k) xi_out[k] = auxiliary[k];
                for (int k=0; k<6; ++k) physical_response[k] = physical[k];
                for (int k=0; k<9; ++k) rotation[k] = nominal_rotation[k];
                for (int k=0; k<9; ++k) rotation_response[k] = result[k];
                for (int k=0; k<3; ++k) work[k] = local_work[k];
                residuals[0] = nominal_residual;
                residuals[1] = auxiliary_residual;
                residuals[2] = passive_residual;
                residuals[3] = actual;
                return CAVITY_NATIVE_OK;
            }
        }
    }
    return CAVITY_NATIVE_ITERATIONS;
}

// Optional product in the NEW global Schott-spin coordinate only.
GPUFUN
int CavityNative_spin_chart_product(const double generator[9],
        const double response[9], const double forcing[9], double product[9]) {
    double result[9];
    for (int row=0; row<3; ++row)
        for (int col=0; col<3; ++col) {
            double value = forcing[row*3+col];
            for (int k=0; k<3; ++k)
                value += generator[row*3+k]*response[k*3+col];
            if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
            result[row*3+col] = value;
        }
    for (int k=0; k<9; ++k) product[k] = result[k];
    return CAVITY_NATIVE_OK;
}

// Another, explicitly DIFFERENT radiative finite word: GLOBAL Y=P-K*R0,
// K=skew(k), using the independently proved Schott-spin source27.
// Initialize Yin=Pin-Kin*Rin ONCE at the global entrance, then carry Yout
// directly across all cells. Physical P=Y+K*R0 is reconstructed one-sided
// from each cell's actual endpoint profile. Raw discontinuous cached faces
// therefore never cause a hidden per-cell rematch/reset of Y or xi.
//
// Accepted local passive stages are multiplied by Rin; the actual global
// defining residual is separately checked against absolute spin_tol.
// Accepted xi stages are recorded directly, NEVER recovered as (xi+s)-s.
// Yi=Yin+h sum_j Aij(G0j Yj+skew(Dqg0j*xij+Bj)*R0j).
// Rout=U0*Rin, Pout=Yout+Kout*Rout. Incoming R/Y are arbitrary finite
// matrices; very large inputs can fail the explicit absolute residual gates.
// No normalization/projection, finite-rho feedback or stochastic law.
// residuals=(nominal,xi,local_passive,global_passive,Y), not error bounds.
// Every output is untouched on failure. Profiles40 are TRUE Re/Im0..4;
// no material-D5 padding or evaluation is part of this NEW consumer.
GPUFUN
int CavityNative_gauss_schott_spin_chart(const double q[6],
        const double xi_in[6], const double rotation_in[9],
        const double shifted_spin_in[9], double h, double Q, double anomaly,
        const double profiles[40], double aperture, int64_t max_iter,
        double tol, double response_tol, double spin_tol, double shifted_tol,
        double out[6], double xi_out[6], double physical_response[6],
        double rotation_out[9], double shifted_spin_out[9],
        double physical_spin_response[9], double work[3], double residuals[5]) {
    if (!isfinite(shifted_tol) || shifted_tol <= 0.)
        return CAVITY_NATIVE_ARGUMENT;
    for (int k=0; k<9; ++k)
        if (!isfinite(rotation_in[k]) || !isfinite(shifted_spin_in[k]))
            return CAVITY_NATIVE_NONFINITE;
    double spin_endpoint[6], mean_endpoint[6], local_rotation[9];
    double spin_qstages[12], mean_qstages[12], local_spin_stages[18];
    double auxiliary_stages[12], auxiliary[6], physical[6], local_work[3];
    double nominal_residual, mean_nominal_residual, auxiliary_residual;
    double local_passive_residual;
    int status = CavityNative_gauss_spin_stages(q, h, Q, anomaly, profiles,
        aperture, max_iter, tol, spin_tol, spin_endpoint, local_rotation,
        &nominal_residual, &local_passive_residual, spin_qstages,
        local_spin_stages);
    if (status != CAVITY_NATIVE_OK) return status;
    status = CavityNative_gauss_schott_response_stage_data(q, xi_in, h, Q,
        profiles, aperture, max_iter, tol, response_tol, mean_endpoint,
        auxiliary, physical, local_work, &mean_nominal_residual,
        &auxiliary_residual, mean_qstages, 0, auxiliary_stages);
    if (status != CAVITY_NATIVE_OK) return status;
    for (int k=0; k<6; ++k)
        if (spin_endpoint[k] != mean_endpoint[k]
                || signbit(spin_endpoint[k]) != signbit(mean_endpoint[k]))
            return CAVITY_NATIVE_ARGUMENT;
    for (int k=0; k<12; ++k)
        if (spin_qstages[k] != mean_qstages[k]
                || signbit(spin_qstages[k]) != signbit(mean_qstages[k]))
            return CAVITY_NATIVE_ARGUMENT;
    const double C[2] = {0.21132486540518713, 0.7886751345948129};
    const double A[2][2] = {{0.25, -0.038675134594812866},
                            {0.5386751345948129, 0.25}};
    double global_stages[2][9], generators[2][9], global_products[2][9];
    double global_rotation[9];
    for (int row=0; row<3; ++row)
        for (int col=0; col<3; ++col) {
            double value = 0.;
            for (int k=0; k<3; ++k)
                value += local_rotation[row*3+k]*rotation_in[k*3+col];
            if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
            global_rotation[row*3+col] = value;
        }
    for (int i=0; i<2; ++i) {
        status = CavityNative_spin_generator(spin_qstages+i*6, Q, anomaly,
            profiles+(i+1)*10, aperture, generators[i]);
        if (status != CAVITY_NATIVE_OK) return status;
        for (int row=0; row<3; ++row)
            for (int col=0; col<3; ++col) {
                double value = 0.;
                for (int k=0; k<3; ++k)
                    value += local_spin_stages[i*9+row*3+k]
                        *rotation_in[k*3+col];
                if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
                global_stages[i][row*3+col] = value;
            }
        const double zero[9] = {0.,0.,0.,0.,0.,0.,0.,0.,0.};
        status = CavityNative_spin_chart_product(generators[i],
            global_stages[i], zero, global_products[i]);
        if (status != CAVITY_NATIVE_OK) return status;
    }
    double global_passive_residual = 0.;
    for (int i=0; i<2; ++i)
        for (int k=0; k<9; ++k) {
            double defect = global_stages[i][k]-rotation_in[k]-h*(
                A[i][0]*global_products[0][k]+A[i][1]*global_products[1][k]);
            if (!isfinite(defect)) return CAVITY_NATIVE_NONFINITE;
            global_passive_residual = fmax(global_passive_residual,
                                           fabs(defect));
        }
    if (!(global_passive_residual < spin_tol))
        return CAVITY_NATIVE_ITERATIONS;
    double entrance_generator[9], entrance_packet[27], entrance_vector[3];
    status = CavityNative_spin_generator(q, Q, anomaly, profiles, aperture,
                                         entrance_generator);
    if (status != CAVITY_NATIVE_OK) return status;
    status = CavityNative_spin_schott_source(q, Q, anomaly, profiles,
                                            aperture, entrance_packet);
    if (status != CAVITY_NATIVE_OK) return status;
    for (int row=0; row<3; ++row) {
        double value = entrance_packet[24+row];
        for (int k=0; k<6; ++k)
            value += entrance_packet[3+row*6+k]*xi_in[k];
        if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
        entrance_vector[row] = value;
    }
    double entrance_matrix[9] = {
        0., -entrance_vector[2], entrance_vector[1],
        entrance_vector[2], 0., -entrance_vector[0],
        -entrance_vector[1], entrance_vector[0], 0.};
    double entrance_forcing[9], entrance_rhs[9], forcing[2][9];
    for (int row=0; row<3; ++row)
        for (int col=0; col<3; ++col) {
            double value = 0.;
            for (int k=0; k<3; ++k)
                value += entrance_matrix[row*3+k]*rotation_in[k*3+col];
            if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
            entrance_forcing[row*3+col] = value;
        }
    status = CavityNative_spin_chart_product(entrance_generator,
        shifted_spin_in, entrance_forcing, entrance_rhs);
    if (status != CAVITY_NATIVE_OK) return status;
    for (int i=0; i<2; ++i) {
        double packet[27], vector[3];
        status = CavityNative_spin_schott_source(spin_qstages+i*6, Q, anomaly,
            profiles+(i+1)*10, aperture, packet);
        if (status != CAVITY_NATIVE_OK) return status;
        for (int row=0; row<3; ++row) {
            double value = packet[24+row];
            for (int k=0; k<6; ++k)
                value += packet[3+row*6+k]*auxiliary_stages[i*6+k];
            if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
            vector[row] = value;
        }
        double matrix[9] = {0., -vector[2], vector[1],
            vector[2], 0., -vector[0], -vector[1], vector[0], 0.};
        for (int row=0; row<3; ++row)
            for (int col=0; col<3; ++col) {
                double value = 0.;
                for (int k=0; k<3; ++k)
                    value += matrix[row*3+k]*global_stages[i][k*3+col];
                if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
                forcing[i][row*3+col] = value;
            }
    }
    double shifted[2][9], proposed[2][9], products[2][9];
    for (int i=0; i<2; ++i)
        for (int k=0; k<9; ++k) {
            shifted[i][k] = shifted_spin_in[k]+C[i]*h*entrance_rhs[k];
            if (!isfinite(shifted[i][k])) return CAVITY_NATIVE_NONFINITE;
        }
    for (int64_t iteration=0; iteration<max_iter; ++iteration) {
        for (int i=0; i<2; ++i) {
            status = CavityNative_spin_chart_product(generators[i],
                shifted[i], forcing[i], products[i]);
            if (status != CAVITY_NATIVE_OK) return status;
        }
        double change = 0.;
        for (int i=0; i<2; ++i)
            for (int k=0; k<9; ++k) {
                proposed[i][k] = shifted_spin_in[k]+h*(
                    A[i][0]*products[0][k]+A[i][1]*products[1][k]);
                if (!isfinite(proposed[i][k])) return CAVITY_NATIVE_NONFINITE;
                change = fmax(change, fabs(proposed[i][k]-shifted[i][k]));
            }
        for (int i=0; i<2; ++i)
            for (int k=0; k<9; ++k) shifted[i][k] = proposed[i][k];
        if (change < shifted_tol) {
            for (int i=0; i<2; ++i) {
                status = CavityNative_spin_chart_product(generators[i],
                    shifted[i], forcing[i], products[i]);
                if (status != CAVITY_NATIVE_OK) return status;
            }
            double actual = 0.;
            for (int i=0; i<2; ++i)
                for (int k=0; k<9; ++k) {
                    double defect = shifted[i][k]-shifted_spin_in[k]-h*(
                        A[i][0]*products[0][k]+A[i][1]*products[1][k]);
                    if (!isfinite(defect)) return CAVITY_NATIVE_NONFINITE;
                    actual = fmax(actual, fabs(defect));
                }
            if (actual < shifted_tol) {
                double result[9], endpoint_packet[27], physical_spin[9];
                status = CavityNative_spin_schott_source(spin_endpoint, Q,
                    anomaly, profiles+30, aperture, endpoint_packet);
                if (status != CAVITY_NATIVE_OK) return status;
                double k_matrix[9] = {0., -endpoint_packet[23],
                    endpoint_packet[22], endpoint_packet[23], 0.,
                    -endpoint_packet[21], -endpoint_packet[22],
                    endpoint_packet[21], 0.};
                for (int k=0; k<9; ++k) {
                    result[k] = shifted_spin_in[k]
                        +h*((products[0][k]+products[1][k])*0.5);
                    if (!isfinite(result[k])) return CAVITY_NATIVE_NONFINITE;
                }
                for (int row=0; row<3; ++row)
                    for (int col=0; col<3; ++col) {
                        double value = result[row*3+col];
                        for (int k=0; k<3; ++k)
                            value += k_matrix[row*3+k]
                                *global_rotation[k*3+col];
                        if (!isfinite(value)) return CAVITY_NATIVE_NONFINITE;
                        physical_spin[row*3+col] = value;
                    }
                for (int k=0; k<6; ++k) out[k] = spin_endpoint[k];
                for (int k=0; k<6; ++k) xi_out[k] = auxiliary[k];
                for (int k=0; k<6; ++k) physical_response[k] = physical[k];
                for (int k=0; k<9; ++k) rotation_out[k] = global_rotation[k];
                for (int k=0; k<9; ++k) shifted_spin_out[k] = result[k];
                for (int k=0; k<9; ++k)
                    physical_spin_response[k] = physical_spin[k];
                for (int k=0; k<3; ++k) work[k] = local_work[k];
                residuals[0] = nominal_residual;
                residuals[1] = auxiliary_residual;
                residuals[2] = local_passive_residual;
                residuals[3] = global_passive_residual;
                residuals[4] = actual;
                return CAVITY_NATIVE_OK;
            }
        }
    }
    return CAVITY_NATIVE_ITERATIONS;
}
#endif
