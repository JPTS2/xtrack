// Private conservative cavity diagnostic; no public Cavity dispatch uses it.
#ifndef XTRACK_CAVITY_NATIVE_ADAPTER_H
#define XTRACK_CAVITY_NATIVE_ADAPTER_H

#include <float.h>
#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/cavity_native_gauss.h"

// Validate mutable native data before any array indexing. This is not a
// source/preparation certificate; profiles are explicitly supplied finite data.
GPUFUN
int CavityNative_adapter_controls(CavityNativeDiagnosticData el) {
    const int64_t cells = CavityNativeDiagnosticData_get_cells(el);
    const int64_t mesh_length = CavityNativeDiagnosticData_len_mesh(el);
    const int64_t profile_length = CavityNativeDiagnosticData_len_profiles(el);
    const double k = CavityNativeDiagnosticData_get_k(el);
    const double omega = CavityNativeDiagnosticData_get_omega(el);
    const double mass = CavityNativeDiagnosticData_get_source_mass_eV(el);
    const double tolerance = CavityNativeDiagnosticData_get_tolerance(el);
    const double aperture = CavityNativeDiagnosticData_get_aperture(el);
    const double length = CavityNativeDiagnosticData_get_length(el);
    if (!(cells>0 && mesh_length>1 && cells==mesh_length-1
            && profile_length%40==0 && cells==profile_length/40
            && isfinite(k) && k>0. && isfinite(omega) && omega>0.
            && isfinite(CavityNativeDiagnosticData_get_anchor_s(el))
            && isfinite(CavityNativeDiagnosticData_get_clock_origin(el))
            && isfinite(mass) && mass>0.
            && isfinite(tolerance) && tolerance>0.
            && CavityNativeDiagnosticData_get_maximum_iterations(el)>0
            && isfinite(aperture) && aperture>0.
            && isfinite(length) && length>0.)) return CAVITY_NATIVE_ARGUMENT;
    for (int64_t j=0; j<mesh_length; ++j) {
        const double value = CavityNativeDiagnosticData_get_mesh(el, j);
        if (!isfinite(value) || (j>0
                && !(value>CavityNativeDiagnosticData_get_mesh(el,j-1))))
            return CAVITY_NATIVE_ARGUMENT;
    }
    const double computed_length = (CavityNativeDiagnosticData_get_mesh(el,cells)
        -CavityNativeDiagnosticData_get_mesh(el,0))/k;
    if (!isfinite(computed_length)
            || fabs(length-computed_length)>16.*DBL_EPSILON
                *fmax(length,computed_length)
            || fabs(omega/k-C_LIGHT)>16.*DBL_EPSILON*C_LIGHT)
        return CAVITY_NATIVE_ARGUMENT;
    return CAVITY_NATIVE_OK;
}

GPUFUN
void CavityNative_adapter_profiles(CavityNativeDiagnosticData el, int64_t cell,
                                   int reverse, double data[40]) {
    for (int plane=0; plane<4; ++plane) {
        const int source_plane = reverse ? 3-plane : plane;
        for (int j=0; j<10; ++j)
            data[plane*10+j] = CavityNativeDiagnosticData_get_profiles(
                el, cell*40+source_plane*10+j);
    }
}

// Physical zero-native-A chart. Interior px/py here store mechanical momenta;
// this is not a declaration of standard canonical native Twiss coordinates.
GPUFUN
int CavityNative_adapter_input(CavityNativeDiagnosticData el, LocalParticle* part,
        double Z, const double profile[10], double q[6], double* Q,
        double* kappa0, double* beta0, double* clock) {
    const double k = CavityNativeDiagnosticData_get_k(el);
    const double mass0 = LocalParticle_get_mass0(part);
    const double chi = LocalParticle_get_chi(part);
    const double mass_ratio = chi!=0.
        ? LocalParticle_get_charge_ratio(part)/chi : -1.;
    const double s = LocalParticle_get_s(part);
    const double sigma = 1.+LocalParticle_get_delta(part);
    const double anchor_s = CavityNativeDiagnosticData_get_anchor_s(el);
    const double anchor_Z = CavityNativeDiagnosticData_get_mesh(el,0);
    *beta0 = LocalParticle_get_beta0(part);
    *kappa0 = LocalParticle_get_p0c(part)/mass0;
    *Q = LocalParticle_get_q0(part)*chi
        *CavityNativeDiagnosticData_get_source_mass_eV(el)/mass0;
    *clock = CavityNativeDiagnosticData_get_omega(el)
        *(LocalParticle_get_time_s(part)+anchor_s/(*beta0*C_LIGHT))
        +CavityNativeDiagnosticData_get_clock_origin(el);
    if (!(isfinite(mass0) && mass0>0.
            && isfinite(*Q) && isfinite(*kappa0) && *kappa0>0.
            && isfinite(*beta0) && *beta0>0. && *beta0<1.
            && isfinite(mass_ratio) && mass_ratio>0.
            && isfinite(LocalParticle_get_time_s(part)) && isfinite(s)
            && sigma>0.)) return CAVITY_NATIVE_NONFINITE;
    const double expected_s = anchor_s+(Z-anchor_Z)/k;
    if (LocalParticle_get_ax(part)!=0. || LocalParticle_get_ay(part)!=0.
            || fabs(s-expected_s)>32.*DBL_EPSILON
                *fmax(1.,fmax(fabs(s),fabs(expected_s))))
        return CAVITY_NATIVE_ARGUMENT;
    q[0] = k*LocalParticle_get_x(part);
    q[1] = k*LocalParticle_get_y(part);
    q[2] = *clock+(Z-anchor_Z)/ *beta0
        -k*LocalParticle_get_zeta(part)/ *beta0;
    q[3] = *kappa0*LocalParticle_get_px(part);
    q[4] = *kappa0*LocalParticle_get_py(part);
    q[5] = hypot(1.,*kappa0*sigma);
    double potential[4];
    const int status = CavityNative_potential(q,profile,potential);
    if (status!=CAVITY_NATIVE_OK) return status;
    q[3] += *Q*potential[0];
    q[4] += *Q*potential[1];
    q[5] += *Q*potential[3];
    return CAVITY_NATIVE_OK;
}

GPUFUN
int CavityNative_adapter_output(CavityNativeDiagnosticData el, const double q[6],
        double Z, const double profile[10], double Q, double kappa0,
        double beta0, double clock, double result[6]) {
    double potential[4];
    const int status = CavityNative_potential(q,profile,potential);
    if (status!=CAVITY_NATIVE_OK) return status;
    const double gamma = q[5]-Q*potential[3];
    const double ux = q[3]-Q*potential[0], uy = q[4]-Q*potential[1];
    const double total2 = (gamma-1.)*(gamma+1.);
    if (!(gamma>1. && total2>ux*ux+uy*uy)) return CAVITY_NATIVE_DOMAIN;
    const double k = CavityNativeDiagnosticData_get_k(el);
    result[0] = q[0]/k;
    result[1] = ux/kappa0;
    result[2] = q[1]/k;
    result[3] = uy/kappa0;
    result[4] = beta0/k*(clock
        +(Z-CavityNativeDiagnosticData_get_mesh(el,0))/beta0-q[2]);
    result[5] = sqrt(total2)/kappa0-1.;
    for (int j=0; j<6; ++j)
        if (!isfinite(result[j])) return CAVITY_NATIVE_NONFINITE;
    return CAVITY_NATIVE_OK;
}

GPUFUN
void CavityNativeDiagnostic_track_local_particle(
        CavityNativeDiagnosticData el, LocalParticle* part0) {
    const int controls = CavityNative_adapter_controls(el);
    const int64_t cells = CavityNativeDiagnosticData_get_cells(el);
    const int reverse = LocalParticle_check_track_flag(part0,XS_FLAG_BACKTRACK);
    START_PER_PARTICLE_BLOCK(part0,part);
        int status = controls;
        double q[6], Q, kappa0, beta0, clock, data[40], result[6];
        if (status==CAVITY_NATIVE_OK) {
            const int64_t first = reverse ? cells-1 : 0;
            CavityNative_adapter_profiles(el,first,reverse,data);
            status = CavityNative_adapter_input(el,part,
                CavityNativeDiagnosticData_get_mesh(el,reverse ? cells : 0),
                data,q,&Q,&kappa0,&beta0,&clock);
        }
        for (int64_t order=0; status==CAVITY_NATIVE_OK && order<cells; ++order) {
            const int64_t cell = reverse ? cells-1-order : order;
            const double h = CavityNativeDiagnosticData_get_mesh(el,cell+1)
                -CavityNativeDiagnosticData_get_mesh(el,cell);
            double out[6], residual;
            CavityNative_adapter_profiles(el,cell,reverse,data);
            status = CavityNative_gauss(q,reverse ? -h : h,Q,data,
                CavityNativeDiagnosticData_get_aperture(el),
                CavityNativeDiagnosticData_get_maximum_iterations(el),
                CavityNativeDiagnosticData_get_tolerance(el),out,&residual);
            if (status==CAVITY_NATIVE_OK)
                for (int j=0; j<6; ++j) q[j]=out[j];
        }
        if (status==CAVITY_NATIVE_OK)
            status = CavityNative_adapter_output(el,q,
                CavityNativeDiagnosticData_get_mesh(el,reverse ? 0 : cells),
                data+30,Q,kappa0,beta0,clock,result);
        if (status==CAVITY_NATIVE_OK) {
            LocalParticle_set_x(part,result[0]);
            LocalParticle_set_px(part,result[1]);
            LocalParticle_set_y(part,result[2]);
            LocalParticle_set_py(part,result[3]);
            LocalParticle_set_zeta(part,result[4]);
            LocalParticle_update_delta(part,result[5]);
            LocalParticle_add_to_s(part,(reverse ? -1. : 1.)
                *CavityNativeDiagnosticData_get_length(el));
        } else {
            // Test-only statuses; no claim of reserved public loss codes.
            LocalParticle_set_state(part,-99700+status);
        }
    END_PER_PARTICLE_BLOCK;
}

#ifdef XTRACK_CAVITY_NATIVE_TANGENT
#include "xtrack/beam_elements/elements_src/cavity_native_tangent.h"

// A separate observational consumer: no J workspace/call in the orbit path.
// Layout per physical particle slot: native endpoint6, native M36, residual,
// derivative residual. It differentiates the implicit finite root at the
// accepted approximate stages, not floating stopping branches or an ODE.
GPUFUN
void CavityNativeDiagnostic_tangent_local_particle(CavityNativeDiagnosticData el,
        LocalParticle* part0, int64_t reverse, GPUGLMEM double* output,
        GPUGLMEM int64_t* statuses) {
    const int controls = CavityNative_adapter_controls(el);
    const int64_t cells = CavityNativeDiagnosticData_get_cells(el);
    const double k = CavityNativeDiagnosticData_get_k(el);
    const double derivative_tol = CavityNativeDiagnosticData_get_derivative_tolerance(el);
    START_PER_PARTICLE_BLOCK(part0,part);
        int status = controls;
        double q[6], Q, kappa0, beta0, clock, data[40], result[6];
        double matrix[36], product[36], potential[4], jet[12];
        double worst=0., derivative_worst=0.;
        if (!(isfinite(derivative_tol) && derivative_tol>0.)
                || (reverse!=0 && reverse!=1)) status=CAVITY_NATIVE_ARGUMENT;
        if (status==CAVITY_NATIVE_OK) {
            CavityNative_adapter_profiles(el,reverse ? cells-1 : 0,reverse,data);
            status = CavityNative_adapter_input(el,part,
                CavityNativeDiagnosticData_get_mesh(el,reverse ? cells : 0),
                data,q,&Q,&kappa0,&beta0,&clock);
        }
        if (status==CAVITY_NATIVE_OK)
            status = CavityNative_potential_jacobian(q,data,jet);
        if (status==CAVITY_NATIVE_OK)
            status = CavityNative_potential(q,data,potential);
        if (status==CAVITY_NATIVE_OK) {
            for (int j=0; j<36; ++j) matrix[j]=0.;
            matrix[0]=k; matrix[1*6+2]=k; matrix[2*6+4]=-k/beta0;
            const int columns[3]={0,2,4};
            const double scales[3]={k,k,-k/beta0};
            for (int axis=0; axis<3; ++axis) {
                matrix[3*6+columns[axis]]=Q*jet[axis]*scales[axis];
                matrix[4*6+columns[axis]]=Q*jet[3+axis]*scales[axis];
                matrix[5*6+columns[axis]]=Q*jet[9+axis]*scales[axis];
            }
            matrix[3*6+1]=matrix[4*6+3]=kappa0;
            matrix[5*6+5]=kappa0*kappa0*(1.+LocalParticle_get_delta(part))
                /(q[5]-Q*potential[3]);
        }
        for (int64_t order=0; status==CAVITY_NATIVE_OK && order<cells; ++order) {
            const int64_t cell = reverse ? cells-1-order : order;
            const double h = CavityNativeDiagnosticData_get_mesh(el,cell+1)
                -CavityNativeDiagnosticData_get_mesh(el,cell);
            double out[6], local[36], residual, derivative_residual;
            CavityNative_adapter_profiles(el,cell,reverse,data);
            status = CavityNative_gauss_tangent(q,reverse ? -h : h,Q,data,
                CavityNativeDiagnosticData_get_aperture(el),
                CavityNativeDiagnosticData_get_maximum_iterations(el),
                CavityNativeDiagnosticData_get_tolerance(el),derivative_tol,
                out,local,&residual,&derivative_residual);
            if (status==CAVITY_NATIVE_OK) {
                for (int i=0; i<6; ++i) for (int j=0; j<6; ++j) {
                    product[i*6+j]=0.;
                    for (int a=0; a<6; ++a)
                        product[i*6+j]+=local[i*6+a]*matrix[a*6+j];
                }
                for (int j=0; j<36; ++j) matrix[j]=product[j];
                for (int j=0; j<6; ++j) q[j]=out[j];
                worst=fmax(worst,residual);
                derivative_worst=fmax(derivative_worst,derivative_residual);
            }
        }
        if (status==CAVITY_NATIVE_OK)
            status = CavityNative_adapter_output(el,q,
                CavityNativeDiagnosticData_get_mesh(el,reverse ? 0 : cells),
                data+30,Q,kappa0,beta0,clock,result);
        if (status==CAVITY_NATIVE_OK)
            status = CavityNative_potential_jacobian(q,data+30,jet);
        if (status==CAVITY_NATIVE_OK)
            status = CavityNative_potential(q,data+30,potential);
        if (status==CAVITY_NATIVE_OK) {
            double boundary[36];
            for (int j=0; j<36; ++j) boundary[j]=0.;
            boundary[0]=boundary[2*6+1]=1./k;
            boundary[4*6+2]=-beta0/k;
            boundary[1*6+3]=boundary[3*6+4]=1./kappa0;
            const double gamma=q[5]-Q*potential[3];
            const double factor=gamma/(kappa0*sqrt((gamma-1.)*(gamma+1.)));
            for (int axis=0; axis<3; ++axis) {
                boundary[1*6+axis]=-Q*jet[axis]/kappa0;
                boundary[3*6+axis]=-Q*jet[3+axis]/kappa0;
                boundary[5*6+axis]=-factor*Q*jet[9+axis];
            }
            boundary[5*6+5]=factor;
            for (int i=0; i<6; ++i) for (int j=0; j<6; ++j) {
                product[i*6+j]=0.;
                for (int a=0; a<6; ++a)
                    product[i*6+j]+=boundary[i*6+a]*matrix[a*6+j];
                if (!isfinite(product[i*6+j])) status=CAVITY_NATIVE_NONFINITE;
            }
        }
        statuses[part->ipart]=status;
        if (status==CAVITY_NATIVE_OK) {
            GPUGLMEM double* target=output+44*part->ipart;
            for (int j=0; j<6; ++j) target[j]=result[j];
            for (int j=0; j<36; ++j) target[6+j]=product[j];
            target[42]=worst; target[43]=derivative_worst;
        }
    END_PER_PARTICLE_BLOCK;
}
#endif

#ifdef XTRACK_CAVITY_NATIVE_SECONDARY
#include "xtrack/beam_elements/elements_src/cavity_native_secondary.h"

// Same-event source observation only. Profile input is compact Re0..5/Im0..5;
// the boundary source uses Re0..4/Im0..4 and therefore needs an explicit remap.
// Actual rho is reconstructed from runtime species AFTER any wrapper reordering.
// No source-Q-squared shortcut or separately indexed species buffer is used.
GPUFUN
void CavityNativeDiagnostic_secondary_local_particle(CavityNativeDiagnosticData el,
        LocalParticle* part0, double position, GPUGLMEM const double* profile,
        double rho_unit_eV, GPUGLMEM double* output, GPUGLMEM int64_t* statuses) {
    int controls = CavityNative_adapter_controls(el);
    if (!isfinite(position) || !isfinite(rho_unit_eV) || rho_unit_eV<0.)
        controls=CAVITY_NATIVE_ARGUMENT;
    double local_profile[12], orbit_profile[10];
    for (int j=0; j<12; ++j) local_profile[j]=profile[j];
    for (int j=0; j<5; ++j) {
        orbit_profile[j]=local_profile[j];
        orbit_profile[j+5]=local_profile[j+6];
    }
    START_PER_PARTICLE_BLOCK(part0,part);
        int status=controls;
        double q[6], Q, kappa0, beta0, clock, result[25];
        if (status==CAVITY_NATIVE_OK)
            status=CavityNative_adapter_input(el,part,position,orbit_profile,
                q,&Q,&kappa0,&beta0,&clock);
        if (status==CAVITY_NATIVE_OK) {
            const double mass=LocalParticle_get_mass0(part)
                *LocalParticle_get_charge_ratio(part)/LocalParticle_get_chi(part);
            const double charge=LocalParticle_get_q0(part)
                *LocalParticle_get_charge_ratio(part);
            const double rho=rho_unit_eV*charge*charge/mass;
            if (!(isfinite(mass) && mass>0. && isfinite(charge)
                    && isfinite(rho) && rho>=0.))
                status=CAVITY_NATIVE_NONFINITE;
            else
                status=CavityNative_secondary(q,Q,rho,
                    LocalParticle_get_anomalous_magnetic_moment(part),
                    local_profile,CavityNativeDiagnosticData_get_aperture(el),result);
        }
        statuses[part->ipart]=status;
        if (status==CAVITY_NATIVE_OK)
            for (int j=0; j<25; ++j) output[25*part->ipart+j]=result[j];
    END_PER_PARTICLE_BLOCK;
}
#endif
#endif
