// Private conservative cavity diagnostic; no public Cavity dispatch uses it.
#ifndef XTRACK_CAVITY_NATIVE_ADAPTER_H
#define XTRACK_CAVITY_NATIVE_ADAPTER_H

#include <float.h>
#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/cavity_native_gauss.h"

GPUFUN
void NativeCavityProbe_track_local_particle(
        NativeCavityProbeData el, LocalParticle* part0) {
    const int64_t cells = NativeCavityProbeData_get_cells(el);
    const int64_t maximum_iterations =
        NativeCavityProbeData_get_maximum_iterations(el);
    const double k = NativeCavityProbeData_get_k(el);
    const double omega = NativeCavityProbeData_get_omega(el);
    const double anchor_s = NativeCavityProbeData_get_anchor_s(el);
    const double clock_origin = NativeCavityProbeData_get_clock_origin(el);
    const double mass_scale = NativeCavityProbeData_get_source_mass_eV(el);
    const double tolerance = NativeCavityProbeData_get_tolerance(el);
    const double aperture = NativeCavityProbeData_get_aperture(el);
    const double length = NativeCavityProbeData_get_length(el);
    const int64_t mesh_length = NativeCavityProbeData_len_mesh(el);
    const int64_t profile_length = NativeCavityProbeData_len_profiles(el);
    int controls = cells>0 && mesh_length>1 && cells==mesh_length-1
        && profile_length%40==0 && cells==profile_length/40
        && isfinite(k) && k>0. && isfinite(omega) && omega>0.
        && isfinite(anchor_s) && isfinite(clock_origin)
        && isfinite(mass_scale) && mass_scale>0.
        && isfinite(tolerance) && tolerance>0. && maximum_iterations>0
        && isfinite(aperture) && aperture>0.
        && isfinite(length) && length>0.;
    if (!controls) {
        START_PER_PARTICLE_BLOCK(part0, part);
            LocalParticle_set_state(part, -99704);
        END_PER_PARTICLE_BLOCK;
        return;
    }
    GPUGLMEM const double* mesh = NativeCavityProbeData_getp1_mesh(el, 0);
    GPUGLMEM const double* profiles = NativeCavityProbeData_getp1_profiles(el, 0);
    for (int64_t j=0; controls && j<mesh_length; ++j)
        if (!isfinite(mesh[j]) || (j>0 && !(mesh[j]>mesh[j-1]))) controls=0;
    const double computed_length = (mesh[cells]-mesh[0])/k;
    if (!isfinite(computed_length)
            || fabs(length-computed_length)>16.*DBL_EPSILON
                *fmax(length, computed_length)
            || fabs(omega/k-C_LIGHT)>16.*DBL_EPSILON*C_LIGHT)
        controls=0;
    if (!controls) {
        START_PER_PARTICLE_BLOCK(part0, part);
            LocalParticle_set_state(part, -99704);
        END_PER_PARTICLE_BLOCK;
        return;
    }
    const int reverse = LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK);

    START_PER_PARTICLE_BLOCK(part0, part);
        int status = 1;
        const double beta0 = LocalParticle_get_beta0(part);
        const double mass0 = LocalParticle_get_mass0(part);
        const double chi = LocalParticle_get_chi(part);
        const double charge_ratio = LocalParticle_get_charge_ratio(part);
        const double mass_ratio = chi != 0. ? charge_ratio/chi : -1.;
        const double kappa0 = LocalParticle_get_p0c(part) / mass0;
        const double Q = LocalParticle_get_q0(part)*chi*mass_scale/mass0;
        const double s_in = LocalParticle_get_s(part);
        const double time_s = LocalParticle_get_time_s(part);
        const double Z_in = reverse ? mesh[cells] : mesh[0];
        const double expected_s = anchor_s + (Z_in-mesh[0])/k;
        const double clock = omega * (time_s + anchor_s/(beta0*C_LIGHT))
            + clock_origin;
        const double sigma = 1. + LocalParticle_get_delta(part);
        double q[6] = {
            k*LocalParticle_get_x(part), k*LocalParticle_get_y(part),
            clock + (Z_in-mesh[0])/beta0
                - k*LocalParticle_get_zeta(part)/beta0,
            kappa0*LocalParticle_get_px(part),
            kappa0*LocalParticle_get_py(part), hypot(1., kappa0*sigma)};
        double boundary[10], potential[4];
        const int64_t entrance_offset = reverse ? 40*(cells-1)+30 : 0;
        for (int j=0; j<10; ++j) boundary[j] = profiles[entrance_offset+j];
        if (!(isfinite(Q) && isfinite(kappa0) && kappa0>0.
                && isfinite(beta0) && beta0>0. && beta0<1.
                && isfinite(mass_ratio) && mass_ratio>0.
                && isfinite(time_s) && isfinite(s_in) && sigma>0.)) {
            status = -3;
        } else if (LocalParticle_get_ax(part) != 0.
                || LocalParticle_get_ay(part) != 0.
                || fabs(s_in-expected_s) > 32.*DBL_EPSILON
                    *fmax(1., fmax(fabs(s_in), fabs(expected_s)))) {
            // This private adapter declares the zero native boundary gauge
            // and an explicit physical anchor; it never ignores either.
            status = -4;
        } else {
            status = CavityNative_potential(q, boundary, potential);
            if (status == 1) {
                q[3] += Q*potential[0];
                q[4] += Q*potential[1];
                q[5] += Q*potential[3];
            }
        }
        // Work entirely on local trial state until every cell and the final
        // boundary chart succeeds. In particular exhaustion cannot leak s,
        // delta, or dependent pzeta/rpp/rvv updates into the particle.
        for (int64_t order=0; status==1 && order<cells; ++order) {
            const int64_t cell = reverse ? cells-1-order : order;
            const double h = reverse ? mesh[cell]-mesh[cell+1]
                                     : mesh[cell+1]-mesh[cell];
            double data[40], out[6], residual;
            for (int plane=0; plane<4; ++plane) {
                const int source_plane = reverse ? 3-plane : plane;
                for (int j=0; j<10; ++j)
                    data[plane*10+j] = profiles[cell*40+source_plane*10+j];
            }
            status = CavityNative_gauss(q, h, Q, data, aperture,
                maximum_iterations, tolerance, out, &residual);
            if (status==1) for (int j=0; j<6; ++j) q[j] = out[j];
        }
        if (status==1) {
            const int64_t exit_offset = reverse ? 0 : 40*(cells-1)+30;
            for (int j=0; j<10; ++j) boundary[j] = profiles[exit_offset+j];
            status = CavityNative_potential(q, boundary, potential);
        }
        double delta_out=0., px_out=0., py_out=0., zeta_out=0.;
        const double Z_out = reverse ? mesh[0] : mesh[cells];
        if (status==1) {
            const double gamma = q[5]-Q*potential[3];
            const double ux = q[3]-Q*potential[0];
            const double uy = q[4]-Q*potential[1];
            const double total2 = (gamma-1.)*(gamma+1.);
            if (!(gamma>1. && total2>ux*ux+uy*uy)) status=-1;
            else {
                delta_out = sqrt(total2)/kappa0-1.;
                px_out = ux/kappa0;
                py_out = uy/kappa0;
                zeta_out = beta0/k*(clock+(Z_out-mesh[0])/beta0-q[2]);
                if (!(isfinite(delta_out) && isfinite(px_out)
                        && isfinite(py_out) && isfinite(zeta_out))) status=-3;
            }
        }
        if (status==1) {
            LocalParticle_set_x(part, q[0]/k);
            LocalParticle_set_y(part, q[1]/k);
            LocalParticle_set_px(part, px_out);
            LocalParticle_set_py(part, py_out);
            LocalParticle_set_zeta(part, zeta_out);
            LocalParticle_update_delta(part, delta_out);
            LocalParticle_add_to_s(part, reverse ? -length : length);
        } else {
            // Test-only statuses; no claim of reserved public loss codes.
            LocalParticle_set_state(part, -99700+status);
        }
    END_PER_PARTICLE_BLOCK;
}
#endif
