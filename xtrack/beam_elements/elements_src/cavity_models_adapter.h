// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_ADAPTER_H
#define XTRACK_CAVITY_MODELS_ADAPTER_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/cavity_models.h"

// Public source-based Cavity uses the ordinary canonical native chart.
// This deliberately differs from the older private zero-native-A diagnostic.
// Native momenta are P_actual/(mass_ratio*p_ref); source momenta are
// P_actual/(m_actual*c), equivalently the mass-scaled momentum/(m_ref*c).
// ax/ay report Q*A_perp/kappa_ref for physical kinetic readback, without
// changing canonical P. Q=q_ref*chi carries charge and mass-ratio scaling.
// One physical parent source/calibration is used throughout all thick slices.
GPUFUN
void CavityModels_track_local_particle(CavityData el, LocalParticle* part0,
                                       double weight, double slice_start) {
    const double length=CavityData_get_length(el);
    const int model=CavityData_get_model(el);
    const int integrator=CavityData_get_integrator(el);
    const int64_t requested=CavityData_get_num_kicks(el);
    int controls=CAVITY_MODEL_OK;
    if (!(isfinite(length) && length>0. && isfinite(weight) && weight>0.
          && isfinite(slice_start) && slice_start>=0.
          && weight<=1. && slice_start+weight<=1.+8e-15)
        || CavityData_get_rf_field_order(el)!=2
        || (integrator!=0 && integrator!=6) || requested<0
        || model<CAVITY_MODEL_FULL || model>CAVITY_MODEL_LONGITUDINAL
        || CavityData_get_order(el)<0
        || CavityData_len_knl(el)<=CavityData_get_order(el)
        || CavityData_len_ksl(el)<=CavityData_get_order(el))
        controls=CAVITY_MODEL_ARGUMENT;
    const int reverse=LocalParticle_check_track_flag(part0,XS_FLAG_BACKTRACK);
    const int taper_analysis=LocalParticle_check_track_flag(part0,XS_FLAG_SR_TAPER);
    const int64_t parent_cells=requested>0 ? requested : 64;
    const int64_t cells=controls==CAVITY_MODEL_OK
        ? (int64_t)ceil(parent_cells*weight) : 1;
    const double start_fraction=slice_start+(reverse ? weight : 0.);
    const double direction=reverse ? -1. : 1.;
    const double end_fraction=slice_start+(reverse ? 0. : weight);
    const int clear_face=reverse ? CavityData_get_edge_entry_active(el)
                                : CavityData_get_edge_exit_active(el);
    const int physical_outer=reverse ? slice_start==0.
        : fabs(slice_start+weight-1.)<8e-15;

    START_PER_PARTICLE_BLOCK(part0,part);
        int status=controls;
        const double mass0=LocalParticle_get_mass0(part);
        const double q0=LocalParticle_get_q0(part);
        const double kappa0=LocalParticle_get_p0c(part)/mass0;
        const double beta0=LocalParticle_get_beta0(part);
        const double chi=LocalParticle_get_chi(part);
        const double mass_ratio=chi!=0.
            ? LocalParticle_get_charge_ratio(part)/chi : -1.;
        const double Q=q0*chi;
        if (!(isfinite(mass0) && mass0>0. && isfinite(q0) && q0!=0.
              && isfinite(kappa0) && kappa0>0. && isfinite(beta0)
              && beta0>0. && beta0<1. && isfinite(Q)
              && isfinite(mass_ratio) && mass_ratio>0.))
            status=CAVITY_MODEL_ARGUMENT;
        xt_float_or_tpsa frequency=CavityData_get_frequency(el);
        const xt_float_or_tpsa harmonic=CavityData_get_harmonic(el);
        if (!xt_float_or_tpsa_is_zero(harmonic)) {
            if (!(part->line_length>0.)) status=CAVITY_MODEL_ARGUMENT;
            else frequency+=harmonic*beta0*C_LIGHT/part->line_length;
        }
        xt_float_or_tpsa phase=CavityData_get_phase(el)
            +DEG2RAD*CavityData_get_lag(el);
        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
            phase+=CavityData_get_phase_taper(el)
                  +DEG2RAD*CavityData_get_lag_taper(el);
        #endif
        const xt_float_or_tpsa voltage=CavityData_get_voltage(el);
        const int dc_analysis=taper_analysis
            && xt_float_or_tpsa_is_zero(frequency);
        xt_float_or_tpsa wave_number=CavityData_get_wave_number(el);
        if (CavityData_get_wave_number_from_frequency(el))
            wave_number=(2.*PI/C_LIGHT)*frequency;
        CavityModelSource source;
        if (status==CAVITY_MODEL_OK)
            status=CavityModel_prepare(frequency,dc_analysis ? 0. : voltage,
                length,phase,CavityData_get_eta(el),
                CavityData_get_counter_phase(el),
                CavityData_get_calibration_beta(el),wave_number,mass0,q0,&source);
        xt_float_or_tpsa q[6], clock_base=0.;
        xt_float_or_tpsa z=0., h=0.;
        if (status==CAVITY_MODEL_OK) {
            source.ks=CavityData_get_ks(el);
            source.magnetic_scale=LocalParticle_get_p0c(part)/(q0*mass0);
            source.order=CavityData_get_order(el);
            source.knl=CavityData_getp1_knl(el,0);
            source.ksl=CavityData_getp1_ksl(el,0);
            source.reference_beta=beta0;
            source.freeze_energy=LocalParticle_check_track_flag(part,XS_FLAG_KILL_CAVITY_KICK)
                || XT_LP_IS_FROZEN(delta) || XT_LP_IS_FROZEN(pzeta);
            source.freeze_zeta=XT_LP_IS_FROZEN(zeta);
            // Existing taper preparation uses a zero-frequency voltage action.
            // This explicitly flagged uniform-DC analysis retains the same
            // static transport; it is not a calibrated zero-frequency RF source.
            if (dc_analysis)
                source.dc_gradient=copysign(1.,q0)*voltage*sin(phase)
                    /(mass0*source.k*length);
            z=source.k*length*(start_fraction-.5);
            h=direction*source.k*length*weight/cells;
            clock_base=source.clock_origin;
            if (CavityData_get_absolute_time(el)) {
                const xt_float_or_tpsa parent_s=LocalParticle_get_s(part)
                    -length*start_fraction;
                clock_base+=source.omega*LocalParticle_get_time_s(part)
                    +source.k*parent_s/beta0;
            }
            q[0]=source.k*LocalParticle_get_x(part);
            q[1]=source.k*LocalParticle_get_y(part);
            q[2]=clock_base+(z+source.half_span)/beta0
                -source.k*LocalParticle_get_zeta(part)/beta0;
            q[3]=kappa0*LocalParticle_get_px(part);
            q[4]=kappa0*LocalParticle_get_py(part);
            const xt_float_or_tpsa momentum=kappa0*(1.+LocalParticle_get_delta(part));
            q[5]=sqrt(1.+momentum*momentum);
        }
        for (int64_t cell=0; status==CAVITY_MODEL_OK && cell<cells; ++cell) {
            xt_float_or_tpsa next[6];
            status=CavityModel_gauss(q,z,h,Q,&source,model,80,2e-13,next);
            if (status==CAVITY_MODEL_OK) {
                for (int j=0;j<6;++j) q[j]=next[j];
                z+=h;
            }
        }
        xt_float_or_tpsa result[6], potential[4];
        if (status==CAVITY_MODEL_OK) {
            // The longitudinal-projected generator still samples its full
            // retained physical field for mechanical readback. Its auxiliary
            // geometric slope is a separate approximation, not kin_px/p_s.
            status=CavityModel_physical_potential(q,z,&source,model,potential);
        }
        if (status==CAVITY_MODEL_OK) {
            const xt_float_or_tpsa energy=q[5]-Q*potential[3];
            const xt_float_or_tpsa total2=(energy-1.)*(energy+1.);
            const xt_float_or_tpsa ux=q[3]-Q*potential[0];
            const xt_float_or_tpsa uy=q[4]-Q*potential[1];
            if (status==CAVITY_MODEL_OK
                && (xt_float_or_tpsa_const_part(energy)<=1.
                    || xt_float_or_tpsa_const_part(total2-ux*ux-uy*uy)<=0.))
                status=CAVITY_MODEL_DOMAIN;
            if (status==CAVITY_MODEL_OK) {
                result[0]=q[0]/source.k;
                result[1]=q[3]/kappa0;
                result[2]=q[1]/source.k;
                result[3]=q[4]/kappa0;
                result[4]=beta0/source.k*(clock_base
                    +(source.k*length*(end_fraction-.5)+source.half_span)/beta0-q[2]);
                result[5]=sqrt(total2)/kappa0-1.;
                if (!CavityModel_vector_finite(result,6)) status=CAVITY_MODEL_NONFINITE;
            }
        }
        // Commit once: an invalid source/root must not partly update a lane.
        if (status==CAVITY_MODEL_OK) {
            LocalParticle_set_x(part,result[0]);
            LocalParticle_set_px(part,result[1]);
            LocalParticle_set_y(part,result[2]);
            LocalParticle_set_py(part,result[3]);
            LocalParticle_set_zeta(part,result[4]);
            LocalParticle_update_delta(part,result[5]);
            LocalParticle_add_to_s(part,direction*length*weight);
            if (clear_face && physical_outer) {
                LocalParticle_set_ax(part,0.);
                LocalParticle_set_ay(part,0.);
            } else {
                LocalParticle_set_ax(part,Q*potential[0]/kappa0);
                LocalParticle_set_ay(part,Q*potential[1]/kappa0);
            }
        } else {
            // -99741 domain, -99742 stage exhaustion, -99743 nonfinite,
            // -99744 invalid/unfinished source or numerical configuration.
            LocalParticle_set_state(part,-99740+status);
        }
    END_PER_PARTICLE_BLOCK;
}

#endif
