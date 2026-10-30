// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_ADAPTER_H
#define XTRACK_CAVITY_MODELS_ADAPTER_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/cavity_models.h"
#include "xtrack/beam_elements/elements_src/cavity_models_frozen.h"
#include "xtrack/beam_elements/elements_src/cavity_models_averaged.h"

GPUFUN
int CavityModels_physical_shell(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg charge,const xt_float_or_tpsa potential[4]) {
    if (!CavityModel_vector_finite(q,6) || !CavityModel_vector_finite(potential,4))
        return CAVITY_MODEL_NONFINITE;
    const xt_float_or_tpsa energy=q[5]-charge*potential[3];
    const xt_float_or_tpsa px=q[3]-charge*potential[0];
    const xt_float_or_tpsa py=q[4]-charge*potential[1];
    const xt_float_or_tpsa square=(energy-1.)*(energy+1.)-px*px-py*py;
    if (!CavityModel_scalar_finite(square)) return CAVITY_MODEL_NONFINITE;
    return xt_float_or_tpsa_const_part(energy)>1.
        && xt_float_or_tpsa_const_part(square)>0. ? CAVITY_MODEL_OK : CAVITY_MODEL_DOMAIN;
}

GPUFUN
int CavityModels_static_present(const CavityModelSource *source) {
    if (!xt_float_or_tpsa_is_zero(source->ks)) return 1;
    for (int64_t i=0;i<=source->order;++i)
        if (!xt_float_or_tpsa_is_zero(CavityModel_static_coefficient(source,i,0))
            || !xt_float_or_tpsa_is_zero(CavityModel_static_coefficient(source,i,1)))
            return 1;
    return 0;
}

GPUFUN
int64_t CavityModels_freeze_mask(const CavityModelSource *source) {
    int64_t mask=0;
    if (source->freeze_x) mask|=(int64_t)1<<0;
    if (source->freeze_y) mask|=(int64_t)1<<1;
    if (source->freeze_zeta) mask|=(int64_t)1<<2;
    if (source->freeze_px) mask|=(int64_t)1<<3;
    if (source->freeze_py) mask|=(int64_t)1<<4;
    if (source->freeze_energy) mask|=(int64_t)1<<5;
    return mask;
}

// Effects require genuinely zero static hardware, including represented
// array entries outside the retained tracking multipole order.
GPUFUN
int CavityModels_effects_static_present(CavityData el) {
    if(!xt_float_or_tpsa_is_zero(CavityData_get_ks(el)))return 1;
    for(int64_t i=0;i<CavityData_len_knl(el);++i)
        if(CavityData_get_knl(el,i)!=0.)return 1;
    for(int64_t i=0;i<CavityData_len_ksl(el);++i)
        if(CavityData_get_ksl(el,i)!=0.)return 1;
    return 0;
}

// Physical kinetic lift follows each model's retained source projection.
GPUFUN
int CavityModels_potential(const xt_float_or_tpsa q[6],xt_float_or_tpsa_arg z,
        const CavityModelSource *source,int model,xt_float_or_tpsa out[4]) {
    int valid=CavityModel_static_validate(source);if(valid!=CAVITY_MODEL_OK)return valid;
    if(model==CAVITY_MODEL_QUADRATIC_FIRST||model==CAVITY_MODEL_QUADRATIC_SECOND)
        CavityQuadratic_potential(q,z,source,out);
    else if(model==CAVITY_MODEL_GAUSSIAN_MOVING_REFERENCE
        ||model==CAVITY_MODEL_GAUSSIAN_DIPOLE
        ||model==CAVITY_MODEL_GAUSSIAN_MAGNETIC||model==CAVITY_MODEL_GAUSSIAN_FRINGE
        ||model==CAVITY_MODEL_PREPARED_ENVELOPE)
        CavityMag_potential(q,z,source,out);
    else return CavityModel_physical_potential(q,z,source,model,out);
    return CavityModel_vector_finite(out,4)?CAVITY_MODEL_OK:CAVITY_MODEL_NONFINITE;
}

// A prepared scalar is fixed physical hardware, not a second set of live
// profile knobs. Scalar/phase/strength derivatives remain native; varying its
// coefficient-defining coordinates requires preparing a replacement table.
GPUFUN
int CavityModels_envelope_match(double actual,double expected) {
    const double tolerance=8.*2.2204460492503131e-16
        *fmax(1.,fmax(fabs(actual),fabs(expected)));
    return isfinite(actual)&&isfinite(expected)&&fabs(actual-expected)<=tolerance;
}
GPUFUN
int CavityModels_envelope_table(CavityData el,const CavityModelSource *source,
        XTCavityEnvelopeTable *table) {
    EnvelopeTableDataData data=CavityData_getp_envelope_table(el);
    if(data==NULL||source->rf_profile!=2||CavityModels_static_present(source)
        ||!xt_float_or_tpsa_is_zero(source->dc_gradient)
        ||!CavityModel_scalar_constant(source->k)
        ||!CavityModel_scalar_constant(source->carrier)
        ||!CavityModel_scalar_constant(source->profile_sigma)
        ||!CavityModel_scalar_constant(source->profile_center))return CAVITY_MODEL_ARGUMENT;
    table->energy_min=EnvelopeTableDataData_get_energy_min(data);
    table->energy_max=EnvelopeTableDataData_get_energy_max(data);
    table->position_min=EnvelopeTableDataData_get_position_min(data);
    table->position_max=EnvelopeTableDataData_get_position_max(data);
    table->nu=EnvelopeTableDataData_get_nu(data);
    table->anchor=EnvelopeTableDataData_get_anchor(data);
    table->profile_width=EnvelopeTableDataData_get_profile_width(data);
    table->profile_centre=EnvelopeTableDataData_get_profile_centre(data);
    table->n_modes=EnvelopeTableDataData_get_n_modes(data);
    table->n_energy=EnvelopeTableDataData_get_n_energy(data);
    table->n_position=EnvelopeTableDataData_get_n_position(data);
    const double k=EnvelopeTableDataData_get_normalization_wave_number(data);
    const double frequency=EnvelopeTableDataData_get_normalization_frequency(data);
    if(!(k>0.&&frequency>0.)
        ||!CavityModels_envelope_match(k,xt_float_or_tpsa_const_part(source->k))
        ||!CavityModels_envelope_match(frequency,xt_float_or_tpsa_const_part(source->omega)/(2.*PI))
        ||!CavityModels_envelope_match(table->position_min,-xt_float_or_tpsa_const_part(source->half_span))
        ||!CavityModels_envelope_match(table->position_max,xt_float_or_tpsa_const_part(source->half_span))
        ||!CavityModels_envelope_match(table->nu,xt_float_or_tpsa_const_part(source->carrier))
        ||!CavityModels_envelope_match(table->profile_width,xt_float_or_tpsa_const_part(source->profile_sigma))
        ||!CavityModels_envelope_match(table->profile_centre,xt_float_or_tpsa_const_part(source->profile_center)))
        return CAVITY_MODEL_ARGUMENT;
    if(table->n_modes<=0||table->n_energy<=0||table->n_position<=0
        ||table->n_energy>INT64_MAX/table->n_position)return CAVITY_MODEL_ARGUMENT;
    int64_t size=table->n_energy*table->n_position;
    if(table->n_modes>INT64_MAX/size/20)return CAVITY_MODEL_ARGUMENT;
    size*=20*table->n_modes;
    if(EnvelopeTableDataData_len_modes_m(data)!=table->n_modes
        ||EnvelopeTableDataData_len_modes_n(data)!=table->n_modes
        ||EnvelopeTableDataData_len_body_real(data)!=size
        ||EnvelopeTableDataData_len_body_imag(data)!=size
        ||EnvelopeTableDataData_len_boundary_real(data)!=size
        ||EnvelopeTableDataData_len_boundary_imag(data)!=size)return CAVITY_MODEL_ARGUMENT;
    table->modes_m=EnvelopeTableDataData_getp1_modes_m(data,0);
    table->modes_n=EnvelopeTableDataData_getp1_modes_n(data,0);
    table->body_real=EnvelopeTableDataData_getp1_body_real(data,0);
    table->body_imag=EnvelopeTableDataData_getp1_body_imag(data,0);
    table->boundary_real=EnvelopeTableDataData_getp1_boundary_real(data,0);
    table->boundary_imag=EnvelopeTableDataData_getp1_boundary_imag(data,0);
    return CAVITY_MODEL_OK;
}

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
    const int quadratic=model==CAVITY_MODEL_QUADRATIC_FIRST
        || model==CAVITY_MODEL_QUADRATIC_SECOND;
    const int magnetic=model==CAVITY_MODEL_GAUSSIAN_MAGNETIC
        ||model==CAVITY_MODEL_GAUSSIAN_FRINGE;
    const int envelope=model==CAVITY_MODEL_PREPARED_ENVELOPE;
    const int dipole=model==CAVITY_MODEL_GAUSSIAN_DIPOLE;
    const int overlap=model==CAVITY_MODEL_GAUSSIAN_MOVING_REFERENCE;
    const int profile=CavityData_get_rf_profile(el);
    const int magnetic_profile=CavityData_get_magnetic_profile(el);
    const int integrator=CavityData_get_integrator(el);
    const int64_t requested=CavityData_get_num_kicks(el);
    const int radiation=CavityData_get_radiation_flag(el);
    const int spin_tracking=CavityData_get_spin_tracking(el);
    const int effects=radiation!=0||spin_tracking!=0;
    int controls=CAVITY_MODEL_OK;
    if (!(isfinite(length) && length>0. && isfinite(weight) && weight>0.
          && isfinite(slice_start) && slice_start>=0.
          && weight<=1. && slice_start+weight<=1.+8e-15)
        || CavityData_get_rf_field_order(el)!=2
        || radiation<0||radiation>1||spin_tracking<0||spin_tracking>1
        ||(effects&&(model!=CAVITY_MODEL_FULL||profile!=0||magnetic_profile!=0
            ||CavityData_get__tpsa_enabled(el)!=0))
        || requested<0
        || model<CAVITY_MODEL_FULL || model>CAVITY_MODEL_GAUSSIAN_MOVING_REFERENCE
        || profile<0 || profile>2
        || magnetic_profile<0||magnetic_profile>1
        ||(model==CAVITY_MODEL_GAUSSIAN_FRINGE&&magnetic_profile!=1)
        ||(magnetic_profile==1&&!(model<=CAVITY_MODEL_LONGITUDINAL
            ||model==CAVITY_MODEL_GAUSSIAN_FRINGE))
        ||(magnetic_profile==1&&(CavityData_get_edge_entry_active(el)
            ||CavityData_get_edge_exit_active(el)))
        || ((model==CAVITY_MODEL_FROZEN||quadratic)&&profile!=1)
        || ((magnetic||envelope||dipole||overlap)&&profile!=2)
        ||(overlap&&(CavityData_get_num_reference_steps(el)<=0
            ||CavityData_get_edge_entry_active(el)||CavityData_get_edge_exit_active(el)))
        || CavityData_get_order(el)<0
        || CavityData_len_knl(el)<=CavityData_get_order(el)
        || CavityData_len_ksl(el)<=CavityData_get_order(el))
        controls=CAVITY_MODEL_ARGUMENT;
    // The draft effects law consumes the scalar full physical source only.
#if defined(XTRACK_TPSA_TRACK) || (!defined(XO_CONTEXT_CPU_SERIAL) && !defined(XO_CONTEXT_CPU_OPENMP)) || defined(XTRACK_MULTIPOLE_NO_SYNRAD)
    if(effects)controls=CAVITY_MODEL_ARGUMENT;
#endif
    if(effects&&controls==CAVITY_MODEL_OK&&CavityModels_effects_static_present(el))
        controls=CAVITY_MODEL_ARGUMENT;
    if (model==CAVITY_MODEL_FROZEN) {
        if ((integrator!=0 && integrator!=7)
            || !xt_float_or_tpsa_is_zero(CavityData_get_eta(el))
            || !xt_float_or_tpsa_is_zero(CavityData_get_calibration_beta(el)-1.)
            || !CavityData_get_wave_number_from_frequency(el))
            controls=CAVITY_MODEL_ARGUMENT;
    } else if ((quadratic||magnetic||envelope||dipole||overlap) ? (integrator!=0 && integrator!=8)
                         : (integrator!=0 && integrator!=6)) {
        controls=CAVITY_MODEL_ARGUMENT;
    }
    if (model>=CAVITY_MODEL_AVERAGED_SECOND_ORDER
        && CavityData_get_num_factor_steps(el)<=0)
        controls=CAVITY_MODEL_ARGUMENT;
    const int reverse=LocalParticle_check_track_flag(part0,XS_FLAG_BACKTRACK);
    const int taper_analysis=LocalParticle_check_track_flag(part0,XS_FLAG_SR_TAPER);
    if(effects&&(taper_analysis||(radiation&&reverse)||weight!=1.||slice_start!=0.))
        controls=CAVITY_MODEL_ARGUMENT;
    // Reference taper preparation already excludes sliced cavities. Refuse a
    // partial invocation here too, rather than resample a changed pilot delta
    // and silently give every slice a different physical hardware strength.
    if (taper_analysis && (weight!=1. || slice_start!=0.))
        controls=CAVITY_MODEL_ARGUMENT;
    // Hardware static taper is frozen once per parent invocation. The
    // dedicated reference-particle preparation may replace the stored setting;
    // its pilot arrival delta is a scalar anchor, not a particle-state knob.
    xt_float_or_tpsa static_factor=1.+CavityData_get_delta_taper(el);
    if (taper_analysis)
        static_factor=1.+xt_float_or_tpsa_const_part(LocalParticle_get_delta(part0));
    if (!CavityModel_scalar_finite(static_factor)) controls=CAVITY_MODEL_NONFINITE;
    const int64_t parent_cells=requested>0 ? requested : 64;
    const int64_t cells=controls==CAVITY_MODEL_OK
        ? (int64_t)ceil(parent_cells*weight) : 1;
    // Slicing sums can differ from a parent boundary by a few ulps. Use
    // the same admitted endpoint tolerance for finite face/prefix semantics.
    const double left_fraction=fabs(slice_start)<8e-15 ? 0. : slice_start;
    const double raw_right=slice_start+weight;
    const double right_fraction=fabs(raw_right-1.)<8e-15 ? 1. : raw_right;
    const double start_fraction=reverse ? right_fraction : left_fraction;
    const double direction=reverse ? -1. : 1.;
    const double end_fraction=reverse ? left_fraction : right_fraction;
    const int clear_face=reverse ? CavityData_get_edge_entry_active(el)
                                : CavityData_get_edge_exit_active(el);
    const int physical_outer=reverse ? left_fraction==0. : right_fraction==1.;

    START_PER_PARTICLE_BLOCK(part0,part);
        int status=controls;
        double effects_rho=0.,anomaly=0.,spin[3]={0.,0.,0.};
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
        const xt_float_or_tpsa sigma=1.+LocalParticle_get_delta(part);
        if (status==CAVITY_MODEL_OK && !CavityModel_scalar_finite(sigma))
            status=CAVITY_MODEL_NONFINITE;
        if (status==CAVITY_MODEL_OK
            && xt_float_or_tpsa_const_part(sigma)<=0.)
            status=CAVITY_MODEL_DOMAIN;
        xt_float_or_tpsa frequency=CavityData_get_frequency(el);
        const xt_float_or_tpsa harmonic=CavityData_get_harmonic(el);
        if (!xt_float_or_tpsa_is_zero(harmonic)) {
            if (!(part->line_length>0.)) status=CAVITY_MODEL_ARGUMENT;
            else frequency+=harmonic*beta0*C_LIGHT/part->line_length;
        }
        xt_float_or_tpsa phase=CavityData_get_phase(el)
            +DEG2RAD*CavityData_get_lag(el)
            +CavityData_get_phase_taper(el)
            +DEG2RAD*CavityData_get_lag_taper(el);
        // New physical sources treat taper offsets as deterministic hardware
        // phase inputs, independent of whether radiation is compiled. The
        // legacy track_rf path retains its historical compile-flag policy.
        const xt_float_or_tpsa voltage=CavityData_get_voltage(el);
        const int dc_analysis=taper_analysis
            && xt_float_or_tpsa_is_zero(frequency);
        // Dedicated taper preparation is a constrained uniform-DC predictor,
        // with the selected hardware's static transport. It is not a
        // zero-frequency continuation of any selected RF approximation.
        const int effective_model=dc_analysis ? CAVITY_MODEL_FULL : model;
        const int effective_quadratic=!dc_analysis&&quadratic;
        const int effective_magnetic=!dc_analysis&&magnetic;
        if (model==CAVITY_MODEL_FROZEN && !dc_analysis
            && !xt_float_or_tpsa_is_zero(voltage)
            && xt_float_or_tpsa_const_part(frequency)<=0.)
            status=CAVITY_MODEL_ARGUMENT;
        xt_float_or_tpsa wave_number=CavityData_get_wave_number(el);
        if (CavityData_get_wave_number_from_frequency(el))
            wave_number=(2.*PI/C_LIGHT)*frequency;
        CavityModelSource source;
        if (status==CAVITY_MODEL_OK) {
            if(dc_analysis)
                status=CavityModel_prepare(frequency,0.,length,phase,
                    CavityData_get_eta(el),CavityData_get_counter_phase(el),
                    CavityData_get_calibration_beta(el),wave_number,mass0,q0,&source);
            else
                status=CavityModel_prepare_profile(frequency,
                    model==CAVITY_MODEL_FROZEN?0.:voltage,
                    length,phase,CavityData_get_eta(el),CavityData_get_counter_phase(el),
                    CavityData_get_calibration_beta(el),wave_number,mass0,q0,profile,
                    CavityData_get_rf_profile_sigma(el),CavityData_get_rf_profile_center(el),&source);
        }
        xt_float_or_tpsa q[6], clock_base=0.;
        xt_float_or_tpsa z=0., h=0.;
        if (status==CAVITY_MODEL_OK) {
            source.ks=CavityData_get_ks(el);
            source.magnetic_profile=magnetic_profile;
            source.magnetic_sigma=source.k*CavityData_get_magnetic_profile_sigma(el);
            if (effective_model==CAVITY_MODEL_FROZEN)
                source.voltage_norm=voltage/mass0;
            source.magnetic_scale=static_factor*LocalParticle_get_p0c(part)/(q0*mass0);
            source.order=CavityData_get_order(el);
            source.knl=CavityData_getp1_knl(el,0);
            source.ksl=CavityData_getp1_ksl(el,0);
            source.knl_tpsa=CavityData_getp1_knl_tpsa(el,0);
            source.ksl_tpsa=CavityData_getp1_ksl_tpsa(el,0);
            source.reference_beta=beta0;
            status=CavityModel_static_validate(&source);
            source.freeze_energy=LocalParticle_check_track_flag(part,XS_FLAG_KILL_CAVITY_KICK)
                || XT_LP_IS_FROZEN(delta) || XT_LP_IS_FROZEN(pzeta);
            source.freeze_zeta=XT_LP_IS_FROZEN(zeta);
            source.freeze_x=XT_LP_IS_FROZEN(x);
            source.freeze_px=XT_LP_IS_FROZEN(px);
            source.freeze_y=XT_LP_IS_FROZEN(y);
            source.freeze_py=XT_LP_IS_FROZEN(py);
            if(effects) {
                if(CavityModels_freeze_mask(&source)||XT_LP_IS_FROZEN(s)
                    ||XT_LP_IS_FROZEN(rpp)||XT_LP_IS_FROZEN(rvv)
                    ||XT_LP_IS_FROZEN(ax)||XT_LP_IS_FROZEN(ay)
                    ||XT_LP_IS_FROZEN(spin_x)||XT_LP_IS_FROZEN(spin_y)||XT_LP_IS_FROZEN(spin_z))
                    status=CAVITY_MODEL_ARGUMENT;
#ifndef XTRACK_TPSA_TRACK
                if(status==CAVITY_MODEL_OK&&radiation)
                    status=CavityModel_species_rho(source.omega,mass0*mass_ratio,
                        q0*LocalParticle_get_charge_ratio(part),&effects_rho);
                if(status==CAVITY_MODEL_OK&&spin_tracking) {
                    anomaly=LocalParticle_get_anomalous_magnetic_moment(part);
                    spin[0]=LocalParticle_get_spin_x(part);spin[1]=LocalParticle_get_spin_y(part);
                    spin[2]=LocalParticle_get_spin_z(part);
                    if(!isfinite(anomaly))status=CAVITY_MODEL_ARGUMENT;
                    for(int j=0;j<3;++j)if(!isfinite(spin[j]))status=CAVITY_MODEL_NONFINITE;
                }
#endif
            }
            if ((effective_model==CAVITY_MODEL_FROZEN
                    ||effective_model==CAVITY_MODEL_GAUSSIAN_FRINGE)
                && (source.freeze_x || source.freeze_px
                    || source.freeze_y || source.freeze_py))
                status=CAVITY_MODEL_ARGUMENT;
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
            const xt_float_or_tpsa momentum=kappa0*sigma;
            q[5]=sqrt(1.+momentum*momentum);
        }
        if (status==CAVITY_MODEL_OK && effective_model>=CAVITY_MODEL_AVERAGED_SECOND_ORDER
            && effective_model!=CAVITY_MODEL_AVERAGED_FULL_SLOW && !effective_magnetic
            && effective_model!=CAVITY_MODEL_GAUSSIAN_DIPOLE
            && effective_model!=CAVITY_MODEL_GAUSSIAN_MOVING_REFERENCE && CavityModels_static_present(&source))
            status=CAVITY_MODEL_ARGUMENT;
        if (status==CAVITY_MODEL_OK && effective_model>=CAVITY_MODEL_AVERAGED_SECOND_ORDER) {
            xt_float_or_tpsa initial_potential[4];
            status=CavityModels_potential(q,z,&source,effective_model,initial_potential);
            if (status==CAVITY_MODEL_OK)
                status=CavityModels_physical_shell(q,Q,initial_potential);
        }
        if (status==CAVITY_MODEL_OK && effective_model>=CAVITY_MODEL_AVERAGED_SECOND_ORDER
            && effective_model<=CAVITY_MODEL_AVERAGED_FULL_SLOW) {
            const xt_float_or_tpsa slow=Q*source.forward;
            const xt_float_or_tpsa fast=Q*source.backward;
            // These canonical products admit every finite longitudinal
            // wave number; unlike the compact 1/nu facades they have no
            // positive-carrier denominator chart.
            if (status==CAVITY_MODEL_OK) {
                CavityAveragedSource averaged={0};
                averaged.half_span=source.half_span;
                averaged.entrance=-source.half_span;
                averaged.exit=source.half_span;
                averaged.slow=slow; averaged.fast=fast;
                averaged.carrier=source.carrier;
                averaged.counter_phase=source.counter_phase;
                // Prefixes share the physical parent's characteristic mesh.
                // Replacing this count by ceil(weight*N) would change M[s].
                averaged.characteristic_steps=parent_cells;
                averaged.factor_steps=CavityData_get_num_factor_steps(el);
                averaged.tolerance=2e-13;
                averaged.reference_beta=beta0;
                averaged.dc_gradient=Q*source.dc_gradient;
                averaged.static_source=effective_model==CAVITY_MODEL_AVERAGED_FULL_SLOW ? &source : NULL;
                averaged.static_charge=effective_model==CAVITY_MODEL_AVERAGED_FULL_SLOW ? Q : 0.;
                averaged.profile_source=&source;
                averaged.frozen_coordinates=CavityModels_freeze_mask(&source);
                if (effective_model==CAVITY_MODEL_AVERAGED_FULL_SLOW)
                    status=CavityFull_interval(q,&averaged,
                        source.k*length*(left_fraction-.5),
                        source.k*length*(right_fraction-.5),reverse);
                else
                    // Original CanonicalF4/CubicF4/QuarticF4 finite products,
                    // evaluated by the compact divided-jet consumer. The
                    // literal dense consumers are independent scratch oracles.
                    status=CavityProduct_interval(q,&averaged,
                        source.k*length*(left_fraction-.5),
                        source.k*length*(right_fraction-.5),effective_model-12,reverse);
                z=source.k*length*(end_fraction-.5);
            }
        }
        if (status==CAVITY_MODEL_OK && effective_quadratic) {
            CavityQuadraticSource uniform={0};
            uniform.half_span=source.half_span;
            uniform.slow=Q*source.forward;
            uniform.fast=Q*source.backward;
            uniform.counter_phase=source.counter_phase;
            uniform.carrier=source.carrier;
            uniform.body_steps=parent_cells;
            uniform.boundary_steps=CavityData_get_num_factor_steps(el);
            uniform.boundary_order=effective_model==CAVITY_MODEL_QUADRATIC_FIRST ? 1 : 2;
            uniform.frozen_coordinates=CavityModels_freeze_mask(&source);
            uniform.reference_beta=beta0;
            uniform.tolerance=2e-13;
            status=CavityQuadratic_interval(q,source.k*length*(left_fraction-.5),
                source.k*length*(right_fraction-.5),&uniform,reverse);
            z=source.k*length*(end_fraction-.5);
        }
        if(status==CAVITY_MODEL_OK&&effective_magnetic) {
            CavityMagSource gaussian={0};
            const int fringe=effective_model==CAVITY_MODEL_GAUSSIAN_FRINGE;
            status=CavityMag_bind_hardware(&source,Q,parent_cells,CavityData_get_num_factor_steps(el),
                CavityModels_freeze_mask(&source),fringe,&gaussian);
            if(status==CAVITY_MODEL_OK) {
                if(fringe)status=CavityMag_fringe_interval(q,
                    source.k*length*(left_fraction-.5),source.k*length*(right_fraction-.5),&gaussian,reverse);
                else status=CavityMag_interval(q,
                    source.k*length*(left_fraction-.5),source.k*length*(right_fraction-.5),&gaussian,reverse);
            }
            z=source.k*length*(end_fraction-.5);
        }
        if(status==CAVITY_MODEL_OK&&effective_model==CAVITY_MODEL_GAUSSIAN_MOVING_REFERENCE) {
            CavityOverlapSource moving={0};
            const xt_float_or_tpsa reference[4]={source.k*CavityData_get_reference_x(el),
                source.k*CavityData_get_reference_y(el),kappa0*CavityData_get_reference_px(el),
                kappa0*CavityData_get_reference_py(el)};
            status=CavityOverlap_bind(&source,Q,reference,parent_cells,
                CavityData_get_num_factor_steps(el),CavityData_get_num_reference_steps(el),
                CavityModels_freeze_mask(&source),&moving);
            if(status==CAVITY_MODEL_OK)status=CavityOverlap_interval(&moving,q,
                source.k*length*(left_fraction-.5),source.k*length*(right_fraction-.5),reverse);
            z=source.k*length*(end_fraction-.5);
        }
        if(status==CAVITY_MODEL_OK&&effective_model==CAVITY_MODEL_GAUSSIAN_DIPOLE) {
            CavityDipoleSource dipole_source={0};
            // Fixed canonical entrance programme in native mass-scaled units;
            // never substitute the tracked lane's current px or delta.
            status=CavityDipole_bind(&source,Q,kappa0*CavityData_get_reference_px(el),
                parent_cells,CavityData_get_num_factor_steps(el),CavityModels_freeze_mask(&source),&dipole_source);
            if(status==CAVITY_MODEL_OK)status=CavityDipole_interval(&dipole_source,q,
                source.k*length*(left_fraction-.5),source.k*length*(right_fraction-.5),reverse);
            z=source.k*length*(end_fraction-.5);
        }
        if(status==CAVITY_MODEL_OK&&effective_model==CAVITY_MODEL_PREPARED_ENVELOPE) {
            XTCavityEnvelopeTable table;
            status=CavityModels_envelope_table(el,&source,&table);
            if(status==CAVITY_MODEL_OK) {
                CavityEnvelopeSource prepared={0};prepared.table=&table;
                prepared.slow=Q*source.forward;prepared.fast=Q*source.backward;
                prepared.phase_s=0.;prepared.phase_f=source.counter_phase;
                prepared.body_steps=parent_cells;
                prepared.boundary_steps=CavityData_get_num_factor_steps(el);
                prepared.frozen_coordinates=CavityModels_freeze_mask(&source);
                prepared.tolerance=2e-13;prepared.reference_beta=beta0;
                status=CavityEnvelope_interval(q,
                    xt_float_or_tpsa_const_part(source.k)*length*(left_fraction-.5),
                    xt_float_or_tpsa_const_part(source.k)*length*(right_fraction-.5),&prepared,reverse);
                z=source.k*length*(end_fraction-.5);
            }
        }
        if (status==CAVITY_MODEL_OK && effective_model==CAVITY_MODEL_FROZEN) {
            xt_float_or_tpsa next[6];
            // The nominal programme sees the parent-reference arrival phase,
            // including the laboratory anchor, never the particle's zeta or
            // evolving delta. Actual species charge enters the word via Q.
            const xt_float_or_tpsa nominal_gain=CavityData_get_nominal_energy_gain_from_rf(el)
                ? fabs(q0)*source.voltage_norm
                    *sin(source.phase+clock_base-source.clock_origin)
                : CavityData_get_nominal_energy_gain(el)/mass0;
            // Explicit eV/reference-particle programme already includes the
            // reference charge. Actual-species chi never retunes this input.
            if (!CavityModel_scalar_finite(nominal_gain))
                status=CAVITY_MODEL_NONFINITE;
            else
                status=CavityModel_frozen_interval(q,Q,&source,kappa0,nominal_gain,requested,
                    CavityData_get_edge_entry_active(el),
                    CavityData_get_edge_exit_active(el),left_fraction,right_fraction,
                    reverse,next);
            if (status==CAVITY_MODEL_OK) {
                for (int j=0;j<6;++j) q[j]=next[j];
                z=source.k*length*(end_fraction-.5);
            }
        }
        for (int64_t cell=0; effective_model<=CAVITY_MODEL_LONGITUDINAL
                && status==CAVITY_MODEL_OK && cell<cells; ++cell) {
            xt_float_or_tpsa next[6];
#ifndef XTRACK_TPSA_TRACK
            if(effects) {
                double next_spin[3];
                status=CavityModel_effects_cell(q,spin,z,h,Q,&source,anomaly,effects_rho,
                    radiation,spin_tracking,next,next_spin);
                if(status==CAVITY_MODEL_OK)for(int j=0;j<3;++j)spin[j]=next_spin[j];
            } else
#endif
                status=CavityModel_gauss(q,z,h,Q,&source,effective_model,80,2e-13,next);
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
            status=CavityModels_potential(q,z,&source,effective_model,potential);
        }
        if (status==CAVITY_MODEL_OK) {
            const xt_float_or_tpsa energy=q[5]-Q*potential[3];
            const xt_float_or_tpsa total2=(energy-1.)*(energy+1.);
            status=CavityModels_physical_shell(q,Q,potential);
            if (status==CAVITY_MODEL_OK) {
                result[0]=q[0]/source.k;
                result[1]=q[3]/kappa0;
                result[2]=q[1]/source.k;
                result[3]=q[4]/kappa0;
                result[4]=beta0/source.k*(clock_base
                    +(source.k*length*(end_fraction-.5)+source.half_span)/beta0-q[2]);
                result[5]=sqrt(total2)/kappa0-1.;
                if (!CavityModel_vector_finite(result,6)) status=CAVITY_MODEL_NONFINITE;
                if (status==CAVITY_MODEL_OK
                    && xt_float_or_tpsa_const_part(1.+result[5])<=0.)
                    status=CAVITY_MODEL_DOMAIN;
            }
        }
        // A declared magnetic face removes only the static gauge contribution.
        // Gaussian/uniform RF tails are NOT field-free observation planes.
        xt_float_or_tpsa readback[4];
        if(status==CAVITY_MODEL_OK) {
            for(int i=0;i<4;++i)readback[i]=potential[i];
            if(clear_face&&physical_outer&&source.magnetic_profile==0) {
                CavityModelSource rf_only=source;
                rf_only.ks=0.;rf_only.magnetic_scale=0.;rf_only.order=-1;
                rf_only.knl=NULL;rf_only.ksl=NULL;rf_only.knl_tpsa=NULL;rf_only.ksl_tpsa=NULL;
                status=CavityModels_potential(q,z,&rf_only,effective_model,readback);
                if(status==CAVITY_MODEL_OK)status=CavityModels_physical_shell(q,Q,readback);
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
            LocalParticle_set_ax(part,Q*readback[0]/kappa0);
            LocalParticle_set_ay(part,Q*readback[1]/kappa0);
#ifndef XTRACK_TPSA_TRACK
            if(spin_tracking){LocalParticle_set_spin_x(part,spin[0]);
                LocalParticle_set_spin_y(part,spin[1]);LocalParticle_set_spin_z(part,spin[2]);}
#endif
        } else {
            // -99741 domain, -99742 stage exhaustion, -99743 nonfinite,
            // -99744 invalid/unfinished source or numerical configuration.
            LocalParticle_set_state(part,-99740+status);
        }
    END_PER_PARTICLE_BLOCK;
}

#endif
