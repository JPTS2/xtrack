// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_ADAPTER_H
#define XTRACK_CAVITY_MODELS_ADAPTER_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_rf.h"
#include "xtrack/beam_elements/elements_src/cavity_models.h"
#include "xtrack/beam_elements/elements_src/cavity_models_frozen.h"
#include "xtrack/beam_elements/elements_src/cavity_models_averaged.h"
#include "xtrack/beam_elements/elements_src/cavity_models_paraxial.h"

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

// Inspect every represented hardware entry, including entries above order.
// First effects admit a uniform solenoid and normal/skew quadrupole only.
GPUFUN
int CavityModels_effects_static_present(const RFModelFields* el) {
    if(!xt_float_or_tpsa_is_zero(el->ks))return 1;
    for(int64_t i=0;i<el->knl_len;++i)
        if(el->knl[i]!=0.)return 1;
    for(int64_t i=0;i<el->ksl_len;++i)
        if(el->ksl[i]!=0.)return 1;
    return 0;
}

GPUFUN
int CavityModels_effects_static_unsupported(const RFModelFields* el) {
    for(int64_t i=0;i<el->knl_len;++i)
        if((i!=1||el->order<1)&&el->knl[i]!=0.)return 1;
    for(int64_t i=0;i<el->ksl_len;++i)
        if((i!=1||el->order<1)&&el->ksl[i]!=0.)return 1;
    return 0;
}

GPUFUN
int CavityModels_transformations_active(const RFModelFields* el) {
    return el->shift_x!=0.||el->shift_y!=0.
        ||el->shift_s!=0.||el->rot_x_rad!=0.
        ||el->rot_y_rad!=0.||el->rot_s_rad!=0.
        ||el->rot_s_rad_no_frame!=0.;
}

// Physical kinetic lift follows each model's retained source projection.
GPUFUN
int CavityModels_potential(const xt_float_or_tpsa q[6],xt_float_or_tpsa_arg z,
        const CavityModelSource *source,int model,xt_float_or_tpsa out[4]) {
    return CavityModel_physical_potential(q,z,source,model,out);
}

// Public source-based Cavity uses the ordinary canonical native chart.
// This deliberately differs from the older private zero-native-A diagnostic.
// Native momenta are P_actual/(mass_ratio*p_ref); source momenta are
// P_actual/(m_actual*c), equivalently the mass-scaled momentum/(m_ref*c).
// ax/ay report Q*A_perp/kappa_ref for physical kinetic readback, without
// changing canonical P. Q=q_ref*chi carries charge and mass-ratio scaling.
// One physical parent source/calibration is used throughout all thick slices.
// Triple-jump compositions of a symmetric fourth-order step: w1 = 1/(2-2^(1/5))
// and w0 = 1-2*w1 give order 6; nesting with 2^(1/7) gives order 8.
#define CAVITY_W6_OUTER 1.1746717580893635
#define CAVITY_W6_INNER (-1.349343516178727)
#define CAVITY_W8_OUTER 1.1161829393253857
#define CAVITY_W8_INNER (-1.2323658786507714)
static const double CAVITY_COMPOSITION_ORDER4[1]={1.};
static const double CAVITY_COMPOSITION_ORDER6[3]={
    CAVITY_W6_OUTER, CAVITY_W6_INNER, CAVITY_W6_OUTER};
static const double CAVITY_COMPOSITION_ORDER8[9]={
    CAVITY_W8_OUTER*CAVITY_W6_OUTER, CAVITY_W8_OUTER*CAVITY_W6_INNER, CAVITY_W8_OUTER*CAVITY_W6_OUTER,
    CAVITY_W8_INNER*CAVITY_W6_OUTER, CAVITY_W8_INNER*CAVITY_W6_INNER, CAVITY_W8_INNER*CAVITY_W6_OUTER,
    CAVITY_W8_OUTER*CAVITY_W6_OUTER, CAVITY_W8_OUTER*CAVITY_W6_INNER, CAVITY_W8_OUTER*CAVITY_W6_OUTER};

// The averaged map takes two steps per half-wavelength of the uniform field.
// The paraxial map integrates its generator through each step: two steps are
// enough for a travelling wave, four once a counter wave turns its phase by 2k. Its
// kick error is then about (dE/E per step)^2, so it takes a step for each 2 % of
// energy change, from V/E, the crest voltage over the total energy, and at least 8 to
// resolve a short cavity: the step error of the kick is then below 1e-4, at any
// frequency, where 4 steps give about 1e-3 and one step 10%.
// The full map takes 16 steps per half-wavelength, as 8 lose particles an accelerating
// orbit keeps, and at least 64 for a short cavity, where the wave alone would ask for
// one to three. These minima were measured on smooth envelopes.
GPUFUN
int64_t CavityPlan_cells(int model, int64_t requested, double frequency, double length,
                         double spatial_half_waves, double counter_wave_amplitude,
                         double weight, int controls,
                         double voltage_over_energy) {
    int64_t parent_cells=requested>0 ? requested : 64;
    if (requested==0) {
        const double half_wave=frequency>0. ? .5*C_LIGHT/frequency : 0.;
        // Cells shorter than the light-line half wave turn the phase faster.
        const double half_cells=fmax(frequency>0. ? length/half_wave : 0.,spatial_half_waves);
        if (model==CAVITY_MODEL_AVERAGED || model==CAVITY_MODEL_PARAXIAL) {
            double per_half=2.;
            if (model==CAVITY_MODEL_PARAXIAL && counter_wave_amplitude>0.) per_half=4.;
            parent_cells=half_cells>.5 ? (int64_t)ceil(per_half*half_cells) : 1;
            if (model==CAVITY_MODEL_PARAXIAL)
                parent_cells=(int64_t)fmax(fmax((double)parent_cells,8.),ceil(voltage_over_energy/.02));
        } else if (model==CAVITY_MODEL_FULL) {
            parent_cells=(int64_t)fmax(64.,ceil(16.*half_cells));
        }
    }
    return controls==CAVITY_MODEL_OK ? (int64_t)ceil(parent_cells*weight) : 1;
}

GPUFUN
int CavityPlan_controls(const RFModelFields* el, LocalParticle* part0, int model, int integrator,
                        int64_t requested, double weight, double slice_start) {
    const double length=el->length;
    const int profile=el->rf_profile;
    int controls=CAVITY_MODEL_OK;
    if (!(isfinite(length) && length>0. && isfinite(weight) && weight>0.
          && isfinite(slice_start) && slice_start>=0.
          && weight<=1. && slice_start+weight<=1.+8e-15)
        || el->n_cells<0
        || requested<0
        || (model!=CAVITY_MODEL_FULL && model!=CAVITY_MODEL_FROZEN
            && model!=CAVITY_MODEL_AVERAGED && model!=CAVITY_MODEL_PARAXIAL)
        || profile<0 || profile>1
        || (profile==1 && (model!=CAVITY_MODEL_FULL || el->num_harmonics<=0))
        || (el->knl_len>0 && (el->order<0 || el->knl_len<=el->order
                              || el->ksl_len<=el->order)))
        controls=CAVITY_MODEL_ARGUMENT;
    if (model==CAVITY_MODEL_FROZEN) {
        if ((integrator!=0 && integrator!=7 && integrator!=3)
            || !xt_float_or_tpsa_is_zero(el->counter_wave_amplitude)
            || el->n_cells!=0)
            controls=CAVITY_MODEL_ARGUMENT;
    } else if (model==CAVITY_MODEL_AVERAGED || model==CAVITY_MODEL_PARAXIAL) {
        if (integrator!=0 && integrator!=3 && integrator!=7)
            controls=CAVITY_MODEL_ARGUMENT;
    } else if (integrator==1 || integrator<0 || integrator>7 || integrator==6) {
        controls=CAVITY_MODEL_ARGUMENT;
    }
    // The whole-cavity taper pass takes the longitudinal map (track_cavity_plan);
    // tapering refuses cavity slices.
    if(LocalParticle_check_track_flag(part0,XS_FLAG_SR_TAPER))
        controls=CAVITY_MODEL_ARGUMENT;
    return controls;
}

// Spin follows the magnets: it is transported when spin or radiation is compiled in,
// for particles whose spin is not zero. Where this cavity cannot carry it, such a
// particle is lost rather than passed through unturned.
GPUFUN
int CavityPlan_spin_supported(const RFModelFields* el, LocalParticle* part0, int model,
                              double weight, double slice_start) {
#if !defined(XO_CONTEXT_CPU_SERIAL) && !defined(XO_CONTEXT_CPU_OPENMP)
    return 0;
#else
    if(el->tpsa_enabled||weight!=1.||slice_start!=0.
        ||LocalParticle_check_track_flag(part0,XS_FLAG_SR_KICK_SAME_AS_FIRST)
        ||LocalParticle_check_track_flag(part0,XS_FLAG_KILL_CAVITY_KICK))return 0;
    if(!CavityModels_effects_static_present(el))return 1;
    return model==CAVITY_MODEL_FULL&&!CavityModels_effects_static_unsupported(el)
        &&!el->edge_entry_active&&!el->edge_exit_active
        &&!CavityModels_transformations_active(el);
#endif
}

// The half wavelengths of the field's fastest spatial variation along the element.
GPUFUN
double CavityPlan_spatial_half_waves(const RFModelFields* el) {
    double half_waves=(double)el->n_cells;
    for (int64_t m=0;m<el->num_harmonics;++m)
        half_waves=fmax(half_waves,fabs(el->harmonic_wave_number[m])*el->length/PI);
    return half_waves;
}

// Reads only the track flags of the particle, as RFPlan_prepare does.
GPUFUN
void CavityPlan_prepare(LocalParticle* part0, double weight, double slice_start, CavityPlan* plan) {
    const RFModelFields* el=&plan->fields;
    const int model=el->model;
    plan->model=model;
    plan->thin_slice=0;
    if (model<10) return;

    const double length=el->length;
    const double frequency=xt_float_or_tpsa_const_part(el->frequency);
    const double counter_wave_amplitude=xt_float_or_tpsa_const_part(el->counter_wave_amplitude);
    const int integrator=el->integrator;
    const int64_t requested=el->num_kicks;
    plan->integrator=integrator; plan->requested=requested; plan->counter_wave_amplitude=counter_wave_amplitude;
    plan->profile=el->rf_profile;
    plan->spin_supported=CavityPlan_spin_supported(el,part0,model,weight,slice_start);
    const int controls=CavityPlan_controls(el,part0,model,integrator,requested,weight,slice_start);
    plan->choice.controls=controls;
    plan->choice.cells=CavityPlan_cells(model,requested,frequency,length,CavityPlan_spatial_half_waves(el),
        counter_wave_amplitude,weight,controls,0.);

    const int reverse=LocalParticle_check_track_flag(part0,XS_FLAG_BACKTRACK);
    // Slicing sums can differ from a parent boundary by a few ulps. Use
    // the same admitted endpoint tolerance for finite face/prefix semantics.
    const double left_fraction=fabs(slice_start)<8e-15 ? 0. : slice_start;
    const double raw_right=slice_start+weight;
    const double right_fraction=fabs(raw_right-1.)<8e-15 ? 1. : raw_right;
    plan->length=length; plan->weight=weight;
    plan->left_fraction=left_fraction; plan->right_fraction=right_fraction;
    plan->start_fraction=reverse ? right_fraction : left_fraction;
    plan->end_fraction=reverse ? left_fraction : right_fraction;
    plan->direction=reverse ? -1. : 1.;
    plan->reverse=reverse;
    plan->clear_face=reverse ? el->edge_entry_active
                             : el->edge_exit_active;
    plan->physical_outer=reverse ? left_fraction==0. : right_fraction==1.;
}


// Gauge transformation between the phase-resolved temporal RF potential and
// the field-free-outside gauge of the end-plane impulse. The gauge
// function Lambda = (r^2/4) S1 - (r^4/64)(S1+S3) gives the temporal A_perp =
// grad Lambda and the outside gauge's phi = d_T Lambda; to_temporal applies
// P += Q grad Lambda, E -= Q d_T Lambda, otherwise the inverse.
GPUFUN
void CavityModels_rf_gauge(xt_float_or_tpsa q[6], const CavityModelSource *source,
                           xt_float_or_tpsa_arg Q, xt_float_or_tpsa_arg z, int to_temporal) {
    xt_float_or_tpsa real[5], imag[5];
    CavityModel_profile(z, source, real, imag);
    const xt_float_or_tpsa c=cos(q[2]), s=sin(q[2]);
    const xt_float_or_tpsa S1=real[1]*s-imag[1]*c, S3=real[3]*s-imag[3]*c;
    const xt_float_or_tpsa C1=real[1]*c+imag[1]*s, C3=real[3]*c+imag[3]*s;
    const xt_float_or_tpsa r2=q[0]*q[0]+q[1]*q[1];
    const xt_float_or_tpsa radial=S1*.5-r2*(S1+S3)/16.;
    const xt_float_or_tpsa scalar=r2*C1/4.-r2*r2*(C1+C3)/64.;
    xt_float_or_tpsa charge=Q;
    if (!to_temporal) charge=-Q;
    q[3]+=charge*q[0]*radial;
    q[4]+=charge*q[1]*radial;
    if (!source->freeze_energy) q[5]-=charge*scalar;
}

// The physical models for one particle, shared by the Cavity and its slices.
GPUFUN
void track_cavity_single_particle(const CavityPlan* plan, int model,
                                  LocalParticle* part, CavityMagnusCache* magnus_cache) {
    const RFModelFields* el=&plan->fields;
    const int controls=plan->choice.controls;
    int64_t cells=plan->choice.cells;
    const int integrator=plan->integrator;
    const int profile=plan->profile;
    int spin_tracking=0;
    const int64_t requested=plan->requested;
    const double length=plan->length;
    const double weight=plan->weight;
    const double left_fraction=plan->left_fraction;
    const double right_fraction=plan->right_fraction;
    const double start_fraction=plan->start_fraction;
    const double end_fraction=plan->end_fraction;
    const double direction=plan->direction;
    const int reverse=plan->reverse;
    const int clear_face=plan->clear_face;
    const int physical_outer=plan->physical_outer;

    int status=controls;
    double anomaly=0.,spin[3]={0.,0.,0.};
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
    xt_float_or_tpsa frequency=el->frequency;
    const xt_float_or_tpsa harmonic=el->harmonic;
    if (!xt_float_or_tpsa_is_zero(harmonic)) {
        if (!(part->line_length>0.)) status=CAVITY_MODEL_ARGUMENT;
        else frequency+=harmonic*beta0*C_LIGHT/part->line_length;
    }
    // A harmonic frequency and the energy change follow this particle's reference.
    if (!xt_float_or_tpsa_is_zero(harmonic) || (model==CAVITY_MODEL_PARAXIAL && requested==0))
        cells=CavityPlan_cells(model,requested,xt_float_or_tpsa_const_part(frequency),
            length,CavityPlan_spatial_half_waves(el),
            plan->counter_wave_amplitude,weight,controls,
            fabs(q0*xt_float_or_tpsa_const_part(el->voltage))
                /sqrt(mass0*mass0+LocalParticle_get_p0c(part)*LocalParticle_get_p0c(part)));
    xt_float_or_tpsa phase=el->phase
        +DEG2RAD*el->lag
        +el->phase_taper
        +DEG2RAD*el->lag_taper;
    // New physical sources treat taper offsets as deterministic hardware
    // phase inputs, independent of whether radiation is compiled. The
    // legacy track_rf path retains its historical compile-flag policy.
    const xt_float_or_tpsa voltage=el->voltage;
    if (model==CAVITY_MODEL_FROZEN
        && !xt_float_or_tpsa_is_zero(voltage)
        && xt_float_or_tpsa_const_part(frequency)<=0.)
        status=CAVITY_MODEL_ARGUMENT;
    // Cells are pi-mode cells of fixed length: they set the spatial wave number,
    // and voltage is the crest gain at their synchronous speed, at most c.
    xt_float_or_tpsa wave_number=(2.*PI/C_LIGHT)*frequency;
    xt_float_or_tpsa calibration_beta=el->calibration_beta;
    if (el->n_cells>0) {
        wave_number=PI*(double)el->n_cells/length;
        const xt_float_or_tpsa synchronous=frequency*(2.*length/(C_LIGHT*(double)el->n_cells));
        if (xt_float_or_tpsa_const_part(synchronous)<1.) calibration_beta=synchronous;
    }
    CavityModelSource source;
    if (status==CAVITY_MODEL_OK && profile==1) {
        status=CavityModel_prepare_harmonics(frequency,voltage,length,phase,calibration_beta,
                mass0,q0,el->num_harmonics,el->harmonic_wave_number,
                el->harmonic_re,el->harmonic_im,&source);
    } else if (status==CAVITY_MODEL_OK) {
        status=CavityModel_prepare(frequency,
                model==CAVITY_MODEL_FROZEN?0.:voltage,
                length,phase,el->counter_wave_amplitude,el->counter_wave_phase,
                calibration_beta,wave_number,mass0,q0,&source);
    }
    xt_float_or_tpsa q[6], clock_base=0.;
    xt_float_or_tpsa z=0., h=0.;
    if (status==CAVITY_MODEL_OK) {
        source.ks=el->ks;
        if (model==CAVITY_MODEL_FROZEN)
            source.voltage_norm=voltage/mass0;
        source.magnetic_scale=LocalParticle_get_p0c(part)/(q0*mass0);
        source.order=el->order;
        source.knl=el->knl;
        source.ksl=el->ksl;
        source.knl_tpsa=el->knl_tpsa;
        source.ksl_tpsa=el->ksl_tpsa;
        source.reference_beta=beta0;
        source.uniform_mesh=integrator==3;
        source.freeze_energy=LocalParticle_check_track_flag(part,XS_FLAG_KILL_CAVITY_KICK)
            || XT_LP_IS_FROZEN(delta) || XT_LP_IS_FROZEN(pzeta);
        source.freeze_zeta=XT_LP_IS_FROZEN(zeta);
        source.freeze_x=XT_LP_IS_FROZEN(x);
        source.freeze_px=XT_LP_IS_FROZEN(px);
        source.freeze_y=XT_LP_IS_FROZEN(y);
        source.freeze_py=XT_LP_IS_FROZEN(py);
#if !defined(XTRACK_TPSA_TRACK) && !defined(XTRACK_MULTIPOLE_NO_SYNRAD)
        if(LocalParticle_get_spin_x(part)!=0.||LocalParticle_get_spin_y(part)!=0.
            ||LocalParticle_get_spin_z(part)!=0.) {
            if(plan->spin_supported) spin_tracking=1;
            else status=CAVITY_MODEL_ARGUMENT;
        }
#endif
        if(spin_tracking) {
            if(CavityModels_freeze_mask(&source)||XT_LP_IS_FROZEN(s)
                ||XT_LP_IS_FROZEN(rpp)||XT_LP_IS_FROZEN(rvv)
                ||XT_LP_IS_FROZEN(ax)||XT_LP_IS_FROZEN(ay)
                ||XT_LP_IS_FROZEN(spin_x)||XT_LP_IS_FROZEN(spin_y)||XT_LP_IS_FROZEN(spin_z))
                status=CAVITY_MODEL_ARGUMENT;
#ifndef XTRACK_TPSA_TRACK
            if(status==CAVITY_MODEL_OK) {
                anomaly=LocalParticle_get_anomalous_magnetic_moment(part);
                spin[0]=LocalParticle_get_spin_x(part);spin[1]=LocalParticle_get_spin_y(part);
                spin[2]=LocalParticle_get_spin_z(part);
                if(!isfinite(anomaly))status=CAVITY_MODEL_ARGUMENT;
                for(int j=0;j<3;++j)if(!isfinite(spin[j]))status=CAVITY_MODEL_NONFINITE;
            }
#endif
        }
        if (model==CAVITY_MODEL_FROZEN
            && (source.freeze_x || source.freeze_px
                || source.freeze_y || source.freeze_py))
            status=CAVITY_MODEL_ARGUMENT;
        z=source.k*length*(start_fraction-.5);
        h=direction*source.k*length*weight/cells;
        clock_base=source.clock_origin;
        if (el->absolute_time) {
            const xt_float_or_tpsa parent_s=LocalParticle_get_s(part)
                -length*start_fraction;
            // The RF phase accumulated at time_s, as in track_rf.
            if (!xt_float_or_tpsa_is_zero(harmonic))
                clock_base+=2.*PI*(el->frequency*LocalParticle_get_time_s(part)
                    +harmonic*LocalParticle_get_reference_turns(part));
            else clock_base+=source.omega*LocalParticle_get_time_s(part);
            clock_base+=source.k*parent_s/beta0;
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
    if (status==CAVITY_MODEL_OK
        && (model==CAVITY_MODEL_AVERAGED || model==CAVITY_MODEL_PARAXIAL)) {
        xt_float_or_tpsa next[6];
        if (source.freeze_x || source.freeze_px || source.freeze_y || source.freeze_py)
            status=CAVITY_MODEL_ARGUMENT;
        else
        {
            CavitySpin step_spin={spin_tracking ? spin : NULL, anomaly};
            const xt_float_or_tpsa gain=fabs(q0*xt_float_or_tpsa_const_part(el->voltage))*weight/mass0;
            const xt_float_or_tpsa entry=sqrt(1.+kappa0*kappa0);
            const xt_float_or_tpsa grading=gain*(1./entry+1./(entry+gain));
            if (model==CAVITY_MODEL_AVERAGED)
                status=CavityModel_averaged_interval(q,Q,&source,cells,
                    left_fraction,right_fraction,reverse,
                    el->edge_entry_active,el->edge_exit_active,
                    kappa0,integrator,grading,&step_spin,next);
            else
                status=CavityModel_paraxial_interval(q,Q,&source,cells,
                    left_fraction,right_fraction,reverse,
                    el->edge_entry_active,el->edge_exit_active,
                    kappa0,integrator,grading,&step_spin,magnus_cache,next);
        }
        if (status==CAVITY_MODEL_OK) {
            for (int j=0;j<6;++j) q[j]=next[j];
            z=source.k*length*(end_fraction-.5);
        }
    }
    if (status==CAVITY_MODEL_OK && model==CAVITY_MODEL_FROZEN) {
        xt_float_or_tpsa next[6];
        // The nominal programme sees the parent-reference arrival phase,
        // including the laboratory anchor, never the particle's zeta or
        // evolving delta. Actual species charge enters the word via Q.
        const xt_float_or_tpsa nominal_gain=el->nominal_energy_gain_from_rf
            ? fabs(q0)*source.voltage_norm
                *sin(source.phase+clock_base-source.clock_origin)
            : el->nominal_energy_gain/mass0;
        // Explicit eV/reference-particle programme already includes the
        // reference charge. Actual-species chi never retunes this input.
        CavitySpin frozen_spin={spin_tracking ? spin : NULL, anomaly};
        if (!CavityModel_scalar_finite(nominal_gain))
            status=CAVITY_MODEL_NONFINITE;
        else
            status=CavityModel_frozen_interval(q,Q,&source,kappa0,nominal_gain,requested,
                el->edge_entry_active,
                el->edge_exit_active,left_fraction,right_fraction,
                reverse,&frozen_spin,next);
        if (status==CAVITY_MODEL_OK) {
            for (int j=0;j<6;++j) q[j]=next[j];
            z=source.k*length*(end_fraction-.5);
        }
    }
    // A whole cavity's end planes are hard edges of both waves:
    // the impulse in the field-free-outside gauge, then the temporal gauge.
    const int physical_start=reverse ? right_fraction==1. : left_fraction==0.;
    if (status==CAVITY_MODEL_OK && model==CAVITY_MODEL_FULL && physical_start) {
        xt_float_or_tpsa envelope[5];
        CavityModel_source_envelope(z, &source, envelope);
        CavitySpin edge_spin={spin_tracking ? spin : NULL, anomaly};
        status=CavityMagnus_edge(q, &source, Q, z, direction*envelope[0], direction, &edge_spin);
        if (status==CAVITY_MODEL_OK) CavityModels_rf_gauge(q, &source, Q, z, 1);
    }
    // Composition of the fourth-order Gauss cell: yoshida-6 is one
    // triple jump, yoshida-8 two; geometric grades the cell lengths.
    const double *substeps=CAVITY_COMPOSITION_ORDER4;
    int num_substeps=1;
    if (integrator==2) {substeps=CAVITY_COMPOSITION_ORDER6; num_substeps=3;}
    else if (integrator==5) {substeps=CAVITY_COMPOSITION_ORDER8; num_substeps=9;}
    xt_float_or_tpsa grading=0.,grading_log=0.;
    if (model==CAVITY_MODEL_FULL && integrator==7 && status==CAVITY_MODEL_OK) {
        const xt_float_or_tpsa entry=sqrt(1.+kappa0*kappa0);
        const xt_float_or_tpsa gain=fabs(q0*xt_float_or_tpsa_const_part(el->voltage))*weight/mass0;
        grading=gain*(1./entry+1./(entry+gain));
        grading_log=CavityModel_log1p(grading);
    }
    for (int64_t cell=0; model==CAVITY_MODEL_FULL
            && status==CAVITY_MODEL_OK && cell<cells; ++cell) {
        xt_float_or_tpsa cell_h=h;
        if (integrator==7)
            cell_h=h*cells*CavityModel_frozen_weight(reverse ? cells-1-cell : cell,
                                                     cells,grading,grading_log);
        for (int sub=0; status==CAVITY_MODEL_OK && sub<num_substeps; ++sub) {
            const xt_float_or_tpsa step=cell_h*substeps[sub];
            xt_float_or_tpsa next[6];
#ifndef XTRACK_TPSA_TRACK
            if(spin_tracking) {
                double next_spin[3];
                status=CavityModel_spin_cell(q,spin,z,step,Q,&source,anomaly,next,next_spin);
                if(status==CAVITY_MODEL_OK)for(int j=0;j<3;++j)spin[j]=next_spin[j];
            } else
#endif
                status=CavityModel_gauss(q,z,step,Q,&source,model,80,2e-13,next);
            if (status==CAVITY_MODEL_OK) {
                for (int j=0;j<6;++j) q[j]=next[j];
                z+=step;
            }
        }
    }
    xt_float_or_tpsa result[6], potential[4];
    CavityModelSource field_source=source;
    if (status==CAVITY_MODEL_OK) {
        // Observation planes are fixed by the parent interval, as in s
        // and the zeta conversion below. Repeated z+=h can finish a few
        // ulps inside a face and manufacture a tiny cached
        // RF potential. Keep the accepted stages/orbit unchanged, but
        // evaluate the physical readback at the declared endpoint.
        if (model==CAVITY_MODEL_FULL)
            z=source.k*length*(end_fraction-.5);
        // The exit plane of a whole cavity: back to the outside gauge and
        // the closing impulse, after which only static fields remain.
        if (model==CAVITY_MODEL_FULL && physical_outer) {
            xt_float_or_tpsa envelope[5];
            CavityModel_source_envelope(z, &source, envelope);
            CavityModels_rf_gauge(q, &source, Q, z, 0);
            CavitySpin edge_spin={spin_tracking ? spin : NULL, anomaly};
            status=CavityMagnus_edge(q, &source, Q, z, -direction*envelope[0], direction, &edge_spin);
            field_source.forward=0.; field_source.backward=0.;
        }
        // The longitudinal-projected generator still samples its full
        // retained physical field for mechanical readback. Its auxiliary
        // geometric slope is a separate approximation, not kin_px/p_s.
        if (status==CAVITY_MODEL_OK)
            status=CavityModels_potential(q,z,&field_source,model,potential);
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
    // A declared magnetic face removes only the static gauge contribution;
    // internal slice planes keep the RF window chart.
    xt_float_or_tpsa readback[4];
    if(status==CAVITY_MODEL_OK) {
        for(int i=0;i<4;++i)readback[i]=potential[i];
        if(clear_face&&physical_outer) {
            CavityModelSource rf_only=field_source;
            rf_only.ks=0.;rf_only.magnetic_scale=0.;rf_only.order=-1;
            rf_only.knl=NULL;rf_only.ksl=NULL;rf_only.knl_tpsa=NULL;rf_only.ksl_tpsa=NULL;
            status=CavityModels_potential(q,z,&rf_only,model,readback);
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
}

// Cavity and its thick and thin slices. In a thin layout the physical models
// are transported on the source-bearing drift slices: the thin slices are
// identity partition markers there, not RF kicks.
GPUFUN
void track_cavity_plan(LocalParticle* part0, const RFPlan* plan) {
    const MagnetTransform* transform=&plan->transform;
    if (transform->active && !magnet_transform(part0,transform,0)) return;
    const CavityPlan* cavity=&plan->cavity;
    // The taper pass (tapering.py) only shares the energy loss among the
    // cavities at zero frequency: every model uses the longitudinal map there.
    if (LocalParticle_check_track_flag(part0,XS_FLAG_SR_TAPER)
        || cavity->model<10) {
        track_rf_body_particles(part0,&plan->body);
    } else {
        // Reference-only paraxial step generators, reused across the call.
        CavityMagnusCache magnus_cache;
        CavityMagnusCache_init(&magnus_cache);
        START_PER_PARTICLE_BLOCK(part0,part);
            const int model=cavity->model;
            if (model<10) track_rf_body_single_particle(part,&plan->body);
            else if (!cavity->thin_slice)
                track_cavity_single_particle(cavity,model,part,&magnus_cache);
        END_PER_PARTICLE_BLOCK;
        CavityMagnusCache_free(&magnus_cache);
    }
    if (transform->active) magnet_transform(part0,transform,1);
}

#endif
