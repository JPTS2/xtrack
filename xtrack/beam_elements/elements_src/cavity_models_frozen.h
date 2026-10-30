// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_FROZEN_H
#define XTRACK_CAVITY_MODELS_FROZEN_H
#include "xtrack/beam_elements/elements_src/cavity_models.h"

// First-RF-interaction, transverse-quadratic voltage actions with a fixed
// nominal inverse-momentum coefficient and collapsed physical RF faces.
// The geometrically weighted exact-drift/voltage/clock word is the retained
// TCAV-connected recipe, not a Gauss step through the smooth P16 source.
// Algebra owner: verification/physics/orbit/sad_reduction.py:tcav_factor_map.
// Source face continuation is deliberately kept, including its cubic defect
// relative to the quadratic canonical face generator. Its inverse below is
// the inverse of that actual finite face, not an assumed symplectic inverse.

GPUFUN
xt_float_or_tpsa CavityModel_expm1(xt_float_or_tpsa_arg x) {
    xt_float_or_tpsa value=exp(x);
    // The backend supplies exp but not expm1. Replacing its constant term
    // preserves every represented Taylor coefficient, without a finite
    // polynomial approximation or cancellation in the scalar result.
    return value-xt_float_or_tpsa_const_part(value)
        +expm1(xt_float_or_tpsa_const_part(x));
}

GPUFUN
xt_float_or_tpsa CavityModel_log1p(xt_float_or_tpsa_arg x) {
    xt_float_or_tpsa value=log(1.+x);
    return value-xt_float_or_tpsa_const_part(value)
        +log1p(xt_float_or_tpsa_const_part(x));
}

GPUFUN
int CavityModel_frozen_shell(const xt_float_or_tpsa state[6],
        xt_float_or_tpsa *momentum, xt_float_or_tpsa *longitudinal) {
    if (!CavityModel_vector_finite(state,6)) return CAVITY_MODEL_NONFINITE;
    xt_float_or_tpsa p2=(state[5]-1.)*(state[5]+1.);
    xt_float_or_tpsa ps2=p2-state[3]*state[3]-state[4]*state[4];
    if (xt_float_or_tpsa_const_part(state[5])<=1.
        || xt_float_or_tpsa_const_part(ps2)<=0.) return CAVITY_MODEL_DOMAIN;
    *momentum=sqrt(p2);*longitudinal=sqrt(ps2);
    return CAVITY_MODEL_OK;
}

GPUFUN
int CavityModel_frozen_drift(xt_float_or_tpsa state[6],
        xt_float_or_tpsa_arg distance, xt_float_or_tpsa_arg beta0,
        int freeze_zeta) {
    xt_float_or_tpsa p,ps;
    int status=CavityModel_frozen_shell(state,&p,&ps);
    if (status!=CAVITY_MODEL_OK) return status;
    state[0]+=distance*state[3]/ps;
    state[1]+=distance*state[4]/ps;
    if (!freeze_zeta) state[2]+=distance*(state[5]/ps-1./beta0);
    return CAVITY_MODEL_OK;
}

GPUFUN
int CavityModel_frozen_clock(xt_float_or_tpsa state[6],
        xt_float_or_tpsa_arg clock, int freeze_zeta) {
    if (freeze_zeta) return CAVITY_MODEL_OK;
    xt_float_or_tpsa p,ps;
    int status=CavityModel_frozen_shell(state,&p,&ps);
    if (status!=CAVITY_MODEL_OK) return status;
    state[2]+=clock*state[5]/p;
    return CAVITY_MODEL_OK;
}

GPUFUN
int CavityModel_frozen_face(xt_float_or_tpsa state[6],
        xt_float_or_tpsa_arg voltage, xt_float_or_tpsa_arg length,
        xt_float_or_tpsa_arg wave_number, xt_float_or_tpsa_arg phase,
        int inverse, int freeze_energy) {
    xt_float_or_tpsa p,ps;
    int status=CavityModel_frozen_shell(state,&p,&ps);
    if (status!=CAVITY_MODEL_OK) return status;
    xt_float_or_tpsa psi=wave_number*state[2]-phase;
    xt_float_or_tpsa kick=voltage*sin(psi)/(2.*length);
    xt_float_or_tpsa dh=-wave_number*voltage
        *(state[0]*state[0]+state[1]*state[1])*cos(psi)/(4.*length);
    if (freeze_energy) dh=0.;
    xt_float_or_tpsa energy=state[5]+dh;
    if (inverse) energy=state[5]-dh;
    if (xt_float_or_tpsa_const_part(energy)<=1.) return CAVITY_MODEL_DOMAIN;
    xt_float_or_tpsa ratio=sqrt((energy-1.)*(energy+1.))/p;
    for (int plane=0;plane<2;++plane) {
        if (inverse) state[plane+3]=ratio*state[plane+3]-kick*state[plane];
        else state[plane+3]=ratio*(state[plane+3]+kick*state[plane]);
    }
    state[5]=energy;
    return CavityModel_frozen_shell(state,&p,&ps);
}

GPUFUN
int CavityModel_frozen_kick(xt_float_or_tpsa state[6],
        xt_float_or_tpsa_arg weight, xt_float_or_tpsa_arg voltage,
        xt_float_or_tpsa_arg radial, xt_float_or_tpsa_arg wave_number,
        xt_float_or_tpsa_arg phase, int inverse, int freeze_energy) {
    xt_float_or_tpsa psi=wave_number*state[2]-phase;
    xt_float_or_tpsa signed_weight=weight;
    if (inverse) signed_weight=-weight;
    if (!freeze_energy)
        state[5]-=signed_weight*(voltage+radial
            *(state[0]*state[0]+state[1]*state[1]))*sin(psi);
    state[3]-=signed_weight*2.*radial*state[0]*cos(psi)/wave_number;
    state[4]-=signed_weight*2.*radial*state[1]*cos(psi)/wave_number;
    xt_float_or_tpsa p,ps;
    return CavityModel_frozen_shell(state,&p,&ps);
}

GPUFUN
xt_float_or_tpsa CavityModel_frozen_weight(int64_t index, int64_t count,
        xt_float_or_tpsa_arg ratio, xt_float_or_tpsa_arg logarithm) {
    if (xt_float_or_tpsa_const_part(ratio)<=1.e-6) return 1./count;
    return CavityModel_expm1(logarithm/count)*exp(index*logarithm/count)/ratio;
}

// Full physical parent only. nominal_momentum and nominal_gain are fixed
// reference-programme data, independent of the particle's evolving energy.
// The caller supplies them in the same positive mass normalization as q.
// q uses the global source clock. Internally d=c(t-t_ref) uses the parent's
// entrance clock; the returned T includes the nominal path advance once.
// This first connection is RF-only; static source-factor extensions are
// separate and cannot be replaced silently by another model.
GPUFUN
int CavityModel_frozen(const xt_float_or_tpsa q[6], xt_float_or_tpsa_arg Q,
        const CavityModelSource *source,
        xt_float_or_tpsa_arg nominal_momentum,
        xt_float_or_tpsa_arg nominal_gain,
        int64_t requested_cells,
        int entry_face, int exit_face, int reverse,
        xt_float_or_tpsa output[6]) {
    if (!xt_float_or_tpsa_is_zero(source->backward)
        || !xt_float_or_tpsa_is_zero(source->ks)
        || !xt_float_or_tpsa_is_zero(source->dc_gradient)) return CAVITY_MODEL_ARGUMENT;
    for (int64_t i=0;i<=source->order;++i)
        if ((source->knl && source->knl[i]!=0.)
            || (source->ksl && source->ksl[i]!=0.)) return CAVITY_MODEL_ARGUMENT;
    if (xt_float_or_tpsa_const_part(nominal_momentum)<=0. || requested_cells<0)
        return CAVITY_MODEL_ARGUMENT;
    xt_float_or_tpsa h0=sqrt(1.+nominal_momentum*nominal_momentum);
    xt_float_or_tpsa he=h0+nominal_gain;
    if (xt_float_or_tpsa_const_part(he)<=1.) return CAVITY_MODEL_DOMAIN;
    xt_float_or_tpsa pe=sqrt((he-1.)*(he+1.));
    xt_float_or_tpsa beta0=nominal_momentum/h0;
    xt_float_or_tpsa inverse_p=(1./nominal_momentum+1./pe)*.5;
    xt_float_or_tpsa voltage=copysign(1.,source->reference_charge)*Q*source->voltage_norm;
    xt_float_or_tpsa k=source->k;
    xt_float_or_tpsa radial=voltage*(k*inverse_p)*(k*inverse_p)/4.;
    xt_float_or_tpsa phase=3.141592653589793238462643383279502884-source->phase;
    xt_float_or_tpsa ratio=nominal_gain*(1./h0+1./he);
    if (xt_float_or_tpsa_const_part(ratio)<0.) ratio=-ratio;
    const double r=xt_float_or_tpsa_const_part(ratio);
    int64_t count=1+(int64_t)fmin(fabs(xt_float_or_tpsa_const_part(k*source->length)),r/sqrt(.003));
    if (count>1000) count=1000;
    xt_float_or_tpsa logarithm=CavityModel_log1p(ratio);
    if (r>1.e-6) {
        int64_t geometric=(int64_t)floor(10.*xt_float_or_tpsa_const_part(logarithm)+.5);
        if (geometric>count) count=geometric;
        if (count>1000) count=1000;
    }
    // A positive requested resolution is a different, explicitly selected
    // finite recipe. Automatic zero retains the source's capped mesh.
    if (requested_cells>0) count=requested_cells;
    xt_float_or_tpsa state[6];
    state[0]=q[0]/k;state[1]=q[1]/k;
    state[2]=(q[2]-source->clock_origin)/k;
    if (reverse) state[2]-=source->length/beta0;
    for (int i=3;i<6;++i) state[i]=q[i];
    xt_float_or_tpsa p,ps;
    int status=CavityModel_frozen_shell(state,&p,&ps);
    if (status!=CAVITY_MODEL_OK) return status;
    if ((!reverse && entry_face) || (reverse && exit_face)) {
        xt_float_or_tpsa face_voltage=voltage;
        if (reverse) face_voltage=-voltage;
        status=CavityModel_frozen_face(state,face_voltage,
            source->length,k,phase,reverse,source->freeze_energy);
        if (status!=CAVITY_MODEL_OK) return status;
    }
    if (!reverse) {
        xt_float_or_tpsa nominal=h0,previous=0.;
        for (int64_t i=0;i<count;++i) {
            xt_float_or_tpsa weight=CavityModel_frozen_weight(i,count,ratio,logarithm);
            xt_float_or_tpsa distance=source->length*(previous+weight)*.5;
            status=CavityModel_frozen_drift(state,distance,beta0,source->freeze_zeta);
            if (status!=CAVITY_MODEL_OK) return status;
            nominal+=nominal_gain*weight;
            xt_float_or_tpsa beta=sqrt((nominal-1.)*(nominal+1.))/nominal;
            xt_float_or_tpsa clock=(beta/beta0-1.)*source->length*weight*.5;
            status=CavityModel_frozen_clock(state,clock,source->freeze_zeta);
            if (status!=CAVITY_MODEL_OK) return status;
            status=CavityModel_frozen_kick(state,weight,voltage,radial,k,phase,0,source->freeze_energy);
            if (status!=CAVITY_MODEL_OK) return status;
            status=CavityModel_frozen_clock(state,clock,source->freeze_zeta);
            if (status!=CAVITY_MODEL_OK) return status;
            previous=weight;
        }
        status=CavityModel_frozen_drift(state,source->length*previous*.5,beta0,source->freeze_zeta);
    } else {
        xt_float_or_tpsa last=CavityModel_frozen_weight(count-1,count,ratio,logarithm);
        status=CavityModel_frozen_drift(state,-source->length*last*.5,beta0,source->freeze_zeta);
        if (status!=CAVITY_MODEL_OK) return status;
        for (int64_t i=count-1;i>=0;--i) {
            xt_float_or_tpsa weight=CavityModel_frozen_weight(i,count,ratio,logarithm);
            // Closed geometric prefix avoids a second O(N) prefix loop and
            // retains the reference programme's parameter derivatives.
            xt_float_or_tpsa prefix=(i+1.)/count;
            if (r>1.e-6) prefix=CavityModel_expm1((i+1)*logarithm/count)/ratio;
            xt_float_or_tpsa nominal=h0+nominal_gain*prefix;
            xt_float_or_tpsa beta=sqrt((nominal-1.)*(nominal+1.))/nominal;
            xt_float_or_tpsa clock=(beta/beta0-1.)*source->length*weight*.5;
            status=CavityModel_frozen_clock(state,-clock,source->freeze_zeta);
            if (status!=CAVITY_MODEL_OK) return status;
            status=CavityModel_frozen_kick(state,weight,voltage,radial,k,phase,1,source->freeze_energy);
            if (status!=CAVITY_MODEL_OK) return status;
            status=CavityModel_frozen_clock(state,-clock,source->freeze_zeta);
            if (status!=CAVITY_MODEL_OK) return status;
            xt_float_or_tpsa previous=0.;
            if (i) previous=CavityModel_frozen_weight(i-1,count,ratio,logarithm);
            status=CavityModel_frozen_drift(state,-source->length*(previous+weight)*.5,beta0,source->freeze_zeta);
            if (status!=CAVITY_MODEL_OK) return status;
        }
    }
    if (status!=CAVITY_MODEL_OK) return status;
    if ((!reverse && exit_face) || (reverse && entry_face)) {
        xt_float_or_tpsa face_voltage=-voltage;
        if (reverse) face_voltage=voltage;
        status=CavityModel_frozen_face(state,face_voltage,
            source->length,k,phase,reverse,source->freeze_energy);
        if (status!=CAVITY_MODEL_OK) return status;
    }
    output[0]=k*state[0];output[1]=k*state[1];
    output[2]=source->clock_origin+k*state[2];
    if (!reverse) output[2]+=k*source->length/beta0;
    for (int i=3;i<6;++i) output[i]=state[i];
    return CavityModel_vector_finite(output,6) ? CAVITY_MODEL_OK : CAVITY_MODEL_NONFINITE;
}
#endif
