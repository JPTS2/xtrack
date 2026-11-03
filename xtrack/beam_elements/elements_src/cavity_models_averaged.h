// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_AVERAGED_H
#define XTRACK_CAVITY_MODELS_AVERAGED_H

// Phase-averaged (Sagan-type) cavity: each step is the symmetric word
// half drift, ponderomotive half-kick, synchronous radial term, energy kick,
// ponderomotive half-kick, half drift, with a canonical E/p edge for every
// jump of the stepped forward-wave gradient. Coordinates are those of the
// phase-resolved model, q = (X, Y, T, Px, Py, E) with k = m c = 1, and the
// forward wave on axis is Q f e(Z) cos(T - nu Z).
#include "xtrack/beam_elements/elements_src/cavity_models.h"
#include "xtrack/beam_elements/elements_src/cavity_models_compat.h"

#define CAVITY_AVERAGED_MIDPOINT_ITERATIONS 60

// Spin (CPU scalar only): each factor turns the spin by its Thomas-BMT
// rotation (PLAN P4, D15). spin is NULL when spin is not tracked.
typedef struct {
    double *spin;
    double anomaly;
} CavityAveragedSpin;

GPUFUN
void CavityModel_averaged_rotate(double spin[3], double ax, double ay, double az) {
    const double angle=sqrt(ax*ax+ay*ay+az*az);
    if (angle==0.) return;
    const double n[3]={ax/angle,ay/angle,az/angle};
    const double c=cos(angle), s=sin(angle);
    const double dot=n[0]*spin[0]+n[1]*spin[1]+n[2]*spin[2];
    const double cross[3]={n[1]*spin[2]-n[2]*spin[1],n[2]*spin[0]-n[0]*spin[2],
                           n[0]*spin[1]-n[1]*spin[0]};
    for (int i=0;i<3;++i) spin[i]=spin[i]*c+cross[i]*s+n[i]*dot*(1.-c);
}

// Rotation with a transverse kick (dpx, dpy) at energy E; kind 0 is an edge
// (radial electric field), kind 1 the ponderomotive kick.
GPUFUN
void CavityModel_averaged_kick_spin(const CavityAveragedSpin *spin, double dpx,
                                    double dpy, double energy, int kind) {
    if (!spin || !spin->spin) return;
    const double a=spin->anomaly, gamma=energy, beta=sqrt(1.-1./(gamma*gamma));
    const double p=beta*gamma;
    double factor;
    if (kind==1) {
        const double radial=((a+1./gamma)+beta*(a+1./(gamma+1.)))/(1.+beta);
        factor=beta*gamma*(radial-2.*(a+1./(gamma+1.)))+2.;
    } else {
        factor=gamma*beta*beta*(a+1./(gamma+1.));
    }
    CavityModel_averaged_rotate(spin->spin, -factor*dpy/p, factor*dpx/p, 0.);
}
#define CAVITY_AVERAGED_MIDPOINT_TOLERANCE 1e-15

GPUFUN
int CavityModel_averaged_drift(xt_float_or_tpsa q[6], xt_float_or_tpsa_arg length,
                               int freeze_zeta) {
    const xt_float_or_tpsa square=q[5]*q[5]-1.-q[3]*q[3]-q[4]*q[4];
    if (xt_float_or_tpsa_const_part(square)<=0.) return CAVITY_MODEL_DOMAIN;
    const xt_float_or_tpsa pz=sqrt(square);
    q[0]+=length*q[3]/pz;
    q[1]+=length*q[4]/pz;
    if (!freeze_zeta) q[2]+=length*q[5]/pz;
    return CAVITY_MODEL_OK;
}

// Flow of c R^2 / p.
GPUFUN
int CavityModel_averaged_ponderomotive(xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg coefficient, int freeze_zeta,
        const CavityAveragedSpin *spin) {
    const double px0=xt_float_or_tpsa_const_part(q[3]);
    const double py0=xt_float_or_tpsa_const_part(q[4]);
    const xt_float_or_tpsa square=(q[5]-1.)*(q[5]+1.);
    if (xt_float_or_tpsa_const_part(square)<=0.) return CAVITY_MODEL_DOMAIN;
    const xt_float_or_tpsa p=sqrt(square);
    const xt_float_or_tpsa r2=q[0]*q[0]+q[1]*q[1];
    if (!freeze_zeta) q[2]+=coefficient*q[5]*r2/(p*p*p);
    q[3]-=2.*coefficient*q[0]/p;
    q[4]-=2.*coefficient*q[1]/p;
    CavityModel_averaged_kick_spin(spin, xt_float_or_tpsa_const_part(q[3])-px0,
        xt_float_or_tpsa_const_part(q[4])-py0, xt_float_or_tpsa_const_part(q[5]), 1);
    return CAVITY_MODEL_OK;
}

// Implicit midpoint (symplectic) for a generator a w(E) trig(T - phase) of
// unit duration: kind 0 is the edge, a cos with w = E/p; kind 1 the
// synchronous radial term, a sin with w = E/p - 1. Positions are invariant
// and the momenta change by c (X, Y).
GPUFUN
int CavityModel_averaged_midpoint(xt_float_or_tpsa q[6], xt_float_or_tpsa_arg a,
        xt_float_or_tpsa_arg momentum_factor, xt_float_or_tpsa_arg phase, int kind,
        int freeze_energy, int freeze_zeta) {
    xt_float_or_tpsa T1=q[2], E1=q[5], c=0.;
    int converged=0, sweeps=CAVITY_MODEL_JET_SWEEPS;
    for (int iteration=0; iteration<CAVITY_AVERAGED_MIDPOINT_ITERATIONS
                          +CAVITY_MODEL_JET_SWEEPS; ++iteration) {
        const xt_float_or_tpsa Tm=(q[2]+T1)*.5, Em=(q[5]+E1)*.5;
        const xt_float_or_tpsa square=(Em-1.)*(Em+1.);
        if (xt_float_or_tpsa_const_part(square)<=0.) return CAVITY_MODEL_DOMAIN;
        const xt_float_or_tpsa p=sqrt(square);
        const xt_float_or_tpsa theta=Tm-phase;
        const xt_float_or_tpsa dw=-1./(p*p*p);
        xt_float_or_tpsa w=Em/p, dT, dE;
        if (kind==0) {
            dT=-a*dw*cos(theta);
            dE=-a*w*sin(theta);
            c=-momentum_factor*w*cos(theta);
        } else {
            w-=1.;
            dT=-a*dw*sin(theta);
            dE=a*w*cos(theta);
            c=-momentum_factor*w*sin(theta);
        }
        if (freeze_zeta) dT=0.;
        if (freeze_energy) dE=0.;
        const xt_float_or_tpsa next_T=q[2]+dT, next_E=q[5]+dE;
        const double change=fmax(
            CavityModel_primal_residual(next_T-T1)/(1.+fabs(xt_float_or_tpsa_const_part(q[2]))),
            CavityModel_primal_residual(next_E-E1)/xt_float_or_tpsa_const_part(q[5]));
        T1=next_T; E1=next_E;
        if (converged) {
            if (--sweeps<0) break;
        } else if (change<=CAVITY_AVERAGED_MIDPOINT_TOLERANCE) {
            converged=1;
            if (sweeps==0) break;
        }
    }
    if (!converged) return CAVITY_MODEL_ITERATIONS;
    q[2]=T1; q[5]=E1;
    q[3]+=c*q[0];
    q[4]+=c*q[1];
    return CAVITY_MODEL_OK;
}

// Canonical edge for a jump of the forward gradient at Z, finite-beta E/p.
GPUFUN
int CavityModel_averaged_edge(xt_float_or_tpsa q[6], xt_float_or_tpsa_arg jump,
        xt_float_or_tpsa_arg z, const CavityModelSource *source,
        const CavityAveragedSpin *spin) {
    if (xt_float_or_tpsa_is_zero(jump)) return CAVITY_MODEL_OK;
    const double px0=xt_float_or_tpsa_const_part(q[3]);
    const double py0=xt_float_or_tpsa_const_part(q[4]);
    const double energy=xt_float_or_tpsa_const_part(q[5]);
    const xt_float_or_tpsa r2=q[0]*q[0]+q[1]*q[1];
    const int status=CavityModel_averaged_midpoint(q, jump*r2/4., jump/2.,
        source->carrier*z, 0, source->freeze_energy, source->freeze_zeta);
    if (status==CAVITY_MODEL_OK)
        CavityModel_averaged_kick_spin(spin, xt_float_or_tpsa_const_part(q[3])-px0,
            xt_float_or_tpsa_const_part(q[4])-py0, energy, 0);
    return status;
}

// The counter-propagating wave's direct energy gain over one step, as the
// complex amplitude Q b int e(Z) exp(i (1/beta0 + nu) Z) dZ on [z_mid-h/2,
// z_mid+h/2]; it does not average away in a short cavity, and the shared
// calibration includes it. Eight-point Gauss-Legendre per step.
GPUFUN
void CavityModel_averaged_counter_gain(xt_float_or_tpsa_arg h,
        xt_float_or_tpsa_arg z_mid, xt_float_or_tpsa_arg Q,
        const CavityModelSource *source, xt_float_or_tpsa out[2]) {
    const double nodes[4]={.1834346424956498,.5255324099163290,
                           .7966664774136267,.9602898564975363};
    const double weights[4]={.3626837833783620,.3137066458778873,
                             .2223810344533745,.1012285362903763};
    const xt_float_or_tpsa detuning=1./source->reference_beta+source->carrier;
    out[0]=0.; out[1]=0.;
    xt_float_or_tpsa envelope[5];
    for (int i=0;i<4;++i) {
        for (int side=-1; side<=1; side+=2) {
            const xt_float_or_tpsa z=z_mid+side*nodes[i]*h*.5;
            CavityModel_source_envelope(z, source, envelope);
            const xt_float_or_tpsa w=weights[i]*h*.5*envelope[0];
            out[0]+=w*cos(detuning*z);
            out[1]+=w*sin(detuning*z);
        }
    }
    const xt_float_or_tpsa scale=Q*source->backward;
    out[0]*=scale; out[1]*=scale;
}

// One symmetric step of signed length h around z_mid; a negative length
// applies the inverse word.
GPUFUN
int CavityModel_averaged_step(xt_float_or_tpsa q[6], xt_float_or_tpsa_arg h,
        xt_float_or_tpsa_arg z_mid, xt_float_or_tpsa_arg gradient,
        xt_float_or_tpsa_arg counter_gradient, xt_float_or_tpsa_arg Q,
        const CavityModelSource *source, const CavityAveragedSpin *spin) {
    const int freeze_energy=source->freeze_energy, freeze_zeta=source->freeze_zeta;
    const xt_float_or_tpsa coefficient=h*counter_gradient*counter_gradient/32.;
    const xt_float_or_tpsa phase=source->carrier*z_mid;
    int status=CavityModel_averaged_drift(q, .5*h, freeze_zeta);
    if (status==CAVITY_MODEL_OK)
        status=CavityModel_averaged_ponderomotive(q, coefficient, freeze_zeta, spin);
    for (int factor=0; factor<2 && status==CAVITY_MODEL_OK; ++factor) {
        // The inverse word takes the energy kick before the radial term.
        const int energy_first=xt_float_or_tpsa_const_part(h)<0.;
        if (factor==energy_first) {
            const xt_float_or_tpsa r2=q[0]*q[0]+q[1]*q[1];
            status=CavityModel_averaged_midpoint(q, gradient*h*r2/4.,
                gradient*h/2., phase, 1, freeze_energy, freeze_zeta);
        } else if (!freeze_energy) {
            if (spin && spin->spin) {
                // Carrier precession in the forward wave's radial fields,
                // then the longitudinal-field rotation of the energy kick.
                const double a=spin->anomaly, E=xt_float_or_tpsa_const_part(q[5]);
                const double beta=sqrt(1.-1./(E*E));
                const double carrier=xt_float_or_tpsa_const_part(gradient*h
                    *sin(q[2]-phase))/(2.*beta)*((a+1./E)-beta*(a+1./(E+1.)));
                CavityModel_averaged_rotate(spin->spin,
                    -carrier*xt_float_or_tpsa_const_part(q[1]),
                    carrier*xt_float_or_tpsa_const_part(q[0]), 0.);
            }
            const double px=xt_float_or_tpsa_const_part(q[3]);
            const double py=xt_float_or_tpsa_const_part(q[4]);
            const double E0=xt_float_or_tpsa_const_part(q[5]);
            xt_float_or_tpsa counter_gain[2];
            CavityModel_averaged_counter_gain(h, z_mid, Q, source, counter_gain);
            const xt_float_or_tpsa counter_phase=q[2]-z_mid/source->reference_beta
                +source->counter_phase;
            q[5]+=h*gradient*cos(q[2]-phase)
                +counter_gain[0]*cos(counter_phase)-counter_gain[1]*sin(counter_phase);
            if (spin && spin->spin) {
                const double pz=sqrt(E0*E0-1.-px*px-py*py);
                const double factor=(spin->anomaly+1./(E0+1.))
                    *(xt_float_or_tpsa_const_part(q[5])-E0)/(pz/E0);
                CavityModel_averaged_rotate(spin->spin, factor*py/E0, -factor*px/E0, 0.);
            }
        }
    }
    if (status==CAVITY_MODEL_OK)
        status=CavityModel_averaged_ponderomotive(q, coefficient, freeze_zeta, spin);
    if (status==CAVITY_MODEL_OK)
        status=CavityModel_averaged_drift(q, .5*h, freeze_zeta);
    return status;
}

// Interval [left, right] of the parent (fractions), cells equal steps. A
// slice edge inside the parent closes against the envelope at its boundary,
// so adjacent slices telescope to the parent's jumps.
GPUFUN
int CavityModel_averaged_interval(const xt_float_or_tpsa q_in[6],
        xt_float_or_tpsa_arg Q, const CavityModelSource *source, int64_t cells,
        double left, double right, int reverse, const CavityAveragedSpin *spin,
        xt_float_or_tpsa q[6]) {
    if (cells<=0 || !(right>left)) return CAVITY_MODEL_ARGUMENT;
    if (!xt_float_or_tpsa_is_zero(source->dc_gradient)) return CAVITY_MODEL_ARGUMENT;
    for (int i=0;i<6;++i) q[i]=q_in[i];
    const xt_float_or_tpsa length=source->k*source->length;
    const xt_float_or_tpsa z_left=length*(left-.5), z_right=length*(right-.5);
    const xt_float_or_tpsa h=(z_right-z_left)/cells;
    xt_float_or_tpsa envelope[5];
    xt_float_or_tpsa outer_left=0., outer_right=0.;
    if (left>0.) {
        CavityModel_source_envelope(z_left, source, envelope);
        outer_left=Q*source->forward*envelope[0];
    }
    if (right<1.) {
        CavityModel_source_envelope(z_right, source, envelope);
        outer_right=Q*source->forward*envelope[0];
    }
    int status=CAVITY_MODEL_OK;
    xt_float_or_tpsa signed_h=h;
    if (reverse) signed_h=-h;
    xt_float_or_tpsa previous=reverse ? outer_right : outer_left;
    for (int64_t n=0; n<cells && status==CAVITY_MODEL_OK; ++n) {
        const int64_t j=reverse ? cells-1-n : n;
        const xt_float_or_tpsa z_mid=z_left+(j+.5)*h;
        CavityModel_source_envelope(z_mid, source, envelope);
        const xt_float_or_tpsa gradient=Q*source->forward*envelope[0];
        const xt_float_or_tpsa counter=Q*source->backward*envelope[0];
        const xt_float_or_tpsa boundary=reverse ? z_left+(j+1.)*h : z_left+j*h;
        // The inverse word crosses the same boundaries with the same jump
        // gradient-previous: the edge inverse negates the jump it undoes.
        status=CavityModel_averaged_edge(q, gradient-previous, boundary, source, spin);
        if (status==CAVITY_MODEL_OK)
            status=CavityModel_averaged_step(q, signed_h, z_mid,
                                             gradient, counter, Q, source, spin);
        previous=gradient;
    }
    if (status==CAVITY_MODEL_OK) {
        const xt_float_or_tpsa outer=reverse ? outer_left : outer_right;
        const xt_float_or_tpsa boundary=reverse ? z_left : z_right;
        status=CavityModel_averaged_edge(q, outer-previous, boundary, source, spin);
    }
    return status;
}

#endif
