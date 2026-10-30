// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_FROZEN_H
#define XTRACK_CAVITY_MODELS_FROZEN_H

// Phase-resolved source conventions shared with the other families.
#include "xtrack/beam_elements/elements_src/cavity_models.h"

// Authenticated static factors of the frozen nominal-coefficient word.
// Authenticated straight static factors for the frozen-voltage word.
// State=(x[m],y[m],d=c(t-t_ref)[m],Px/(mc),Py/(mc),E/(mc^2)).
// They transcribe sad_reduction._solenoid_body_flow,
// _quadrupole_body_flow and _quadrupole_face. RF/static profiles, smooth
// fringe radiation and physical source qualification remain separate.

// Entire C(u)=cos(sqrt(u)), S(u)=sin(sqrt(u))/sqrt(u). Negative u
// gives the hyperbolic continuation. There is no sqrt(strength) singularity
// at zero gradient; the TPSA backend supplies the entire functions directly.
GPUFUN
void CavityModel_static_cs(xt_float_or_tpsa_arg u,
        xt_float_or_tpsa *cosine, xt_float_or_tpsa *sinc) {
#ifdef XTRACK_TPSA_TRACK
    auto pair=mad::sincosq(u);
    *sinc=pair.first; *cosine=pair.second;
#else
    if (u>0.) {
        double root=sqrt(u); *cosine=cos(root);*sinc=sin(root)/root;
    } else if (u<0.) {
        double root=sqrt(-u);*cosine=cosh(root);*sinc=sinh(root)/root;
    } else { *cosine=1.;*sinc=1.; }
#endif
}

GPUFUN
int CavityModel_static_shell(const xt_float_or_tpsa state[6],
        xt_float_or_tpsa_arg solenoid, xt_float_or_tpsa *momentum,
        xt_float_or_tpsa *longitudinal) {
    if (!CavityModel_vector_finite(state,6)) return CAVITY_MODEL_NONFINITE;
    xt_float_or_tpsa px=state[3]+solenoid*state[1]*.5;
    xt_float_or_tpsa py=state[4]-solenoid*state[0]*.5;
    xt_float_or_tpsa p2=(state[5]-1.)*(state[5]+1.);
    xt_float_or_tpsa square=p2-px*px-py*py;
    if (!CavityModel_scalar_finite(square)) return CAVITY_MODEL_NONFINITE;
    if (xt_float_or_tpsa_const_part(state[5])<=1.
        || xt_float_or_tpsa_const_part(square)<=0.) return CAVITY_MODEL_DOMAIN;
    *momentum=sqrt(p2);*longitudinal=sqrt(square);
    return CAVITY_MODEL_OK;
}

// Exact full-positive-root helix in A=(-b*y/2,b*x/2,0), b=q*Bz/(mc).
// Both observation planes are inside the uniform field; no SOL face is added.
GPUFUN
int CavityModel_static_solenoid(xt_float_or_tpsa state[6],
        xt_float_or_tpsa_arg distance, xt_float_or_tpsa_arg beta0,
        xt_float_or_tpsa_arg strength, int freeze_zeta) {
    xt_float_or_tpsa p,ps;
    int status=CavityModel_static_shell(state,strength,&p,&ps);
    if (status!=CAVITY_MODEL_OK) return status;
    xt_float_or_tpsa px=state[3]+strength*state[1]*.5;
    xt_float_or_tpsa py=state[4]-strength*state[0]*.5;
    xt_float_or_tpsa angle=strength*distance/ps;
    xt_float_or_tpsa sine=sin(angle),cosine=cos(angle);
    xt_float_or_tpsa sinc=CavityModel_static_sinc(angle);
    xt_float_or_tpsa half_sinc=CavityModel_static_sinc(angle*.5);
    xt_float_or_tpsa half=angle*half_sinc*half_sinc*.5;
    xt_float_or_tpsa linear=distance/ps;
    state[0]+=linear*(sinc*px+half*py);
    state[1]+=linear*(-half*px+sinc*py);
    state[3]=cosine*px+sine*py-strength*state[1]*.5;
    state[4]=-sine*px+cosine*py+strength*state[0]*.5;
    if (!freeze_zeta) state[2]+=distance*(state[5]/ps-1./beta0);
    return CavityModel_static_shell(state,strength,&p,&ps);
}

// Exact flow of Q=-p+P_perp^2/(2p)+g*(x^2-y^2)/2, including its
// energy derivative. The two plane integrals below are actual integrals
// of P_x(s)^2 and P_y(s)^2, not endpoint trapezoidal approximations.
GPUFUN
int CavityModel_static_quadratic(xt_float_or_tpsa state[6],
        xt_float_or_tpsa_arg distance, xt_float_or_tpsa_arg beta0,
        xt_float_or_tpsa_arg gradient, int freeze_zeta) {
    xt_float_or_tpsa p,ps;
    int status=CavityModel_static_shell(state,0.,&p,&ps);
    if (status!=CAVITY_MODEL_OK) return status;
    xt_float_or_tpsa u=gradient*distance*distance/p;
    xt_float_or_tpsa Cx,Sx,Cy,Sy,C4x,S4x,C4y,S4y;
    CavityModel_static_cs(u,&Cx,&Sx);
    CavityModel_static_cs(-u,&Cy,&Sy);
    CavityModel_static_cs(4.*u,&C4x,&S4x);
    CavityModel_static_cs(-4.*u,&C4y,&S4y);
    xt_float_or_tpsa x=state[0],y=state[1],px=state[3],py=state[4];
    xt_float_or_tpsa integral_x=distance*.5*(px*px*(1.+S4x)
         +p*gradient*x*x*(1.-S4x))-p*x*px*u*Sx*Sx;
    xt_float_or_tpsa integral_y=distance*.5*(py*py*(1.+S4y)
         -p*gradient*y*y*(1.-S4y))+p*y*py*u*Sy*Sy;
    state[0]=Cx*x+distance/p*Sx*px;
    state[1]=Cy*y+distance/p*Sy*py;
    state[3]=-gradient*distance*Sx*x+Cx*px;
    state[4]=gradient*distance*Sy*y+Cy*py;
    if (!freeze_zeta) state[2]+=distance*(state[5]/p-1./beta0)
        +state[5]*(integral_x+integral_y)/(2.*p*p*p);
    return CavityModel_static_shell(state,0.,&p,&ps);
}

// Exact shear of R=-sqrt(p^2-P_perp^2)+p-P_perp^2/(2p).
GPUFUN
int CavityModel_static_remainder(xt_float_or_tpsa state[6],
        xt_float_or_tpsa_arg distance, int freeze_zeta) {
    xt_float_or_tpsa p,ps;
    int status=CavityModel_static_shell(state,0.,&p,&ps);
    if (status!=CAVITY_MODEL_OK) return status;
    xt_float_or_tpsa angle2=(state[3]*state[3]+state[4]*state[4])/(p*p);
    xt_float_or_tpsa root=ps/p;
    xt_float_or_tpsa defect=angle2/(root*(1.+root));
    state[0]+=distance*state[3]*defect/p;
    state[1]+=distance*state[4]*defect/p;
    if (!freeze_zeta) state[2]+=distance*state[5]/p*(defect-angle2*.5);
    return CAVITY_MODEL_OK;
}

// tsolqu's Bz=0 normal-quadrupole R/2,Q,R,...,Q,R/2 word (default EPS).
// This is not silently replaced by an exact full-root quadrupole ODE.
GPUFUN
int CavityModel_static_quadrupole(xt_float_or_tpsa state[6],
        xt_float_or_tpsa_arg distance, xt_float_or_tpsa_arg beta0,
        xt_float_or_tpsa_arg gradient,
        xt_float_or_tpsa_arg reference_momentum, int freeze_zeta) {
    const int64_t count=1+(int64_t)(fabs(xt_float_or_tpsa_const_part(
        gradient*distance*distance/reference_momentum))/.2);
    xt_float_or_tpsa step=distance/count;
    int status=CavityModel_static_remainder(state,step*.5,freeze_zeta);
    if (status!=CAVITY_MODEL_OK) return status;
    for (int64_t i=0;i<count;++i) {
        status=CavityModel_static_quadratic(state,step,beta0,gradient,freeze_zeta);
        if (status!=CAVITY_MODEL_OK) return status;
        xt_float_or_tpsa remainder=step;
        if (i==count-1) remainder=step*.5;
        status=CavityModel_static_remainder(state,remainder,freeze_zeta);
        if (status!=CAVITY_MODEL_OK) return status;
    }
    return CAVITY_MODEL_OK;
}

// Source normal-quadrupole face q'=f(q,E), P'=Df^-T P,
// d'=d-P'.partial_E f. The matrix is Df=[[xx,cross],[-cross,yy]].
GPUFUN
void CavityModel_static_quad_face_values(xt_float_or_tpsa_arg x,
        xt_float_or_tpsa_arg y, xt_float_or_tpsa_arg coefficient,
        xt_float_or_tpsa *dx, xt_float_or_tpsa *dy,
        xt_float_or_tpsa *xx, xt_float_or_tpsa *yy,
        xt_float_or_tpsa *cross, xt_float_or_tpsa *correction) {
    xt_float_or_tpsa a=coefficient*x*x,b=coefficient*y*y,difference=a-b;
    *correction=difference*difference/6.;
    *dx=x*(a/3.+b+*correction);
    *dy=-y*(a+b/3.-*correction);
    *xx=1.+a+b+difference*(5.*a-b)/6.;
    *yy=1.-a-b+difference*(a-5.*b)/6.;
    *cross=2.*coefficient*x*y*(1.-difference/3.);
}

GPUFUN
int CavityModel_static_quad_face(xt_float_or_tpsa state[6],
        xt_float_or_tpsa_arg gradient, int inverse, int freeze_zeta) {
    xt_float_or_tpsa p,ps;
    int status=CavityModel_static_shell(state,0.,&p,&ps);
    if (status!=CAVITY_MODEL_OK) return status;
    xt_float_or_tpsa coefficient=gradient/(4.*p);
    xt_float_or_tpsa x=state[0],y=state[1],dx,dy,xx,yy,cross,correction;
    int converged=!inverse;
    if (inverse) {
        // Invert the actual finite coordinate transformation. Negating g is
        // not its exact inverse. The accepted root is differentiated by TPSA.
        for (int iteration=0;iteration<80;++iteration) {
            CavityModel_static_quad_face_values(x,y,coefficient,
                &dx,&dy,&xx,&yy,&cross,&correction);
            xt_float_or_tpsa rx=x+dx-state[0],ry=y+dy-state[1];
            if (fmax(CavityModel_scalar_norm(rx),CavityModel_scalar_norm(ry))<2e-13) {
                converged=1;break;
            }
            xt_float_or_tpsa determinant=xx*yy+cross*cross;
            if (xt_float_or_tpsa_const_part(determinant)==0.)
                return CAVITY_MODEL_DOMAIN;
            xt_float_or_tpsa change_x=(yy*rx-cross*ry)/determinant;
            xt_float_or_tpsa change_y=(cross*rx+xx*ry)/determinant;
            x-=change_x;y-=change_y;
            if (!CavityModel_scalar_finite(x) || !CavityModel_scalar_finite(y))
                return CAVITY_MODEL_NONFINITE;
        }
    }
    if (!converged) return CAVITY_MODEL_ITERATIONS;
    CavityModel_static_quad_face_values(x,y,coefficient,
        &dx,&dy,&xx,&yy,&cross,&correction);
    xt_float_or_tpsa determinant=xx*yy+cross*cross;
    if (!CavityModel_scalar_finite(determinant)
        || xt_float_or_tpsa_const_part(determinant)==0.) return CAVITY_MODEL_DOMAIN;
    xt_float_or_tpsa px=state[3],py=state[4];
    if (inverse) {
        state[0]=x;state[1]=y;
        state[3]=xx*px-cross*py;state[4]=cross*px+yy*py;
        if (!freeze_zeta) state[2]-=state[5]/(p*p)
            *(px*(dx+correction*x)+py*(dy+correction*y));
    } else {
        state[0]+=dx;state[1]+=dy;
        state[3]=(px*yy+py*cross)/determinant;
        state[4]=(py*xx-px*cross)/determinant;
        if (!freeze_zeta) state[2]+=state[5]/(p*p)
            *(state[3]*(dx+correction*x)+state[4]*(dy+correction*y));
    }
    return CavityModel_static_shell(state,0.,&p,&ps);
}

// Frozen voltage actions, finite word and parent-prefix composition.
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
        xt_float_or_tpsa_arg clock, int freeze_zeta,
        xt_float_or_tpsa_arg solenoid) {
    if (freeze_zeta) return CAVITY_MODEL_OK;
    xt_float_or_tpsa p,ps;
    int status=CavityModel_static_shell(state,solenoid,&p,&ps);
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
        xt_float_or_tpsa_arg phase, int inverse, int freeze_energy,
        xt_float_or_tpsa_arg solenoid) {
    xt_float_or_tpsa psi=wave_number*state[2]-phase;
    xt_float_or_tpsa signed_weight=weight;
    if (inverse) signed_weight=-weight;
    if (!freeze_energy)
        state[5]-=signed_weight*(voltage+radial
            *(state[0]*state[0]+state[1]*state[1]))*sin(psi);
    state[3]-=signed_weight*2.*radial*state[0]*cos(psi)/wave_number;
    state[4]-=signed_weight*2.*radial*state[1]*cos(psi)/wave_number;
    xt_float_or_tpsa p,ps;
    return CavityModel_static_shell(state,solenoid,&p,&ps);
}

GPUFUN
int CavityModel_frozen_body(xt_float_or_tpsa state[6],
        xt_float_or_tpsa_arg distance, xt_float_or_tpsa_arg beta0,
        xt_float_or_tpsa_arg solenoid, xt_float_or_tpsa_arg gradient,
        xt_float_or_tpsa_arg reference_momentum, int freeze_zeta) {
    if (!xt_float_or_tpsa_is_zero(solenoid))
        return CavityModel_static_solenoid(state,distance,beta0,solenoid,freeze_zeta);
    if (!xt_float_or_tpsa_is_zero(gradient))
        return CavityModel_static_quadrupole(state,distance,beta0,gradient,
                                             reference_momentum,freeze_zeta);
    return CavityModel_frozen_drift(state,distance,beta0,freeze_zeta);
}

GPUFUN
xt_float_or_tpsa CavityModel_frozen_weight(int64_t index, int64_t count,
        xt_float_or_tpsa_arg ratio, xt_float_or_tpsa_arg logarithm) {
    if (xt_float_or_tpsa_const_part(ratio)<=1.e-6) return 1./count;
    return CavityModel_expm1(logarithm/count)*exp(index*logarithm/count)/ratio;
}

// Anchored parent prefix. nominal_momentum and nominal_gain are fixed
// reference-programme data, independent of the particle's evolving energy.
// The caller supplies them in the same positive mass normalization as q.
// q uses the global source clock. Internally d=c(t-t_ref) uses the parent's
// entrance clock; the returned T includes the nominal path advance once.
// Authenticated static subsets are a uniform SOL body with RF faces off,
// or a straight normal QUAD with its own source faces. Combined SOL/QUAD,
// dipoles/skew/higher multipoles require their own finite source factors.
GPUFUN
int CavityModel_frozen_prefix(const xt_float_or_tpsa q[6], xt_float_or_tpsa_arg Q,
        const CavityModelSource *source,
        xt_float_or_tpsa_arg nominal_momentum,
        xt_float_or_tpsa_arg nominal_gain,
        int64_t requested_cells,
        int entry_face, int exit_face, double fraction, int reverse,
        xt_float_or_tpsa output[6]) {
    if (!isfinite(fraction) || fraction<0. || fraction>1.)
        return CAVITY_MODEL_ARGUMENT;
    if (fraction==0.) {
        for (int i=0;i<6;++i) output[i]=q[i];
        return CAVITY_MODEL_OK;
    }
    exit_face=exit_face && fraction==1.;
    if (!xt_float_or_tpsa_is_zero(source->backward)
        || !xt_float_or_tpsa_is_zero(source->dc_gradient)) return CAVITY_MODEL_ARGUMENT;
    // Transverse frozen-coordinate analysis of the static finite factors
    // requires its own constrained word. Silently freezing only the final
    // particle setters would execute a different internal trajectory.
    if (source->freeze_x || source->freeze_px || source->freeze_y || source->freeze_py)
        return CAVITY_MODEL_ARGUMENT;
    for (int64_t i=0;i<=source->order;++i)
        if ((i!=1 && !xt_float_or_tpsa_is_zero(CavityModel_static_coefficient(source,i,0)))
            || !xt_float_or_tpsa_is_zero(CavityModel_static_coefficient(source,i,1)))
            return CAVITY_MODEL_ARGUMENT;
    xt_float_or_tpsa solenoid=Q*source->magnetic_scale*source->ks;
    xt_float_or_tpsa gradient=0.;
    if (source->order>=1)
        gradient=Q*source->magnetic_scale
            *CavityModel_static_coefficient(source,1,0)/source->length;
    const int solenoid_present=!xt_float_or_tpsa_is_zero(solenoid);
    const int quad_present=!xt_float_or_tpsa_is_zero(gradient);
    if (solenoid_present && (quad_present || entry_face || exit_face))
        return CAVITY_MODEL_ARGUMENT;
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
    if (reverse) state[2]-=fraction*source->length/beta0;
    for (int i=3;i<6;++i) state[i]=q[i];
    xt_float_or_tpsa p,ps;
    int status=CavityModel_static_shell(state,solenoid,&p,&ps);
    if (status!=CAVITY_MODEL_OK) return status;
    if ((!reverse && entry_face) || (reverse && exit_face)) {
        xt_float_or_tpsa face_voltage=voltage;
        if (reverse) face_voltage=-voltage;
        status=CavityModel_frozen_face(state,face_voltage,
            source->length,k,phase,reverse,source->freeze_energy);
        if (status!=CAVITY_MODEL_OK) return status;
        if (quad_present) {
            xt_float_or_tpsa face_gradient=gradient;
            if (reverse) face_gradient=-gradient;
            status=CavityModel_static_quad_face(state,face_gradient,reverse,
                                                 source->freeze_zeta);
            if (status!=CAVITY_MODEL_OK) return status;
        }
    }
    // The parent mesh, voltage and nominal programme never change with a
    // prefix. Only the observation plane moves. Each kick belongs to its
    // right-continuous prefix; topology decisions use the constant chart.
    xt_float_or_tpsa target=fraction*source->length;
    if (!reverse) {
        xt_float_or_tpsa previous_node=0.,previous_weight=0.;
        for (int64_t i=0;i<count;++i) {
            xt_float_or_tpsa weight=CavityModel_frozen_weight(i,count,ratio,logarithm);
            xt_float_or_tpsa prefix=(i+1.)/count;
            if (r>1.e-6) prefix=CavityModel_expm1((i+1)*logarithm/count)/ratio;
            xt_float_or_tpsa node=source->length*(prefix-.5*weight);
            if (fraction<1. && xt_float_or_tpsa_const_part(node-target)>0.) break;
            // Preserve the original full-parent body calls and arithmetic;
            // a normal-QUAD finite factor is not a drift semigroup.
            status=CavityModel_frozen_body(state,source->length*(previous_weight+weight)*.5,beta0,solenoid,gradient,
                                            nominal_momentum,source->freeze_zeta);
            if (status!=CAVITY_MODEL_OK) return status;
            xt_float_or_tpsa nominal=h0+nominal_gain*prefix;
            xt_float_or_tpsa beta=sqrt((nominal-1.)*(nominal+1.))/nominal;
            xt_float_or_tpsa clock=(beta/beta0-1.)*source->length*weight*.5;
            status=CavityModel_frozen_clock(state,clock,source->freeze_zeta,solenoid);
            if (status!=CAVITY_MODEL_OK) return status;
            status=CavityModel_frozen_kick(state,weight,voltage,radial,k,phase,0,
                                          source->freeze_energy,solenoid);
            if (status!=CAVITY_MODEL_OK) return status;
            status=CavityModel_frozen_clock(state,clock,source->freeze_zeta,solenoid);
            if (status!=CAVITY_MODEL_OK) return status;
            previous_node=node;previous_weight=weight;
        }
        xt_float_or_tpsa final_distance=target-previous_node;
        if (fraction==1.) final_distance=source->length*previous_weight*.5;
        status=CavityModel_frozen_body(state,final_distance,beta0,
                                       solenoid,gradient,nominal_momentum,source->freeze_zeta);
    } else {
        xt_float_or_tpsa previous_node=target,previous_weight=0.;
        int first_node=1;
        for (int64_t i=count-1;i>=0;--i) {
            xt_float_or_tpsa weight=CavityModel_frozen_weight(i,count,ratio,logarithm);
            // Closed geometric prefix retains the full reference programme
            // and avoids a second O(N) prefix loop on every reverse stage.
            xt_float_or_tpsa prefix=(i+1.)/count;
            if (r>1.e-6) prefix=CavityModel_expm1((i+1)*logarithm/count)/ratio;
            xt_float_or_tpsa node=source->length*(prefix-.5*weight);
            if (fraction<1. && xt_float_or_tpsa_const_part(node-target)>0.) continue;
            xt_float_or_tpsa distance=node-previous_node;
            if (first_node && fraction==1.) distance=-source->length*weight*.5;
            if (!first_node) distance=-source->length*(previous_weight+weight)*.5;
            status=CavityModel_frozen_body(state,distance,beta0,
                solenoid,gradient,nominal_momentum,source->freeze_zeta);
            if (status!=CAVITY_MODEL_OK) return status;
            xt_float_or_tpsa nominal=h0+nominal_gain*prefix;
            xt_float_or_tpsa beta=sqrt((nominal-1.)*(nominal+1.))/nominal;
            xt_float_or_tpsa clock=(beta/beta0-1.)*source->length*weight*.5;
            status=CavityModel_frozen_clock(state,-clock,source->freeze_zeta,solenoid);
            if (status!=CAVITY_MODEL_OK) return status;
            status=CavityModel_frozen_kick(state,weight,voltage,radial,k,phase,1,
                                          source->freeze_energy,solenoid);
            if (status!=CAVITY_MODEL_OK) return status;
            status=CavityModel_frozen_clock(state,-clock,source->freeze_zeta,solenoid);
            if (status!=CAVITY_MODEL_OK) return status;
            previous_node=node;previous_weight=weight;first_node=0;
        }
        xt_float_or_tpsa final_distance=-previous_node;
        if (!first_node) final_distance=-source->length*previous_weight*.5;
        status=CavityModel_frozen_body(state,final_distance,beta0,solenoid,gradient,
                                      nominal_momentum,source->freeze_zeta);
    }
    if (status!=CAVITY_MODEL_OK) return status;
    if ((!reverse && exit_face) || (reverse && entry_face)) {
        if (quad_present) {
            xt_float_or_tpsa face_gradient=-gradient;
            if (reverse) face_gradient=gradient;
            status=CavityModel_static_quad_face(state,face_gradient,reverse,
                                                 source->freeze_zeta);
            if (status!=CAVITY_MODEL_OK) return status;
        }
        xt_float_or_tpsa face_voltage=-voltage;
        if (reverse) face_voltage=voltage;
        status=CavityModel_frozen_face(state,face_voltage,
            source->length,k,phase,reverse,source->freeze_energy);
        if (status!=CAVITY_MODEL_OK) return status;
    }
    output[0]=k*state[0];output[1]=k*state[1];
    output[2]=source->clock_origin+k*state[2];
    if (!reverse) output[2]+=k*fraction*source->length/beta0;
    for (int i=3;i<6;++i) output[i]=state[i];
    return CavityModel_vector_finite(output,6) ? CAVITY_MODEL_OK : CAVITY_MODEL_NONFINITE;
}
// The complete retained word remains the fraction-one prefix.
GPUFUN
int CavityModel_frozen(const xt_float_or_tpsa q[6], xt_float_or_tpsa_arg Q,
        const CavityModelSource *source, xt_float_or_tpsa_arg nominal_momentum,
        xt_float_or_tpsa_arg nominal_gain, int64_t requested_cells,
        int entry_face, int exit_face, int reverse, xt_float_or_tpsa output[6]) {
    return CavityModel_frozen_prefix(q,Q,source,nominal_momentum,nominal_gain,
        requested_cells,entry_face,exit_face,1.,reverse,output);
}

// Any child is the interval M_b M_a^-1 of the same anchored parent family.
// This preserves the nominal energy programme and source finite word rather
// than recalibrating independent short cavities. Physical faces occur once.
GPUFUN
int CavityModel_frozen_interval(const xt_float_or_tpsa q[6], xt_float_or_tpsa_arg Q,
        const CavityModelSource *source, xt_float_or_tpsa_arg nominal_momentum,
        xt_float_or_tpsa_arg nominal_gain, int64_t requested_cells,
        int entry_face, int exit_face, double entrance, double exit,
        int reverse, xt_float_or_tpsa output[6]) {
    if (!isfinite(entrance) || !isfinite(exit) || entrance<0. || exit>1.
        || entrance>exit) return CAVITY_MODEL_ARGUMENT;
    if (entrance==exit) {
        for (int i=0;i<6;++i) output[i]=q[i];
        return CAVITY_MODEL_OK;
    }
    xt_float_or_tpsa temporary[6];
    double first=entrance,second=exit;
    if (reverse) { first=exit;second=entrance; }
    int status=CavityModel_frozen_prefix(q,Q,source,nominal_momentum,nominal_gain,
        requested_cells,entry_face,exit_face,first,1,temporary);
    if (status!=CAVITY_MODEL_OK) return status;
    return CavityModel_frozen_prefix(temporary,Q,source,nominal_momentum,nominal_gain,
        requested_cells,entry_face,exit_face,second,0,output);
}

#endif
