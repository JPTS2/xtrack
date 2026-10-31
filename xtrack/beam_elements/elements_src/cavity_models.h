// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_H
#define XTRACK_CAVITY_MODELS_H

// Full, paraxial and longitudinal phase-resolved source/maps.
// Conservative source-based RF maps. Coordinates are (X,Y,T,Px,Py,E),
// scaled by (1/k,1/k,1/(kc),mc,mc,mc^2); the T--E block has negative sign.
// k is the coordinate scale omega/c, independent of the physical carrier.
// Source ancestry: cavity-modelling phase_resolved/conservative_source.py,
// kinetic/transport.py, voltage.py and phase_resolved/localized.py.
// This header supplies numerical maps, not physical/numerical admission.
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include "xtrack/headers/constants.h"
#include "xtrack/beam_elements/elements_src/cavity_models_compat.h"

enum CavityModelStatus {
    CAVITY_MODEL_OK = 1,
    CAVITY_MODEL_DOMAIN = -1,
    CAVITY_MODEL_ITERATIONS = -2,
    CAVITY_MODEL_NONFINITE = -3,
    CAVITY_MODEL_ARGUMENT = -4
};

enum CavityPhysicalModel {
    CAVITY_MODEL_FULL = 10,
    CAVITY_MODEL_PARAXIAL = 11,
    CAVITY_MODEL_LONGITUDINAL = 12,
    CAVITY_MODEL_FROZEN = 13,
    CAVITY_MODEL_AVERAGED_SECOND_ORDER = 14,
    CAVITY_MODEL_AVERAGED_CUBIC = 15,
    CAVITY_MODEL_AVERAGED_QUARTIC = 16,
    CAVITY_MODEL_AVERAGED_FULL_SLOW = 17
};

typedef struct {
    // omega is the coordinate-clock scale c*k. With RF off and f=0, k=1/m
    // is an arbitrary unit choice, not a nonzero physical RF frequency.
    xt_float_or_tpsa k, omega, half_span, length;
    xt_float_or_tpsa forward, backward, carrier, counter_phase, clock_origin;
    xt_float_or_tpsa voltage_norm, phase, dc_gradient;
    // Built-in profile: 0 P16, 1 uniform, 2 Gaussian. Width/centre in Z=k*z.
    int rf_profile;
    xt_float_or_tpsa profile_sigma, profile_center;
    // 0: uniform longitudinal gauge; 1: Gaussian original fringe donor.
    int magnetic_profile;
    xt_float_or_tpsa magnetic_sigma; // normalized k*sigma_m, centred at Z=0
    double reference_charge, reference_beta;
    int freeze_energy, freeze_zeta;
    int freeze_x, freeze_px, freeze_y, freeze_py;
    // q_ref A/(p_ref) conventions: magnetic_scale=p_ref/(q_ref*m_source*c).
    // knl/ksl are the ordinary integrated normal/skew magnetic strengths.
    xt_float_or_tpsa magnetic_scale, ks;
    int64_t order;
    GPUGLMEM const double *knl, *ksl;
    // Optional process-local handles. The ordinary Float64 arrays retain
    // scalar mirrors; CPU TPSA reads the live indexed series when present.
    GPUGLMEM const uint64_t *knl_tpsa, *ksl_tpsa;
} CavityModelSource;

GPUFUN
xt_float_or_tpsa CavityModel_static_coefficient(const CavityModelSource *source,
                                               int64_t index,int skew) {
    GPUGLMEM const double *values=skew ? source->ksl : source->knl;
    GPUGLMEM const uint64_t *handles=skew ? source->ksl_tpsa : source->knl_tpsa;
    if (handles && handles[index]) {
#ifdef XTRACK_TPSA_TRACK
        uint64_t handle=handles[index];
        return xt_float_or_tpsa_get(&handle,1);
#elif defined(XTRACK_TPSA_COMPAT_H)
        return xt_float_or_tpsa_get_double(handles[index],1);
#endif
    }
    return values ? values[index] : 0.;
}

typedef struct {
    xt_float_or_tpsa a[4];
    // Derivatives at fixed Z, columns X,Y,T; rows Ax,Ay,Az,Phi.
    xt_float_or_tpsa d[4][3];
} CavityModelPotential;

GPUFUN
int CavityModel_vector_finite(const xt_float_or_tpsa *value, int count) {
    for (int i=0; i<count; ++i)
        if (!CavityModel_scalar_finite(value[i])) return 0;
    return 1;
}

// Exact Fourier integral of the P16 envelope. For x=H*detuning,
// I/H=B(1/2,17)*0F1(;35/2;-x*x/4)=2^17*16!*j_16(x)/x^16.
// The regular series avoids a removable singularity at zero. Forward
// spherical-Bessel recurrence is used only in its oscillatory regime x>=16.
// All source-control arithmetic, including the zero-base-point terms, remains
// in the scalar algebra. This is not the former fixed-node quadrature.
GPUFUN
xt_float_or_tpsa CavityModel_p16_integral(xt_float_or_tpsa_arg H,
                                         xt_float_or_tpsa_arg detuning) {
    xt_float_or_tpsa x=H*detuning;
    if(fabs(xt_float_or_tpsa_const_part(x))<=16.) {
        const xt_float_or_tpsa square=-x*x*.25;
        xt_float_or_tpsa term=1.,sum=1.,correction=0.;
        // TPSA retains the full series length: a zero primal term can still
        // carry higher coefficients. Scalar arithmetic may stop on the
        // alternating-series next-term bound once its terms decrease.
        for(int n=1;n<=96;++n) {
            term*=square/(n*(16.5+n));
            const xt_float_or_tpsa adjusted=term-correction;
            const xt_float_or_tpsa next=sum+adjusted;
            correction=(next-sum)-adjusted;sum=next;
#ifndef XTRACK_TPSA_TRACK
            const double ratio=fabs(square)/((n+1.)*(n+17.5));
            if(ratio<1.&&fabs(term)*ratio<=2.2204460492503131e-16*fabs(sum)) {
                term=0.;break;
            }
#endif
        }
        if(!CavityModel_scalar_finite(sum)
            ||CavityModel_scalar_norm(term)>8.*2.2204460492503131e-16
                *CavityModel_scalar_norm(sum))return NAN;
        return H*.433055299379213223749060294983877685*sum;
    }
    if(xt_float_or_tpsa_const_part(x)<0.)x=-x;
    xt_float_or_tpsa previous=sin(x)/x;
    xt_float_or_tpsa current=(previous-cos(x))/x;
    for(int n=1;n<16;++n) {
        const xt_float_or_tpsa next=(2.*n+1.)/x*current-previous;
        previous=current;current=next;
    }
    xt_float_or_tpsa factor=2.;
    for(int n=1;n<=16;++n)factor*=2.*n/x;
    return H*factor*current;
}

// Live fixed-beta electric-crest calibration. No evolving particle beta
// retunes the hardware. wave_number is already resolved by the caller and
// may be zero; it is not the coordinate normalization k=omega/c.
// Only RF entries are assigned; the caller supplies static strengths.
GPUFUN
int CavityModel_prepare(
        xt_float_or_tpsa_arg frequency, xt_float_or_tpsa_arg voltage,
        xt_float_or_tpsa_arg length, xt_float_or_tpsa_arg phase,
        xt_float_or_tpsa_arg eta, xt_float_or_tpsa_arg counter_phase,
        xt_float_or_tpsa_arg calibration_beta,
        xt_float_or_tpsa_arg wave_number,
        double source_mass_eV, double q0, CavityModelSource *source) {
    const double f=xt_float_or_tpsa_const_part(frequency);
    const double L=xt_float_or_tpsa_const_part(length);
    const double beta=xt_float_or_tpsa_const_part(calibration_beta);
    const double mixture=xt_float_or_tpsa_const_part(eta);
    if (!isfinite(f) || f<0. || !isfinite(L) || L<=0.
        || !isfinite(source_mass_eV) || source_mass_eV<=0.
        || !isfinite(q0) || q0==0.
        || !CavityModel_scalar_finite(voltage)
        || !CavityModel_scalar_finite(phase)
        || !CavityModel_scalar_finite(counter_phase)
        || !CavityModel_scalar_finite(wave_number)
        || !isfinite(beta) || beta<=0. || beta>1.
        || isnan(mixture) || mixture<0.)
        return CAVITY_MODEL_ARGUMENT;
    // The eta endpoints are distinct charts. Even at zero RF voltage a
    // seeded endpoint would have nonanalytic mixed voltage/eta derivatives.
    if ((mixture==0. && !xt_float_or_tpsa_is_zero(eta))
        || (isinf(mixture) && !CavityModel_scalar_constant(eta)))
        return CAVITY_MODEL_ARGUMENT;
    const double c=299792458.;
    const double pi=3.141592653589793238462643383279502884;
    source->k=1.;
    if (f>0.) source->k=2.*pi*frequency/c;
    source->omega = c*source->k;
    source->length=length;
    source->half_span=source->k*length*.5;
    source->carrier=wave_number/source->k;
    source->counter_phase=counter_phase;
    source->clock_origin=phase-copysign(pi*.5,q0);
    source->voltage_norm=voltage/source_mass_eV;
    source->phase=phase;
    source->reference_charge=q0;
    source->reference_beta=1.;
    source->rf_profile=0;source->profile_sigma=0.;source->profile_center=0.;
    source->magnetic_profile=0;source->magnetic_sigma=0.;
    source->freeze_energy=0; source->freeze_zeta=0;
    source->freeze_x=0; source->freeze_px=0;
    source->freeze_y=0; source->freeze_py=0;
    source->dc_gradient=0.;
    source->knl_tpsa=NULL;source->ksl_tpsa=NULL;
    if (xt_float_or_tpsa_is_zero(voltage)) {
        source->forward=0.; source->backward=0.;
        return CAVITY_MODEL_OK;
    }
    if (f<=0.) return CAVITY_MODEL_ARGUMENT;
    xt_float_or_tpsa forward_weight=0., backward_weight=1.;
    if (isfinite(mixture)) {
        xt_float_or_tpsa ratio=0.;
        if (!xt_float_or_tpsa_is_zero(eta)) ratio=sqrt(eta);
        forward_weight=1./(1.+ratio);
        backward_weight=ratio/(1.+ratio);
    }
    xt_float_or_tpsa slow=0.,fast=0.;
    if(!xt_float_or_tpsa_is_zero(forward_weight))
        slow=forward_weight*CavityModel_p16_integral(
            source->half_span,1./calibration_beta-source->carrier);
    if(!xt_float_or_tpsa_is_zero(backward_weight))
        fast=backward_weight*CavityModel_p16_integral(
            source->half_span,1./calibration_beta+source->carrier);
    xt_float_or_tpsa real=slow+fast*cos(counter_phase);
    xt_float_or_tpsa imag=fast*sin(counter_phase);
    xt_float_or_tpsa magnitude_square=real*real+imag*imag;
    const double scale=fabs(xt_float_or_tpsa_const_part(slow))
                       +fabs(xt_float_or_tpsa_const_part(fast));
    const double threshold=64.*2.2204460492503131e-16*scale;
    if (!CavityModel_scalar_finite(magnitude_square)
        || xt_float_or_tpsa_const_part(magnitude_square)<=threshold*threshold)
        return CAVITY_MODEL_ARGUMENT;
    xt_float_or_tpsa magnitude=sqrt(magnitude_square);
    xt_float_or_tpsa amplitude=voltage/(source_mass_eV*magnitude);
    source->forward=amplitude*forward_weight;
    source->backward=amplitude*backward_weight;
    // Arg[exp(i H/beta)*(real+i imag)] is evaluated with the same branch
    // as the prepared source, retaining every calibration derivative.
    xt_float_or_tpsa angle=source->half_span/calibration_beta;
    xt_float_or_tpsa phasor_real=real*cos(angle)-imag*sin(angle);
    xt_float_or_tpsa phasor_imag=real*sin(angle)+imag*cos(angle);
    source->clock_origin -= atan2(phasor_imag,phasor_real);
    return CAVITY_MODEL_OK;
}

// Exact polynomial P16 envelope derivatives, evaluated without subtracting
// expanded degree-32 polynomials near the faces. They vanish outside support.
GPUFUN
void CavityModel_envelope(xt_float_or_tpsa_arg z,
        xt_float_or_tpsa_arg H, xt_float_or_tpsa derivatives[5]) {
    xt_float_or_tpsa u=z/H;
    if (fabs(xt_float_or_tpsa_const_part(u))>=1.) {
        for (int i=0;i<5;++i) derivatives[i]=0.;
        return;
    }
    xt_float_or_tpsa u2=u*u, v=1.-u2;
    xt_float_or_tpsa v2=v*v, v4=v2*v2, v8=v4*v4;
    xt_float_or_tpsa v12=v8*v4, v13=v12*v, v14=v12*v2;
    xt_float_or_tpsa v15=v14*v;
    xt_float_or_tpsa inverse=1./H;
    derivatives[0]=v8*v8;
    derivatives[1]=-32.*u*v15*inverse;
    derivatives[2]=32.*(31.*u2-1.)*v14*inverse*inverse;
    derivatives[3]=960.*u*(3.-31.*u2)*v13*inverse*inverse*inverse;
    derivatives[4]=960.*(3.-174.*u2+899.*u2*u2)*v12
                    *inverse*inverse*inverse*inverse;
}

// Built-in source profiles. Gaussian tails are NOT clipped at observation
// boundaries; uniform/Gaussian endpoints generally have nonzero potentials.
GPUFUN
void CavityModel_source_envelope(xt_float_or_tpsa_arg z,
        const CavityModelSource *source,xt_float_or_tpsa p[5]) {
    if (source->rf_profile==0) {
        CavityModel_envelope(z,source->half_span,p);
    } else if (source->rf_profile==1) {
        p[0]=1.;for(int i=1;i<5;++i)p[i]=0.;
    } else if (source->rf_profile==2
            && xt_float_or_tpsa_const_part(source->profile_sigma)>0.) {
        const xt_float_or_tpsa inv=1./source->profile_sigma;
        const xt_float_or_tpsa u=(z-source->profile_center)*inv,u2=u*u;
        const xt_float_or_tpsa g=exp(-.5*u2),i2=inv*inv;
        p[0]=g;p[1]=-u*g*inv;p[2]=(u2-1.)*g*i2;
        p[3]=u*(3.-u2)*g*i2*inv;
        p[4]=(u2*u2-6.*u2+3.)*g*i2*i2;
    } else {
        for(int i=0;i<5;++i)p[i]=NAN;
    }
}

GPUFUN
void CavityModel_profile(xt_float_or_tpsa_arg z,
        const CavityModelSource *source, xt_float_or_tpsa real[5],
        xt_float_or_tpsa imag[5]) {
    xt_float_or_tpsa envelope[5];
    CavityModel_source_envelope(z,source,envelope);
    const int binomial[5][5]={{1,0,0,0,0},{1,1,0,0,0},{1,2,1,0,0},
                              {1,3,3,1,0},{1,4,6,4,1}};
    xt_float_or_tpsa phases[2]={source->carrier*z,
                               -source->carrier*z-source->counter_phase};
    xt_float_or_tpsa strengths[2]={source->forward,source->backward};
    for (int j=0;j<5;++j) { real[j]=0.;imag[j]=0.; }
    for (int wave=0;wave<2;++wave) {
        xt_float_or_tpsa nu=source->carrier;
        if (wave) nu=-nu;
        xt_float_or_tpsa cosine=cos(phases[wave]), sine=sin(phases[wave]);
        xt_float_or_tpsa power[5]={1.,nu,nu*nu,nu*nu*nu,nu*nu*nu*nu};
        for (int j=0;j<5;++j)
            for (int ell=0;ell<=j;++ell) {
                const int degree=j-ell, quadrant=degree%4;
                xt_float_or_tpsa coefficient=strengths[wave]
                    *binomial[j][ell]*envelope[ell]*power[degree];
                if (quadrant==0) {real[j]+=coefficient*cosine;imag[j]+=coefficient*sine;}
                if (quadrant==1) {real[j]-=coefficient*sine;imag[j]+=coefficient*cosine;}
                if (quadrant==2) {real[j]-=coefficient*cosine;imag[j]-=coefficient*sine;}
                if (quadrant==3) {real[j]+=coefficient*sine;imag[j]-=coefficient*cosine;}
            }
    }
}

// Retained M2 temporal-gauge RF potential and its exact first derivatives.
// This is the compact form of native_rf_source()['potentials']; no force
// fit or independent radial-focusing coefficient is introduced.
GPUFUN
void CavityModel_rf_potential(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z, const CavityModelSource *source,
        int model, CavityModelPotential *potential) {
    xt_float_or_tpsa real[5],imag[5],S[5],C[5];
    CavityModel_profile(z,source,real,imag);
    xt_float_or_tpsa sine=sin(q[2]),cosine=cos(q[2]);
    for (int j=0;j<5;++j) {
        S[j]=real[j]*sine-imag[j]*cosine;
        C[j]=real[j]*cosine+imag[j]*sine;
    }
    for (int i=0;i<4;++i) {
        potential->a[i]=0.;
        for (int j=0;j<3;++j) potential->d[i][j]=0.;
    }
    // Frozen voltage actions have their own body chart and collapsed RF
    // faces. A smooth-P16 interior gauge must never be attached to them.
    if (model==CAVITY_MODEL_FROZEN) return;
    if (model==CAVITY_MODEL_LONGITUDINAL) {
        potential->a[2]=-S[0];
        potential->d[2][2]=-C[0];
        return;
    }
    xt_float_or_tpsa X=q[0],Y=q[1],r2=X*X+Y*Y;
    xt_float_or_tpsa radial=S[1]*.5-r2*(S[1]+S[3])/16.;
    xt_float_or_tpsa radial_T=C[1]*.5-r2*(C[1]+C[3])/16.;
    xt_float_or_tpsa radial_r2=-(S[1]+S[3])/16.;
    xt_float_or_tpsa axial_r2=(S[0]+S[2])/4.;
    xt_float_or_tpsa axial_r4=-(S[0]+2.*S[2]+S[4])/64.;
    potential->a[0]=X*radial; potential->a[1]=Y*radial;
    potential->a[2]=-S[0]+r2*axial_r2+r2*r2*axial_r4;
    potential->d[0][0]=radial+2.*X*X*radial_r2;
    potential->d[0][1]=2.*X*Y*radial_r2;
    potential->d[0][2]=X*radial_T;
    potential->d[1][0]=potential->d[0][1];
    potential->d[1][1]=radial+2.*Y*Y*radial_r2;
    potential->d[1][2]=Y*radial_T;
    potential->d[2][0]=2.*X*(axial_r2+2.*r2*axial_r4);
    potential->d[2][1]=2.*Y*(axial_r2+2.*r2*axial_r4);
    potential->d[2][2]=-C[0]+r2*(C[0]+C[2])/4.
                           -r2*r2*(C[0]+2.*C[2]+C[4])/64.;
}

// The finite-parent integral preserves ordinary integrated knl/ksl.
// ks remains the signed peak solenoid coefficient. Gaussian tails and the
// original donor's asymptotic pure gauge are never clipped at the planes.
GPUFUN
xt_float_or_tpsa CavityModel_erf(xt_float_or_tpsa_arg value) {
#ifdef XTRACK_TPSA_TRACK
    return mad::erf(value);
#else
    return erf(value);
#endif
}
GPUFUN
xt_float_or_tpsa CavityModel_magnetic_integral(const CavityModelSource *source) {
    if(source->magnetic_profile==0)return source->length;
    const xt_float_or_tpsa width=source->magnetic_sigma;
    return width/source->k*sqrt(2.*PI)
        *CavityModel_erf(source->half_span/(sqrt(2.)*width));
}
GPUFUN
int CavityModel_static_validate(const CavityModelSource *source) {
    if(source->magnetic_profile==0)return CAVITY_MODEL_OK;
    if(source->magnetic_profile!=1||!CavityModel_scalar_finite(source->magnetic_sigma)
        ||xt_float_or_tpsa_const_part(source->magnetic_sigma)<=0.)return CAVITY_MODEL_ARGUMENT;
    for(int64_t i=0;i<=source->order;++i)
        if(i!=1&&(!xt_float_or_tpsa_is_zero(CavityModel_static_coefficient(source,i,0))
            ||!xt_float_or_tpsa_is_zero(CavityModel_static_coefficient(source,i,1))))
            return CAVITY_MODEL_ARGUMENT;
    const xt_float_or_tpsa integral=CavityModel_magnetic_integral(source);
    return CavityModel_scalar_finite(integral)&&xt_float_or_tpsa_const_part(integral)>0.
        ?CAVITY_MODEL_OK:CAVITY_MODEL_ARGUMENT;
}

// Actual GaussianFringeReference original potential through static radial
// order one. The compact quadratic map omits the cubic terms; mechanical
// readback and the FULL taper predictor keep this SAME retained donor.
GPUFUN
void CavityModel_add_gaussian_static(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z,const CavityModelSource *source,CavityModelPotential *potential) {
    const xt_float_or_tpsa width=source->magnetic_sigma,scaled=z/width;
    const xt_float_or_tpsa profile=exp(-scaled*scaled/2.);
    const xt_float_or_tpsa B=source->magnetic_scale*source->ks/source->k*profile;
    const xt_float_or_tpsa Bzz=(scaled*scaled-1.)*B/(width*width);
    const xt_float_or_tpsa factor=source->magnetic_scale/
        (source->k*source->k*CavityModel_magnetic_integral(source));
    xt_float_or_tpsa gn=0.,gs=0.;
    if(source->order>=1){gn=factor*CavityModel_static_coefficient(source,1,0);
        gs=-factor*CavityModel_static_coefficient(source,1,1);}
    const xt_float_or_tpsa primitive=width*sqrt(PI/2.)
        *CavityModel_erf(z/(sqrt(2.)*width))/2.;
    const xt_float_or_tpsa sn=gn*primitive,ss=gs*primitive;
    const xt_float_or_tpsa gnz=-scaled*gn*profile/width,gsz=-scaled*gs*profile/width;
    const xt_float_or_tpsa x=q[0],y=q[1],r2=x*x+y*y;
    const xt_float_or_tpsa H=2.*gnz*x*y+gsz*(y*y-x*x);
    const xt_float_or_tpsa Hx=2.*gnz*y-2.*gsz*x,Hy=2.*gnz*x+2.*gsz*y;
    const xt_float_or_tpsa Fx=-(Hx*r2+2.*x*H)/24.,Fy=-(Hy*r2+2.*y*H)/24.;
    const xt_float_or_tpsa Fxx=-(-2.*gsz*r2+4.*x*Hx+2.*H)/24.;
    const xt_float_or_tpsa Fyy=-( 2.*gsz*r2+4.*y*Hy+2.*H)/24.;
    const xt_float_or_tpsa Fxy=-(2.*gnz*r2+2.*x*Hy+2.*y*Hx)/24.;
    potential->a[0]+=-B*y/2.+Bzz*y*r2/16.+2.*(sn*x+ss*y)+Fy;
    potential->a[1]+= B*x/2.-Bzz*x*r2/16.+2.*(ss*x-sn*y)-Fx;
    potential->d[0][0]+=Bzz*x*y/8.+2.*sn+Fxy;
    potential->d[0][1]+=-B/2.+Bzz*(x*x+3.*y*y)/16.+2.*ss+Fyy;
    potential->d[1][0]+= B/2.-Bzz*(3.*x*x+y*y)/16.+2.*ss-Fxx;
    potential->d[1][1]+=-Bzz*x*y/8.-2.*sn-Fxy;
}

// Straight static multipoles and uniform solenoid in the same potential.
// knl[1]>0 gives focusing in x for the reference charge. Source coefficients
// remain uncharged; actual signed Q enters the Hamiltonian exactly once.
GPUFUN
void CavityModel_add_static(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z,const CavityModelSource *source, CavityModelPotential *potential) {
    // Explicit taper-compensation analysis only: A_z=-g*T gives constant
    // longitudinal voltage while retaining the same static transport.
    // Ordinary harmonic preparation initializes g=0; it never invents a DC
    // continuation of the calibrated harmonic field at nonzero V, f=0.
    potential->a[2]-=source->dc_gradient*q[2];
    potential->d[2][2]-=source->dc_gradient;
    if(source->magnetic_profile==1) {
        CavityModel_add_gaussian_static(q,z,source,potential);return;
    }
    xt_float_or_tpsa x=q[0]/source->k,y=q[1]/source->k;
    xt_float_or_tpsa solenoid=source->magnetic_scale*source->ks*.5;
    potential->a[0]-=solenoid*y; potential->a[1]+=solenoid*x;
    potential->d[0][1]-=solenoid/source->k;
    potential->d[1][0]+=solenoid/source->k;
    xt_float_or_tpsa real=1.,imag=0.;
    xt_float_or_tpsa factor=source->magnetic_scale/source->length;
    for (int64_t n=0;n<=source->order;++n) {
        const xt_float_or_tpsa normal=CavityModel_static_coefficient(source,n,0);
        const xt_float_or_tpsa skew=CavityModel_static_coefficient(source,n,1);
        potential->d[2][0]-=factor*(normal*real-skew*imag)/source->k;
        potential->d[2][1]+=factor*(normal*imag+skew*real)/source->k;
        xt_float_or_tpsa next_real=(real*x-imag*y)/(n+1.);
        imag=(real*y+imag*x)/(n+1.); real=next_real;
        potential->a[2]-=factor*(normal*real-skew*imag);
    }
}

GPUFUN
int CavityModel_potential(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z, const CavityModelSource *source,
        int model, xt_float_or_tpsa output[4]) {
    if (!CavityModel_vector_finite(q,6)) return CAVITY_MODEL_NONFINITE;
    int static_status=CavityModel_static_validate(source);
    if(static_status!=CAVITY_MODEL_OK)return static_status;
    CavityModelPotential potential;
    CavityModel_rf_potential(q,z,source,model,&potential);
    CavityModel_add_static(q,z,source,&potential);
    if (!CavityModel_vector_finite(potential.a,4)) return CAVITY_MODEL_NONFINITE;
    for (int i=0;i<4;++i) output[i]=potential.a[i];
    return CAVITY_MODEL_OK;
}

// The longitudinal generator is projected, but its physical mechanical lift
// still uses the complete retained RF potential. This is the same distinction
// as VoltageTransport.mechanical(), not an auxiliary momentum readback.
GPUFUN
int CavityModel_physical_potential(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z, const CavityModelSource *source,
        int model, xt_float_or_tpsa output[4]) {
    const int physical_model=model==CAVITY_MODEL_LONGITUDINAL
        ? CAVITY_MODEL_FULL : model;
    return CavityModel_potential(q,z,source,physical_model,output);
}

// Full: H=-Q Az-sqrt(epsilon^2-1-|P-Q A_perp|^2).
// H2:   H=-Q Az-p0+|P-Q A_perp|^2/(2 p0), p0=sqrt(epsilon^2-1).
// Longitudinal changes ONLY the RF potential above and retains the full root.
// In particular, H2 has its own paraxial vacuum drift, not the full-root drift.
GPUFUN
int CavityModel_flow(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z, xt_float_or_tpsa_arg Q,
        const CavityModelSource *source, int model,
        xt_float_or_tpsa flow[6]) {
    if (model<CAVITY_MODEL_FULL || model>CAVITY_MODEL_LONGITUDINAL)
        return CAVITY_MODEL_ARGUMENT;
    if (source->freeze_zeta && !(source->reference_beta>0.
                                && source->reference_beta<=1.))
        return CAVITY_MODEL_ARGUMENT;
    if (!CavityModel_vector_finite(q,6)) return CAVITY_MODEL_NONFINITE;
    int static_status=CavityModel_static_validate(source);
    if(static_status!=CAVITY_MODEL_OK)return static_status;
    CavityModelPotential potential;
    CavityModel_rf_potential(q,z,source,model,&potential);
    CavityModel_add_static(q,z,source,&potential);
    if (!CavityModel_vector_finite(potential.a,4)) return CAVITY_MODEL_NONFINITE;
    if (model==CAVITY_MODEL_LONGITUDINAL) {
        xt_float_or_tpsa physical[4];
        int status=CavityModel_physical_potential(q,z,source,model,physical);
        if (status!=CAVITY_MODEL_OK) return status;
        xt_float_or_tpsa physical_energy=q[5]-Q*physical[3];
        xt_float_or_tpsa physical_px=q[3]-Q*physical[0];
        xt_float_or_tpsa physical_py=q[4]-Q*physical[1];
        xt_float_or_tpsa physical_square=(physical_energy-1.)*(physical_energy+1.)
            -physical_px*physical_px-physical_py*physical_py;
        if (!CavityModel_scalar_finite(physical_square))
            return CAVITY_MODEL_NONFINITE;
        if (xt_float_or_tpsa_const_part(physical_energy)<=1.
            || xt_float_or_tpsa_const_part(physical_square)<=0.)
            return CAVITY_MODEL_DOMAIN;
    }
    xt_float_or_tpsa energy=q[5]-Q*potential.a[3];
    xt_float_or_tpsa px=q[3]-Q*potential.a[0];
    xt_float_or_tpsa py=q[4]-Q*potential.a[1];
    xt_float_or_tpsa p02=(energy-1.)*(energy+1.);
    xt_float_or_tpsa transverse=px*px+py*py;
    xt_float_or_tpsa square=p02-transverse;
    if (xt_float_or_tpsa_const_part(energy)<=1.
        || xt_float_or_tpsa_const_part(square)<=0.) return CAVITY_MODEL_DOMAIN;
    xt_float_or_tpsa denominator;
    if (model==CAVITY_MODEL_PARAXIAL) {
        denominator=sqrt(p02);
        flow[2]=energy/denominator*(1.+transverse/(2.*p02));
    } else {
        denominator=sqrt(square); flow[2]=energy/denominator;
    }
    flow[0]=px/denominator; flow[1]=py/denominator;
    for (int coordinate=0;coordinate<3;++coordinate) {
        xt_float_or_tpsa derivative=Q*(potential.d[2][coordinate]
             +flow[0]*potential.d[0][coordinate]
             +flow[1]*potential.d[1][coordinate]
             -flow[2]*potential.d[3][coordinate]);
        flow[coordinate+3]=derivative;
        if (coordinate==2) flow[coordinate+3]=-derivative;
    }
    // Named constrained analysis maps used by normal Line workflows.
    // The constraints apply inside every implicit stage, not just setters.
    if (source->freeze_energy) flow[5]=0.;
    if (source->freeze_zeta) flow[2]=1./source->reference_beta;
    if (source->freeze_x) flow[0]=0.;
    if (source->freeze_px) flow[3]=0.;
    if (source->freeze_y) flow[1]=0.;
    if (source->freeze_py) flow[4]=0.;
    return CavityModel_vector_finite(flow,6) ? CAVITY_MODEL_OK : CAVITY_MODEL_NONFINITE;
}

// Two-stage Gauss--Legendre: the same equations and constants as the existing
// Python owner. Signed h reverses the deterministic word. All outputs are
// transactional. The defining residual is checked after reassignment, and
// TPSA convergence includes its represented coefficients, not only the orbit.
// Stage unknowns are increments about q: this exact translation avoids
// subtracting a large base energy/clock in the defining residual. The physical
// primal increment retains the same absolute tolerance; reconstructed stages
// and the final rounded output still have their floating-point ULP floor.
GPUFUN
int CavityModel_gauss_stages(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z, xt_float_or_tpsa_arg h,
        xt_float_or_tpsa_arg Q, const CavityModelSource *source, int model,
        int64_t max_iter, double tolerance, xt_float_or_tpsa output[6],
        xt_float_or_tpsa *accepted_stages) {
    if (max_iter<=0 || !isfinite(tolerance) || tolerance<=0.)
        return CAVITY_MODEL_ARGUMENT;
    const double C[2]={.21132486540518713,.7886751345948129};
    const double A[2][2]={{.25,-.038675134594812866},{.5386751345948129,.25}};
    xt_float_or_tpsa initial[6],stages[2][6],values[2][6],proposed[2][6];
    int status=CavityModel_flow(q,z,Q,source,model,initial);
    if (status!=CAVITY_MODEL_OK) return status;
    for (int i=0;i<2;++i)
        for (int j=0;j<6;++j) stages[i][j]=C[i]*h*initial[j];
    for (int64_t iteration=0;iteration<max_iter;++iteration) {
        for (int i=0;i<2;++i) {
            for (int j=0;j<6;++j) initial[j]=q[j]+stages[i][j];
            status=CavityModel_flow(initial,z+C[i]*h,Q,source,model,values[i]);
            if (status!=CAVITY_MODEL_OK) return status;
        }
        double change=0.;
        for (int i=0;i<2;++i)
            for (int j=0;j<6;++j) {
                proposed[i][j]=h*(A[i][0]*values[0][j]+A[i][1]*values[1][j]);
                change=fmax(change,CavityModel_scalar_residual(
                    proposed[i][j]-stages[i][j],q[j]+proposed[i][j],1));
                stages[i][j]=proposed[i][j];
            }
        if (change>=tolerance) continue;
        for (int i=0;i<2;++i) {
            for (int j=0;j<6;++j) initial[j]=q[j]+stages[i][j];
            status=CavityModel_flow(initial,z+C[i]*h,Q,source,model,values[i]);
            if (status!=CAVITY_MODEL_OK) return status;
        }
        double residual=0.;
        for (int i=0;i<2;++i)
            for (int j=0;j<6;++j)
                residual=fmax(residual,CavityModel_scalar_residual(stages[i][j]
                    -h*(A[i][0]*values[0][j]+A[i][1]*values[1][j]),q[j]+stages[i][j],1));
        if (residual>=tolerance) continue;
        xt_float_or_tpsa result[6],check[6];
        for (int j=0;j<6;++j) result[j]=q[j]+h*(values[0][j]+values[1][j])*.5;
        status=CavityModel_flow(result,z+h,Q,source,model,check);
        if (status!=CAVITY_MODEL_OK) return status;
        if(accepted_stages)
            for(int i=0;i<2;++i)for(int j=0;j<6;++j)
                accepted_stages[6*i+j]=q[j]+stages[i][j];
        for (int j=0;j<6;++j) output[j]=result[j];
        return CAVITY_MODEL_OK;
    }
#ifdef CAVITY_MODEL_DEBUG
    for (int i=0;i<2;++i) {
        for (int j=0;j<6;++j) initial[j]=q[j]+stages[i][j];
        CavityModel_flow(initial,z+C[i]*h,Q,source,model,values[i]);
    }
    CavityModel_debug_stages("physical-gauss",q,stages[0],stages[1],
                             values[0],values[1],h,6);
#endif
    return CAVITY_MODEL_ITERATIONS;
}

GPUFUN
int CavityModel_gauss(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z,xt_float_or_tpsa_arg h,xt_float_or_tpsa_arg Q,
        const CavityModelSource *source,int model,int64_t max_iter,double tolerance,
        xt_float_or_tpsa output[6]) {
    return CavityModel_gauss_stages(q,z,h,Q,source,model,max_iter,tolerance,output,NULL);
}

// Entire sinc shared by uniform calibration and frozen static factors.
GPUFUN
xt_float_or_tpsa CavityModel_static_sinc(xt_float_or_tpsa_arg angle) {
#ifdef XTRACK_TPSA_TRACK
    return mad::sinc(angle);
#else
    if (angle==0.) return 1.;
    return sin(angle)/angle;
#endif
}


// Explicit built-in RF profiles and their finite-parent crest calibration.
GPUFUN
xt_float_or_tpsa CavityModel_erfc(xt_float_or_tpsa_arg value) {
#ifdef XTRACK_TPSA_TRACK
    return mad::erfc(value);
#else
    return erfc(value);
#endif
}

// Finite-parent Gaussian Fourier integral. Identically zero detuning has
// its exact erf/erfc formula. A zero primal with a seeded detuning must still
// take the general rule, preserving phase/carrier/calibration derivatives.
// Otherwise a depth-first composite 16-point Gauss rule uses a Cauchy
// truncation bound on each complete panel. l below is FULL panel length:
// E <= l*(l/R)^32/[33*binom(32,16)^2]
//      *exp(-D^2/(2*sigma^2)+R^2/(2*sigma^2)+abs(detuning)*R),
// D=max(0,distance(panel,centre)-R). The Gaussian and phase are entire;
// Cauchy's derivative bound and the Gauss remainder give this expression.
// Bounds below are evaluated in double, not outward interval arithmetic:
// they control quadrature truncation, not binary/TPSA derivative admission.
GPUFUN
int CavityModel_profile_integral(xt_float_or_tpsa_arg H,
        xt_float_or_tpsa_arg detuning,const CavityModelSource *source,
        double absolute_tolerance,xt_float_or_tpsa out[2],double *error) {
    out[0]=0.;out[1]=0.;*error=0.;
    if(source->rf_profile==1) {
        out[0]=2.*H*CavityModel_static_sinc(detuning*H);
        return CAVITY_MODEL_OK;
    }
    if(source->rf_profile==0) {
        out[0]=CavityModel_p16_integral(H,detuning);
        return CavityModel_scalar_finite(out[0])?CAVITY_MODEL_OK:CAVITY_MODEL_NONFINITE;
    }
    const xt_float_or_tpsa sigma=source->profile_sigma;
    const xt_float_or_tpsa centre=source->profile_center;
    const double h=xt_float_or_tpsa_const_part(H);
    const double width=xt_float_or_tpsa_const_part(sigma);
    const double offset=xt_float_or_tpsa_const_part(centre);
    const double frequency=fabs(xt_float_or_tpsa_const_part(detuning));
    if(!(isfinite(h)&&h>0.&&isfinite(width)&&width>0.
        &&isfinite(offset)&&isfinite(frequency)
        &&isfinite(absolute_tolerance)&&absolute_tolerance>0.))
        return CAVITY_MODEL_ARGUMENT;
    if(xt_float_or_tpsa_is_zero(detuning)) {
        const xt_float_or_tpsa left=(-H-centre)/(sqrt(2.)*sigma);
        const xt_float_or_tpsa right=(H-centre)/(sqrt(2.)*sigma);
        xt_float_or_tpsa first,second;
        int separated=0;
        // erfc avoids subtracting two rounded ones in a one-sided tail.
        if(xt_float_or_tpsa_const_part(left)>=0.) {
            first=CavityModel_erfc(left);second=CavityModel_erfc(right);
        } else if(xt_float_or_tpsa_const_part(right)<=0.) {
            first=CavityModel_erfc(-right);second=CavityModel_erfc(-left);
        } else {
            first=CavityModel_erf(right);second=CavityModel_erf(left);
            separated=1; // opposite-sign primals: no cancellation in subtraction
        }
        out[0]=first-second;
        // Close same-side endpoints can cancel even in erfc. Fall back to
        // the bounded integral instead of magnifying subtraction roundoff.
        if(separated||fabs(xt_float_or_tpsa_const_part(out[0]))>=.125*
            (fabs(xt_float_or_tpsa_const_part(first))+fabs(xt_float_or_tpsa_const_part(second)))) {
            out[0]*=sigma*sqrt(PI/2.);
            return CavityModel_scalar_finite(out[0])?CAVITY_MODEL_OK:CAVITY_MODEL_NONFINITE;
        }
        out[0]=0.;
    }
    const double nodes[8]={
        .095012509837637440185,.28160355077925891323,
        .45801677765722738634,.61787624440264374845,
        .75540440835500303390,.86563120238783174388,
        .94457502307323257608,.98940093499164993260};
    const double weights[8]={
        .18945061045506849629,.18260341504492358887,
        .16915651939500253819,.14959598881657673208,
        .12462897125553387205,.095158511682492784810,
        .062253523938647892863,.027152459411754094852};
    // This central radius minimizes -32log(R)+R²/(2sigma²)+abs(d)*R.
    // It remains a valid Cauchy radius for panels away from the centre.
    const double scale=hypot(frequency,sqrt(128.)/width);
    const double radius=(64./scale)/(1.+frequency/scale);
    if(!(isfinite(radius)&&radius>0.))return CAVITY_MODEL_NONFINITE;
    const double log_coefficient=log(33.)+2.*log(601080390.);
    xt_float_or_tpsa correction[2]={0.,0.};
    uint64_t index=0;int depth=0;
    // Dyadic traversal needs no recursion or per-thread interval stack.
    for(int work=0;work<65536;++work) {
        const double fraction=ldexp(1.,1-depth);
        const double left=-1.+index*fraction,right=left+fraction;
        const double length=h*fraction;
        const double distance=fmax(0.,fmax(h*left-offset,offset-h*right));
        const double separated=fmax(0.,distance-radius)/width;
        const double ratio=radius/width;
        double log_error=log(length)+32.*log(length/radius)-log_coefficient
            -.5*separated*separated+.5*ratio*ratio+frequency*radius;
        // Positivity of Gauss weights also gives |I-Q|<=2*l*max|f|.
        const double scaled_distance=distance/width;
        log_error=fmin(log_error,log(2.*length)-.5*scaled_distance*scaled_distance);
        const double budget=absolute_tolerance*fraction*.5;
        if(log_error>log(budget)) {
            if(depth>=48)return CAVITY_MODEL_ITERATIONS;
            index*=2;++depth;continue;
        }
        *error+=exp(log_error);
        for(int j=0;j<8;++j)for(int sign=-1;sign<=1;sign+=2) {
            const xt_float_or_tpsa z=H*((left+right)*.5+sign*fraction*.5*nodes[j]);
            const xt_float_or_tpsa u=(z-centre)/sigma;
            const xt_float_or_tpsa weighted=H*(fraction*.5*weights[j])*exp(-.5*u*u);
            const xt_float_or_tpsa phase=detuning*z;
            const xt_float_or_tpsa terms[2]={weighted*cos(phase),weighted*sin(phase)};
            for(int k=0;k<2;++k) {
                const xt_float_or_tpsa adjusted=terms[k]-correction[k];
                const xt_float_or_tpsa next=out[k]+adjusted;
                correction[k]=(next-out[k])-adjusted;out[k]=next;
            }
        }
        ++index;
        while(depth>0&&(index&1)==0){index/=2;--depth;}
        if(depth==0&&index==1)
            return CavityModel_vector_finite(out,2)?CAVITY_MODEL_OK:CAVITY_MODEL_NONFINITE;
    }
    return CAVITY_MODEL_ITERATIONS;
}

// Same fixed-beta integrated electric-crest convention as the built-in P16.
// Gaussian quadrature budgets are refined against the combined calibration
// phasor, including cancellation. An exhausted rule refuses; it never returns
// the last unresolved calibration. Physical/source and binary qualification
// remain separate from the quadrature-truncation estimate.
GPUFUN
int CavityModel_prepare_profile(xt_float_or_tpsa_arg frequency,
        xt_float_or_tpsa_arg voltage,xt_float_or_tpsa_arg length,
        xt_float_or_tpsa_arg phase,xt_float_or_tpsa_arg eta,
        xt_float_or_tpsa_arg counter_phase,xt_float_or_tpsa_arg calibration_beta,
        xt_float_or_tpsa_arg wave_number,double mass0,double q0,int profile,
        xt_float_or_tpsa_arg sigma_m,xt_float_or_tpsa_arg center_m,
        CavityModelSource *source) {
    if(profile==0) return CavityModel_prepare(frequency,voltage,length,phase,
        eta,counter_phase,calibration_beta,wave_number,mass0,q0,source);
    if(profile<0||profile>2||!CavityModel_scalar_finite(sigma_m)
        ||!CavityModel_scalar_finite(center_m)
        ||(profile==2&&xt_float_or_tpsa_const_part(sigma_m)<=0.))
        return CAVITY_MODEL_ARGUMENT;
    int status=CavityModel_prepare(frequency,0.,length,phase,eta,counter_phase,
        calibration_beta,wave_number,mass0,q0,source);
    if(status!=CAVITY_MODEL_OK)return status;
    source->rf_profile=profile;
    source->profile_sigma=source->k*sigma_m;
    source->profile_center=source->k*center_m;
    source->voltage_norm=voltage/mass0;
    if(xt_float_or_tpsa_is_zero(voltage))return CAVITY_MODEL_OK;
    if(xt_float_or_tpsa_const_part(frequency)<=0.)return CAVITY_MODEL_ARGUMENT;
    xt_float_or_tpsa forward=0.,backward=1.;
    if(isfinite(xt_float_or_tpsa_const_part(eta))) {
        xt_float_or_tpsa ratio=0.;if(!xt_float_or_tpsa_is_zero(eta))ratio=sqrt(eta);
        forward=1./(1.+ratio);backward=ratio/(1.+ratio);
    }
    xt_float_or_tpsa slow[2]={0.,0.},fast[2]={0.,0.};
    xt_float_or_tpsa real=0.,imag=0.,square=0.;
    const double eps=2.2204460492503131e-16;
    double tolerance=8.*eps*fmin(2.*xt_float_or_tpsa_const_part(source->half_span),
        sqrt(2.*PI)*xt_float_or_tpsa_const_part(source->profile_sigma));
    if(profile==1)tolerance=8.*eps*2.*xt_float_or_tpsa_const_part(source->half_span);
    int resolved=0;
    for(int refinement=0;refinement<4;++refinement) {
        double slow_error=0.,fast_error=0.;
        if(!xt_float_or_tpsa_is_zero(forward)) {
            status=CavityModel_profile_integral(source->half_span,
                1./calibration_beta-source->carrier,source,tolerance,slow,&slow_error);
            if(status!=CAVITY_MODEL_OK)return status;
        }
        if(!xt_float_or_tpsa_is_zero(backward)) {
            status=CavityModel_profile_integral(source->half_span,
                1./calibration_beta+source->carrier,source,tolerance,fast,&fast_error);
            if(status!=CAVITY_MODEL_OK)return status;
        }
        real=forward*slow[0]+backward*(fast[0]*cos(counter_phase)-fast[1]*sin(counter_phase));
        imag=forward*slow[1]+backward*(fast[1]*cos(counter_phase)+fast[0]*sin(counter_phase));
        square=real*real+imag*imag;
        const double magnitude=sqrt(xt_float_or_tpsa_const_part(square));
        const double error=fabs(xt_float_or_tpsa_const_part(forward))*slow_error
            +fabs(xt_float_or_tpsa_const_part(backward))*fast_error;
        if(error<=8.*eps*fmax(0.,magnitude-error)) {resolved=1;break;}
        tolerance=8.*eps*fmax(magnitude,error);
    }
    if(!resolved)return CAVITY_MODEL_ITERATIONS;
    const double scale=fabs(xt_float_or_tpsa_const_part(forward))*hypot(xt_float_or_tpsa_const_part(slow[0]),xt_float_or_tpsa_const_part(slow[1]))
        +fabs(xt_float_or_tpsa_const_part(backward))*hypot(xt_float_or_tpsa_const_part(fast[0]),xt_float_or_tpsa_const_part(fast[1]));
    const double floor=64.*2.2204460492503131e-16*scale;
    if(!CavityModel_scalar_finite(square)||xt_float_or_tpsa_const_part(square)<=floor*floor)
        return CAVITY_MODEL_ARGUMENT;
    const xt_float_or_tpsa amplitude=voltage/(mass0*sqrt(square));
    source->forward=amplitude*forward;source->backward=amplitude*backward;
    const xt_float_or_tpsa angle=source->half_span/calibration_beta;
    const xt_float_or_tpsa rotated_re=real*cos(angle)-imag*sin(angle);
    const xt_float_or_tpsa rotated_im=real*sin(angle)+imag*cos(angle);
    source->clock_origin-=atan2(rotated_im,rotated_re);
    return (CavityModel_scalar_finite(source->forward)&&CavityModel_scalar_finite(source->backward)
        &&CavityModel_scalar_finite(source->clock_origin))?CAVITY_MODEL_OK:CAVITY_MODEL_NONFINITE;
}


#ifndef XTRACK_TPSA_TRACK
// Applied-effects local laws: full-root P16/M2 plus uniform SOL/QUAD overlap.
// Static overlap is a body law between in-field observation planes. It does
// not supply hard-edge spin rotations or radiation through field jumps.
// Fields are uncharged in source mass units. q=(X,Y,T,Px,Py,gamma).
// Derivative rows T,X,Y,Z; rho is NOT absorbed in this law.
typedef struct {
    double E[3],B[3],dE[4][3],dB[4][3];
} CavityModelEffectsField;
GPUFUN
void CavityModel_effects_cross(const double a[3],const double b[3],double c[3]) {
    c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];
    c[2]=a[0]*b[1]-a[1]*b[0];
}
GPUFUN
int CavityModel_effects_profile(double T,double z,const CavityModelSource *source,
                               double S[6],double C[6]) {
    const double H=xt_float_or_tpsa_const_part(source->half_span);
    if(source->rf_profile!=0 || !(H>0.) || !isfinite(H))return CAVITY_MODEL_ARGUMENT;
    double f[6]={0.,0.,0.,0.,0.,0.};
    if(fabs(z)<H) {
        const double u=z/H,a=1.-u,b=1.+u;
        // Leibniz derivatives of (1-u)^16(1+u)^16 through true D5.
        for(int n=0;n<=5;++n) {
            double choose=1.,fall_left=1.;
            for(int j=0;j<=n;++j) {
                double fall_right=1.;for(int v=0;v<n-j;++v)fall_right*=16-v;
                f[n]+=choose*((j%2)?-1.:1.)*fall_left*fall_right
                    *pow(a,16-j)*pow(b,16-n+j)/pow(H,n);
                choose*=((double)(n-j))/(j+1.);fall_left*=16-j;
            }
        }
    }
    const double nu=xt_float_or_tpsa_const_part(source->carrier);
    const double amp[2]={xt_float_or_tpsa_const_part(source->forward),
                         xt_float_or_tpsa_const_part(source->backward)};
    const double cp=xt_float_or_tpsa_const_part(source->counter_phase);
    for(int n=0;n<6;++n) {S[n]=0.;C[n]=0.;
        for(int wave=0;wave<2;++wave) {
            const double w=wave?nu:-nu,theta=T+w*z+(wave?cp:0.);
            double choose=1.;
            for(int j=0;j<=n;++j) {
                const int degree=n-j;
                const double factor=amp[wave]*choose*f[j]*pow(w,degree);
                const double angle=theta+degree*(PI/2.);
                S[n]+=factor*sin(angle);C[n]+=factor*cos(angle);
                choose*=((double)(n-j))/(j+1.);
            }
        }
        if(!isfinite(S[n])||!isfinite(C[n]))return CAVITY_MODEL_NONFINITE;
    }
    return CAVITY_MODEL_OK;
}
// Uniform longitudinal-gauge static overlap only. Fields remain uncharged;
// actual signed Q is applied by the Lorentz/LL/BMT laws, never here.
GPUFUN
int CavityModel_effects_static_components(const CavityModelSource *source,
                                         double *b,double *gn,double *gs) {
    if(source->magnetic_profile!=0)return CAVITY_MODEL_ARGUMENT;
    const double ks=source->ks;
    if(!isfinite(ks))return CAVITY_MODEL_NONFINITE;
    double normal=0.,skew=0.;int present=ks!=0.;
    for(int64_t n=0;n<=source->order;++n) {
        // Applied effects have no TPSA execution/derivative interpretation.
        if((source->knl_tpsa&&source->knl_tpsa[n])
            ||(source->ksl_tpsa&&source->ksl_tpsa[n]))return CAVITY_MODEL_ARGUMENT;
        const double kn=CavityModel_static_coefficient(source,n,0);
        const double sn=CavityModel_static_coefficient(source,n,1);
        if(!isfinite(kn)||!isfinite(sn))return CAVITY_MODEL_NONFINITE;
        if(n!=1&&(kn!=0.||sn!=0.))return CAVITY_MODEL_ARGUMENT;
        if(n==1){normal=kn;skew=sn;present|=kn!=0.||sn!=0.;}
    }
    *b=0.;*gn=0.;*gs=0.;
    // Raw RF-only point queries historically need no magnetic geometry.
    if(!present)return CAVITY_MODEL_OK;
    const double k=source->k,length=source->length,scale=source->magnetic_scale;
    if(!isfinite(k)||k<=0.||!isfinite(length)||length<=0.||!isfinite(scale))
        return CAVITY_MODEL_ARGUMENT;
    *b=scale*ks/k;*gn=scale*normal/(length*k*k);*gs=scale*skew/(length*k*k);
    return isfinite(*b)&&isfinite(*gn)&&isfinite(*gs)
        ?CAVITY_MODEL_OK:CAVITY_MODEL_NONFINITE;
}
GPUFUN
int CavityModel_effects_fields(const double q[6],double z,
        const CavityModelSource *source,CavityModelEffectsField *field) {
    double S[6],C[6];int status=CavityModel_effects_profile(q[2],z,source,S,C);
    if(status!=CAVITY_MODEL_OK)return status;
    const double x=q[0],y=q[1],r2=x*x+y*y,r4=r2*r2;
    const double K=C[1]/2.-r2*(C[1]+C[3])/16.;
    const double L=S[0]/2.-r2*(S[0]+S[2])/16.;
    field->E[0]=-x*K;field->E[1]=-y*K;
    field->E[2]=C[0]-r2*(C[0]+C[2])/4.
               +r4*(C[0]+2.*C[2]+C[4])/64.
               +xt_float_or_tpsa_const_part(source->dc_gradient);
    field->B[0]=y*L;field->B[1]=-x*L;field->B[2]=0.;
    for(int axis=0;axis<4;++axis) {
        const double dx=axis==1,dy=axis==2,dr=2.*(x*dx+y*dy);
        double ds[5]={0.},dc[5]={0.};
        for(int j=0;j<5;++j) {
            if(axis==0){ds[j]=C[j];dc[j]=-S[j];}
            if(axis==3){ds[j]=S[j+1];dc[j]=C[j+1];}
        }
        const double dK=dc[1]/2.-(dr*(C[1]+C[3])+r2*(dc[1]+dc[3]))/16.;
        const double dL=ds[0]/2.-(dr*(S[0]+S[2])+r2*(ds[0]+ds[2]))/16.;
        field->dE[axis][0]=-dx*K-x*dK;field->dE[axis][1]=-dy*K-y*dK;
        field->dE[axis][2]=dc[0]-(dr*(C[0]+C[2])+r2*(dc[0]+dc[2]))/4.
            +(2.*r2*dr*(C[0]+2.*C[2]+C[4])+r4*(dc[0]+2.*dc[2]+dc[4]))/64.;
        field->dB[axis][0]=dy*L+y*dL;field->dB[axis][1]=-dx*L-x*dL;
        field->dB[axis][2]=0.;
    }
    // Curl of the SAME uniform static potential used by full orbit tracking:
    // A=(-bY/2,bX/2,-gn(X^2-Y^2)/2+gsXY). Static E vanishes.
    double b,gn,gs;status=CavityModel_effects_static_components(source,&b,&gn,&gs);
    if(status!=CAVITY_MODEL_OK)return status;
    field->B[0]+=gs*x+gn*y;field->B[1]+=gn*x-gs*y;field->B[2]+=b;
    field->dB[1][0]+=gs;field->dB[1][1]+=gn;
    field->dB[2][0]+=gn;field->dB[2][1]-=gs;
    for(int j=0;j<3;++j) {
        if(!isfinite(field->E[j])||!isfinite(field->B[j]))return CAVITY_MODEL_NONFINITE;
        for(int axis=0;axis<4;++axis)
            if(!isfinite(field->dE[axis][j])||!isfinite(field->dB[axis][j]))
                return CAVITY_MODEL_NONFINITE;
    }
    return CAVITY_MODEL_OK;
}
GPUFUN
int CavityModel_effects_law(const double u[3],double Q,double anomaly,
        const CavityModelEffectsField *field,double S1[6],
        double omega[3],double thomas[3]) {
    if(!isfinite(Q)||!isfinite(anomaly))return CAVITY_MODEL_ARGUMENT;
    double u2=0.;for(int j=0;j<3;++j){if(!isfinite(u[j]))return CAVITY_MODEL_NONFINITE;u2+=u[j]*u[j];}
    const double gamma=sqrt(1.+u2);if(!(u[2]>0.)||!isfinite(gamma))return CAVITY_MODEL_DOMAIN;
    double beta[3],DE[3],DB[3],G[3],bxB[3],ExB[3],BxBxBeta[3],bxe[3];
    double bE=0.,bB=0.,g2=0.;
    for(int j=0;j<3;++j)beta[j]=u[j]/gamma;
    CavityModel_effects_cross(beta,field->B,bxB);
    CavityModel_effects_cross(field->E,field->B,ExB);
    CavityModel_effects_cross(field->B,bxB,BxBxBeta);
    // B cross (B cross beta) = -B cross (beta cross B).
    CavityModel_effects_cross(beta,field->E,bxe);
    for(int j=0;j<3;++j){G[j]=field->E[j]+bxB[j];g2+=G[j]*G[j];
        bE+=beta[j]*field->E[j];bB+=beta[j]*field->B[j];
        DE[j]=field->dE[0][j];DB[j]=field->dB[0][j];
        for(int a=0;a<3;++a){DE[j]+=beta[a]*field->dE[a+1][j];DB[j]+=beta[a]*field->dB[a+1][j];}}
    double bxdB[3],R[3],Rxu[3];CavityModel_effects_cross(beta,DB,bxdB);
    const double emitted=Q*Q*gamma*gamma*(g2-bE*bE),clock=gamma/u[2];
    double energy=0.;
    for(int j=0;j<3;++j){
        R[j]=Q*gamma*(DE[j]+bxdB[j])
            +Q*Q*(ExB[j]-BxBxBeta[j]+field->E[j]*bE)-emitted*beta[j];
        energy+=beta[j]*R[j];
        omega[j]=-Q*((anomaly+1./gamma)*field->B[j]
            -anomaly*gamma/(gamma+1.)*bB*beta[j]
            -(anomaly+1./(gamma+1.))*bxe[j]);
    }
    CavityModel_effects_cross(R,u,Rxu);
    for(int j=0;j<3;++j){S1[j]=0.;thomas[j]=Rxu[j]/(gamma+1.);}
    S1[3]=clock*R[0];S1[4]=clock*R[1];S1[5]=clock*energy;
    for(int j=0;j<6;++j)if(!isfinite(S1[j]))return CAVITY_MODEL_NONFINITE;
    for(int j=0;j<3;++j)if(!isfinite(omega[j])||!isfinite(thomas[j]))return CAVITY_MODEL_NONFINITE;
    return CAVITY_MODEL_OK;
}
GPUFUN
int CavityModel_effects(const double q[6],double z,double Q,
        const CavityModelSource *source,double anomaly,double S1[6],
        double omega[3],double thomas[3]) {
    CavityModelEffectsField field;int status=CavityModel_effects_fields(q,z,source,&field);
    if(status!=CAVITY_MODEL_OK)return status;
    double b,gn,gs;status=CavityModel_effects_static_components(source,&b,&gn,&gs);
    if(status!=CAVITY_MODEL_OK)return status;
    CavityModelPotential potential;
    CavityModel_rf_potential(q,z,source,CAVITY_MODEL_FULL,&potential);
    if(b!=0.||gn!=0.||gs!=0.)CavityModel_add_static(q,z,source,&potential);
    else potential.a[2]-=source->dc_gradient*q[2];
    // Kinetic momentum comes from TOTAL A, including the solenoid's Ax/Ay.
    // Az affects the conservative generator; it is not a transverse kick.
    const double gamma=q[5]-Q*potential.a[3];
    double u[3]={q[3]-Q*potential.a[0],q[4]-Q*potential.a[1],0.};
    const double shell=gamma*gamma-1.-u[0]*u[0]-u[1]*u[1];
    if(!(gamma>1.)||!(shell>0.))return CAVITY_MODEL_DOMAIN;
    u[2]=sqrt(shell);return CavityModel_effects_law(u,Q,anomaly,&field,S1,omega,thomas);
}
// Actual species, independently of the source normalization Q. These SI
// constants match the retained private observer (e exact; CODATA2018 eps0).
GPUFUN
int CavityModel_species_rho(double omega,double actual_mass_eV,
        double actual_charge_e,double *rho) {
    if(!isfinite(omega)||omega<=0.||!isfinite(actual_mass_eV)||actual_mass_eV<=0.
        ||!isfinite(actual_charge_e))return CAVITY_MODEL_ARGUMENT;
    const double elementary_charge=1.602176634e-19,epsilon0=8.8541878128e-12;
    const double result=elementary_charge*omega/(6.*PI*epsilon0*C_LIGHT)
        *actual_charge_e*actual_charge_e/actual_mass_eV;
    if(!isfinite(result)||result<0.)return CAVITY_MODEL_NONFINITE;
    *rho=result;return CAVITY_MODEL_OK;
}

// Applied-effects draft: full-root P16 plus uniform SOL/QUAD, CPU scalar.
// The LL law is already spatial; BMT/Thomas lab-time rates receive the
// same-event mechanical clock exactly once here. No photon law is implied.
GPUFUN
int CavityModel_effects_rhs(const double q[6],double z,double Q,
        const CavityModelSource *source,double rho,double out[6]) {
    int status=CavityModel_flow(q,z,Q,source,CAVITY_MODEL_FULL,out);
    if(status!=CAVITY_MODEL_OK)return status;
    double reaction[6],omega[3],thomas[3];
    status=CavityModel_effects(q,z,Q,source,0.,reaction,omega,thomas);
    if(status!=CAVITY_MODEL_OK)return status;
    for(int j=0;j<6;++j){out[j]+=rho*reaction[j];if(!isfinite(out[j]))return CAVITY_MODEL_NONFINITE;}
    return CAVITY_MODEL_OK;
}

// Nonlinear reacted G2 stages, not a first-rho endpoint correction.
GPUFUN
int CavityModel_mean_gauss_stages(const double q[6],double z,double h,double Q,
        const CavityModelSource *source,double rho,int64_t maximum,double tolerance,
        double out[6],double accepted[12]) {
    if(!isfinite(rho)||rho<0.||maximum<=0||!isfinite(tolerance)||tolerance<=0.)
        return CAVITY_MODEL_ARGUMENT;
    const double C[2]={.21132486540518713,.7886751345948129};
    const double A[2][2]={{.25,-.038675134594812866},{.5386751345948129,.25}};
    double initial[6],u[2][6],values[2][6],full[6];
    int status=CavityModel_effects_rhs(q,z,Q,source,rho,initial);if(status!=CAVITY_MODEL_OK)return status;
    for(int i=0;i<2;++i)for(int j=0;j<6;++j)u[i][j]=C[i]*h*initial[j];
    for(int64_t iteration=0;iteration<maximum;++iteration) {
        for(int i=0;i<2;++i){for(int j=0;j<6;++j)full[j]=q[j]+u[i][j];
            status=CavityModel_effects_rhs(full,z+C[i]*h,Q,source,rho,values[i]);
            if(status!=CAVITY_MODEL_OK)return status;}
        double change=0.;
        for(int i=0;i<2;++i)for(int j=0;j<6;++j){
            const double next=h*(A[i][0]*values[0][j]+A[i][1]*values[1][j]);
            change=fmax(change,fabs(next-u[i][j]));u[i][j]=next;}
        if(change>=tolerance)continue;
        for(int i=0;i<2;++i){for(int j=0;j<6;++j)full[j]=q[j]+u[i][j];
            status=CavityModel_effects_rhs(full,z+C[i]*h,Q,source,rho,values[i]);
            if(status!=CAVITY_MODEL_OK)return status;}
        double residual=0.;
        for(int i=0;i<2;++i)for(int j=0;j<6;++j)
            residual=fmax(residual,fabs(u[i][j]-h*(A[i][0]*values[0][j]+A[i][1]*values[1][j])));
        if(residual>=tolerance)continue;
        double endpoint[6],check[6];
        for(int j=0;j<6;++j)endpoint[j]=q[j]+h*(values[0][j]+values[1][j])*.5;
        status=CavityModel_effects_rhs(endpoint,z+h,Q,source,rho,check);
        if(status!=CAVITY_MODEL_OK)return status;
        for(int i=0;i<2;++i)for(int j=0;j<6;++j)accepted[6*i+j]=q[j]+u[i][j];
        for(int j=0;j<6;++j)out[j]=endpoint[j];
        return CAVITY_MODEL_OK;
    }
    return CAVITY_MODEL_ITERATIONS;
}

GPUFUN
int CavityModel_spin_generator(const double q[6],double z,double Q,
        const CavityModelSource *source,double anomaly,double rho,double G[9]) {
    double reaction[6],omega[3],thomas[3],flow[6];
    int status=CavityModel_effects(q,z,Q,source,anomaly,reaction,omega,thomas);
    if(status!=CAVITY_MODEL_OK)return status;
    status=CavityModel_flow(q,z,Q,source,CAVITY_MODEL_FULL,flow);
    if(status!=CAVITY_MODEL_OK)return status;
    double w[3];for(int j=0;j<3;++j){w[j]=flow[2]*(omega[j]+rho*thomas[j]);
        if(!isfinite(w[j]))return CAVITY_MODEL_NONFINITE;}
    const double skew[9]={0.,-w[2],w[1],w[2],0.,-w[0],-w[1],w[0],0.};
    for(int j=0;j<9;++j)G[j]=skew[j];return CAVITY_MODEL_OK;
}

// Solve the two 3-vector collocation stages for all three input columns.
// The accepted orbit stages are supplied; no orbit or spin normalization is
// substituted. A fresh OPERATOR residual is checked independently of LU.
GPUFUN
int CavityModel_spin_gauss_matrix(const double stages[12],double z,double h,double Q,
        const CavityModelSource *source,double anomaly,double rho,double tolerance,
        double out[9]) {
    if(!isfinite(anomaly)||!isfinite(rho)||rho<0.||!isfinite(tolerance)||tolerance<=0.)
        return CAVITY_MODEL_ARGUMENT;
    const double C[2]={.21132486540518713,.7886751345948129};
    const double A[2][2]={{.25,-.038675134594812866},{.5386751345948129,.25}};
    double G[2][9],matrix[6][9],solution[6][3];
    for(int i=0;i<2;++i){int status=CavityModel_spin_generator(stages+6*i,z+C[i]*h,Q,
        source,anomaly,rho,G[i]);if(status!=CAVITY_MODEL_OK)return status;}
    for(int i=0;i<2;++i)for(int row=0;row<3;++row){const int r=3*i+row;
        for(int j=0;j<2;++j)for(int col=0;col<3;++col)
            matrix[r][3*j+col]=(i==j&&row==col?1.:0.)-h*A[i][j]*G[j][3*row+col];
        for(int col=0;col<3;++col)matrix[r][6+col]=row==col?1.:0.;}
    for(int pivot=0;pivot<6;++pivot){
        int best=pivot;for(int row=pivot+1;row<6;++row)
            if(fabs(matrix[row][pivot])>fabs(matrix[best][pivot]))best=row;
        if(!isfinite(matrix[best][pivot])||matrix[best][pivot]==0.)return CAVITY_MODEL_ITERATIONS;
        if(best!=pivot)for(int col=pivot;col<9;++col){double saved=matrix[pivot][col];
            matrix[pivot][col]=matrix[best][col];matrix[best][col]=saved;}
        for(int row=pivot+1;row<6;++row){const double factor=matrix[row][pivot]/matrix[pivot][pivot];
            for(int col=pivot+1;col<9;++col)matrix[row][col]-=factor*matrix[pivot][col];
            matrix[row][pivot]=0.;}
    }
    for(int row=5;row>=0;--row)for(int col=0;col<3;++col){
        double value=matrix[row][6+col];for(int j=row+1;j<6;++j)value-=matrix[row][j]*solution[j][col];
        solution[row][col]=value/matrix[row][row];
        if(!isfinite(solution[row][col]))return CAVITY_MODEL_NONFINITE;}
    double products[2][9],residual=0.;
    for(int i=0;i<2;++i)for(int row=0;row<3;++row)for(int col=0;col<3;++col){
        double value=0.;for(int k=0;k<3;++k)value+=G[i][3*row+k]*solution[3*i+k][col];
        products[i][3*row+col]=value;}
    for(int i=0;i<2;++i)for(int row=0;row<3;++row)for(int col=0;col<3;++col){
        const double defect=solution[3*i+row][col]-(row==col?1.:0.)
            -h*(A[i][0]*products[0][3*row+col]+A[i][1]*products[1][3*row+col]);
        if(!isfinite(defect))return CAVITY_MODEL_NONFINITE;
        residual=fmax(residual,fabs(defect));}
    if(residual>=tolerance)return CAVITY_MODEL_ITERATIONS;
    double result[9];for(int row=0;row<3;++row)for(int col=0;col<3;++col){
        const int j=3*row+col;result[j]=(row==col?1.:0.)+h*(products[0][j]+products[1][j])*.5;
        if(!isfinite(result[j]))return CAVITY_MODEL_NONFINITE;}
    for(int j=0;j<9;++j)out[j]=result[j];return CAVITY_MODEL_OK;
}

GPUFUN
int CavityModel_effects_cell(const double q[6],const double spin[3],double z,double h,double Q,
        const CavityModelSource *source,double anomaly,double rho,int apply_mean,int apply_spin,
        double out[6],double spin_out[3]) {
    double endpoint[6],stages[12];int status;
    if(apply_mean)status=CavityModel_mean_gauss_stages(q,z,h,Q,source,rho,80,2e-13,endpoint,stages);
    else status=CavityModel_gauss_stages(q,z,h,Q,source,CAVITY_MODEL_FULL,80,2e-13,endpoint,stages);
    if(status!=CAVITY_MODEL_OK)return status;
    double next[3]={spin[0],spin[1],spin[2]};
    if(apply_spin){double rotation[9];status=CavityModel_spin_gauss_matrix(stages,z,h,Q,source,
            anomaly,apply_mean?rho:0.,2e-13,rotation);if(status!=CAVITY_MODEL_OK)return status;
        for(int i=0;i<3;++i){next[i]=0.;for(int j=0;j<3;++j)next[i]+=rotation[3*i+j]*spin[j];
            if(!isfinite(next[i]))return CAVITY_MODEL_NONFINITE;}}
    for(int j=0;j<6;++j)out[j]=endpoint[j];for(int j=0;j<3;++j)spin_out[j]=next[j];
    return CAVITY_MODEL_OK;
}

#endif // CPU-scalar applied effects; no TPSA scalarization

#endif
