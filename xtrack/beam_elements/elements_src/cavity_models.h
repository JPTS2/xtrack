// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_H
#define XTRACK_CAVITY_MODELS_H

// Phase-resolved source and maps.
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

// The phase-resolved map keeps E_z to r^radial_order, M = radial_order/2
// terms in r^2, from the profile jets 0..2M (2M+1 for the effect fields). The
// field table holds jets 0..TABLE_JETS-1: each interpolated jet takes the next
// two as its derivatives.
#define CAVITY_MODEL_MAX_RADIAL_ORDER 8
#define CAVITY_MODEL_JETS (CAVITY_MODEL_MAX_RADIAL_ORDER+2)
#define CAVITY_MODEL_TABLE_JETS (CAVITY_MODEL_JETS+2)
#define CAVITY_MODEL_TABLE_ROW (2*CAVITY_MODEL_TABLE_JETS)

enum CavityPhysicalModel {
    CAVITY_MODEL_FULL = 10,
    CAVITY_MODEL_AVERAGED = 11,
    CAVITY_MODEL_PARAXIAL = 12,
    CAVITY_MODEL_FROZEN = 13
};

typedef struct {
    // omega is the coordinate-clock scale c*k. With RF off and f=0, k=1/m
    // is an arbitrary unit choice, not a nonzero physical RF frequency.
    xt_float_or_tpsa k, omega, half_span, length;
    xt_float_or_tpsa forward, backward, carrier, counter_phase, clock_origin;
    xt_float_or_tpsa voltage_norm, phase, dc_gradient;
    // RF field: 0 uniform envelope with the two waves, 1 space harmonics.
    int rf_profile;
    int64_t radial_order;
    // Space harmonics: P(Z) = sum_m (re_m + i im_m) exp(i kz_m Z/k), times forward,
    // on the element; kz_m in rad/m.
    int64_t num_harmonics;
    GPUGLMEM const double *harmonic_wave_number, *harmonic_re, *harmonic_im;
    // Optional table of the unit-amplitude jets in physical z (track_rf.h).
    int64_t table_points;
    double table_z0, table_dz;
    GPUGLMEM const double *table;
    double reference_charge, reference_beta;
    int freeze_energy, freeze_zeta;
    int freeze_x, freeze_px, freeze_y, freeze_py;
    // Frozen model: equal cells instead of the geometric (SAD) grading.
    int uniform_mesh;
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

// Live fixed-beta electric-crest calibration. No evolving particle beta
// retunes the hardware. wave_number is already resolved by the caller and
// may be zero; it is not the coordinate normalization k=omega/c.
// Only RF entries are assigned; the caller supplies static strengths.
GPUFUN
int CavityModel_prepare(
        xt_float_or_tpsa_arg frequency, xt_float_or_tpsa_arg voltage,
        xt_float_or_tpsa_arg length, xt_float_or_tpsa_arg phase,
        xt_float_or_tpsa_arg counter_amplitude, xt_float_or_tpsa_arg counter_phase,
        xt_float_or_tpsa_arg calibration_beta,
        xt_float_or_tpsa_arg wave_number,
        double source_mass_eV, double q0, CavityModelSource *source) {
    const double f=xt_float_or_tpsa_const_part(frequency);
    const double L=xt_float_or_tpsa_const_part(length);
    const double beta=xt_float_or_tpsa_const_part(calibration_beta);
    const double mixture=xt_float_or_tpsa_const_part(counter_amplitude);
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
    // A pure counter wave is a distinct chart: its weights have no derivative.
    if (isinf(mixture) && !CavityModel_scalar_constant(counter_amplitude))
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
    source->rf_profile=0;
    source->radial_order=6;
    source->num_harmonics=0;source->harmonic_wave_number=NULL;
    source->harmonic_re=NULL;source->harmonic_im=NULL;
    source->table_points=0;source->table_z0=0.;source->table_dz=0.;source->table=NULL;
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
        forward_weight=1./(1.+counter_amplitude);
        backward_weight=counter_amplitude/(1.+counter_amplitude);
    }
    xt_float_or_tpsa slow=0.,fast=0.;
    if(!xt_float_or_tpsa_is_zero(forward_weight))
        slow=forward_weight*2.*source->half_span*CavityModel_static_sinc(
            (1./calibration_beta-source->carrier)*source->half_span);
    if(!xt_float_or_tpsa_is_zero(backward_weight))
        fast=backward_weight*2.*source->half_span*CavityModel_static_sinc(
            (1./calibration_beta+source->carrier)*source->half_span);
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

// The uniform envelope: the field ends at the faces, which the maps apply.
GPUFUN
void CavityModel_source_envelope(xt_float_or_tpsa_arg z,
        const CavityModelSource *source,xt_float_or_tpsa p[5]) {
    p[0]=1.;for(int i=1;i<5;++i)p[i]=0.;
}

// Jets 0..orders-1 of the harmonic profile in the normalized Z = k z from the
// table, by quintic Hermite interpolation of each jet with the next two as its
// derivatives: CAVITY_MODEL_TABLE_ROW doubles per row, the real then imaginary
// parts of jets 0..CAVITY_MODEL_TABLE_JETS-1 of the unit-amplitude field in
// metres. Returns 0 where the table does not apply.
GPUFUN
int CavityModel_table_jets(xt_float_or_tpsa_arg z, const CavityModelSource *source,
        xt_float_or_tpsa real[], xt_float_or_tpsa imag[], int orders) {
    if (!source->table) return 0;
    const xt_float_or_tpsa t=(z/source->k-source->table_z0)/source->table_dz;
    const double t0=xt_float_or_tpsa_const_part(t);
    if (!(t0>=0.) || t0>(double)(source->table_points-1)) return 0;
    int64_t n=(int64_t)floor(t0);
    if (n>source->table_points-2) n=source->table_points-2;
    const xt_float_or_tpsa u=t-(double)n, u2=u*u, u3=u2*u, u4=u3*u, u5=u4*u;
    const double h=source->table_dz, hh=h*h;
    const xt_float_or_tpsa w0=1.-10.*u3+15.*u4-6.*u5, w1=h*(u-6.*u3+8.*u4-3.*u5);
    const xt_float_or_tpsa w2=hh*(.5*u2-1.5*u3+1.5*u4-.5*u5), w3=10.*u3-15.*u4+6.*u5;
    const xt_float_or_tpsa w4=h*(-4.*u3+7.*u4-3.*u5), w5=hh*(.5*u3-u4+.5*u5);
    GPUGLMEM const double *a=source->table+CAVITY_MODEL_TABLE_ROW*n;
    GPUGLMEM const double *b=a+CAVITY_MODEL_TABLE_ROW;
    GPUGLMEM const double *c=a+CAVITY_MODEL_TABLE_JETS, *d=b+CAVITY_MODEL_TABLE_JETS;
    xt_float_or_tpsa scale=source->forward;
    for (int j=0;j<orders;++j) {
        real[j]=scale*(w0*a[j]+w1*a[j+1]+w2*a[j+2]+w3*b[j]+w4*b[j+1]+w5*b[j+2]);
        imag[j]=scale*(w0*c[j]+w1*c[j+1]+w2*c[j+2]+w3*d[j]+w4*d[j+1]+w5*d[j+2]);
        scale=scale/source->k;
    }
    return 1;
}

GPUFUN
void CavityModel_profile(xt_float_or_tpsa_arg z,
        const CavityModelSource *source, xt_float_or_tpsa real[],
        xt_float_or_tpsa imag[], int orders) {
    if (source->rf_profile==1) {
        if (CavityModel_table_jets(z,source,real,imag,orders)) return;
        for (int j=0;j<orders;++j) { real[j]=0.;imag[j]=0.; }
        for (int64_t m=0;m<source->num_harmonics;++m) {
            const xt_float_or_tpsa kappa=source->harmonic_wave_number[m]/source->k;
            const xt_float_or_tpsa c=cos(kappa*z), s=sin(kappa*z);
            const double re=source->harmonic_re[m], im=source->harmonic_im[m];
            // (i kappa)^j a e^{i kappa Z}: each derivative turns the phasor by i.
            xt_float_or_tpsa term_re=source->forward*(re*c-im*s);
            xt_float_or_tpsa term_im=source->forward*(re*s+im*c);
            for (int j=0;j<orders;++j) {
                real[j]+=term_re; imag[j]+=term_im;
                const xt_float_or_tpsa turned=term_re;
                term_re=-kappa*term_im; term_im=kappa*turned;
            }
        }
        return;
    }
    xt_float_or_tpsa phases[2]={source->carrier*z,
                               -source->carrier*z-source->counter_phase};
    xt_float_or_tpsa strengths[2]={source->forward,source->backward};
    for (int j=0;j<orders;++j) { real[j]=0.;imag[j]=0.; }
    for (int wave=0;wave<2;++wave) {
        xt_float_or_tpsa nu=source->carrier;
        if (wave) nu=-nu;
        // (i nu)^j strength e^{i phase}: each derivative turns the phasor by i.
        xt_float_or_tpsa term_re=strengths[wave]*cos(phases[wave]);
        xt_float_or_tpsa term_im=strengths[wave]*sin(phases[wave]);
        for (int j=0;j<orders;++j) {
            real[j]+=term_re; imag[j]+=term_im;
            const xt_float_or_tpsa turned=term_re;
            term_re=-nu*term_im; term_im=nu*turned;
        }
    }
}

// Retained temporal-gauge RF potential to M = radial_order/2 terms in r^2 and
// its exact first derivatives, with P_n = (1 + d_Z^2)^n of the phase profile:
// A_z = -sum_{n<=M} (-1/4)^n/n!^2 r^2n P_n, A_perp = r_perp sum_{n<M}
// (-1/4)^n/(2 n!(n+1)!) r^2n P_n'. M = 2 is the compact form of
// native_rf_source()['potentials']; no force fit or independent
// radial-focusing coefficient is introduced.
GPUFUN
void CavityModel_rf_potential(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z, const CavityModelSource *source,
        int model, CavityModelPotential *potential) {
    const int M=(int)(source->radial_order/2), jets=2*M+1;
    xt_float_or_tpsa real[CAVITY_MODEL_JETS],imag[CAVITY_MODEL_JETS];
    xt_float_or_tpsa S[CAVITY_MODEL_JETS],C[CAVITY_MODEL_JETS];
    CavityModel_profile(z,source,real,imag,jets);
    xt_float_or_tpsa sine=sin(q[2]),cosine=cos(q[2]);
    for (int j=0;j<jets;++j) {
        S[j]=real[j]*sine-imag[j]*cosine;
        C[j]=real[j]*cosine+imag[j]*sine;
    }
    for (int i=0;i<4;++i) {
        potential->a[i]=0.;
        for (int j=0;j<3;++j) potential->d[i][j]=0.;
    }
    // Frozen voltage actions have their own body chart and collapsed RF
    // faces. The phase-resolved interior gauge must never be attached to them.
    if (model==CAVITY_MODEL_FROZEN || model==CAVITY_MODEL_AVERAGED
        || model==CAVITY_MODEL_PARAXIAL) return;
    xt_float_or_tpsa X=q[0],Y=q[1],r2=X*X+Y*Y;
    // Sums and their r^2 derivatives; S, C turn into P_n in place, P_n
    // keeping jets 0..2(M-n).
    xt_float_or_tpsa axial=0.,axial_T=0.,axial_r2=0.;
    xt_float_or_tpsa radial=0.,radial_T=0.,radial_r2=0.;
    xt_float_or_tpsa power=1.,slope=0.;
    double alpha=1.,beta=.5;
    for (int n=0;n<=M;++n) {
        axial+=alpha*power*S[0]; axial_T+=alpha*power*C[0];
        axial_r2+=alpha*slope*S[0];
        if (n<M) {
            radial+=beta*power*S[1]; radial_T+=beta*power*C[1];
            radial_r2+=beta*slope*S[1];
        }
        for (int j=0;j+2<jets-2*n;++j) { S[j]+=S[j+2]; C[j]+=C[j+2]; }
        slope=(double)(n+1)*power; power=power*r2;
        alpha*=-.25/((n+1.)*(n+1.)); beta*=-.25/((n+1.)*(n+2.));
    }
    potential->a[0]=X*radial; potential->a[1]=Y*radial;
    potential->a[2]=-axial;
    potential->d[0][0]=radial+2.*X*X*radial_r2;
    potential->d[0][1]=2.*X*Y*radial_r2;
    potential->d[0][2]=X*radial_T;
    potential->d[1][0]=potential->d[0][1];
    potential->d[1][1]=radial+2.*Y*Y*radial_r2;
    potential->d[1][2]=Y*radial_T;
    potential->d[2][0]=-2.*X*axial_r2;
    potential->d[2][1]=-2.*Y*axial_r2;
    potential->d[2][2]=-axial_T;
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
    return CavityModel_potential(q,z,source,model,output);
}

// H=-Q Az-sqrt(epsilon^2-1-|P-Q A_perp|^2).
GPUFUN
int CavityModel_flow(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z, xt_float_or_tpsa_arg Q,
        const CavityModelSource *source, int model,
        xt_float_or_tpsa flow[6]) {
    if (model!=CAVITY_MODEL_FULL) return CAVITY_MODEL_ARGUMENT;
    if (source->freeze_zeta && !(source->reference_beta>0.
                                && source->reference_beta<=1.))
        return CAVITY_MODEL_ARGUMENT;
    if (!CavityModel_vector_finite(q,6)) return CAVITY_MODEL_NONFINITE;
    CavityModelPotential potential;
    CavityModel_rf_potential(q,z,source,model,&potential);
    CavityModel_add_static(q,z,source,&potential);
    if (!CavityModel_vector_finite(potential.a,4)) return CAVITY_MODEL_NONFINITE;
    xt_float_or_tpsa energy=q[5]-Q*potential.a[3];
    xt_float_or_tpsa px=q[3]-Q*potential.a[0];
    xt_float_or_tpsa py=q[4]-Q*potential.a[1];
    xt_float_or_tpsa p02=(energy-1.)*(energy+1.);
    xt_float_or_tpsa transverse=px*px+py*py;
    xt_float_or_tpsa square=p02-transverse;
    if (xt_float_or_tpsa_const_part(energy)<=1.
        || xt_float_or_tpsa_const_part(square)<=0.) return CAVITY_MODEL_DOMAIN;
    xt_float_or_tpsa denominator=sqrt(square);
    flow[2]=energy/denominator;
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

#ifndef CAVITY_MODEL_JET_SWEEPS
#ifdef XTRACK_TPSA_TRACK
#define CAVITY_MODEL_JET_SWEEPS 4
#else
#define CAVITY_MODEL_JET_SWEEPS 0
#endif
#endif

// Two-stage Gauss--Legendre: the same equations and constants as the existing
// Python owner. Signed h reverses the deterministic word. All outputs are
// transactional. The defining residual is checked after reassignment. Both
// tests use the primal only; under TPSA the coefficients lag the primal by a
// few contractions, so fixed sweeps after primal convergence settle them.
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
                change=fmax(change,CavityModel_primal_residual(
                    proposed[i][j]-stages[i][j]));
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
                residual=fmax(residual,CavityModel_primal_residual(stages[i][j]
                    -h*(A[i][0]*values[0][j]+A[i][1]*values[1][j])));
        if (residual>=tolerance) continue;
        for (int sweep=0;sweep<CAVITY_MODEL_JET_SWEEPS;++sweep) {
            for (int i=0;i<2;++i)
                for (int j=0;j<6;++j)
                    proposed[i][j]=h*(A[i][0]*values[0][j]+A[i][1]*values[1][j]);
            for (int i=0;i<2;++i) {
                for (int j=0;j<6;++j) {
                    stages[i][j]=proposed[i][j];
                    initial[j]=q[j]+stages[i][j];
                }
                status=CavityModel_flow(initial,z+C[i]*h,Q,source,model,values[i]);
                if (status!=CAVITY_MODEL_OK) return status;
            }
        }
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
    return CAVITY_MODEL_ITERATIONS;
}

GPUFUN
int CavityModel_gauss(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z,xt_float_or_tpsa_arg h,xt_float_or_tpsa_arg Q,
        const CavityModelSource *source,int model,int64_t max_iter,double tolerance,
        xt_float_or_tpsa output[6]) {
    return CavityModel_gauss_stages(q,z,h,Q,source,model,max_iter,tolerance,output,NULL);
}



// Space harmonics over the element, with the same fixed-beta crest calibration:
// the crest phasor is sum_m conj(a_m) 2H sinc((1/beta - kappa_m) H).
GPUFUN
int CavityModel_prepare_harmonics(xt_float_or_tpsa_arg frequency,
        xt_float_or_tpsa_arg voltage,xt_float_or_tpsa_arg length,
        xt_float_or_tpsa_arg phase,xt_float_or_tpsa_arg calibration_beta,
        double mass0,double q0,int64_t num_harmonics,
        GPUGLMEM const double *wave_number,GPUGLMEM const double *re,
        GPUGLMEM const double *im,CavityModelSource *source) {
    int status=CavityModel_prepare(frequency,0.,length,phase,0.,0.,calibration_beta,
        (2.*PI/C_LIGHT)*frequency,mass0,q0,source);
    if(status!=CAVITY_MODEL_OK)return status;
    if(num_harmonics<=0||!wave_number||!re||!im)return CAVITY_MODEL_ARGUMENT;
    double scale=0.;
    for(int64_t m=0;m<num_harmonics;++m) {
        if(!isfinite(wave_number[m])||!isfinite(re[m])||!isfinite(im[m]))
            return CAVITY_MODEL_ARGUMENT;
        scale+=hypot(re[m],im[m]);
    }
    source->rf_profile=1;
    source->num_harmonics=num_harmonics;
    source->harmonic_wave_number=wave_number;
    source->harmonic_re=re;source->harmonic_im=im;
    source->voltage_norm=voltage/mass0;
    if(xt_float_or_tpsa_is_zero(voltage))return CAVITY_MODEL_OK;
    if(xt_float_or_tpsa_const_part(frequency)<=0.||!(scale>0.))return CAVITY_MODEL_ARGUMENT;
    const xt_float_or_tpsa H=source->half_span;
    xt_float_or_tpsa real=0.,imag=0.;
    for(int64_t m=0;m<num_harmonics;++m) {
        const xt_float_or_tpsa kappa=wave_number[m]/source->k;
        const xt_float_or_tpsa integral=2.*H*CavityModel_static_sinc((1./calibration_beta-kappa)*H);
        real+=re[m]*integral;imag-=im[m]*integral;
    }
    const xt_float_or_tpsa square=real*real+imag*imag;
    const double floor=64.*2.2204460492503131e-16*2.*xt_float_or_tpsa_const_part(H)*scale;
    if(!CavityModel_scalar_finite(square)||xt_float_or_tpsa_const_part(square)<=floor*floor)
        return CAVITY_MODEL_ARGUMENT;
    source->forward=voltage/(mass0*sqrt(square));source->backward=0.;
    const xt_float_or_tpsa angle=H/calibration_beta;
    const xt_float_or_tpsa rotated_re=real*cos(angle)-imag*sin(angle);
    const xt_float_or_tpsa rotated_im=real*sin(angle)+imag*cos(angle);
    source->clock_origin-=atan2(rotated_im,rotated_re);
    return (CavityModel_scalar_finite(source->forward)&&CavityModel_scalar_finite(source->clock_origin))
        ?CAVITY_MODEL_OK:CAVITY_MODEL_NONFINITE;
}

#ifndef XTRACK_TPSA_TRACK
// Applied-effects local laws: full-root M2 plus uniform SOL/QUAD overlap.
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
                               int jets,double S[],double C[]) {
    const double H=xt_float_or_tpsa_const_part(source->half_span);
    if(source->rf_profile<0||source->rf_profile>1 || !(H>0.) || !isfinite(H))
        return CAVITY_MODEL_ARGUMENT;
    if(source->rf_profile==1) {
        // C + iS is e^{iT} conj(P^(n)), as for the two waves below.
        const double forward=xt_float_or_tpsa_const_part(source->forward);
        const double k=xt_float_or_tpsa_const_part(source->k);
        // The field continues past the faces as in the orbit's profile: the
        // compositions' substeps reach beyond them, and the faces are impulses.
        for(int n=0;n<jets;++n){S[n]=0.;C[n]=0.;}
        double jet_re[CAVITY_MODEL_JETS],jet_im[CAVITY_MODEL_JETS];
        if(CavityModel_table_jets(z,source,jet_re,jet_im,jets)) {
            for(int n=0;n<jets;++n) {
                C[n]=cos(T)*jet_re[n]+sin(T)*jet_im[n];
                S[n]=sin(T)*jet_re[n]-cos(T)*jet_im[n];
                if(!isfinite(S[n])||!isfinite(C[n]))return CAVITY_MODEL_NONFINITE;
            }
            return CAVITY_MODEL_OK;
        }
        for(int64_t m=0;m<source->num_harmonics;++m) {
            const double kappa=source->harmonic_wave_number[m]/k;
            const double c=cos(kappa*z),s=sin(kappa*z);
            double p_re=forward*(source->harmonic_re[m]*c-source->harmonic_im[m]*s);
            double p_im=forward*(source->harmonic_re[m]*s+source->harmonic_im[m]*c);
            for(int n=0;n<jets;++n) {
                C[n]+=cos(T)*p_re+sin(T)*p_im;
                S[n]+=sin(T)*p_re-cos(T)*p_im;
                const double turned=p_re;p_re=-kappa*p_im;p_im=kappa*turned;
            }
        }
        for(int n=0;n<jets;++n)if(!isfinite(S[n])||!isfinite(C[n]))return CAVITY_MODEL_NONFINITE;
        return CAVITY_MODEL_OK;
    }
    double f[CAVITY_MODEL_JETS]={0.};
    f[0]=1.;
    const double nu=xt_float_or_tpsa_const_part(source->carrier);
    const double amp[2]={xt_float_or_tpsa_const_part(source->forward),
                         xt_float_or_tpsa_const_part(source->backward)};
    const double cp=xt_float_or_tpsa_const_part(source->counter_phase);
    // Match the conservative source: keep event time separate from local phase.
    const double sinT=sin(T),cosT=cos(T);
    double sine[2],cosine[2];
    for(int wave=0;wave<2;++wave) {
        const double w=wave?nu:-nu,spatial=w*z+(wave?cp:0.);
        const double ss=sin(spatial),cs=cos(spatial);
        sine[wave]=sinT*cs+cosT*ss;
        cosine[wave]=cosT*cs-sinT*ss;
    }
    for(int n=0;n<jets;++n) {S[n]=0.;C[n]=0.;
        for(int wave=0;wave<2;++wave) {
            const double w=wave?nu:-nu;
            double choose=1.;
            for(int j=0;j<=n;++j) {
                const int degree=n-j;
                const double factor=amp[wave]*choose*f[j]*pow(w,degree);
                const int quadrant=degree%4;
                if(quadrant==0){S[n]+=factor*sine[wave];C[n]+=factor*cosine[wave];}
                if(quadrant==1){S[n]+=factor*cosine[wave];C[n]-=factor*sine[wave];}
                if(quadrant==2){S[n]-=factor*sine[wave];C[n]-=factor*cosine[wave];}
                if(quadrant==3){S[n]-=factor*cosine[wave];C[n]+=factor*sine[wave];}
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
// The radial sums of the effect fields from jets 0..2M of a phase profile F,
// with P_n = (1 + d_Z^2)^n F: E_z's sum_{n<=M} (-1/4)^n/n!^2 r^2n P_n, then
// sum_{n<M} (-1/4)^n/(2 n!(n+1)!) r^2n of P_n and of P_n', each followed by
// its r^2 derivative.
GPUFUN
void CavityModel_effects_series(const double F[],int M,double r2,double sums[6]) {
    double P[CAVITY_MODEL_JETS];
    for(int j=0;j<=2*M;++j)P[j]=F[j];
    for(int i=0;i<6;++i)sums[i]=0.;
    double power=1.,slope=0.,alpha=1.,beta=.5;
    for(int n=0;n<=M;++n) {
        sums[0]+=alpha*power*P[0];sums[1]+=alpha*slope*P[0];
        if(n<M) {
            sums[2]+=beta*power*P[0];sums[3]+=beta*slope*P[0];
            sums[4]+=beta*power*P[1];sums[5]+=beta*slope*P[1];
        }
        for(int j=0;j+2<=2*(M-n);++j)P[j]+=P[j+2];
        slope=(double)(n+1)*power;power*=r2;
        alpha*=-.25/((n+1.)*(n+1.));beta*=-.25/((n+1.)*(n+2.));
    }
}
GPUFUN
int CavityModel_effects_fields(const double q[6],double z,
        const CavityModelSource *source,CavityModelEffectsField *field) {
    const int M=(int)(source->radial_order/2);
    double S[CAVITY_MODEL_JETS],C[CAVITY_MODEL_JETS];
    int status=CavityModel_effects_profile(q[2],z,source,2*M+2,S,C);
    if(status!=CAVITY_MODEL_OK)return status;
    const double x=q[0],y=q[1],r2=x*x+y*y;
    double sine[6],cosine[6];
    CavityModel_effects_series(S,M,r2,sine);CavityModel_effects_series(C,M,r2,cosine);
    const double K=cosine[4],L=sine[2];
    field->E[0]=-x*K;field->E[1]=-y*K;
    field->E[2]=cosine[0]+xt_float_or_tpsa_const_part(source->dc_gradient);
    field->B[0]=y*L;field->B[1]=-x*L;field->B[2]=0.;
    for(int axis=0;axis<4;++axis) {
        const double dx=axis==1,dy=axis==2,dr=2.*(x*dx+y*dy);
        double ds[CAVITY_MODEL_JETS]={0.},dc[CAVITY_MODEL_JETS]={0.};
        for(int j=0;j<=2*M;++j) {
            if(axis==0){ds[j]=C[j];dc[j]=-S[j];}
            if(axis==3){ds[j]=S[j+1];dc[j]=C[j+1];}
        }
        double dsine[6],dcosine[6];
        CavityModel_effects_series(ds,M,r2,dsine);
        CavityModel_effects_series(dc,M,r2,dcosine);
        const double dK=dr*cosine[5]+dcosine[4],dL=dr*sine[3]+dsine[2];
        field->dE[axis][0]=-dx*K-x*dK;field->dE[axis][1]=-dy*K-y*dK;
        field->dE[axis][2]=dr*cosine[1]+dcosine[0];
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
    double beta[3],DE[3],DB[3],GxB[3],bxe[3];
    double bE=0.,bB=0.;
    const double momentum=hypot(hypot(u[0],u[1]),u[2]);
    double direction[3],perpendicular[3],perpendicular_force[3],nxB[3];
    double parallel=0.;
    for(int j=0;j<3;++j){direction[j]=momentum>0.?u[j]/momentum:0.;
        parallel+=direction[j]*field->E[j];}
    for(int j=0;j<3;++j)perpendicular[j]=field->E[j]-parallel*direction[j];
    for(int j=0;j<3;++j)beta[j]=u[j]/gamma;

    CavityModel_effects_cross(beta,field->E,bxe);
    for(int j=0;j<3;++j){
        bE+=beta[j]*field->E[j];bB+=beta[j]*field->B[j];
        DE[j]=field->dE[0][j];DB[j]=field->dB[0][j];
        for(int a=0;a<3;++a){DE[j]+=beta[a]*field->dE[a+1][j];DB[j]+=beta[a]*field->dB[a+1][j];}}
    // beta_magnitude+deficit=1 by the same positive mass shell.
    // This form retains the small force when E_perp and n cross B cancel.
    const double beta_magnitude=momentum/gamma;
    const double deficit=(1./gamma)/(gamma+momentum);
    CavityModel_effects_cross(direction,field->B,nxB);
    double transverse_square=0.;
    for(int j=0;j<3;++j){
        perpendicular_force[j]=beta_magnitude*(perpendicular[j]+nxB[j])
            +deficit*perpendicular[j];
        transverse_square+=perpendicular_force[j]*perpendicular_force[j];}
    CavityModel_effects_cross(perpendicular_force,field->B,GxB);
    double bxdB[3],R[3],Rxu[3];CavityModel_effects_cross(beta,DB,bxdB);
    // The longitudinal electric quadratic term cancels its parallel drag.
    // This remaining transverse drag is not the total emitted power.
    const double transverse_drag=Q*Q*gamma*gamma*transverse_square,clock=gamma/u[2];
    double pre_drag[3];
    double energy=0.;
    for(int j=0;j<3;++j){
        pre_drag[j]=Q*gamma*(DE[j]+bxdB[j])+Q*Q*(GxB[j]+perpendicular_force[j]*bE+((parallel/gamma)/gamma)*nxB[j]);
        R[j]=pre_drag[j]-transverse_drag*beta[j];
        energy+=beta[j]*R[j];
        omega[j]=-Q*((anomaly+1./gamma)*field->B[j]
            -anomaly*gamma/(gamma+1.)*bB*beta[j]
            -(anomaly+1./(gamma+1.))*bxe[j]);
    }
    // Drag is parallel to u and contributes exactly zero Thomas rotation.
    CavityModel_effects_cross(pre_drag,u,Rxu);
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
GPUFUN
int CavityModel_spin_generator(const double q[6],double z,double Q,
        const CavityModelSource *source,double anomaly,double G[9]) {
    double reaction[6],omega[3],thomas[3],flow[6];
    int status=CavityModel_effects(q,z,Q,source,anomaly,reaction,omega,thomas);
    if(status!=CAVITY_MODEL_OK)return status;
    status=CavityModel_flow(q,z,Q,source,CAVITY_MODEL_FULL,flow);
    if(status!=CAVITY_MODEL_OK)return status;
    double w[3];for(int j=0;j<3;++j){w[j]=flow[2]*omega[j];
        if(!isfinite(w[j]))return CAVITY_MODEL_NONFINITE;}
    const double skew[9]={0.,-w[2],w[1],w[2],0.,-w[0],-w[1],w[0],0.};
    for(int j=0;j<9;++j)G[j]=skew[j];return CAVITY_MODEL_OK;
}

// Solve the two 3-vector collocation stages for all three input columns.
// The accepted orbit stages are supplied; no orbit or spin normalization is
// substituted. A fresh OPERATOR residual is checked independently of LU.
GPUFUN
int CavityModel_spin_gauss_matrix(const double stages[12],double z,double h,double Q,
        const CavityModelSource *source,double anomaly,double tolerance,
        double out[9]) {
    if(!isfinite(anomaly)||!isfinite(tolerance)||tolerance<=0.)
        return CAVITY_MODEL_ARGUMENT;
    const double C[2]={.21132486540518713,.7886751345948129};
    const double A[2][2]={{.25,-.038675134594812866},{.5386751345948129,.25}};
    double G[2][9],matrix[6][9],solution[6][3];
    for(int i=0;i<2;++i){int status=CavityModel_spin_generator(stages+6*i,z+C[i]*h,Q,
        source,anomaly,G[i]);if(status!=CAVITY_MODEL_OK)return status;}
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
int CavityModel_spin_cell(const double q[6],const double spin[3],double z,double h,double Q,
        const CavityModelSource *source,double anomaly,double out[6],double spin_out[3]) {
    double endpoint[6],stages[12];
    int status=CavityModel_gauss_stages(q,z,h,Q,source,CAVITY_MODEL_FULL,80,2e-13,endpoint,stages);
    if(status!=CAVITY_MODEL_OK)return status;
    double rotation[9],next[3];
    status=CavityModel_spin_gauss_matrix(stages,z,h,Q,source,anomaly,2e-13,rotation);
    if(status!=CAVITY_MODEL_OK)return status;
    for(int i=0;i<3;++i){next[i]=0.;for(int j=0;j<3;++j)next[i]+=rotation[3*i+j]*spin[j];
        if(!isfinite(next[i]))return CAVITY_MODEL_NONFINITE;}
    for(int j=0;j<6;++j)out[j]=endpoint[j];for(int j=0;j<3;++j)spin_out[j]=next[j];
    return CAVITY_MODEL_OK;
}

#endif // CPU-scalar applied effects; no TPSA scalarization

#endif
