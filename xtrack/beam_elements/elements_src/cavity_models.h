// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_H
#define XTRACK_CAVITY_MODELS_H

// Conservative source-based RF maps. Coordinates are (X,Y,T,Px,Py,E),
// scaled by (1/k,1/k,1/(kc),mc,mc,mc^2); the T--E block has negative sign.
// k is the coordinate scale omega/c, independent of the physical carrier.
// Source ancestry: cavity-modelling phase_resolved/conservative_source.py,
// kinetic/transport.py, voltage.py and phase_resolved/localized.py.
// This header supplies numerical maps, not physical/numerical admission.
#include <math.h>
#include <stdint.h>
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
    CAVITY_MODEL_FROZEN = 13
};

typedef struct {
    // omega is the coordinate-clock scale c*k. With RF off and f=0, k=1/m
    // is an arbitrary unit choice, not a nonzero physical RF frequency.
    xt_float_or_tpsa k, omega, half_span, length;
    xt_float_or_tpsa forward, backward, carrier, counter_phase, clock_origin;
    xt_float_or_tpsa voltage_norm, phase, dc_gradient;
    double reference_charge, reference_beta;
    int freeze_energy, freeze_zeta;
    // q_ref A/(p_ref) conventions: magnetic_scale=p_ref/(q_ref*m_source*c).
    // knl/ksl are the ordinary integrated normal/skew magnetic strengths.
    xt_float_or_tpsa magnetic_scale, ks;
    int64_t order;
    GPUGLMEM const double *knl, *ksl;
} CavityModelSource;

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

// The same fixed 64-point Gauss rule as the private preparation factory.
// Evenness leaves 32 positive nodes. These are quadrature constants, not a
// frequency/profile fit. Their derivatives remain in the scalar algebra.
GPUFUN
xt_float_or_tpsa CavityModel_p16_integral(xt_float_or_tpsa_arg H,
                                         xt_float_or_tpsa_arg frequency) {
    const double nodes[32] = {
        .024350292663424436,.072993121787799042,.12146281929612056,
        .1696444204239928,.21742364374000708,.26468716220876742,
        .31132287199021097,.35722015833766813,.40227015796399157,
        .44636601725346409,.48940314570705296,.53127946401989457,
        .571895646202634,.61115535517239328,.64896547125465731,
        .68523631305423327,.71988185017161077,.75281990726053194,
        .78397235894334139,.81326531512279754,.84062929625258032,
        .86599939815409277,.88931544599511414,.91052213707850282,
        .92956917213193957,.94641137485840277,.96100879965205366,
        .97332682778991098,.98333625388462598,.99101337147674429,
        .99634011677195522,.99930504173577217};
    const double weights[32] = {
        .048690957009139689,.048575467441503428,.048344762234802933,
        .047999388596458303,.047540165714830343,.046968182816210007,
        .046284796581314375,.0454916279274181,.044590558163756601,
        .043583724529323471,.042473515123653542,.041262563242623486,
        .039953741132720454,.038550153178615605,.037055128540240019,
        .035472213256882226,.033805161837141877,.032057928354851412,
        .030234657072402488,.028339672614259424,.026377469715054894,
        .024352702568711086,.022270173808382945,.020134823153530039,
        .017951715775697288,.01572603047602511,.01346304789671912,
        .011168139460131008,.0088467598263633606,.0065044579689784374,
        .0041470332605646485,.0017832807216942642};
    xt_float_or_tpsa result = 0.;
    for (int i=0; i<32; ++i) {
        const double u=nodes[i];
        double envelope=1.-u*u;
        envelope*=envelope; envelope*=envelope;
        envelope*=envelope; envelope*=envelope;
        result += weights[i]*envelope*cos(frequency*H*u);
    }
    return 2.*H*result;
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
    source->freeze_energy=0; source->freeze_zeta=0;
    source->dc_gradient=0.;
    if (xt_float_or_tpsa_is_zero(voltage)) {
        source->forward=0.; source->backward=0.;
        return CAVITY_MODEL_OK;
    }
    if (f<=0.) return CAVITY_MODEL_ARGUMENT;
    // sqrt(eta) is nonanalytic at eta=0. A constant zero endpoint is valid;
    // an eta Taylor series based there must use a different explicit chart.
    if (mixture==0. && !xt_float_or_tpsa_is_zero(eta))
        return CAVITY_MODEL_ARGUMENT;
    xt_float_or_tpsa forward_weight=0., backward_weight=1.;
    if (isfinite(mixture)) {
        xt_float_or_tpsa ratio=0.;
        if (!xt_float_or_tpsa_is_zero(eta)) ratio=sqrt(eta);
        forward_weight=1./(1.+ratio);
        backward_weight=ratio/(1.+ratio);
    }
    xt_float_or_tpsa slow=forward_weight*CavityModel_p16_integral(
        source->half_span,1./calibration_beta-source->carrier);
    xt_float_or_tpsa fast=backward_weight*CavityModel_p16_integral(
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

GPUFUN
void CavityModel_profile(xt_float_or_tpsa_arg z,
        const CavityModelSource *source, xt_float_or_tpsa real[5],
        xt_float_or_tpsa imag[5]) {
    xt_float_or_tpsa envelope[5];
    CavityModel_envelope(z,source->half_span,envelope);
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

// Straight static multipoles and uniform solenoid in the same potential.
// knl[1]>0 gives focusing in x for the reference charge. Source coefficients
// remain uncharged; actual signed Q enters the Hamiltonian exactly once.
GPUFUN
void CavityModel_add_static(const xt_float_or_tpsa q[6],
        const CavityModelSource *source, CavityModelPotential *potential) {
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
        const double normal=source->knl ? source->knl[n] : 0.;
        const double skew=source->ksl ? source->ksl[n] : 0.;
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
    CavityModel_add_static(q,source,&potential);
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
    CavityModelPotential potential;
    CavityModel_rf_potential(q,z,source,model,&potential);
    CavityModel_add_static(q,source,&potential);
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
    return CavityModel_vector_finite(flow,6) ? CAVITY_MODEL_OK : CAVITY_MODEL_NONFINITE;
}

// Two-stage Gauss--Legendre: the same equations and constants as the existing
// Python owner. Signed h reverses the deterministic word. All outputs are
// transactional. The defining residual is checked after reassignment, and
// TPSA convergence includes its represented coefficients, not only the orbit.
GPUFUN
int CavityModel_gauss(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z, xt_float_or_tpsa_arg h,
        xt_float_or_tpsa_arg Q, const CavityModelSource *source, int model,
        int64_t max_iter, double tolerance, xt_float_or_tpsa output[6]) {
    if (max_iter<=0 || !isfinite(tolerance) || tolerance<=0.)
        return CAVITY_MODEL_ARGUMENT;
    const double C[2]={.21132486540518713,.7886751345948129};
    const double A[2][2]={{.25,-.038675134594812866},{.5386751345948129,.25}};
    xt_float_or_tpsa initial[6],stages[2][6],values[2][6],proposed[2][6];
    int status=CavityModel_flow(q,z,Q,source,model,initial);
    if (status!=CAVITY_MODEL_OK) return status;
    for (int i=0;i<2;++i)
        for (int j=0;j<6;++j) stages[i][j]=q[j]+C[i]*h*initial[j];
    for (int64_t iteration=0;iteration<max_iter;++iteration) {
        for (int i=0;i<2;++i) {
            status=CavityModel_flow(stages[i],z+C[i]*h,Q,source,model,values[i]);
            if (status!=CAVITY_MODEL_OK) return status;
        }
        double change=0.;
        for (int i=0;i<2;++i)
            for (int j=0;j<6;++j) {
                proposed[i][j]=q[j]+h*(A[i][0]*values[0][j]+A[i][1]*values[1][j]);
                change=fmax(change,CavityModel_scalar_norm(proposed[i][j]-stages[i][j]));
                stages[i][j]=proposed[i][j];
            }
        if (change>=tolerance) continue;
        for (int i=0;i<2;++i) {
            status=CavityModel_flow(stages[i],z+C[i]*h,Q,source,model,values[i]);
            if (status!=CAVITY_MODEL_OK) return status;
        }
        double residual=0.;
        for (int i=0;i<2;++i)
            for (int j=0;j<6;++j)
                residual=fmax(residual,CavityModel_scalar_norm(stages[i][j]-q[j]
                    -h*(A[i][0]*values[0][j]+A[i][1]*values[1][j])));
        if (residual>=tolerance) continue;
        xt_float_or_tpsa result[6],check[6];
        for (int j=0;j<6;++j) result[j]=q[j]+h*(values[0][j]+values[1][j])*.5;
        status=CavityModel_flow(result,z+h,Q,source,model,check);
        if (status!=CAVITY_MODEL_OK) return status;
        for (int j=0;j<6;++j) output[j]=result[j];
        return CAVITY_MODEL_OK;
    }
    return CAVITY_MODEL_ITERATIONS;
}

#endif
