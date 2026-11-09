// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_PARAXIAL_H
#define XTRACK_CAVITY_MODELS_PARAXIAL_H

// Phase-resolved paraxial cavity: both RF waves without
// averaging, at quadratic transverse order. In the gauge
// phi = (r^2/4) d_z e, A_perp = 0, A_z = -int e dt - (r^2/4) d_t e, with e the
// on-axis E_z, the Hamiltonian is linear in the field; per wave (amplitude a,
// wave number nu, phase phi, psi = T - nu Z + phi)
//     K = -p_z + a [f sin psi + (r^2/4)(E/p) f' cos psi + (r^2/4) f (nu E/p - 1) sin psi].
// One step is body(h/2) exp(Omega) body(h/2), Omega the first- and
// second-order Magnus generator of the RF terms in the drift frame
// X + tau U/p, T + tau/beta0, from eight Gauss nodes and their spectral
// integration matrix; Omega's flow is the implicit midpoint (symplectic). A
// jump df of the envelope (the parent's ends, a slice boundary closing against
// the field) is the impulse (r^2/4)(E/p) a df cos psi of each wave. Coordinates
// are those of the phase-resolved model, q = (X, Y, T, Px, Py, E), k = m c = 1.
// Drifts, static bodies and faces are shared with the phase-averaged model.
#include "xtrack/beam_elements/elements_src/cavity_models_averaged.h"

#define CAVITY_MAGNUS_NODES 8
#define CAVITY_MAGNUS_TERMS 31
#if (defined(XO_CONTEXT_CPU_SERIAL) || defined(XO_CONTEXT_CPU_OPENMP)) && !defined(XTRACK_TPSA_TRACK)
#define CAVITY_MAGNUS_CACHE_STEPS 128
#endif


////////////////////////////////////////
// Complex arithmetic and quadrature
////////////////////////////////////////
typedef struct { xt_float_or_tpsa re, im; } CavityComplex;

GPUFUN
CavityComplex CavityComplex_make(xt_float_or_tpsa_arg re, xt_float_or_tpsa_arg im) {
    CavityComplex out; out.re=re; out.im=im; return out;
}

GPUFUN
CavityComplex CavityComplex_mul(CavityComplex a, CavityComplex b) {
    return CavityComplex_make(a.re*b.re-a.im*b.im, a.re*b.im+a.im*b.re);
}

GPUFUN
CavityComplex CavityComplex_conj(CavityComplex a) {
    return CavityComplex_make(a.re, -a.im);
}

GPUFUN
void CavityComplex_add_scaled(CavityComplex *out, CavityComplex a, double re, double im) {
    out->re+=a.re*re-a.im*im;
    out->im+=a.re*im+a.im*re;
}

// Eight-point Gauss-Legendre nodes and weights on [-1, 1], and the spectral
// integration matrix S[i][j] = int_{-1}^{x_i} l_j for their Lagrange basis.
static const double CAVITY_MAGNUS_X[8]={-0.96028985649753623,-0.79666647741362674,
    -0.52553240991632899,-0.1834346424956498,0.1834346424956498,0.52553240991632899,
    0.79666647741362674,0.96028985649753623};
static const double CAVITY_MAGNUS_W[8]={0.10122853629037626,0.22238103445337447,
    0.31370664587788729,0.36268378337836198,0.36268378337836198,0.31370664587788729,
    0.22238103445337447,0.10122853629037626};
static const double CAVITY_MAGNUS_S[8][8]={
    {0.05061426814518813,-0.01821188661194015,0.012561662294060948,-0.0089660312261095029,
     0.0061569827366535598,-0.0038353505092739047,0.0019455153281185271,-0.00055501665423383845},
    {0.10951864353510864,0.11119051722668724,-0.027279592471563339,0.016299417716721101,
     -0.010430704178294307,0.0062795059709273379,-0.0031298698218978848,0.00088560460868447563},
    {0.097175071997825762,0.24171904999434633,0.15685332293894364,-0.031950206723756863,
     0.016743465440452328,-0.0092869317242089596,0.0044514295505697994,-0.0012376113905010307},
    {0.10373104194116247,0.21238698029669681,0.34134226854910724,0.18134189168918099,
     -0.032042082642050026,0.01448241312244454,-0.0063956286207215406,0.0017184731685297054},
    {0.099510063121846554,0.22877666307409601,0.29922423275544275,0.39472586602041201,
     0.18134189168918099,-0.027635622671219955,0.0099940541566776568,-0.00250250565078621},
    {0.10246614768087729,0.21792960490280467,0.32299357760209625,0.34594031793790966,
     0.39463399010211885,0.15685332293894364,-0.019338015540971864,0.0040534642925504973},
    {0.10034293168169178,0.22551090427527236,0.30742713990695995,0.37311448755665629,
     0.34638436566164088,0.34098623834945063,0.11119051722668724,-0.0082901072447323814},
    {0.1017835529446101,0.22043551912525594,0.31754199638716119,0.35652680064170842,
     0.37164981460447149,0.30114498358382634,0.24059292106531462,0.05061426814518813}};

// Omega = sum Re{e^{i n T} C} E^c p^-d M, M in {1, X^2+Y^2, XU+YV, U^2+V^2};
// columns n, c, d, M. The n = 1 terms are first order in the charge, the
// others second.
static const int CAVITY_MAGNUS_LAYOUT[CAVITY_MAGNUS_TERMS][4]={
    {0,0,1,1},{0,0,2,2},{0,0,3,1},{0,0,3,3},{0,0,4,2},{0,0,5,3},{0,1,2,1},{0,1,3,2},
    {0,1,4,3},{0,2,3,1},{0,2,4,2},{0,2,5,3},{1,0,0,0},{1,0,0,1},{1,0,1,2},{1,0,2,3},
    {1,1,1,1},{1,1,2,2},{1,1,3,3},{2,0,1,1},{2,0,2,2},{2,0,3,1},{2,0,3,3},{2,0,4,2},
    {2,0,5,3},{2,1,2,1},{2,1,3,2},{2,1,4,3},{2,2,3,1},{2,2,4,2},{2,2,5,3}};

GPUFUN
int CavityMagnus_slot(int n, int c, int d, int monomial) {
    for (int i=0;i<CAVITY_MAGNUS_TERMS;++i)
        if (CAVITY_MAGNUS_LAYOUT[i][0]==n && CAVITY_MAGNUS_LAYOUT[i][1]==c
            && CAVITY_MAGNUS_LAYOUT[i][2]==d && CAVITY_MAGNUS_LAYOUT[i][3]==monomial) return i;
    return -1;
}


////////////////////////////////////////
// Step generator
////////////////////////////////////////
typedef struct {
    CavityComplex term[CAVITY_MAGNUS_TERMS];
    // Unit-charge node samples, kept for the spin rotation (scalar CPU only).
    double tau[CAVITY_MAGNUS_NODES], weight[CAVITY_MAGNUS_NODES];
    double sample[3][CAVITY_MAGNUS_NODES][2];
} CavityMagnusStep;

// Unit-charge samples eps = sum_w c_w f, alpha1 = sum_w c_w (i f' + nu_w f),
// alpha0 = -sum_w c_w f, c_w = a_w exp(i(tau/beta0 - nu_w (z_mid + tau) + phi_w)).
GPUFUN
void CavityMagnus_samples(const CavityModelSource *source, xt_float_or_tpsa_arg z_mid,
        xt_float_or_tpsa_arg width, xt_float_or_tpsa tau[CAVITY_MAGNUS_NODES],
        CavityComplex out[3][CAVITY_MAGNUS_NODES]) {
    xt_float_or_tpsa envelope[5];
    for (int i=0;i<CAVITY_MAGNUS_NODES;++i) {
        tau[i]=CAVITY_MAGNUS_X[i]*width*.5;
        CavityModel_source_envelope(z_mid+tau[i], source, envelope);
        for (int k=0;k<3;++k) out[k][i]=CavityComplex_make(0., 0.);
        for (int wave=0;wave<2;++wave) {
            xt_float_or_tpsa amplitude=source->forward, nu=source->carrier, phase=0.;
            if (wave) {amplitude=source->backward; nu=-source->carrier; phase=source->counter_phase;}
            const xt_float_or_tpsa angle=tau[i]/source->reference_beta-nu*(z_mid+tau[i])+phase;
            const CavityComplex carrier=CavityComplex_make(amplitude*cos(angle), amplitude*sin(angle));
            const xt_float_or_tpsa f=envelope[0], df=envelope[1];
            out[0][i].re+=carrier.re*f; out[0][i].im+=carrier.im*f;
            out[1][i]=CavityComplex_make(out[1][i].re+carrier.re*nu*f-carrier.im*df,
                                         out[1][i].im+carrier.im*nu*f+carrier.re*df);
            out[2][i].re-=carrier.re*f; out[2][i].im-=carrier.im*f;
        }
    }
}

// sum_i w_i tau_i^p1 x_i Y_i, with Y the cumulative integral of tau^p2 y.
GPUFUN
CavityComplex CavityMagnus_double(const xt_float_or_tpsa tau[CAVITY_MAGNUS_NODES],
        const xt_float_or_tpsa weight[CAVITY_MAGNUS_NODES],
        CavityComplex cumulative[3][3][CAVITY_MAGNUS_NODES],
        CavityComplex samples[3][CAVITY_MAGNUS_NODES],
        int p1, int x, int conj_x, int p2, int y, int conj_y) {
    CavityComplex total=CavityComplex_make(0., 0.);
    for (int i=0;i<CAVITY_MAGNUS_NODES;++i) {
        CavityComplex a=samples[x][i], b=cumulative[p2][y][i];
        if (conj_x) a=CavityComplex_conj(a);
        if (conj_y) b=CavityComplex_conj(b);
        xt_float_or_tpsa scale=weight[i];
        for (int k=0;k<p1;++k) scale=scale*tau[i];
        const CavityComplex product=CavityComplex_mul(a, b);
        total.re+=scale*product.re; total.im+=scale*product.im;
    }
    return total;
}

GPUFUN
void CavityMagnus_add(CavityMagnusStep *step, int n, int c, int d, int monomial,
                      CavityComplex value, double re, double im) {
    CavityComplex_add_scaled(&step->term[CavityMagnus_slot(n, c, d, monomial)], value, re, im);
}

GPUFUN
void CavityMagnus_prepare_step(const CavityModelSource *source, xt_float_or_tpsa_arg z_mid,
                               xt_float_or_tpsa_arg width, CavityMagnusStep *step) {
    xt_float_or_tpsa tau[CAVITY_MAGNUS_NODES], weight[CAVITY_MAGNUS_NODES];
    CavityComplex samples[3][CAVITY_MAGNUS_NODES];
    CavityComplex cumulative[3][3][CAVITY_MAGNUS_NODES];
    CavityMagnus_samples(source, z_mid, width, tau, samples);
    const xt_float_or_tpsa half=width*.5;
    for (int i=0;i<CAVITY_MAGNUS_NODES;++i) {
        weight[i]=CAVITY_MAGNUS_W[i]*half;
        step->tau[i]=xt_float_or_tpsa_const_part(tau[i]);
        step->weight[i]=xt_float_or_tpsa_const_part(weight[i]);
        for (int k=0;k<3;++k) {
            step->sample[k][i][0]=xt_float_or_tpsa_const_part(samples[k][i].re);
            step->sample[k][i][1]=xt_float_or_tpsa_const_part(samples[k][i].im);
        }
    }
    for (int p=0;p<3;++p)
        for (int k=0;k<3;++k)
            for (int i=0;i<CAVITY_MAGNUS_NODES;++i) {
                CavityComplex sum=CavityComplex_make(0., 0.);
                for (int j=0;j<CAVITY_MAGNUS_NODES;++j) {
                    xt_float_or_tpsa scale=half*CAVITY_MAGNUS_S[i][j];
                    for (int m=0;m<p;++m) scale=scale*tau[j];
                    sum.re+=scale*samples[k][j].re; sum.im+=scale*samples[k][j].im;
                }
                cumulative[p][k][i]=sum;
            }
    for (int i=0;i<CAVITY_MAGNUS_TERMS;++i) step->term[i]=CavityComplex_make(0., 0.);
    // First order: energy kick and the transverse terms with r(tau)^2.
    for (int p=0;p<3;++p) {
        CavityComplex moment[3];
        for (int k=0;k<3;++k) {
            moment[k]=CavityComplex_make(0., 0.);
            for (int i=0;i<CAVITY_MAGNUS_NODES;++i) {
                xt_float_or_tpsa scale=weight[i];
                for (int m=0;m<p;++m) scale=scale*tau[i];
                moment[k].re+=scale*samples[k][i].re; moment[k].im+=scale*samples[k][i].im;
            }
        }
        if (p==0) CavityMagnus_add(step, 1, 0, 0, 0, moment[0], 0., -1.);
        const double scale=p==1 ? .5 : .25;
        CavityMagnus_add(step, 1, 1, 1+p, 1+p, moment[1], 0., -scale);
        CavityMagnus_add(step, 1, 0, p, 1+p, moment[2], 0., -scale);
    }
    // Second order, transverse x transverse: (1/16p)(tau2-tau1)
    // [Re(b1 conj b2) - Re(e^{2iT} b1 b2)][rho^2 + (tau1+tau2)mu/p + tau1 tau2 sigma/p^2],
    // b = (E/p) alpha1 + alpha0; the weights split into separable powers.
    const int split[3][2][2]={{{0,1},{1,0}},{{0,2},{2,0}},{{1,2},{2,1}}};
    for (int m=0;m<3;++m)
        for (int x=0;x<2;++x)
            for (int y=0;y<2;++y) {
                const int kx=x ? 1 : 2, ky=y ? 1 : 2;
                CavityComplex conj_part=CavityComplex_make(0., 0.), plain=CavityComplex_make(0., 0.);
                for (int s=0;s<2;++s) {
                    const double sign=s ? -1. : 1.;
                    CavityComplex_add_scaled(&conj_part, CavityMagnus_double(tau, weight, cumulative,
                        samples, split[m][s][0], kx, 0, split[m][s][1], ky, 1), sign, 0.);
                    CavityComplex_add_scaled(&plain, CavityMagnus_double(tau, weight, cumulative,
                        samples, split[m][s][0], kx, 0, split[m][s][1], ky, 0), sign, 0.);
                }
                CavityMagnus_add(step, 0, x+y, x+y+1+m, 1+m, conj_part, 1./16., 0.);
                CavityMagnus_add(step, 2, x+y, x+y+1+m, 1+m, plain, -1./16., 0.);
            }
    // Second order, energy x transverse through the (-T, E) bracket, with
    // dE H_perp = sum_m Im(e^{iT} gamma_m) M_m, gamma_m = sum_j s_j E^c p^-d tau^t z_j.
    const double pieces[7][6]={{1,-.25,0,3,0,1},{2,-.5,0,4,1,1},{2,-.5,2,4,1,1},{2,-.5,1,3,1,0},
                               {3,-.25,0,5,2,1},{3,-.5,2,5,2,1},{3,-.5,1,4,2,0}};
    for (int j=0;j<7;++j) {
        const int monomial=(int)pieces[j][0], c=(int)pieces[j][2], d=(int)pieces[j][3];
        const int t=(int)pieces[j][4], z=pieces[j][5]>.5 ? 1 : 2;
        const double scale=pieces[j][1]/4.;
        CavityComplex plain=CavityComplex_make(0., 0.), conj_part=CavityComplex_make(0., 0.);
        CavityComplex_add_scaled(&plain, CavityMagnus_double(tau, weight, cumulative, samples,
            0, 0, 0, t, z, 0), -1., 0.);
        CavityComplex_add_scaled(&plain, CavityMagnus_double(tau, weight, cumulative, samples,
            t, z, 0, 0, 0, 0), 1., 0.);
        CavityComplex_add_scaled(&conj_part, CavityMagnus_double(tau, weight, cumulative, samples,
            0, 0, 1, t, z, 0), -1., 0.);
        CavityComplex_add_scaled(&conj_part, CavityMagnus_double(tau, weight, cumulative, samples,
            t, z, 0, 0, 0, 1), 1., 0.);
        CavityMagnus_add(step, 2, c, d, monomial, plain, 0., -scale);
        CavityMagnus_add(step, 0, c, d, monomial, conj_part, 0., -scale);
    }
}

// Hard-edge impulse (r^2/4)(E/p) Re{e^{iT} sum_w a_w df e^{i(phi_w - nu_w Z)}}.
GPUFUN
CavityComplex CavityMagnus_edge_amplitude(const CavityModelSource *source,
        xt_float_or_tpsa_arg z, xt_float_or_tpsa_arg jump) {
    if (source->rf_profile==1) {
        xt_float_or_tpsa real[5], imag[5];
        CavityModel_profile(z, source, real, imag);
        return CavityComplex_make(jump*real[0], -jump*imag[0]);
    }
    const xt_float_or_tpsa forward=-source->carrier*z;
    const xt_float_or_tpsa backward=source->carrier*z+source->counter_phase;
    return CavityComplex_make(jump*(source->forward*cos(forward)+source->backward*cos(backward)),
                              jump*(source->forward*sin(forward)+source->backward*sin(backward)));
}


////////////////////////////////////////
// Flow
////////////////////////////////////////
// Gradient of Omega at q (charge Q); out = (dX, dY, dT, dU, dV, dE) of Omega.
GPUFUN
int CavityMagnus_gradient(const CavityComplex term[CAVITY_MAGNUS_TERMS], xt_float_or_tpsa_arg Q,
                          const xt_float_or_tpsa q[6], xt_float_or_tpsa out[6]) {
    const xt_float_or_tpsa X=q[0], Y=q[1], T=q[2], U=q[3], V=q[4], E=q[5];
    const xt_float_or_tpsa square=(E-1.)*(E+1.);
    if (xt_float_or_tpsa_const_part(square)<=0.) return CAVITY_MODEL_DOMAIN;
    const xt_float_or_tpsa p=sqrt(square), inverse=1./p;
    xt_float_or_tpsa power_e[3], power_p[8];
    power_e[0]=1.; power_e[1]=E; power_e[2]=E*E;
    power_p[0]=1.;
    for (int i=1;i<8;++i) power_p[i]=power_p[i-1]*inverse;
    const xt_float_or_tpsa c1=cos(T), s1=sin(T), c2=c1*c1-s1*s1, s2=2.*s1*c1;
    const xt_float_or_tpsa value[4]={1., X*X+Y*Y, X*U+Y*V, U*U+V*V};
    for (int i=0;i<6;++i) out[i]=0.;
    for (int i=0;i<CAVITY_MAGNUS_TERMS;++i) {
        const int n=CAVITY_MAGNUS_LAYOUT[i][0], c=CAVITY_MAGNUS_LAYOUT[i][1];
        const int d=CAVITY_MAGNUS_LAYOUT[i][2], monomial=CAVITY_MAGNUS_LAYOUT[i][3];
        xt_float_or_tpsa charge=Q;
        if (n!=1) charge=Q*Q;
        xt_float_or_tpsa cn=1., sn=0.;
        if (n==1) {cn=c1; sn=s1;}
        if (n==2) {cn=c2; sn=s2;}
        const xt_float_or_tpsa phase=charge*(term[i].re*cn-term[i].im*sn);
        const xt_float_or_tpsa phase_t=-(double)n*charge*(term[i].re*sn+term[i].im*cn);
        const xt_float_or_tpsa F=power_e[c]*power_p[d];
        xt_float_or_tpsa dF=-(double)d*power_e[c]*E*power_p[d+2];
        if (c==1) dF+=power_p[d];
        if (c==2) dF+=2.*E*power_p[d];
        const xt_float_or_tpsa weight=phase*F;
        if (monomial==1) {out[0]+=2.*X*weight; out[1]+=2.*Y*weight;}
        if (monomial==2) {out[0]+=U*weight; out[1]+=V*weight; out[3]+=X*weight; out[4]+=Y*weight;}
        if (monomial==3) {out[3]+=2.*U*weight; out[4]+=2.*V*weight;}
        out[2]+=phase_t*F*value[monomial];
        out[5]+=phase*dF*value[monomial];
    }
    return CAVITY_MODEL_OK;
}

// Implicit midpoint of the unit-time flow of Omega, pairs (X,U),(Y,V),(-T,E);
// sign -1 applies the inverse. Positions and momenta both move.
GPUFUN
int CavityMagnus_flow(xt_float_or_tpsa q[6], const CavityComplex term[CAVITY_MAGNUS_TERMS],
                      xt_float_or_tpsa_arg Q, double sign, int freeze_energy, int freeze_zeta) {
    xt_float_or_tpsa next[6], mid[6], g[6];
    for (int j=0;j<6;++j) next[j]=q[j];
    int converged=0, sweeps=CAVITY_MODEL_JET_SWEEPS;
    for (int iteration=0; iteration<CAVITY_AVERAGED_MIDPOINT_ITERATIONS+CAVITY_MODEL_JET_SWEEPS;
         ++iteration) {
        for (int j=0;j<6;++j) mid[j]=(q[j]+next[j])*.5;
        const int status=CavityMagnus_gradient(term, Q, mid, g);
        if (status!=CAVITY_MODEL_OK) return status;
        xt_float_or_tpsa candidate[6];
        candidate[0]=q[0]+sign*g[3]; candidate[1]=q[1]+sign*g[4];
        candidate[2]=q[2]; candidate[5]=q[5];
        if (!freeze_zeta) candidate[2]=q[2]-sign*g[5];
        candidate[3]=q[3]-sign*g[0]; candidate[4]=q[4]-sign*g[1];
        if (!freeze_energy) candidate[5]=q[5]+sign*g[2];
        double change=0.;
        for (int j=0;j<6;++j)
            change=fmax(change, CavityModel_primal_residual(candidate[j]-next[j])
                                /(1.+fabs(xt_float_or_tpsa_const_part(q[j]))));
        for (int j=0;j<6;++j) next[j]=candidate[j];
        if (converged) {
            if (--sweeps<0) break;
        } else if (change<=CAVITY_AVERAGED_MIDPOINT_TOLERANCE) {
            converged=1;
            if (sweeps==0) break;
        }
    }
    if (!converged) return CAVITY_MODEL_ITERATIONS;
    for (int j=0;j<6;++j) q[j]=next[j];
    return CAVITY_MODEL_OK;
}

GPUFUN
int CavityMagnus_edge(xt_float_or_tpsa q[6], const CavityModelSource *source,
                      xt_float_or_tpsa_arg Q, xt_float_or_tpsa_arg z, xt_float_or_tpsa_arg jump,
                      double sign, const CavitySpin *spin) {
    if (xt_float_or_tpsa_is_zero(jump)) return CAVITY_MODEL_OK;
    CavityComplex term[CAVITY_MAGNUS_TERMS];
    for (int i=0;i<CAVITY_MAGNUS_TERMS;++i) term[i]=CavityComplex_make(0., 0.);
    const CavityComplex amplitude=CavityMagnus_edge_amplitude(source, z, jump);
    term[CavityMagnus_slot(1, 1, 1, 1)]=CavityComplex_make(.25*amplitude.re, .25*amplitude.im);
    double before[6];
    for (int j=0;j<6;++j) before[j]=xt_float_or_tpsa_const_part(q[j]);
    const int status=CavityMagnus_flow(q, term, Q, sign, source->freeze_energy, source->freeze_zeta);
    if (status!=CAVITY_MODEL_OK || !spin || !spin->spin) return status;
    // Edge precession at the midpoint state: (y, -x, (xV - yU)/(E beta)) rho,
    // rho = c_E Re{e^{iT} C}/2.
    double mid[6];
    for (int j=0;j<6;++j) mid[j]=(before[j]+xt_float_or_tpsa_const_part(q[j]))*.5;
    const double E=mid[5], beta=sqrt(E*E-1.)/E, c_E=spin->anomaly+1./(E+1.);
    const double rho=sign*xt_float_or_tpsa_const_part(Q)*c_E*.5
        *(xt_float_or_tpsa_const_part(amplitude.re)*cos(mid[2])
          -xt_float_or_tpsa_const_part(amplitude.im)*sin(mid[2]));
    CavitySpin_rotate(spin->spin, mid[1]*rho, -mid[0]*rho,
                      (mid[0]*mid[4]-mid[1]*mid[3])/E*rho/beta);
    return CAVITY_MODEL_OK;
}


////////////////////////////////////////
// Spin
////////////////////////////////////////
// First- and second-order Magnus rotation of Thomas-BMT along the drift-frame
// orbit at the step's midpoint state m: rate per unit Z
//     w = (y rho + c_E V e/p, -x rho - c_E U e/p, ((xV - yU)/E) Im{e^{iT} l}),
//     rho = Im{e^{iT}(c_B alpha0/(2 beta) + c_E alpha1/2)},
//     l = c_E alpha1/(2 beta) - a gamma eps/(2(gamma+1)),
// plus the spin-orbit term 1/2 int int_{tau2<tau1} [grad w(1) J grad H(2) - grad w(2) J grad H(1)].
GPUFUN
void CavityMagnus_spin_step(const CavitySpin *spin, const CavityMagnusStep *step,
                            double Q, double sign, const double m[6]) {
    if (!spin || !spin->spin) return;
    const double X=m[0], Y=m[1], U=m[3], V=m[4], E=m[5];
    const double p=sqrt(E*E-1.), beta=p/E, a=spin->anomaly;
    const double c_B=a+1./E, c_E=a+1./(E+1.), dc_B=-1./(E*E), dc_E=-1./((E+1.)*(E+1.));
    const double dbeta=1./(p*E*E), c_A=a*E/(E+1.), dc_A=a/((E+1.)*(E+1.));
    const double axial=c_E/p, axial_E=dc_E/p-c_E*E/(p*p*p);
    const double cT=cos(m[2]), sT=sin(m[2]), ang=(X*V-Y*U)/E;
    double w[3][CAVITY_MAGNUS_NODES], gw[3][6][CAVITY_MAGNUS_NODES], rate[6][CAVITY_MAGNUS_NODES];
    for (int i=0;i<CAVITY_MAGNUS_NODES;++i) {
        const double tau=step->tau[i];
        double e_re[3], e_im[3];
        for (int k=0;k<3;++k) {
            const double re=Q*step->sample[k][i][0], im=Q*step->sample[k][i][1];
            e_re[k]=re*cT-im*sT; e_im[k]=re*sT+im*cT;
        }
        // e^{iT} times eps (0), alpha1 (1), alpha0 (2): real and imaginary parts.
        const double core_re=c_B*e_re[2]/(2.*beta)+c_E*e_re[1]*.5;
        const double core_im=c_B*e_im[2]/(2.*beta)+c_E*e_im[1]*.5;
        const double dcore_im=(dc_B/(2.*beta)-c_B*dbeta/(2.*beta*beta))*e_im[2]+dc_E*e_im[1]*.5;
        const double rho=core_im, rho_T=core_re, rho_E=dcore_im;
        const double e=e_re[0], e_T=-e_im[0];
        const double lz_im=-c_A*.5*e_im[0]+c_E/(2.*beta)*e_im[1];
        const double lz_re=-c_A*.5*e_re[0]+c_E/(2.*beta)*e_re[1];
        const double lz_E=-dc_A*.5*e_im[0]+(dc_E/(2.*beta)-c_E*dbeta/(2.*beta*beta))*e_im[1];
        const double x=X+tau*U/p, y=Y+tau*V/p;
        w[0][i]=y*rho+V*axial*e; w[1][i]=-x*rho-U*axial*e; w[2][i]=ang*lz_im;
        const double row0[6]={0., rho, y*rho_T+V*axial*e_T, 0., tau/p*rho+axial*e,
                              y*rho_E-tau*V*E/(p*p*p)*rho+V*axial_E*e};
        const double row1[6]={-rho, 0., -x*rho_T-U*axial*e_T, -tau/p*rho-axial*e, 0.,
                              -x*rho_E+tau*U*E/(p*p*p)*rho-U*axial_E*e};
        const double row2[6]={V/E*lz_im, -U/E*lz_im, ang*lz_re, -Y/E*lz_im, X/E*lz_im,
                              -ang/E*lz_im+ang*lz_E};
        for (int j=0;j<6;++j) {gw[0][j][i]=row0[j]; gw[1][j][i]=row1[j]; gw[2][j][i]=row2[j];}
        // J grad H_I at the node.
        const double b_re=E/p*e_re[1]+e_re[2], b_im=E/p*e_im[1]+e_im[2];
        const double R=x*x+y*y, imb=.25*b_im;
        const double H_X=2.*x*imb, H_Y=2.*y*imb, H_U=H_X*tau/p, H_V=H_Y*tau/p;
        const double H_T=e_re[0]+.25*b_re*R;
        const double H_E=.25*(-e_im[1]/(p*p*p)*R-b_im*2.*tau*E*(x*U+y*V)/(p*p*p));
        rate[0][i]=H_U; rate[1][i]=H_V; rate[2][i]=-H_E;
        rate[3][i]=-H_X; rate[4][i]=-H_Y; rate[5][i]=H_T;
    }
    const double half=(step->tau[CAVITY_MAGNUS_NODES-1]-step->tau[0])
                      /(CAVITY_MAGNUS_X[CAVITY_MAGNUS_NODES-1]-CAVITY_MAGNUS_X[0]);
    double theta[3]={0., 0., 0.};
    for (int i=0;i<CAVITY_MAGNUS_NODES;++i) {
        double cum_rate[6]={0.,0.,0.,0.,0.,0.}, cum_gw[3][6];
        for (int a3=0;a3<3;++a3) for (int j=0;j<6;++j) cum_gw[a3][j]=0.;
        for (int l=0;l<CAVITY_MAGNUS_NODES;++l) {
            const double s=CAVITY_MAGNUS_S[i][l]*half;
            for (int j=0;j<6;++j) {
                cum_rate[j]+=s*rate[j][l];
                for (int a3=0;a3<3;++a3) cum_gw[a3][j]+=s*gw[a3][j][l];
            }
        }
        for (int a3=0;a3<3;++a3) {
            double second=0.;
            for (int j=0;j<6;++j) second+=gw[a3][j][i]*cum_rate[j]-cum_gw[a3][j]*rate[j][i];
            theta[a3]+=step->weight[i]*(w[a3][i]+.5*second);
        }
    }
    CavitySpin_rotate(spin->spin, sign*theta[0], sign*theta[1], sign*theta[2]);
}


////////////////////////////////////////
// Interval
////////////////////////////////////////
typedef struct {
    int valid;
    int64_t cells;
    double key[6];
#ifdef CAVITY_MAGNUS_CACHE_STEPS
    CavityMagnusStep step[CAVITY_MAGNUS_CACHE_STEPS];
#endif
} CavityMagnusCache;

GPUFUN
void CavityMagnus_key(const CavityModelSource *source, double key[6]) {
    key[0]=xt_float_or_tpsa_const_part(source->forward);
    key[1]=xt_float_or_tpsa_const_part(source->backward);
    key[2]=xt_float_or_tpsa_const_part(source->carrier);
    key[3]=xt_float_or_tpsa_const_part(source->counter_phase);
    key[4]=xt_float_or_tpsa_const_part(source->k);
    key[5]=source->reference_beta;
}

// Interval [left, right] of the parent (fractions), cells steps, uniform or
// graded (integrator 7). Its ends are envelope jumps: the parent's own ends,
// or a slice boundary closing against the field, so adjacent slices telescope.
GPUFUN
int CavityModel_paraxial_interval(const xt_float_or_tpsa q_in[6],
        xt_float_or_tpsa_arg Q, const CavityModelSource *source, int64_t cells,
        double left, double right, int reverse, int entry_face, int exit_face,
        xt_float_or_tpsa_arg reference_momentum, int integrator,
        xt_float_or_tpsa_arg grading, const CavitySpin *spin, CavityMagnusCache *cache,
        xt_float_or_tpsa q[6]) {
    if (cells<=0 || !(right>left)) return CAVITY_MODEL_ARGUMENT;
    if (!xt_float_or_tpsa_is_zero(source->dc_gradient)) return CAVITY_MODEL_ARGUMENT;
    // Supported static fields, as for the frozen model: a uniform solenoid
    // with its faces off, or a normal quadrupole with its own faces.
    for (int64_t i=0;i<=source->order;++i)
        if ((i!=1 && !xt_float_or_tpsa_is_zero(CavityModel_static_coefficient(source,i,0)))
            || !xt_float_or_tpsa_is_zero(CavityModel_static_coefficient(source,i,1)))
            return CAVITY_MODEL_ARGUMENT;
    CavityAveragedStatic field;
    field.solenoid=Q*source->magnetic_scale*source->ks;
    field.gradient=0.;
    if (source->order>=1)
        field.gradient=Q*source->magnetic_scale
            *CavityModel_static_coefficient(source,1,0)/source->length;
    field.reference_momentum=reference_momentum;
    const int static_present=!xt_float_or_tpsa_is_zero(field.solenoid)
                             || !xt_float_or_tpsa_is_zero(field.gradient);
    if (!xt_float_or_tpsa_is_zero(field.solenoid)
        && (!xt_float_or_tpsa_is_zero(field.gradient) || entry_face || exit_face))
        return CAVITY_MODEL_ARGUMENT;
    if (static_present && spin && spin->spin) return CAVITY_MODEL_ARGUMENT;
    entry_face=entry_face && left==0.;
    exit_face=exit_face && right==1.;
    for (int i=0;i<6;++i) q[i]=q_in[i];
    const xt_float_or_tpsa length=source->k*source->length;
    const xt_float_or_tpsa z_left=length*(left-.5), z_right=length*(right-.5);
    const xt_float_or_tpsa h=(z_right-z_left)/cells;
    const xt_float_or_tpsa grading_log=CavityModel_log1p(grading);
    const int graded=integrator==7 && xt_float_or_tpsa_const_part(grading)>1.e-6;
    const double sign=reverse ? -1. : 1.;
    const double Q_spin=xt_float_or_tpsa_const_part(Q);
    // Reference-only step generators are cached once per call on the CPU.
    int cached=0;
#ifdef CAVITY_MAGNUS_CACHE_STEPS
    if (cache && cells<=CAVITY_MAGNUS_CACHE_STEPS) {
        double key[6];
        CavityMagnus_key(source, key);
        int same=cache->valid && cache->cells==cells;
        for (int i=0;i<6 && same;++i) same=key[i]==cache->key[i];
        if (!same) {
            for (int64_t j=0;j<cells;++j) {
                xt_float_or_tpsa start=z_left+j*h, end=z_left+(j+1.)*h;
                if (graded) {
                    start=z_left+(z_right-z_left)*CavityModel_expm1(j*grading_log/cells)/grading;
                    end=z_left+(z_right-z_left)*CavityModel_expm1((j+1.)*grading_log/cells)/grading;
                }
                CavityMagnus_prepare_step(source, (start+end)*.5, end-start, &cache->step[j]);
            }
            cache->valid=1; cache->cells=cells;
            for (int i=0;i<6;++i) cache->key[i]=key[i];
        }
        cached=1;
    }
#else
    (void)cache;
#endif
    xt_float_or_tpsa envelope[5];
    CavityModel_source_envelope(z_left, source, envelope);
    const xt_float_or_tpsa f_left=envelope[0];
    CavityModel_source_envelope(z_right, source, envelope);
    const xt_float_or_tpsa f_right=envelope[0];
    int status=CAVITY_MODEL_OK;
    if ((!reverse && entry_face) || (reverse && exit_face))
        status=CavityModel_averaged_static_face(q, source, &field, reverse, reverse);
    // The inverse word crosses the same steps backwards with negated generators.
    if (status==CAVITY_MODEL_OK) {
        if (reverse) status=CavityMagnus_edge(q, source, Q, z_right, -f_right, -1., spin);
        else status=CavityMagnus_edge(q, source, Q, z_left, f_left, 1., spin);
    }
    for (int64_t n=0; n<cells && status==CAVITY_MODEL_OK; ++n) {
        const int64_t j=reverse ? cells-1-n : n;
        xt_float_or_tpsa start=z_left+j*h, end=z_left+(j+1.)*h;
        if (graded) {
            start=z_left+(z_right-z_left)*CavityModel_expm1(j*grading_log/cells)/grading;
            end=z_left+(z_right-z_left)*CavityModel_expm1((j+1.)*grading_log/cells)/grading;
        }
        const xt_float_or_tpsa width=end-start;
        CavityMagnusStep local;
        const CavityMagnusStep *step=&local;
#ifdef CAVITY_MAGNUS_CACHE_STEPS
        if (cached) step=&cache->step[j];
#endif
        if (!cached) CavityMagnus_prepare_step(source, (start+end)*.5, width, &local);
        status=CavityModel_averaged_body(q, .5*sign*width, source, &field);
        double before[6], mid[6];
        for (int i=0;i<6;++i) before[i]=xt_float_or_tpsa_const_part(q[i]);
        if (status==CAVITY_MODEL_OK)
            status=CavityMagnus_flow(q, step->term, Q, sign, source->freeze_energy,
                                     source->freeze_zeta);
        if (status==CAVITY_MODEL_OK && spin && spin->spin) {
            for (int i=0;i<6;++i) mid[i]=(before[i]+xt_float_or_tpsa_const_part(q[i]))*.5;
            CavityMagnus_spin_step(spin, step, Q_spin, sign, mid);
        }
        if (status==CAVITY_MODEL_OK)
            status=CavityModel_averaged_body(q, .5*sign*width, source, &field);
    }
    if (status==CAVITY_MODEL_OK) {
        if (reverse) status=CavityMagnus_edge(q, source, Q, z_left, f_left, -1., spin);
        else status=CavityMagnus_edge(q, source, Q, z_right, -f_right, 1., spin);
    }
    if (status==CAVITY_MODEL_OK && ((!reverse && exit_face) || (reverse && entry_face)))
        status=CavityModel_averaged_static_face(q, source, &field, !reverse, reverse);
    return status;
}

#endif
