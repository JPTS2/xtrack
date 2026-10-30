// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_AVERAGED_H
#define XTRACK_CAVITY_MODELS_AVERAGED_H

// Shared physical source and finite-step conventions.
#include "xtrack/beam_elements/elements_src/cavity_models.h"
#include "xtrack/headers/constants.h"

// Common physical data and signs, without generated derivative expressions.
// All averaged products retain their own scalar construction and finite word.
typedef struct {
    xt_float_or_tpsa half_span,entrance,exit,slow,fast,carrier,counter_phase;
    int64_t characteristic_steps,factor_steps;
    double tolerance;
    int64_t frozen_coordinates;
    double reference_beta;
    xt_float_or_tpsa dc_gradient;
    const CavityModelSource *static_source;
    xt_float_or_tpsa static_charge;
    // Profile geometry is independent of magnetic-source admission. In
    // particular RF-only products must preserve an explicitly Gaussian RF.
    const CavityModelSource *profile_source;
} CavityAveragedSource;
GPUFUN
void CavityAveraged_constrain(xt_float_or_tpsa flow[6],const CavityAveragedSource *source,int physical_plane) {
    for(int i=0;i<6;++i)if(source->frozen_coordinates&((int64_t)1<<i))flow[i]=0.;
    if(physical_plane&&(source->frozen_coordinates&((int64_t)1<<2)))flow[2]=1./source->reference_beta;
}
GPUFUN
void CavityAveraged_J(const xt_float_or_tpsa g[6],xt_float_or_tpsa f[6]) {
    f[0]=g[3];f[1]=g[4];f[2]=-g[5];f[3]=-g[0];f[4]=-g[1];f[5]=g[2];
}
GPUFUN
int CavityAveraged_J_index(int row){return row<3?row+3:row-3;}
GPUFUN
double CavityAveraged_J_sign(int row){return row==0||row==1||row==5?1.:-1.;}

// The same complete compact Maxwell potential/flow as the phase-resolved
// owner. Signed Q has already entered the RF and static coefficients, so the
// Hamiltonian receives Q=1. Analysis masks apply only to outer factors.
GPUFUN
void CavityAveraged_physical_source(const CavityAveragedSource *source,CavityModelSource *out) {
    CavityModelSource physical={0};physical.k=1.;physical.length=2.*source->half_span;
    const CavityModelSource *profile=source->profile_source?source->profile_source:source->static_source;
    if(profile)physical=*profile;
    physical.magnetic_scale=0.;physical.ks=0.;physical.order=-1;
    physical.knl=NULL;physical.ksl=NULL;physical.knl_tpsa=NULL;physical.ksl_tpsa=NULL;
    if(source->static_source) {
        const CavityModelSource *magnet=source->static_source;
        physical.k=magnet->k;physical.length=magnet->length;
        physical.magnetic_scale=magnet->magnetic_scale*source->static_charge;
        physical.ks=magnet->ks;physical.order=magnet->order;
        physical.knl=magnet->knl;physical.ksl=magnet->ksl;
        physical.knl_tpsa=magnet->knl_tpsa;physical.ksl_tpsa=magnet->ksl_tpsa;
    }
    physical.half_span=source->half_span;physical.carrier=source->carrier;
    physical.counter_phase=source->counter_phase;physical.forward=source->slow;physical.backward=0.;
    physical.dc_gradient=source->dc_gradient;physical.reference_beta=source->reference_beta;
    physical.freeze_x=0;physical.freeze_y=0;physical.freeze_zeta=0;
    physical.freeze_px=0;physical.freeze_py=0;physical.freeze_energy=0;
    *out=physical;
}
GPUFUN
int CavityAveraged_slow(const xt_float_or_tpsa q[6],xt_float_or_tpsa_arg z,
        const CavityAveragedSource *source,xt_float_or_tpsa out[6]) {
    CavityModelSource physical;CavityAveraged_physical_source(source,&physical);
    int status=CavityModel_flow(q,z,1.,&physical,CAVITY_MODEL_FULL,out);
    if(status!=CAVITY_MODEL_OK)return status;
    CavityAveraged_constrain(out,source,1);
    return CavityModel_vector_finite(out,6)?CAVITY_MODEL_OK:CAVITY_MODEL_NONFINITE;
}

// Diagnostic literal scalar exporters use these unchanged P16 profile inputs
// and positive RF-only slow shell. Production divided-jet consumers evaluate
// their potentials directly and never require an expanded source derivative.
GPUFUN
int CavityAveraged_data(const xt_float_or_tpsa q[6],xt_float_or_tpsa_arg z,
        const CavityAveragedSource *source,xt_float_or_tpsa profile[5]) {
    if(source->static_source)return CAVITY_MODEL_ARGUMENT;
    CavityModelSource physical;CavityAveraged_physical_source(source,&physical);
    CavityModel_source_envelope(z,&physical,profile);
    CavityModelPotential potential;CavityModel_rf_potential(q,z,&physical,CAVITY_MODEL_FULL,&potential);
    xt_float_or_tpsa px=q[3]-potential.a[0],py=q[4]-potential.a[1];
    xt_float_or_tpsa square=(q[5]-1.)*(q[5]+1.)-px*px-py*py;
    if(!CavityModel_vector_finite(q,6)||!CavityModel_scalar_finite(square))return CAVITY_MODEL_NONFINITE;
    if(xt_float_or_tpsa_const_part(q[5])<=1.||xt_float_or_tpsa_const_part(square)<=0.)return CAVITY_MODEL_DOMAIN;
    return CAVITY_MODEL_OK;
}

// Shared fixed-size Gauss integration.
// Fixed-size G2 routines share one bounded defining-equation recipe.
// Separate call graphs avoid recursion and avoid 358-series scratch
// in the six-state slow/body/boundary factors. Unknowns u,v are stage
// increments about the cell base; residuals never subtract a large base
// energy/clock. The existing initial-flow array doubles as full-stage scratch.
#define CAVITY_AVERAGED_GAUSS(NAME, RHS, COUNT) \
GPUFUN \
int NAME(xt_float_or_tpsa *value, \
        xt_float_or_tpsa_arg start, xt_float_or_tpsa_arg end, \
        int64_t steps, const CavityAveragedSource *source) { \
    if (steps<=0 || source->tolerance<=0.) return CAVITY_MODEL_ARGUMENT; \
    const double c0=.5-sqrt(3.)/6.,c1=.5+sqrt(3.)/6.; \
    const double a01=.25-sqrt(3.)/6.,a10=.25+sqrt(3.)/6.; \
    xt_float_or_tpsa h=(end-start)/steps; \
    if (CavityModel_scalar_norm(h)==0.) return CAVITY_MODEL_OK; \
    for (int64_t cell=0;cell<steps;++cell) { \
        xt_float_or_tpsa z=start+cell*h; \
        xt_float_or_tpsa u[COUNT],v[COUNT],f[COUNT],g[COUNT],initial[COUNT]; \
        int status=RHS(value,z,source,initial); \
        if (status!=CAVITY_MODEL_OK) return status; \
        for (int i=0;i<COUNT;++i) { \
            u[i]=c0*h*initial[i]; \
            v[i]=c1*h*initial[i]; \
        } \
        int converged=0; \
        for (int iteration=0;iteration<80;++iteration) { \
            for (int i=0;i<COUNT;++i) initial[i]=value[i]+u[i]; \
            status=RHS(initial,z+c0*h,source,f); \
            if (status!=CAVITY_MODEL_OK) return status; \
            for (int i=0;i<COUNT;++i) initial[i]=value[i]+v[i]; \
            status=RHS(initial,z+c1*h,source,g); \
            if (status!=CAVITY_MODEL_OK) return status; \
            double defect=0.; \
            for (int i=0;i<COUNT;++i) { \
                xt_float_or_tpsa next_u=h*(.25*f[i]+a01*g[i]); \
                xt_float_or_tpsa next_v=h*(a10*f[i]+.25*g[i]); \
                double du=CavityModel_scalar_residual(u[i]-next_u,value[i]+next_u,i<6); \
                double dv=CavityModel_scalar_residual(v[i]-next_v,value[i]+next_v,i<6); \
                if (du>defect) defect=du; \
                if (dv>defect) defect=dv; \
                u[i]=next_u; v[i]=next_v; \
            } \
            if (defect<=source->tolerance) { converged=1; break; } \
        } \
    /* Re-evaluate the fresh increment defining equations, not a stale */ \
    /* update metric. Every coefficient enters the weighted infinity norm. */ \
        for (int i=0;i<COUNT;++i) initial[i]=value[i]+u[i]; \
        status=RHS(initial,z+c0*h,source,f); \
        if (status!=CAVITY_MODEL_OK) return status; \
        for (int i=0;i<COUNT;++i) initial[i]=value[i]+v[i]; \
        status=RHS(initial,z+c1*h,source,g); \
        if (status!=CAVITY_MODEL_OK) return status; \
        if (!converged) { \
            CavityModel_debug_stages(#NAME,value,u,v,f,g,h,COUNT); \
            return CAVITY_MODEL_ITERATIONS; \
        } \
        for (int i=0;i<COUNT;++i) { \
            if (CavityModel_scalar_residual(u[i]-h*(.25*f[i]+a01*g[i]),value[i]+u[i],i<6) \
                    >source->tolerance \
                || CavityModel_scalar_residual(v[i]-h*(a10*f[i]+.25*g[i]),value[i]+v[i],i<6) \
                    >source->tolerance) { \
                CavityModel_debug_stages(#NAME,value,u,v,f,g,h,COUNT); \
                return CAVITY_MODEL_ITERATIONS; \
            } \
        } \
        for (int i=0;i<COUNT;++i) value[i]+=.5*h*(f[i]+g[i]); \
    } \
    return CAVITY_MODEL_OK; \
}

// Fixed rank-four divided-jet multi-index plan (indices, not physics data).
GPUFUN
int CavityFull_count(int rank) {
    const int counts[5]={1,7,28,84,210};return counts[rank];
}
GPUFUN
void CavityFull_product(const xt_float_or_tpsa *a,const xt_float_or_tpsa *b,
        int rank,xt_float_or_tpsa *out) {
    const int offset[211]={
        0,1,3,5,7,9,11,13,16,20,23,27,31,34,38,42,46,49,53,57,61,65,68,72,
        76,80,84,88,91,95,101,107,111,117,125,131,137,143,147,153,161,167,175,183,189,195,201,207,
        211,217,225,231,239,247,253,261,269,277,283,289,295,301,307,311,317,325,331,339,347,353,361,369,
        377,383,391,399,407,415,421,427,433,439,445,451,455,460,468,477,485,490,498,510,522,530,539,551,
        560,568,576,581,589,601,613,621,633,649,661,673,685,693,702,714,723,735,747,756,764,772,780,785,
        793,805,817,825,837,853,865,877,889,897,909,925,937,953,969,981,993,1005,1017,1025,1034,1046,1055,1067,
        1079,1088,1100,1112,1124,1133,1141,1149,1157,1165,1170,1178,1190,1202,1210,1222,1238,1250,1262,1274,1282,1294,1310,1322,
        1338,1354,1366,1378,1390,1402,1410,1422,1438,1450,1466,1482,1494,1510,1526,1542,1554,1566,1578,1590,1602,1610,1619,1631,
        1640,1652,1664,1673,1685,1697,1709,1718,1730,1742,1754,1766,1775,1783,1791,1799,1807,1815,1820
    };
    const int left[1820]={
        0,0,1,0,2,0,3,0,4,0,5,0,6,0,1,7,0,1,2,8,0,2,9,0,
        1,3,10,0,2,3,11,0,3,12,0,1,4,13,0,2,4,14,0,3,4,15,0,4,
        16,0,1,5,17,0,2,5,18,0,3,5,19,0,4,5,20,0,5,21,0,1,6,22,
        0,2,6,23,0,3,6,24,0,4,6,25,0,5,6,26,0,6,27,0,1,7,28,0,
        1,7,2,8,29,0,1,2,8,9,30,0,2,9,31,0,1,7,3,10,32,0,1,2,
        8,3,10,11,33,0,2,9,3,11,34,0,1,3,10,12,35,0,2,3,11,12,36,0,
        3,12,37,0,1,7,4,13,38,0,1,2,8,4,13,14,39,0,2,9,4,14,40,0,
        1,3,10,4,13,15,41,0,2,3,11,4,14,15,42,0,3,12,4,15,43,0,1,4,
        13,16,44,0,2,4,14,16,45,0,3,4,15,16,46,0,4,16,47,0,1,7,5,17,
        48,0,1,2,8,5,17,18,49,0,2,9,5,18,50,0,1,3,10,5,17,19,51,0,
        2,3,11,5,18,19,52,0,3,12,5,19,53,0,1,4,13,5,17,20,54,0,2,4,
        14,5,18,20,55,0,3,4,15,5,19,20,56,0,4,16,5,20,57,0,1,5,17,21,
        58,0,2,5,18,21,59,0,3,5,19,21,60,0,4,5,20,21,61,0,5,21,62,0,
        1,7,6,22,63,0,1,2,8,6,22,23,64,0,2,9,6,23,65,0,1,3,10,6,
        22,24,66,0,2,3,11,6,23,24,67,0,3,12,6,24,68,0,1,4,13,6,22,25,
        69,0,2,4,14,6,23,25,70,0,3,4,15,6,24,25,71,0,4,16,6,25,72,0,
        1,5,17,6,22,26,73,0,2,5,18,6,23,26,74,0,3,5,19,6,24,26,75,0,
        4,5,20,6,25,26,76,0,5,21,6,26,77,0,1,6,22,27,78,0,2,6,23,27,
        79,0,3,6,24,27,80,0,4,6,25,27,81,0,5,6,26,27,82,0,6,27,83,0,
        1,7,28,84,0,1,7,28,2,8,29,85,0,1,7,2,8,29,9,30,86,0,1,2,
        8,9,30,31,87,0,2,9,31,88,0,1,7,28,3,10,32,89,0,1,7,2,8,29,
        3,10,32,11,33,90,0,1,2,8,9,30,3,10,11,33,34,91,0,2,9,31,3,11,
        34,92,0,1,7,3,10,32,12,35,93,0,1,2,8,3,10,11,33,12,35,36,94,0,
        2,9,3,11,34,12,36,95,0,1,3,10,12,35,37,96,0,2,3,11,12,36,37,97,
        0,3,12,37,98,0,1,7,28,4,13,38,99,0,1,7,2,8,29,4,13,38,14,39,
        100,0,1,2,8,9,30,4,13,14,39,40,101,0,2,9,31,4,14,40,102,0,1,7,
        3,10,32,4,13,38,15,41,103,0,1,2,8,3,10,11,33,4,13,14,39,15,41,42,
        104,0,2,9,3,11,34,4,14,40,15,42,105,0,1,3,10,12,35,4,13,15,41,43,
        106,0,2,3,11,12,36,4,14,15,42,43,107,0,3,12,37,4,15,43,108,0,1,7,
        4,13,38,16,44,109,0,1,2,8,4,13,14,39,16,44,45,110,0,2,9,4,14,40,
        16,45,111,0,1,3,10,4,13,15,41,16,44,46,112,0,2,3,11,4,14,15,42,16,
        45,46,113,0,3,12,4,15,43,16,46,114,0,1,4,13,16,44,47,115,0,2,4,14,
        16,45,47,116,0,3,4,15,16,46,47,117,0,4,16,47,118,0,1,7,28,5,17,48,
        119,0,1,7,2,8,29,5,17,48,18,49,120,0,1,2,8,9,30,5,17,18,49,50,
        121,0,2,9,31,5,18,50,122,0,1,7,3,10,32,5,17,48,19,51,123,0,1,2,
        8,3,10,11,33,5,17,18,49,19,51,52,124,0,2,9,3,11,34,5,18,50,19,52,
        125,0,1,3,10,12,35,5,17,19,51,53,126,0,2,3,11,12,36,5,18,19,52,53,
        127,0,3,12,37,5,19,53,128,0,1,7,4,13,38,5,17,48,20,54,129,0,1,2,
        8,4,13,14,39,5,17,18,49,20,54,55,130,0,2,9,4,14,40,5,18,50,20,55,
        131,0,1,3,10,4,13,15,41,5,17,19,51,20,54,56,132,0,2,3,11,4,14,15,
        42,5,18,19,52,20,55,56,133,0,3,12,4,15,43,5,19,53,20,56,134,0,1,4,
        13,16,44,5,17,20,54,57,135,0,2,4,14,16,45,5,18,20,55,57,136,0,3,4,
        15,16,46,5,19,20,56,57,137,0,4,16,47,5,20,57,138,0,1,7,5,17,48,21,
        58,139,0,1,2,8,5,17,18,49,21,58,59,140,0,2,9,5,18,50,21,59,141,0,
        1,3,10,5,17,19,51,21,58,60,142,0,2,3,11,5,18,19,52,21,59,60,143,0,
        3,12,5,19,53,21,60,144,0,1,4,13,5,17,20,54,21,58,61,145,0,2,4,14,
        5,18,20,55,21,59,61,146,0,3,4,15,5,19,20,56,21,60,61,147,0,4,16,5,
        20,57,21,61,148,0,1,5,17,21,58,62,149,0,2,5,18,21,59,62,150,0,3,5,
        19,21,60,62,151,0,4,5,20,21,61,62,152,0,5,21,62,153,0,1,7,28,6,22,
        63,154,0,1,7,2,8,29,6,22,63,23,64,155,0,1,2,8,9,30,6,22,23,64,
        65,156,0,2,9,31,6,23,65,157,0,1,7,3,10,32,6,22,63,24,66,158,0,1,
        2,8,3,10,11,33,6,22,23,64,24,66,67,159,0,2,9,3,11,34,6,23,65,24,
        67,160,0,1,3,10,12,35,6,22,24,66,68,161,0,2,3,11,12,36,6,23,24,67,
        68,162,0,3,12,37,6,24,68,163,0,1,7,4,13,38,6,22,63,25,69,164,0,1,
        2,8,4,13,14,39,6,22,23,64,25,69,70,165,0,2,9,4,14,40,6,23,65,25,
        70,166,0,1,3,10,4,13,15,41,6,22,24,66,25,69,71,167,0,2,3,11,4,14,
        15,42,6,23,24,67,25,70,71,168,0,3,12,4,15,43,6,24,68,25,71,169,0,1,
        4,13,16,44,6,22,25,69,72,170,0,2,4,14,16,45,6,23,25,70,72,171,0,3,
        4,15,16,46,6,24,25,71,72,172,0,4,16,47,6,25,72,173,0,1,7,5,17,48,
        6,22,63,26,73,174,0,1,2,8,5,17,18,49,6,22,23,64,26,73,74,175,0,2,
        9,5,18,50,6,23,65,26,74,176,0,1,3,10,5,17,19,51,6,22,24,66,26,73,
        75,177,0,2,3,11,5,18,19,52,6,23,24,67,26,74,75,178,0,3,12,5,19,53,
        6,24,68,26,75,179,0,1,4,13,5,17,20,54,6,22,25,69,26,73,76,180,0,2,
        4,14,5,18,20,55,6,23,25,70,26,74,76,181,0,3,4,15,5,19,20,56,6,24,
        25,71,26,75,76,182,0,4,16,5,20,57,6,25,72,26,76,183,0,1,5,17,21,58,
        6,22,26,73,77,184,0,2,5,18,21,59,6,23,26,74,77,185,0,3,5,19,21,60,
        6,24,26,75,77,186,0,4,5,20,21,61,6,25,26,76,77,187,0,5,21,62,6,26,
        77,188,0,1,7,6,22,63,27,78,189,0,1,2,8,6,22,23,64,27,78,79,190,0,
        2,9,6,23,65,27,79,191,0,1,3,10,6,22,24,66,27,78,80,192,0,2,3,11,
        6,23,24,67,27,79,80,193,0,3,12,6,24,68,27,80,194,0,1,4,13,6,22,25,
        69,27,78,81,195,0,2,4,14,6,23,25,70,27,79,81,196,0,3,4,15,6,24,25,
        71,27,80,81,197,0,4,16,6,25,72,27,81,198,0,1,5,17,6,22,26,73,27,78,
        82,199,0,2,5,18,6,23,26,74,27,79,82,200,0,3,5,19,6,24,26,75,27,80,
        82,201,0,4,5,20,6,25,26,76,27,81,82,202,0,5,21,6,26,77,27,82,203,0,
        1,6,22,27,78,83,204,0,2,6,23,27,79,83,205,0,3,6,24,27,80,83,206,0,
        4,6,25,27,81,83,207,0,5,6,26,27,82,83,208,0,6,27,83,209
    };
    const int right[1820]={
        0,1,0,2,0,3,0,4,0,5,0,6,0,7,1,0,8,2,1,0,9,2,0,10,
        3,1,0,11,3,2,0,12,3,0,13,4,1,0,14,4,2,0,15,4,3,0,16,4,
        0,17,5,1,0,18,5,2,0,19,5,3,0,20,5,4,0,21,5,0,22,6,1,0,
        23,6,2,0,24,6,3,0,25,6,4,0,26,6,5,0,27,6,0,28,7,1,0,29,
        8,2,7,1,0,30,9,8,2,1,0,31,9,2,0,32,10,3,7,1,0,33,11,10,
        3,8,2,1,0,34,11,3,9,2,0,35,12,10,3,1,0,36,12,11,3,2,0,37,
        12,3,0,38,13,4,7,1,0,39,14,13,4,8,2,1,0,40,14,4,9,2,0,41,
        15,13,4,10,3,1,0,42,15,14,4,11,3,2,0,43,15,4,12,3,0,44,16,13,
        4,1,0,45,16,14,4,2,0,46,16,15,4,3,0,47,16,4,0,48,17,5,7,1,
        0,49,18,17,5,8,2,1,0,50,18,5,9,2,0,51,19,17,5,10,3,1,0,52,
        19,18,5,11,3,2,0,53,19,5,12,3,0,54,20,17,5,13,4,1,0,55,20,18,
        5,14,4,2,0,56,20,19,5,15,4,3,0,57,20,5,16,4,0,58,21,17,5,1,
        0,59,21,18,5,2,0,60,21,19,5,3,0,61,21,20,5,4,0,62,21,5,0,63,
        22,6,7,1,0,64,23,22,6,8,2,1,0,65,23,6,9,2,0,66,24,22,6,10,
        3,1,0,67,24,23,6,11,3,2,0,68,24,6,12,3,0,69,25,22,6,13,4,1,
        0,70,25,23,6,14,4,2,0,71,25,24,6,15,4,3,0,72,25,6,16,4,0,73,
        26,22,6,17,5,1,0,74,26,23,6,18,5,2,0,75,26,24,6,19,5,3,0,76,
        26,25,6,20,5,4,0,77,26,6,21,5,0,78,27,22,6,1,0,79,27,23,6,2,
        0,80,27,24,6,3,0,81,27,25,6,4,0,82,27,26,6,5,0,83,27,6,0,84,
        28,7,1,0,85,29,8,2,28,7,1,0,86,30,9,29,8,2,7,1,0,87,31,30,
        9,8,2,1,0,88,31,9,2,0,89,32,10,3,28,7,1,0,90,33,11,32,10,3,
        29,8,2,7,1,0,91,34,33,11,10,3,30,9,8,2,1,0,92,34,11,3,31,9,
        2,0,93,35,12,32,10,3,7,1,0,94,36,35,12,33,11,10,3,8,2,1,0,95,
        36,12,34,11,3,9,2,0,96,37,35,12,10,3,1,0,97,37,36,12,11,3,2,0,
        98,37,12,3,0,99,38,13,4,28,7,1,0,100,39,14,38,13,4,29,8,2,7,1,
        0,101,40,39,14,13,4,30,9,8,2,1,0,102,40,14,4,31,9,2,0,103,41,15,
        38,13,4,32,10,3,7,1,0,104,42,41,15,39,14,13,4,33,11,10,3,8,2,1,
        0,105,42,15,40,14,4,34,11,3,9,2,0,106,43,41,15,13,4,35,12,10,3,1,
        0,107,43,42,15,14,4,36,12,11,3,2,0,108,43,15,4,37,12,3,0,109,44,16,
        38,13,4,7,1,0,110,45,44,16,39,14,13,4,8,2,1,0,111,45,16,40,14,4,
        9,2,0,112,46,44,16,41,15,13,4,10,3,1,0,113,46,45,16,42,15,14,4,11,
        3,2,0,114,46,16,43,15,4,12,3,0,115,47,44,16,13,4,1,0,116,47,45,16,
        14,4,2,0,117,47,46,16,15,4,3,0,118,47,16,4,0,119,48,17,5,28,7,1,
        0,120,49,18,48,17,5,29,8,2,7,1,0,121,50,49,18,17,5,30,9,8,2,1,
        0,122,50,18,5,31,9,2,0,123,51,19,48,17,5,32,10,3,7,1,0,124,52,51,
        19,49,18,17,5,33,11,10,3,8,2,1,0,125,52,19,50,18,5,34,11,3,9,2,
        0,126,53,51,19,17,5,35,12,10,3,1,0,127,53,52,19,18,5,36,12,11,3,2,
        0,128,53,19,5,37,12,3,0,129,54,20,48,17,5,38,13,4,7,1,0,130,55,54,
        20,49,18,17,5,39,14,13,4,8,2,1,0,131,55,20,50,18,5,40,14,4,9,2,
        0,132,56,54,20,51,19,17,5,41,15,13,4,10,3,1,0,133,56,55,20,52,19,18,
        5,42,15,14,4,11,3,2,0,134,56,20,53,19,5,43,15,4,12,3,0,135,57,54,
        20,17,5,44,16,13,4,1,0,136,57,55,20,18,5,45,16,14,4,2,0,137,57,56,
        20,19,5,46,16,15,4,3,0,138,57,20,5,47,16,4,0,139,58,21,48,17,5,7,
        1,0,140,59,58,21,49,18,17,5,8,2,1,0,141,59,21,50,18,5,9,2,0,142,
        60,58,21,51,19,17,5,10,3,1,0,143,60,59,21,52,19,18,5,11,3,2,0,144,
        60,21,53,19,5,12,3,0,145,61,58,21,54,20,17,5,13,4,1,0,146,61,59,21,
        55,20,18,5,14,4,2,0,147,61,60,21,56,20,19,5,15,4,3,0,148,61,21,57,
        20,5,16,4,0,149,62,58,21,17,5,1,0,150,62,59,21,18,5,2,0,151,62,60,
        21,19,5,3,0,152,62,61,21,20,5,4,0,153,62,21,5,0,154,63,22,6,28,7,
        1,0,155,64,23,63,22,6,29,8,2,7,1,0,156,65,64,23,22,6,30,9,8,2,
        1,0,157,65,23,6,31,9,2,0,158,66,24,63,22,6,32,10,3,7,1,0,159,67,
        66,24,64,23,22,6,33,11,10,3,8,2,1,0,160,67,24,65,23,6,34,11,3,9,
        2,0,161,68,66,24,22,6,35,12,10,3,1,0,162,68,67,24,23,6,36,12,11,3,
        2,0,163,68,24,6,37,12,3,0,164,69,25,63,22,6,38,13,4,7,1,0,165,70,
        69,25,64,23,22,6,39,14,13,4,8,2,1,0,166,70,25,65,23,6,40,14,4,9,
        2,0,167,71,69,25,66,24,22,6,41,15,13,4,10,3,1,0,168,71,70,25,67,24,
        23,6,42,15,14,4,11,3,2,0,169,71,25,68,24,6,43,15,4,12,3,0,170,72,
        69,25,22,6,44,16,13,4,1,0,171,72,70,25,23,6,45,16,14,4,2,0,172,72,
        71,25,24,6,46,16,15,4,3,0,173,72,25,6,47,16,4,0,174,73,26,63,22,6,
        48,17,5,7,1,0,175,74,73,26,64,23,22,6,49,18,17,5,8,2,1,0,176,74,
        26,65,23,6,50,18,5,9,2,0,177,75,73,26,66,24,22,6,51,19,17,5,10,3,
        1,0,178,75,74,26,67,24,23,6,52,19,18,5,11,3,2,0,179,75,26,68,24,6,
        53,19,5,12,3,0,180,76,73,26,69,25,22,6,54,20,17,5,13,4,1,0,181,76,
        74,26,70,25,23,6,55,20,18,5,14,4,2,0,182,76,75,26,71,25,24,6,56,20,
        19,5,15,4,3,0,183,76,26,72,25,6,57,20,5,16,4,0,184,77,73,26,22,6,
        58,21,17,5,1,0,185,77,74,26,23,6,59,21,18,5,2,0,186,77,75,26,24,6,
        60,21,19,5,3,0,187,77,76,26,25,6,61,21,20,5,4,0,188,77,26,6,62,21,
        5,0,189,78,27,63,22,6,7,1,0,190,79,78,27,64,23,22,6,8,2,1,0,191,
        79,27,65,23,6,9,2,0,192,80,78,27,66,24,22,6,10,3,1,0,193,80,79,27,
        67,24,23,6,11,3,2,0,194,80,27,68,24,6,12,3,0,195,81,78,27,69,25,22,
        6,13,4,1,0,196,81,79,27,70,25,23,6,14,4,2,0,197,81,80,27,71,25,24,
        6,15,4,3,0,198,81,27,72,25,6,16,4,0,199,82,78,27,73,26,22,6,17,5,
        1,0,200,82,79,27,74,26,23,6,18,5,2,0,201,82,80,27,75,26,24,6,19,5,
        3,0,202,82,81,27,76,26,25,6,20,5,4,0,203,82,27,77,26,6,21,5,0,204,
        83,78,27,22,6,1,0,205,83,79,27,23,6,2,0,206,83,80,27,24,6,3,0,207,
        83,81,27,25,6,4,0,208,83,82,27,26,6,5,0,209,83,27,6,0
    };
    for(int i=0;i<CavityFull_count(rank);++i) {
        out[i]=0.;
        for(int j=offset[i];j<offset[i+1];++j)out[i]+=a[left[j]]*b[right[j]];
    }
}
GPUFUN
void CavityFull_derivative(const xt_float_or_tpsa *a,int axis,int rank,
        xt_float_or_tpsa *out) {
    const int shift[504]={
        6,22,23,24,25,26,27,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,
        80,81,82,83,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,
        174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,
        198,199,200,201,202,203,204,205,206,207,208,209,5,17,18,19,20,21,26,48,49,50,51,52,
        53,54,55,56,57,58,59,60,61,62,73,74,75,76,77,82,119,120,121,122,123,124,125,126,
        127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,
        151,152,153,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,199,200,201,202,203,208,
        4,13,14,15,16,20,25,38,39,40,41,42,43,44,45,46,47,54,55,56,57,61,69,70,
        71,72,76,81,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,
        129,130,131,132,133,134,135,136,137,138,145,146,147,148,152,164,165,166,167,168,169,170,171,172,
        173,180,181,182,183,187,195,196,197,198,202,207,3,10,11,12,15,19,24,32,33,34,35,36,
        37,41,42,43,46,51,52,53,56,60,66,67,68,71,75,80,89,90,91,92,93,94,95,96,
        97,98,103,104,105,106,107,108,112,113,114,117,123,124,125,126,127,128,132,133,134,137,142,143,
        144,147,151,158,159,160,161,162,163,167,168,169,172,177,178,179,182,186,192,193,194,197,201,206,
        2,8,9,11,14,18,23,29,30,31,33,34,36,39,40,42,45,49,50,52,55,59,64,65,
        67,70,74,79,85,86,87,88,90,91,92,94,95,97,100,101,102,104,105,107,110,111,113,116,
        120,121,122,124,125,127,130,131,133,136,140,141,143,146,150,155,156,157,159,160,162,165,166,168,
        171,175,176,178,181,185,190,191,193,196,200,205,1,7,8,10,13,17,22,28,29,30,32,33,
        35,38,39,41,44,48,49,51,54,58,63,64,66,69,73,78,84,85,86,87,89,90,91,93,
        94,96,99,100,101,103,104,106,109,110,112,115,119,120,121,123,124,126,129,130,132,135,139,140,
        142,145,149,154,155,156,158,159,161,164,165,167,170,174,175,177,180,184,189,190,192,195,199,204
    };
    const int factor[504]={
        1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,
        2,2,2,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,
        2,2,2,2,2,2,3,3,3,3,3,4,1,1,1,1,1,2,1,1,1,1,1,1,
        1,1,1,1,1,2,2,2,2,3,1,1,1,1,2,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,
        3,3,4,1,1,1,1,1,1,1,1,1,1,2,2,2,2,3,1,1,1,1,2,1,
        1,1,1,1,2,1,1,1,1,1,1,1,1,2,2,2,3,1,1,1,2,1,1,1,
        1,2,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,4,
        1,1,1,1,1,1,2,2,2,3,1,1,1,2,1,1,1,1,1,1,1,2,2,2,
        3,1,1,1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,1,1,1,1,2,2,
        3,1,1,2,1,1,1,2,1,1,1,1,2,1,1,1,1,1,1,1,2,2,2,3,
        3,4,1,1,1,2,2,3,1,1,2,1,1,1,1,2,2,3,1,1,2,1,1,1,
        2,1,1,1,1,1,2,2,3,1,1,2,1,1,1,2,1,1,1,1,2,1,1,1,
        1,1,2,1,1,1,1,1,2,3,1,2,1,1,2,1,1,1,2,1,1,1,1,2,
        1,1,1,1,1,2,3,4,1,2,3,1,2,1,1,2,3,1,2,1,1,2,1,1,
        1,2,3,1,2,1,1,2,1,1,1,2,1,1,1,1,2,3,1,2,1,1,2,1,
        1,1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,1,1,1,3,2,1,2,1,
        1,2,1,1,1,2,1,1,1,1,2,1,1,1,1,1,4,3,2,1,3,2,1,2,
        1,1,3,2,1,2,1,1,2,1,1,1,3,2,1,2,1,1,2,1,1,1,2,1,
        1,1,1,3,2,1,2,1,1,2,1,1,1,2,1,1,1,1,2,1,1,1,1,1
    };
    for(int i=0;i<CavityFull_count(rank);++i)
        out[i]=factor[84*axis+i]*a[shift[84*axis+i]];
}
GPUFUN
int CavityFull_unit(int axis) {
    const int units[6]={
        6,5,4,3,2,1
    };
    return units[axis];
}

// Divided-jet source algebra and connected global-log4 product.
// Concrete gradient consumer of CanonicalAveraged(body_representation=
// global_log4), M2/P16 with the optional same-source straight static
// solenoid/normal/skew multipole potentials. It retains the complete slow mass shell and
// the connected five-offset dressing/body construction. CanonicalF4 and its
// cubic/quartic extensions remain separate finite products of this source.
// Six-axis divided jets through rank four supply the finite source gradient
// cone. Every coefficient is generic for independent outer TPSA derivatives.
// The fixed plan implements the owner's exact multiindex convolution.
typedef struct { xt_float_or_tpsa c[210]; } CavityFullJet;
// Read-only C++ arguments borrow the jet instead of deep-copying 210 MAD
// objects. Ordinary C keeps its existing by-value calling convention.
#ifdef __cplusplus
typedef const CavityFullJet& CavityFullJet_arg;
#else
typedef const CavityFullJet CavityFullJet_arg;
#endif
typedef struct { CavityFullJet a[3],d[3][3]; } CavityFullPotential;
GPUFUN
CavityFullJet CavityFull_constant(xt_float_or_tpsa_arg value,int rank) {
    CavityFullJet out;for(int i=0;i<CavityFull_count(rank);++i)out.c[i]=0.;
    out.c[0]=value;return out;
}
GPUFUN
CavityFullJet CavityFull_linear(CavityFullJet_arg a,CavityFullJet_arg b,
        xt_float_or_tpsa_arg sa,xt_float_or_tpsa_arg sb,int rank) {
    CavityFullJet out;for(int i=0;i<CavityFull_count(rank);++i)out.c[i]=sa*a.c[i]+sb*b.c[i];
    return out;
}
GPUFUN
CavityFullJet CavityFull_scale(CavityFullJet_arg a,xt_float_or_tpsa_arg scale,int rank) {
    CavityFullJet out;for(int i=0;i<CavityFull_count(rank);++i)out.c[i]=scale*a.c[i];return out;
}
GPUFUN
CavityFullJet CavityFull_mul(CavityFullJet_arg a,CavityFullJet_arg b,int rank) {
    CavityFullJet out;CavityFull_product(a.c,b.c,rank,out.c);return out;
}
GPUFUN
CavityFullJet CavityFull_reciprocal(CavityFullJet_arg a,int rank) {
    CavityFullJet delta=CavityFull_scale(a,-1./a.c[0],rank);delta.c[0]=0.;
    CavityFullJet power=CavityFull_constant(1.,rank),out=power;
    for(int n=1;n<=rank;++n) {
        power=CavityFull_mul(power,delta,rank);
        out=CavityFull_linear(out,power,1.,1.,rank);
    }
    return CavityFull_scale(out,1./a.c[0],rank);
}
GPUFUN
CavityFullJet CavityFull_sqrt(CavityFullJet_arg a,int rank) {
    CavityFullJet delta=CavityFull_scale(a,1./a.c[0],rank);delta.c[0]=0.;
    CavityFullJet power=CavityFull_constant(1.,rank),out=power;
    double binomial=1.;
    for(int n=1;n<=rank;++n) {
        power=CavityFull_mul(power,delta,rank);binomial*=((.5-(n-1))/n);
        out=CavityFull_linear(out,power,1.,binomial,rank);
    }
    return CavityFull_scale(out,sqrt(a.c[0]),rank);
}
GPUFUN
void CavityFull_sincos(CavityFullJet a,int rank,CavityFullJet *sine,CavityFullJet *cosine) {
    xt_float_or_tpsa s=sin(a.c[0]),c=cos(a.c[0]);
    xt_float_or_tpsa st[5]={s,c,-s/2.,-c/6.,s/24.};
    xt_float_or_tpsa ct[5]={c,-s,-c/2.,s/6.,c/24.};
    a.c[0]=0.;CavityFullJet power=CavityFull_constant(1.,rank);
    *sine=CavityFull_constant(st[0],rank);*cosine=CavityFull_constant(ct[0],rank);
    for(int n=1;n<=rank;++n) {
        power=CavityFull_mul(power,a,rank);
        *sine=CavityFull_linear(*sine,power,1.,st[n],rank);
        *cosine=CavityFull_linear(*cosine,power,1.,ct[n],rank);
    }
}
GPUFUN
CavityFullJet CavityFull_pb(CavityFullJet_arg a,CavityFullJet_arg b,int rank) {
    CavityFullJet out=CavityFull_constant(0.,rank),left,right;
    for(int axis=0;axis<6;++axis) {
        CavityFull_derivative(a.c,axis,rank,left.c);
        CavityFull_derivative(b.c,CavityAveraged_J_index(axis),rank,right.c);
        CavityFullJet term=CavityFull_mul(left,right,rank);
        out=CavityFull_linear(out,term,1.,CavityAveraged_J_sign(axis),rank);
    }
    return out;
}

// Same summed static potentials and live normal/skew coefficient getter as
// CavityModel_add_static. Q is absorbed ONCE; the RF slow/fast coefficients
// already contain the same signed species charge. No magnetic expansion.
GPUFUN
void CavityFull_add_static(const CavityFullJet q[6],
        const CavityAveragedSource *source,int rank,CavityFullPotential *out) {
    const CavityModelSource *physical=source->static_source;
    if(!physical)return;
    CavityFullJet x=CavityFull_scale(q[0],1./physical->k,rank);
    CavityFullJet y=CavityFull_scale(q[1],1./physical->k,rank);
    xt_float_or_tpsa solenoid=source->static_charge*physical->magnetic_scale*physical->ks*.5;
    out->a[0]=CavityFull_linear(out->a[0],y,1.,-solenoid,rank);
    out->a[1]=CavityFull_linear(out->a[1],x,1.,solenoid,rank);
    out->d[0][1].c[0]-=solenoid/physical->k;
    out->d[1][0].c[0]+=solenoid/physical->k;
    CavityFullJet real=CavityFull_constant(1.,rank),imag=CavityFull_constant(0.,rank);
    xt_float_or_tpsa factor=source->static_charge*physical->magnetic_scale/physical->length;
    for(int64_t n=0;n<=physical->order;++n) {
        xt_float_or_tpsa normal=CavityModel_static_coefficient(physical,n,0);
        xt_float_or_tpsa skew=CavityModel_static_coefficient(physical,n,1);
        out->d[2][0]=CavityFull_linear(out->d[2][0],
            CavityFull_linear(real,imag,normal,-skew,rank),1.,-factor/physical->k,rank);
        out->d[2][1]=CavityFull_linear(out->d[2][1],
            CavityFull_linear(imag,real,normal,skew,rank),1.,factor/physical->k,rank);
        CavityFullJet next_real=CavityFull_scale(CavityFull_linear(
            CavityFull_mul(real,x,rank),CavityFull_mul(imag,y,rank),1.,-1.,rank),1./(n+1.),rank);
        imag=CavityFull_scale(CavityFull_linear(CavityFull_mul(real,y,rank),
            CavityFull_mul(imag,x,rank),1.,1.,rank),1./(n+1.),rank);
        real=next_real;
        out->a[2]=CavityFull_linear(out->a[2],
            CavityFull_linear(real,imag,normal,-skew,rank),1.,-factor,rank);
    }
}

// This is the same compact Maxwell M2 potential/first derivative formula as
// CavityModel_rf_potential, evaluated in the finite divided-coefficient ring.
GPUFUN
void CavityFull_potential(const CavityFullJet q[6],xt_float_or_tpsa_arg z,
        const CavityAveragedSource *source,xt_float_or_tpsa_arg phase,
        int fast,int rank,CavityFullPotential *out) {
    CavityModelSource physical;CavityAveraged_physical_source(source,&physical);
    physical.half_span=source->half_span;physical.carrier=source->carrier;
    physical.counter_phase=phase;physical.forward=fast?0.:source->slow;
    physical.backward=fast?1.:0.;
    xt_float_or_tpsa real[5],imag[5];CavityModel_profile(z,&physical,real,imag);
    CavityFullJet sine,cosine,S[5],C[5];CavityFull_sincos(q[2],rank,&sine,&cosine);
    for(int i=0;i<5;++i) {
        S[i]=CavityFull_linear(sine,cosine,real[i],-imag[i],rank);
        C[i]=CavityFull_linear(cosine,sine,real[i],imag[i],rank);
    }
    CavityFullJet xx=CavityFull_mul(q[0],q[0],rank),yy=CavityFull_mul(q[1],q[1],rank);
    CavityFullJet r2=CavityFull_linear(xx,yy,1.,1.,rank);
    CavityFullJet radial_r2=CavityFull_linear(S[1],S[3],-1./16.,-1./16.,rank);
    CavityFullJet radial=CavityFull_linear(S[1],CavityFull_mul(r2,radial_r2,rank),.5,1.,rank);
    CavityFullJet radial_t=CavityFull_linear(C[1],CavityFull_mul(r2,CavityFull_linear(C[1],C[3],1.,1.,rank),rank),.5,-1./16.,rank);
    CavityFullJet az2=CavityFull_linear(S[0],S[2],.25,.25,rank);
    CavityFullJet az4=CavityFull_linear(CavityFull_linear(S[0],S[2],1.,2.,rank),S[4],-1./64.,-1./64.,rank);
    out->a[0]=CavityFull_mul(q[0],radial,rank);out->a[1]=CavityFull_mul(q[1],radial,rank);
    out->a[2]=CavityFull_linear(CavityFull_linear(S[0],CavityFull_mul(r2,az2,rank),-1.,1.,rank),CavityFull_mul(CavityFull_mul(r2,r2,rank),az4,rank),1.,1.,rank);
    out->d[0][0]=CavityFull_linear(radial,CavityFull_mul(xx,radial_r2,rank),1.,2.,rank);
    out->d[0][1]=CavityFull_scale(CavityFull_mul(CavityFull_mul(q[0],q[1],rank),radial_r2,rank),2.,rank);
    out->d[0][2]=CavityFull_mul(q[0],radial_t,rank);
    out->d[1][0]=out->d[0][1];
    out->d[1][1]=CavityFull_linear(radial,CavityFull_mul(yy,radial_r2,rank),1.,2.,rank);
    out->d[1][2]=CavityFull_mul(q[1],radial_t,rank);
    CavityFullJet transverse=CavityFull_linear(az2,CavityFull_mul(r2,az4,rank),1.,2.,rank);
    out->d[2][0]=CavityFull_scale(CavityFull_mul(q[0],transverse,rank),2.,rank);
    out->d[2][1]=CavityFull_scale(CavityFull_mul(q[1],transverse,rank),2.,rank);
    CavityFullJet ct2=CavityFull_linear(C[0],C[2],.25,.25,rank);
    CavityFullJet ct4=CavityFull_linear(CavityFull_linear(C[0],C[2],1.,2.,rank),C[4],-1./64.,-1./64.,rank);
    out->d[2][2]=CavityFull_linear(CavityFull_linear(C[0],CavityFull_mul(r2,ct2,rank),-1.,1.,rank),CavityFull_mul(CavityFull_mul(r2,r2,rank),ct4,rank),1.,1.,rank);
    if(!fast) {
        out->a[2]=CavityFull_linear(out->a[2],q[2],1.,-source->dc_gradient,rank);
        out->d[2][2].c[0]-=source->dc_gradient;
        CavityFull_add_static(q,source,rank,out);
    }
}
GPUFUN
int CavityFull_slow_flow(const CavityFullJet q[6],xt_float_or_tpsa_arg z,
        const CavityAveragedSource *source,CavityFullJet out[6],int rank) {
    CavityFullPotential potential;CavityFull_potential(q,z,source,source->counter_phase,0,rank,&potential);
    CavityFullJet px=CavityFull_linear(q[3],potential.a[0],1.,-1.,rank);
    CavityFullJet py=CavityFull_linear(q[4],potential.a[1],1.,-1.,rank);
    CavityFullJet square=CavityFull_linear(CavityFull_mul(q[5],q[5],rank),
        CavityFull_linear(CavityFull_mul(px,px,rank),CavityFull_mul(py,py,rank),1.,1.,rank),1.,-1.,rank);
    square.c[0]-=1.;
    if(xt_float_or_tpsa_const_part(q[5].c[0])<=1. || xt_float_or_tpsa_const_part(square.c[0])<=0.)return CAVITY_MODEL_DOMAIN;
    CavityFullJet inverse=CavityFull_reciprocal(CavityFull_sqrt(square,rank),rank);
    out[0]=CavityFull_mul(px,inverse,rank);out[1]=CavityFull_mul(py,inverse,rank);
    out[2]=CavityFull_mul(q[5],inverse,rank);
    for(int i=0;i<3;++i) {
        out[i+3]=CavityFull_linear(potential.d[2][i],
            CavityFull_linear(CavityFull_mul(out[0],potential.d[0][i],rank),
                               CavityFull_mul(out[1],potential.d[1][i],rank),1.,1.,rank),1.,1.,rank);
        if(i==2)out[i+3]=CavityFull_scale(out[i+3],-1.,rank);
    }
    return CAVITY_MODEL_OK;
}

// Source amplitude recurrence R(lambda)^2=S(lambda), on the positive slow
// root. This is the full mass shell before truncating counter-wave amplitude.
GPUFUN
int CavityFull_local_sources_rank(const CavityFullJet q[6],xt_float_or_tpsa_arg z,
        const CavityAveragedSource *source,xt_float_or_tpsa_arg phase,
        int rank,CavityFullJet out[4]) {
    CavityFullPotential slow,fast;
    CavityFull_potential(q,z,source,phase,0,rank,&slow);
    CavityFull_potential(q,z,source,phase,1,rank,&fast);
    CavityFullJet px=CavityFull_linear(q[3],slow.a[0],1.,-1.,rank);
    CavityFullJet py=CavityFull_linear(q[4],slow.a[1],1.,-1.,rank);
    CavityFullJet square=CavityFull_linear(CavityFull_mul(q[5],q[5],rank),
        CavityFull_linear(CavityFull_mul(px,px,rank),CavityFull_mul(py,py,rank),1.,1.,rank),1.,-1.,rank);
    square.c[0]-=1.;
    if(xt_float_or_tpsa_const_part(q[5].c[0])<=1. || xt_float_or_tpsa_const_part(square.c[0])<=0.)return CAVITY_MODEL_DOMAIN;
    CavityFullJet root=CavityFull_sqrt(square,rank),inverse=CavityFull_scale(CavityFull_reciprocal(root,rank),.5,rank);
    CavityFullJet s1=CavityFull_scale(CavityFull_linear(CavityFull_mul(px,fast.a[0],rank),CavityFull_mul(py,fast.a[1],rank),1.,1.,rank),2.,rank);
    CavityFullJet s2=CavityFull_scale(CavityFull_linear(CavityFull_mul(fast.a[0],fast.a[0],rank),CavityFull_mul(fast.a[1],fast.a[1],rank),1.,1.,rank),-1.,rank);
    CavityFullJet r1=CavityFull_mul(s1,inverse,rank);
    CavityFullJet r2=CavityFull_mul(CavityFull_linear(s2,CavityFull_mul(r1,r1,rank),1.,-1.,rank),inverse,rank);
    out[0]=CavityFull_linear(fast.a[2],r1,-1.,-1.,rank);
    out[1]=CavityFull_scale(r2,-1.,rank-1);
    if(rank>=3) {
        CavityFullJet r3=CavityFull_scale(CavityFull_mul(CavityFull_mul(r1,r2,rank),inverse,rank),-2.,rank);
        out[2]=CavityFull_scale(r3,-1.,rank-2);
        if(rank>=4) {
            CavityFullJet r4=CavityFull_scale(CavityFull_mul(CavityFull_linear(CavityFull_mul(r1,r3,rank),CavityFull_mul(r2,r2,rank),2.,1.,rank),inverse,rank),-1.,rank);
            out[3]=CavityFull_scale(r4,-1.,rank-3);
        }
    }
    return CAVITY_MODEL_OK;
}
GPUFUN
int CavityFull_local_sources(const CavityFullJet q[6],xt_float_or_tpsa_arg z,
        const CavityAveragedSource *source,xt_float_or_tpsa_arg phase,
        CavityFullJet out[4]) {
    return CavityFull_local_sources_rank(q,z,source,phase,4,out);
}

// Coefficient-major layout keeps the six physical primal coordinates first;
// auxiliary coefficients use the same reviewed coefficientwise residual rule.
GPUFUN
int CavityFull_slow_coefficients_rank(const xt_float_or_tpsa value[1260],
        xt_float_or_tpsa_arg z,const CavityAveragedSource *source,
        xt_float_or_tpsa out[1260],int rank) {
    CavityFullJet q[6],flow[6];
    const int count=CavityFull_count(rank);
    for(int axis=0;axis<6;++axis)for(int i=0;i<count;++i)q[axis].c[i]=value[6*i+axis];
    int status=CavityFull_slow_flow(q,z,source,flow,rank);
    if(status!=CAVITY_MODEL_OK)return status;
    for(int axis=0;axis<6;++axis)for(int i=0;i<count;++i)out[6*i+axis]=flow[axis].c[i];
    return CavityModel_vector_finite(out,6*count)?CAVITY_MODEL_OK:CAVITY_MODEL_NONFINITE;
}
GPUFUN
int CavityFull_slow_stage_rank(xt_float_or_tpsa value[1260],xt_float_or_tpsa_arg z,
        xt_float_or_tpsa_arg h,const CavityAveragedSource *source,
        CavityFullJet stages[2][6],int rank) {
    const int count=6*CavityFull_count(rank);
    const double c0=.5-sqrt(3.)/6.,c1=.5+sqrt(3.)/6.;
    const double a01=.25-sqrt(3.)/6.,a10=.25+sqrt(3.)/6.;
    xt_float_or_tpsa u[1260],v[1260],f[1260],g[1260];
    int status=CavityFull_slow_coefficients_rank(value,z,source,f,rank);
    if(status!=CAVITY_MODEL_OK)return status;
    // Solve for stage INCREMENTS. A large reference gamma must not enter
    // the physical residual through subtraction of two rounded energies.
    for(int i=0;i<count;++i){u[i]=c0*h*f[i];v[i]=c1*h*f[i];}
    int converged=0;
    for(int iteration=0;iteration<80;++iteration) {
        for(int i=0;i<count;++i){f[i]=value[i]+u[i];g[i]=value[i]+v[i];}
        // The coefficient RHS copies its full input before writing output,
        // so these two scratch arrays can also hold reconstructed stages.
        status=CavityFull_slow_coefficients_rank(f,z+c0*h,source,f,rank);
        if(status!=CAVITY_MODEL_OK)return status;
        status=CavityFull_slow_coefficients_rank(g,z+c1*h,source,g,rank);
        if(status!=CAVITY_MODEL_OK)return status;
        double defect=0.;
        for(int i=0;i<count;++i) {
            xt_float_or_tpsa next_u=h*(.25*f[i]+a01*g[i]);
            xt_float_or_tpsa next_v=h*(a10*f[i]+.25*g[i]);
            double du=CavityModel_scalar_residual(u[i]-next_u,value[i]+next_u,i<6);
            double dv=CavityModel_scalar_residual(v[i]-next_v,value[i]+next_v,i<6);
            if(du>defect)defect=du;if(dv>defect)defect=dv;
            u[i]=next_u;v[i]=next_v;
        }
        if(defect<=source->tolerance){converged=1;break;}
    }
    if(!converged) {
#ifdef CAVITY_MODEL_DEBUG
        // The last update has replaced u,v. Reconstruct its RHS before
        // reporting a defining residual; stale f,g would print zero.
        for(int i=0;i<count;++i){f[i]=value[i]+u[i];g[i]=value[i]+v[i];}
        status=CavityFull_slow_coefficients_rank(f,z+c0*h,source,f,rank);
        if(status!=CAVITY_MODEL_OK)return status;
        status=CavityFull_slow_coefficients_rank(g,z+c1*h,source,g,rank);
        if(status!=CAVITY_MODEL_OK)return status;
#endif
        CavityModel_debug_stages("full-slow-inner-gauss",value,u,v,f,g,h,count);
        return CAVITY_MODEL_ITERATIONS;
    }
    for(int i=0;i<count;++i){f[i]=value[i]+u[i];g[i]=value[i]+v[i];}
    status=CavityFull_slow_coefficients_rank(f,z+c0*h,source,f,rank);
    if(status!=CAVITY_MODEL_OK)return status;
    status=CavityFull_slow_coefficients_rank(g,z+c1*h,source,g,rank);
    if(status!=CAVITY_MODEL_OK)return status;
    for(int i=0;i<count;++i) {
        xt_float_or_tpsa du=u[i]-h*(.25*f[i]+a01*g[i]);
        xt_float_or_tpsa dv=v[i]-h*(a10*f[i]+.25*g[i]);
        if(CavityModel_scalar_residual(du,value[i]+u[i],i<6)>source->tolerance
            ||CavityModel_scalar_residual(dv,value[i]+v[i],i<6)>source->tolerance) {
            CavityModel_debug_stages("full-slow-inner-fresh",value,u,v,f,g,h,count);
            return CAVITY_MODEL_ITERATIONS;
        }
    }
    // Keep the diagnostic's base immutable until all equations are checked.
    for(int i=0;i<count;++i) {
        stages[0][i%6].c[i/6]=value[i]+u[i];stages[1][i%6].c[i/6]=value[i]+v[i];
        value[i]+=.5*h*(f[i]+g[i]);
    }
    return CAVITY_MODEL_OK;
}

GPUFUN
int CavityFull_slow_stage(xt_float_or_tpsa value[1260],xt_float_or_tpsa_arg z,
        xt_float_or_tpsa_arg h,const CavityAveragedSource *source,
        CavityFullJet stages[2][6]) {
    return CavityFull_slow_stage_rank(value,z,h,source,stages,4);
}

typedef struct {
    CavityFullJet f1[5],f2[5],s3[5],s4[5],ell;
} CavityFullPrimitive;
GPUFUN
CavityFullJet CavityFull_mean(const CavityFullJet values[5],int rank) {
    CavityFullJet out=CavityFull_constant(0.,rank);
    for(int i=0;i<5;++i)out=CavityFull_linear(out,values[i],1.,.2,rank);return out;
}
GPUFUN
CavityFullJet CavityFull_stage(CavityFullJet_arg prior,CavityFullJet_arg first,
        CavityFullJet_arg second,xt_float_or_tpsa_arg h,int stage,int rank) {
    const double a01=.25-sqrt(3.)/6.,a10=.25+sqrt(3.)/6.;
    return CavityFull_linear(prior,CavityFull_linear(first,second,
        stage?a10:.25,stage?.25:a01,rank),1.,h,rank);
}
GPUFUN
CavityFullJet CavityFull_endpoint(CavityFullJet_arg prior,CavityFullJet_arg first,
        CavityFullJet_arg second,xt_float_or_tpsa_arg h,int rank) {
    return CavityFull_linear(prior,CavityFull_linear(first,second,.5,.5,rank),1.,h,rank);
}
GPUFUN
CavityFullJet CavityFull_connection3(CavityFullJet_arg f,CavityFullJet_arg d,
        CavityFullJet_arg fp,CavityFullJet_arg dp,CavityFullJet_arg mp) {
    CavityFullJet first=CavityFull_pb(f,CavityFull_linear(dp,mp,1.,2.,3),2);
    CavityFullJet second=CavityFull_pb(d,fp,2);
    CavityFullJet nested=CavityFull_pb(f,CavityFull_pb(f,fp,3),2);
    return CavityFull_linear(CavityFull_linear(first,second,.5,.5,2),nested,1.,1./6.,2);
}
GPUFUN
CavityFullJet CavityFull_connection4(CavityFullJet_arg f,CavityFullJet_arg d,
        CavityFullJet_arg c,CavityFullJet_arg fp,CavityFullJet_arg dp,CavityFullJet_arg mp,
        CavityFullJet_arg cp) {
    // This is chronological_dressing.connection4, not QuarticF4.gamma4.
    CavityFullJet out=CavityFull_linear(CavityFull_pb(f,cp,1),CavityFull_pb(d,dp,1),.5,.5,1);
    out=CavityFull_linear(out,CavityFull_pb(c,fp,1),1.,.5,1);
    out=CavityFull_linear(out,CavityFull_pb(f,CavityFull_pb(f,dp,2),1),1.,1./6.,1);
    out=CavityFull_linear(out,CavityFull_pb(f,CavityFull_pb(d,fp,2),1),1.,1./6.,1);
    out=CavityFull_linear(out,CavityFull_pb(d,CavityFull_pb(f,fp,2),1),1.,1./6.,1);
    out=CavityFull_linear(out,CavityFull_pb(f,CavityFull_pb(f,CavityFull_pb(f,fp,3),2),1),1.,1./24.,1);
    out=CavityFull_linear(out,CavityFull_pb(d,mp,1),1.,1.,1);
    out=CavityFull_linear(out,CavityFull_pb(f,CavityFull_pb(f,mp,2),1),1.,.5,1);
    return out;
}
GPUFUN
int CavityFull_primitive_step(CavityFullPrimitive *prior,
        const CavityFullJet qstages[2][6],xt_float_or_tpsa_arg z,
        xt_float_or_tpsa_arg h,const CavityAveragedSource *source) {
    const double nodes[2]={.5-sqrt(3.)/6.,.5+sqrt(3.)/6.};
    CavityFullJet local[2][5][4],a1[2][5],r2[2][5],a2[2][5],m[2],mp[2];
    CavityFullJet d[2][5],dp[2][5],r3[2][5],a3[2][5],residual4[2][5];
    CavityFullJet b4[2],r4[2][5],log_rate[2];
    for(int stage=0;stage<2;++stage)for(int offset=0;offset<5;++offset) {
        int status=CavityFull_local_sources(qstages[stage],z+nodes[stage]*h,source,
            source->counter_phase+offset*(2.*PI/5.),local[stage][offset]);
        if(status!=CAVITY_MODEL_OK)return status;
    }
    for(int stage=0;stage<2;++stage)for(int offset=0;offset<5;++offset)
        a1[stage][offset]=CavityFull_stage(prior->f1[offset],local[0][offset][0],local[1][offset][0],h,stage,4);
    for(int stage=0;stage<2;++stage)for(int offset=0;offset<5;++offset)
        r2[stage][offset]=CavityFull_linear(local[stage][offset][1],
            CavityFull_pb(local[stage][offset][0],a1[stage][offset],3),1.,.5,3);
    for(int stage=0;stage<2;++stage) {
        for(int offset=0;offset<5;++offset)
            a2[stage][offset]=CavityFull_stage(prior->f2[offset],r2[0][offset],r2[1][offset],h,stage,3);
        m[stage]=CavityFull_mean(a2[stage],3);mp[stage]=CavityFull_mean(r2[stage],3);
        for(int offset=0;offset<5;++offset) {
            d[stage][offset]=CavityFull_linear(a2[stage][offset],m[stage],1.,-1.,3);
            dp[stage][offset]=CavityFull_linear(r2[stage][offset],mp[stage],1.,-1.,3);
            r3[stage][offset]=CavityFull_linear(local[stage][offset][2],
                CavityFull_connection3(a1[stage][offset],d[stage][offset],
                    local[stage][offset][0],dp[stage][offset],mp[stage]),1.,-1.,2);
        }
    }
    for(int stage=0;stage<2;++stage)for(int offset=0;offset<5;++offset) {
        a3[stage][offset]=CavityFull_stage(prior->s3[offset],r3[0][offset],r3[1][offset],h,stage,2);
        residual4[stage][offset]=CavityFull_linear(local[stage][offset][3],
            CavityFull_connection4(a1[stage][offset],d[stage][offset],a3[stage][offset],
                local[stage][offset][0],dp[stage][offset],mp[stage],r3[stage][offset]),1.,-1.,1);
    }
    for(int stage=0;stage<2;++stage) {
        b4[stage]=CavityFull_mean(residual4[stage],1);
        for(int offset=0;offset<5;++offset)
            r4[stage][offset]=CavityFull_linear(residual4[stage][offset],b4[stage],1.,-1.,1);
        log_rate[stage]=CavityFull_linear(b4[stage],CavityFull_pb(m[stage],mp[stage],1),1.,-.5,1);
    }
    for(int offset=0;offset<5;++offset) {
        prior->f1[offset]=CavityFull_endpoint(prior->f1[offset],local[0][offset][0],local[1][offset][0],h,4);
        prior->f2[offset]=CavityFull_endpoint(prior->f2[offset],r2[0][offset],r2[1][offset],h,3);
        prior->s3[offset]=CavityFull_endpoint(prior->s3[offset],r3[0][offset],r3[1][offset],h,2);
        prior->s4[offset]=CavityFull_endpoint(prior->s4[offset],r4[0][offset],r4[1][offset],h,1);
    }
    prior->ell=CavityFull_endpoint(prior->ell,log_rate[0],log_rate[1],h,1);
    return CAVITY_MODEL_OK;
}
GPUFUN
int CavityFull_generators(const xt_float_or_tpsa q[6],const CavityAveragedSource *source,
        xt_float_or_tpsa body[7],xt_float_or_tpsa boundary[7]) {
    if(source->characteristic_steps<=0 || source->factor_steps<=0 || source->tolerance<=0.)return CAVITY_MODEL_ARGUMENT;
    if(!xt_float_or_tpsa_is_zero(source->dc_gradient)
        &&(!xt_float_or_tpsa_is_zero(source->slow)||!xt_float_or_tpsa_is_zero(source->fast)))return CAVITY_MODEL_ARGUMENT;
    CavityFullPrimitive primitive;
    for(int i=0;i<5;++i) {
        primitive.f1[i]=CavityFull_constant(0.,4);primitive.f2[i]=CavityFull_constant(0.,3);
        primitive.s3[i]=CavityFull_constant(0.,2);primitive.s4[i]=CavityFull_constant(0.,1);
    }
    primitive.ell=CavityFull_constant(0.,1);
    xt_float_or_tpsa value[1260];for(int i=0;i<1260;++i)value[i]=0.;
    for(int axis=0;axis<6;++axis){value[axis]=q[axis];value[6*CavityFull_unit(axis)+axis]=1.;}
    xt_float_or_tpsa h=(source->exit-source->entrance)/source->characteristic_steps;
    for(int64_t i=0;i<source->characteristic_steps;++i) {
        CavityFullJet stages[2][6];xt_float_or_tpsa z=source->entrance+i*h;
        int status=CavityFull_slow_stage(value,z,h,source,stages);
        if(status!=CAVITY_MODEL_OK)return status;
        status=CavityFull_primitive_step(&primitive,stages,z,h,source);
        if(status!=CAVITY_MODEL_OK)return status;
    }
    CavityFullJet mean=CavityFull_mean(primitive.f2,1);
    xt_float_or_tpsa f=source->fast,f2=f*f,f3=f2*f,f4=f2*f2;
    CavityFullJet bodyjet=CavityFull_linear(mean,primitive.ell,f2,f4,1);
    CavityFullJet boundaryjet=CavityFull_linear(primitive.f1[0],
        CavityFull_linear(primitive.f2[0],mean,1.,-1.,1),f,f2,1);
    boundaryjet=CavityFull_linear(boundaryjet,primitive.s3[0],1.,f3,1);
    boundaryjet=CavityFull_linear(boundaryjet,primitive.s4[0],1.,f4,1);
    body[0]=bodyjet.c[0];boundary[0]=boundaryjet.c[0];
    for(int axis=0;axis<6;++axis) {
        body[1+axis]=bodyjet.c[CavityFull_unit(axis)];boundary[1+axis]=boundaryjet.c[CavityFull_unit(axis)];
    }
    return CavityModel_vector_finite(body,7)&&CavityModel_vector_finite(boundary,7)?CAVITY_MODEL_OK:CAVITY_MODEL_NONFINITE;
}
GPUFUN
int CavityFull_factor(int mode,const xt_float_or_tpsa *q,xt_float_or_tpsa_arg tau,
        const CavityAveragedSource *source,xt_float_or_tpsa *out) {
    xt_float_or_tpsa body[7],boundary[7];int status=CavityFull_generators(q,source,body,boundary);
    if(status!=CAVITY_MODEL_OK)return status;
    CavityAveraged_J((mode?boundary:body)+1,out);CavityAveraged_constrain(out,source,0);
    return CavityModel_vector_finite(out,6)?CAVITY_MODEL_OK:CAVITY_MODEL_NONFINITE;
}
GPUFUN
int CavityFull_body(const xt_float_or_tpsa *q,xt_float_or_tpsa_arg tau,const CavityAveragedSource *source,xt_float_or_tpsa *out) {
    return CavityFull_factor(0,q,tau,source,out);
}
GPUFUN
int CavityFull_boundary(const xt_float_or_tpsa *q,xt_float_or_tpsa_arg tau,const CavityAveragedSource *source,xt_float_or_tpsa *out) {
    return CavityFull_factor(1,q,tau,source,out);
}
CAVITY_AVERAGED_GAUSS(CavityFull_body_gauss,CavityFull_body,6)
CAVITY_AVERAGED_GAUSS(CavityFull_boundary_gauss,CavityFull_boundary,6)
GPUFUN
int CavityFull_final_slow(const xt_float_or_tpsa *q,xt_float_or_tpsa_arg z,
        const CavityAveragedSource *source,xt_float_or_tpsa *out) {
    return CavityAveraged_slow(q,z,source,out);
}
CAVITY_AVERAGED_GAUSS(CavityFull_final_slow_gauss,CavityFull_final_slow,6)
GPUFUN
int CavityFull_map(xt_float_or_tpsa q[6],const CavityAveragedSource *source,int reverse) {
    xt_float_or_tpsa value[6];for(int i=0;i<6;++i)value[i]=q[i];int status=CAVITY_MODEL_OK;
    const int present=CavityModel_scalar_norm(source->fast)!=0.;
    if(reverse) {
        status=CavityFull_final_slow_gauss(value,source->exit,source->entrance,source->characteristic_steps,source);
        if(status==CAVITY_MODEL_OK&&present)status=CavityFull_boundary_gauss(value,1.,0.,source->factor_steps,source);
        if(status==CAVITY_MODEL_OK&&present)status=CavityFull_body_gauss(value,1.,0.,source->factor_steps,source);
    } else {
        if(present)status=CavityFull_body_gauss(value,0.,1.,source->factor_steps,source);
        if(status==CAVITY_MODEL_OK&&present)status=CavityFull_boundary_gauss(value,0.,1.,source->factor_steps,source);
        if(status==CAVITY_MODEL_OK)status=CavityFull_final_slow_gauss(value,source->entrance,source->exit,source->characteristic_steps,source);
    }
    if(status==CAVITY_MODEL_OK)for(int i=0;i<6;++i)q[i]=value[i];return status;
}
GPUFUN
int CavityFull_interval(xt_float_or_tpsa q[6],const CavityAveragedSource *parent,
        xt_float_or_tpsa_arg a,xt_float_or_tpsa_arg b,int reverse) {
    if(CavityModel_scalar_norm(b-a)==0.)return CAVITY_MODEL_OK;
    xt_float_or_tpsa value[6];for(int i=0;i<6;++i)value[i]=q[i];
    CavityAveragedSource prefix=*parent;int status=CAVITY_MODEL_OK;
    if(reverse) {
        prefix.exit=b;if(CavityModel_scalar_norm(b-parent->entrance)!=0.)status=CavityFull_map(value,&prefix,1);
        prefix.exit=a;if(status==CAVITY_MODEL_OK&&CavityModel_scalar_norm(a-parent->entrance)!=0.)status=CavityFull_map(value,&prefix,0);
    } else {
        prefix.exit=a;if(CavityModel_scalar_norm(a-parent->entrance)!=0.)status=CavityFull_map(value,&prefix,1);
        prefix.exit=b;if(status==CAVITY_MODEL_OK&&CavityModel_scalar_norm(b-parent->entrance)!=0.)status=CavityFull_map(value,&prefix,0);
    }
    if(status==CAVITY_MODEL_OK)for(int i=0;i<6;++i)q[i]=value[i];return status;
}

// Distinct original second-, third- and fourth-order canonical products.
// Direct divided-jet consumer of the ORIGINAL CanonicalF4/CubicF4/QuarticF4
// products. Two pi/2 quadratures retain their original mean_F2. The distinct
// C3 and C4 factors remain SEPARATE from the C2 boundary/body; this never
// substitutes the connected five-offset global_log4 dressing of model17.
// Source roots and scalar quadratures use the same fixed G2 mesh. Required
// inner canonical ranks are 2,3,4 respectively; all entries remain generic
// for outer TPSA state/hardware derivatives. Literal dense characteristic
// headers remain independent numerical reference consumers.
GPUFUN
CavityFullJet CavityProduct_gamma3(CavityFullJet_arg f,CavityFullJet_arg d,
        CavityFullJet_arg fp,CavityFullJet_arg dp,CavityFullJet_arg mp,int rank) {
    CavityFullJet first=CavityFull_pb(f,CavityFull_linear(dp,mp,1.,2.,rank+1),rank);
    CavityFullJet second=CavityFull_pb(d,fp,rank);
    CavityFullJet nested=CavityFull_pb(f,CavityFull_pb(f,fp,rank+1),rank);
    return CavityFull_linear(CavityFull_linear(first,second,.5,.5,rank),nested,1.,1./6.,rank);
}
GPUFUN
CavityFullJet CavityProduct_gamma4(CavityFullJet_arg f,CavityFullJet_arg d,
        CavityFullJet_arg m,CavityFullJet_arg fp,CavityFullJet_arg dp,CavityFullJet_arg mp,
        CavityFullJet_arg cp) {
    // QuarticF4.gamma4 has {f,cp} and {m,mp}/2; it has NO {c,fp}/2.
    CavityFullJet out=CavityFull_connection4(f,d,CavityFull_constant(0.,2),fp,dp,mp,cp);
    out=CavityFull_linear(out,CavityFull_pb(f,cp,1),1.,.5,1);
    return CavityFull_linear(out,CavityFull_pb(m,mp,1),1.,.5,1);
}
GPUFUN
int CavityProduct_generators(const xt_float_or_tpsa q[6],const CavityAveragedSource *source,
        int degree,xt_float_or_tpsa scalars[4],xt_float_or_tpsa gradients[4][6]) {
    if(degree<2||degree>4||source->characteristic_steps<=0||source->tolerance<=0.
        ||source->static_source)return CAVITY_MODEL_ARGUMENT;
    const int rank1=degree,rank2=degree-1;
    CavityFullJet f1[2],f2[2],c3=CavityFull_constant(0.,1),c4=CavityFull_constant(0.,1);
    for(int i=0;i<2;++i){f1[i]=CavityFull_constant(0.,rank1);f2[i]=CavityFull_constant(0.,rank2);}
    xt_float_or_tpsa value[1260];
    const int count=6*CavityFull_count(degree);
    for(int i=0;i<count;++i)value[i]=0.;
    for(int axis=0;axis<6;++axis){value[axis]=q[axis];value[6*CavityFull_unit(axis)+axis]=1.;}
    xt_float_or_tpsa h=(source->exit-source->entrance)/source->characteristic_steps;
    const double nodes[2]={.5-sqrt(3.)/6.,.5+sqrt(3.)/6.};
    for(int64_t cell=0;cell<source->characteristic_steps;++cell) {
        xt_float_or_tpsa z=source->entrance+cell*h;CavityFullJet stages[2][6];
        int status=CavityFull_slow_stage_rank(value,z,h,source,stages,degree);
        if(status!=CAVITY_MODEL_OK)return status;
        CavityFullJet local[2][2][4],a1[2][2],r2[2][2],a2[2][2],m[2],mp[2],r3[2][2],r4[2];
        for(int stage=0;stage<2;++stage)for(int quadrature=0;quadrature<2;++quadrature) {
            status=CavityFull_local_sources_rank(stages[stage],z+nodes[stage]*h,source,
                source->counter_phase+quadrature*(PI/2.),degree,local[stage][quadrature]);
            if(status!=CAVITY_MODEL_OK)return status;
        }
        // Both source stages must exist before constructing their coupled
        // F1 stage values; no current-stage sample replaces the finite root.
        for(int stage=0;stage<2;++stage)for(int quadrature=0;quadrature<2;++quadrature)
            a1[stage][quadrature]=CavityFull_stage(f1[quadrature],local[0][quadrature][0],local[1][quadrature][0],h,stage,rank1);
        for(int stage=0;stage<2;++stage)for(int quadrature=0;quadrature<2;++quadrature)
            r2[stage][quadrature]=CavityFull_linear(local[stage][quadrature][1],
                CavityFull_pb(local[stage][quadrature][0],a1[stage][quadrature],rank2),1.,.5,rank2);
        for(int stage=0;stage<2;++stage) {
            for(int quadrature=0;quadrature<2;++quadrature)
                a2[stage][quadrature]=CavityFull_stage(f2[quadrature],r2[0][quadrature],r2[1][quadrature],h,stage,rank2);
            m[stage]=CavityFull_linear(a2[stage][0],a2[stage][1],.5,.5,rank2);
            mp[stage]=CavityFull_linear(r2[stage][0],r2[stage][1],.5,.5,rank2);
            if(degree>=3) {
                CavityFullJet d=CavityFull_linear(a2[stage][0],m[stage],1.,-1.,rank2);
                CavityFullJet dp=CavityFull_linear(r2[stage][0],mp[stage],1.,-1.,rank2);
                r3[stage][0]=CavityFull_linear(local[stage][0][2],
                    CavityProduct_gamma3(a1[stage][0],d,local[stage][0][0],dp,mp[stage],degree-2),1.,-1.,degree-2);
                if(degree==4)r4[stage]=CavityFull_linear(local[stage][0][3],
                    CavityProduct_gamma4(a1[stage][0],d,m[stage],local[stage][0][0],dp,mp[stage],r3[stage][0]),1.,-1.,1);
            }
        }
        for(int quadrature=0;quadrature<2;++quadrature) {
            f1[quadrature]=CavityFull_endpoint(f1[quadrature],local[0][quadrature][0],local[1][quadrature][0],h,rank1);
            f2[quadrature]=CavityFull_endpoint(f2[quadrature],r2[0][quadrature],r2[1][quadrature],h,rank2);
        }
        if(degree>=3)c3=CavityFull_endpoint(c3,r3[0][0],r3[1][0],h,1);
        if(degree==4)c4=CavityFull_endpoint(c4,r4[0],r4[1],h,1);
    }
    CavityFullJet mean=CavityFull_linear(f2[0],f2[1],.5,.5,1);
    xt_float_or_tpsa f=source->fast,ff=f*f;CavityFullJet result[4];
    result[0]=CavityFull_scale(mean,ff,1);
    result[1]=CavityFull_linear(f1[0],CavityFull_linear(f2[0],mean,1.,-1.,1),f,ff,1);
    result[2]=CavityFull_scale(c3,ff*f,1);result[3]=CavityFull_scale(c4,ff*ff,1);
    for(int model=0;model<4;++model) {
        if(scalars)scalars[model]=result[model].c[0];
        for(int axis=0;axis<6;++axis)gradients[model][axis]=result[model].c[CavityFull_unit(axis)];
    }
    return CAVITY_MODEL_OK;
}
GPUFUN
int CavityProduct_factor(int degree,int mode,const xt_float_or_tpsa *q,
        xt_float_or_tpsa_arg tau,const CavityAveragedSource *source,xt_float_or_tpsa *out) {
    xt_float_or_tpsa gradients[4][6];
    int status=CavityProduct_generators(q,source,degree,0,gradients);
    if(status!=CAVITY_MODEL_OK)return status;
    CavityAveraged_J(gradients[mode],out);CavityAveraged_constrain(out,source,0);
    return CavityModel_vector_finite(out,6)?CAVITY_MODEL_OK:CAVITY_MODEL_NONFINITE;
}
#define CAVITY_PRODUCT_FACTOR(NAME,DEGREE,MODE) \
GPUFUN int NAME(const xt_float_or_tpsa *q,xt_float_or_tpsa_arg tau, \
        const CavityAveragedSource *source,xt_float_or_tpsa *out) { \
    return CavityProduct_factor(DEGREE,MODE,q,tau,source,out); \
}
CAVITY_PRODUCT_FACTOR(CavityProduct_body,2,0)
CAVITY_PRODUCT_FACTOR(CavityProduct_boundary,2,1)
CAVITY_PRODUCT_FACTOR(CavityProduct_cubic,3,2)
CAVITY_PRODUCT_FACTOR(CavityProduct_quartic,4,3)
#undef CAVITY_PRODUCT_FACTOR
CAVITY_AVERAGED_GAUSS(CavityProduct_body_gauss,CavityProduct_body,6)
CAVITY_AVERAGED_GAUSS(CavityProduct_boundary_gauss,CavityProduct_boundary,6)
CAVITY_AVERAGED_GAUSS(CavityProduct_cubic_gauss,CavityProduct_cubic,6)
CAVITY_AVERAGED_GAUSS(CavityProduct_quartic_gauss,CavityProduct_quartic,6)
GPUFUN
int CavityProduct_map(xt_float_or_tpsa q[6],const CavityAveragedSource *source,int degree,int reverse) {
    if(degree<2||degree>4||source->static_source||source->characteristic_steps<=0
        ||source->factor_steps<=0||source->tolerance<=0.
        ||xt_float_or_tpsa_const_part(source->half_span)<=0.)return CAVITY_MODEL_ARGUMENT;
    if(CavityModel_scalar_norm(source->dc_gradient)!=0.
        &&(CavityModel_scalar_norm(source->slow)!=0.||CavityModel_scalar_norm(source->fast)!=0.))return CAVITY_MODEL_ARGUMENT;
    if((source->frozen_coordinates&((int64_t)1<<2))&&!(source->reference_beta>0.&&source->reference_beta<=1.))return CAVITY_MODEL_ARGUMENT;
    xt_float_or_tpsa value[6];for(int i=0;i<6;++i)value[i]=q[i];int status=CAVITY_MODEL_OK;
    const int present=CavityModel_scalar_norm(source->fast)!=0.;
    if(reverse) {
        status=CavityFull_final_slow_gauss(value,source->exit,source->entrance,source->characteristic_steps,source);
        if(status==CAVITY_MODEL_OK&&present)status=CavityProduct_boundary_gauss(value,0.,-1.,source->factor_steps,source);
        if(status==CAVITY_MODEL_OK&&present)status=CavityProduct_body_gauss(value,0.,-1.,source->factor_steps,source);
        if(status==CAVITY_MODEL_OK&&present&&degree>=3)status=CavityProduct_cubic_gauss(value,0.,-1.,source->factor_steps,source);
        if(status==CAVITY_MODEL_OK&&present&&degree==4)status=CavityProduct_quartic_gauss(value,0.,-1.,source->factor_steps,source);
    } else {
        if(present&&degree==4)status=CavityProduct_quartic_gauss(value,0.,1.,source->factor_steps,source);
        if(status==CAVITY_MODEL_OK&&present&&degree>=3)status=CavityProduct_cubic_gauss(value,0.,1.,source->factor_steps,source);
        if(status==CAVITY_MODEL_OK&&present)status=CavityProduct_body_gauss(value,0.,1.,source->factor_steps,source);
        if(status==CAVITY_MODEL_OK&&present)status=CavityProduct_boundary_gauss(value,0.,1.,source->factor_steps,source);
        if(status==CAVITY_MODEL_OK)status=CavityFull_final_slow_gauss(value,source->entrance,source->exit,source->characteristic_steps,source);
    }
    if(status==CAVITY_MODEL_OK)for(int i=0;i<6;++i)q[i]=value[i];return status;
}
GPUFUN
int CavityProduct_interval(xt_float_or_tpsa q[6],const CavityAveragedSource *parent,
        xt_float_or_tpsa_arg a,xt_float_or_tpsa_arg b,int degree,int reverse) {
    if(CavityModel_scalar_norm(b-a)==0.)return CAVITY_MODEL_OK;
    xt_float_or_tpsa value[6];for(int i=0;i<6;++i)value[i]=q[i];
    CavityAveragedSource prefix=*parent;int status=CAVITY_MODEL_OK;
    if(reverse) {
        prefix.exit=b;if(CavityModel_scalar_norm(b-parent->entrance)!=0.)status=CavityProduct_map(value,&prefix,degree,1);
        prefix.exit=a;if(status==CAVITY_MODEL_OK&&CavityModel_scalar_norm(a-parent->entrance)!=0.)status=CavityProduct_map(value,&prefix,degree,0);
    } else {
        prefix.exit=a;if(CavityModel_scalar_norm(a-parent->entrance)!=0.)status=CavityProduct_map(value,&prefix,degree,1);
        prefix.exit=b;if(status==CAVITY_MODEL_OK&&CavityModel_scalar_norm(b-parent->entrance)!=0.)status=CavityProduct_map(value,&prefix,degree,0);
    }
    if(status==CAVITY_MODEL_OK)for(int i=0;i<6;++i)q[i]=value[i];return status;
}

// Uniform quadratic body and first/second-order boundary scalars.
#include "xtrack/beam_elements/elements_src/cavity_models_compat.h"

GPUFUN
xt_float_or_tpsa CavityQuadratic_power(xt_float_or_tpsa_arg value,int degree) {
    xt_float_or_tpsa result=1.;
    for (int i=0;i<degree;++i) result*=value;
    return result;
}

// Compact rational coefficient and its exact E derivative. P,Q are small
// polynomials in p=sqrt(E^2-1); their coefficients may depend on live nu.
// Return [P(p)+E*Q(p)]/(scale*p^pp*nu^nn*E^ee*(E+nu*p)^dd).
GPUFUN
void CavityQuadratic_ratio(xt_float_or_tpsa_arg E,xt_float_or_tpsa_arg p,
        xt_float_or_tpsa_arg nu,const xt_float_or_tpsa P[7],
        const xt_float_or_tpsa Q[7],int scale,int pp,int nn,int ee,int dd,
        xt_float_or_tpsa out[2]) {
    xt_float_or_tpsa pv=0.,pd=0.,qv=0.,qd=0.;
    for (int j=6;j>=0;--j) {
        pd=pd*p+pv;pv=pv*p+P[j];
        qd=qd*p+qv;qv=qv*p+Q[j];
    }
    const xt_float_or_tpsa rate=E/p,D=E+nu*p;
    const xt_float_or_tpsa value=pv+E*qv;
    const xt_float_or_tpsa derivative=rate*pd+qv+E*rate*qd;
    const xt_float_or_tpsa factor=1./(scale*CavityQuadratic_power(p,pp)
        *CavityQuadratic_power(nu,nn)*CavityQuadratic_power(E,ee)
        *CavityQuadratic_power(D,dd));
    const xt_float_or_tpsa logarithm=-pp*E/(p*p)-ee/E-dd*(1.+nu*rate)/D;
    out[0]=factor*value;
    out[1]=factor*(derivative+logarithm*value);
}

// Three real second-order boundary harmonics: psi-chi, 2psi, psi+chi.
// Each row has reference/radial/shear/drift coefficients multiplying
// sin(angle)*(1,R^2,P^2) and cos(angle)*(X*Px+Y*Py).
GPUFUN
void CavityQuadratic_boundary_coefficients(xt_float_or_tpsa_arg E,
        xt_float_or_tpsa_arg p,xt_float_or_tpsa_arg nu,
        xt_float_or_tpsa packets[3][4][2]) {
    const xt_float_or_tpsa n2=nu*nu,n3=n2*nu,n4=n2*n2;
    const xt_float_or_tpsa numerator[12][7]={
        {-1.}, {-n2+2.,0.,3.*n2-1.},
        {0.,-7.*n3+6.*nu,0.,-n3-3.*nu},
        {-3.*n2-2.,0.,3.*n4-18.*n2-1.,0.,n4+6.*n2+1.},
        {-1.}, {-3.*n2+2.,0.,-3.*n2+1.},
        {0.,n2+2.,0.,-n2-1.}, {0.,3.*n2-14.,0.,n2+1.},
        {-1.}, {-n2+2.,0.,n2+1.},
        {-4.*nu,0.,-3.*n3-6.*nu,0.,-n3-3.*nu},
        {0.,13.*n2-14.,0.,3.*n4+18.*n2-13.,0.,n4+6.*n2+1.}
    };
    const xt_float_or_tpsa companion[12][7]={
        {0.}, {0.,2.*n3}, {-n2+2.,0.,-3.*n2-1.},
        {0.,-12.*n3-8.*nu,0.,4.*n3+4.*nu},
        {0.}, {0.,-2.*n3}, {4.*nu,0.,-2.*nu}, {-12.*nu,0.,2.*nu},
        {0.}, {0.,2.*nu}, {0.,-9.*n2+2.,0.,-3.*n2-1.},
        {-8.*nu,0.,12.*n3-8.*nu,0.,4.*n3+4.*nu}
    };
    // scale, p-power, nu-power, E-power, (E+nu*p)-power.
    const int denominator[12][5]={
        {16,1,1,0,2},{64,1,1,0,2},{64,2,2,0,3},{128,3,3,0,4},
        {32,0,0,0,3},{128,0,0,0,3},{128,1,0,0,4},{256,1,0,0,5},
        {16,0,0,1,2},{64,0,0,1,2},{64,1,0,2,3},{128,1,0,3,4}
    };
    for (int row=0;row<12;++row)
        CavityQuadratic_ratio(E,p,nu,numerator[row],companion[row],
            denominator[row][0],denominator[row][1],denominator[row][2],
            denominator[row][3],denominator[row][4],packets[row/4][row%4]);
}

// Averaged body: free paraxial transport + slow first-order RF + separate
// slow^2 and counter-wave^2 terms. Fast averaging does not erase slow phase.
GPUFUN
void CavityQuadratic_body(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z,xt_float_or_tpsa_arg s,
        xt_float_or_tpsa_arg f,xt_float_or_tpsa_arg nu,
        xt_float_or_tpsa out[7]) {
    const xt_float_or_tpsa X=q[0],Y=q[1],U=q[3],V=q[4],E=q[5];
    const xt_float_or_tpsa p=sqrt((E-1.)*(E+1.)),p2=p*p,p3=p2*p;
    const xt_float_or_tpsa D=E+nu*p,DE=1.+nu*E/p,D2=D*D;
    const xt_float_or_tpsa R2=X*X+Y*Y,P2=U*U+V*V,M=X*U+Y*V;
    const xt_float_or_tpsa sine=sin(q[2]-nu*z),cosine=cos(q[2]-nu*z);
    const xt_float_or_tpsa n2=nu*nu;
    const xt_float_or_tpsa reference=f*f/(4.*p*D2);
    const xt_float_or_tpsa reference_E=reference*(-E/p2-2.*DE/D);
    const xt_float_or_tpsa N=n2*(E*E+2.)+E*E-3.+2.*nu*E*p;
    const xt_float_or_tpsa NE=2.*E*(n2+1.)+2.*nu*(p+E*E/p);
    const xt_float_or_tpsa prefactor=f*f/(8.*p2*D2);
    const xt_float_or_tpsa focusing=prefactor*N;
    const xt_float_or_tpsa focusing_E=prefactor*(NE-N*(2.*E/p2+2.*DE/D));
    const xt_float_or_tpsa DN=E*E+3.*nu*E*p-n2*p2+3.*p2;
    const xt_float_or_tpsa DNE=(8.-2.*n2)*E+3.*nu*(p+E*E/p);
    const xt_float_or_tpsa dprefactor=f*f/(4.*p3*D2*D2);
    const xt_float_or_tpsa drift=dprefactor*DN;
    const xt_float_or_tpsa drift_E=dprefactor*(DNE-DN*(3.*E/p2+4.*DE/D));
    const xt_float_or_tpsa shear=s*nu*cosine/(2.*p);
    const xt_float_or_tpsa radial=p*focusing+s*s*n2*cosine*cosine/(4.*p);
    const xt_float_or_tpsa radial_E=E*focusing/p+p*focusing_E
        -s*s*n2*E*cosine*cosine/(4.*p3);
    const xt_float_or_tpsa radial_T=-s*s*n2*sine*cosine/(2.*p);
    const xt_float_or_tpsa slow_radial=s*(n2-1.)*sine*.5;
    out[0]=-p+P2/(2.*p)+s*sine*(1.+(n2-1.)*R2*.25)
        +shear*M+reference+radial*R2*.5+drift*P2*.5;
    out[1]=(slow_radial+radial)*X+shear*U;
    out[2]=(slow_radial+radial)*Y+shear*V;
    out[3]=s*cosine*(1.+(n2-1.)*R2*.25)-s*nu*sine*M/(2.*p)+radial_T*R2*.5;
    out[4]=(1./p+drift)*U+shear*X;
    out[5]=(1./p+drift)*V+shear*Y;
    out[6]=-E/p-E*P2/(2.*p3)-s*nu*E*cosine*M/(2.*p3)
        +reference_E+radial_E*R2*.5+drift_E*P2*.5;
}

GPUFUN
void CavityQuadratic_first_boundary(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z,xt_float_or_tpsa_arg f,
        xt_float_or_tpsa_arg nu,xt_float_or_tpsa_arg phase,
        xt_float_or_tpsa out[7]) {
    const xt_float_or_tpsa X=q[0],Y=q[1],U=q[3],V=q[4],E=q[5];
    const xt_float_or_tpsa p=sqrt((E-1.)*(E+1.)),D=E+nu*p;
    const xt_float_or_tpsa D2=D*D,D3=D2*D,D4=D2*D2,companion=nu*E+p;
    const xt_float_or_tpsa R2=X*X+Y*Y,P2=U*U+V*V,M=X*U+Y*V;
    const xt_float_or_tpsa sine=sin(q[2]+nu*z+phase),cosine=cos(q[2]+nu*z+phase);
    const xt_float_or_tpsa constant=p/(2.*D),radial=p*(nu*nu-1.)/(8.*D);
    const xt_float_or_tpsa drift=-1./(4.*p*D3),shear=companion/(4.*D2);
    const xt_float_or_tpsa ca=2.*f*(constant+radial*R2+drift*P2),sa=2.*f*shear*M;
    const xt_float_or_tpsa caE=2.*f*(1./(2.*p*D2)+(nu*nu-1.)*R2/(8.*p*D2)
        +(E*D+3.*p*companion)*P2/(4.*p*p*p*D4));
    const xt_float_or_tpsa saE=2.*f*(D2-2.*companion*companion)*M/(4.*p*D3);
    out[0]=ca*cosine+sa*sine;
    out[1]=2.*f*(2.*radial*X*cosine+shear*U*sine);
    out[2]=2.*f*(2.*radial*Y*cosine+shear*V*sine);
    out[3]=-ca*sine+sa*cosine;
    out[4]=2.*f*(2.*drift*U*cosine+shear*X*sine);
    out[5]=2.*f*(2.*drift*V*cosine+shear*Y*sine);
    out[6]=caE*cosine+saE*sine;
}

GPUFUN
void CavityQuadratic_second_boundary(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z,xt_float_or_tpsa_arg s,
        xt_float_or_tpsa_arg f,xt_float_or_tpsa_arg nu,
        xt_float_or_tpsa_arg phase,xt_float_or_tpsa out[7]) {
    const xt_float_or_tpsa X=q[0],Y=q[1],U=q[3],V=q[4],E=q[5];
    const xt_float_or_tpsa p=sqrt((E-1.)*(E+1.));
    const xt_float_or_tpsa R2=X*X+Y*Y,P2=U*U+V*V,M=X*U+Y*V;
    xt_float_or_tpsa packets[3][4][2];
    CavityQuadratic_boundary_coefficients(E,p,nu,packets);
    const xt_float_or_tpsa angles[3]={2.*nu*z+phase,2.*(q[2]+nu*z+phase),2.*q[2]+phase};
    const xt_float_or_tpsa weights[3]={4.*s*f,4.*f*f,4.*s*f};
    const int rates[3]={0,2,2};
    for (int i=0;i<7;++i) out[i]=0.;
    for (int mode=0;mode<3;++mode) {
        const xt_float_or_tpsa sine=sin(angles[mode]),cosine=cos(angles[mode]);
        const xt_float_or_tpsa c=packets[mode][0][0],r=packets[mode][1][0];
        const xt_float_or_tpsa m=packets[mode][2][0],d=packets[mode][3][0];
        const xt_float_or_tpsa amp=c+r*R2+d*P2;
        const xt_float_or_tpsa ampE=packets[mode][0][1]
            +packets[mode][1][1]*R2+packets[mode][3][1]*P2;
        const xt_float_or_tpsa weight=weights[mode];
        out[0]+=weight*(amp*sine+m*M*cosine);
        out[1]+=weight*(2.*r*X*sine+m*U*cosine);
        out[2]+=weight*(2.*r*Y*sine+m*V*cosine);
        out[3]+=weight*rates[mode]*(amp*cosine-m*M*sine);
        out[4]+=weight*(2.*d*U*sine+m*X*cosine);
        out[5]+=weight*(2.*d*V*sine+m*Y*cosine);
        out[6]+=weight*(ampE*sine+packets[mode][2][1]*M*cosine);
    }
}

// Uniform quadratic finite maps and parent prefixes.
#define CAVITY_MODEL_QUADRATIC_FIRST 18
#define CAVITY_MODEL_QUADRATIC_SECOND 19

typedef struct {
    xt_float_or_tpsa half_span, slow, fast, carrier, counter_phase;
    int64_t body_steps, boundary_steps, boundary_order;
    int64_t frozen_coordinates;
    double reference_beta, tolerance;
} CavityQuadraticSource;

// Fixed-beta crest calibration of the actual CONSTANT two-wave envelope.
// eta is the squared amplitude ratio; the shared endpoint guard refuses
// nonanalytic eta Taylor charts while retaining ordinary state derivatives.
GPUFUN
int CavityQuadratic_prepare(xt_float_or_tpsa_arg frequency,
        xt_float_or_tpsa_arg voltage, xt_float_or_tpsa_arg length,
        xt_float_or_tpsa_arg phase, xt_float_or_tpsa_arg eta,
        xt_float_or_tpsa_arg counter_phase,
        xt_float_or_tpsa_arg calibration_beta,
        xt_float_or_tpsa_arg wave_number,double mass0,double q0,
        CavityModelSource *source) {
    int status=CavityModel_prepare(frequency,0.,length,phase,eta,counter_phase,
        calibration_beta,wave_number,mass0,q0,source);
    if (status!=CAVITY_MODEL_OK) return status;
    if (xt_float_or_tpsa_const_part(frequency)<=0.
        || xt_float_or_tpsa_const_part(source->carrier)<=0.)
        return CAVITY_MODEL_ARGUMENT;
    const xt_float_or_tpsa H=source->half_span;
    const xt_float_or_tpsa a=H/calibration_beta;
    xt_float_or_tpsa forward_weight=0.,backward_weight=1.;
    if (isfinite(xt_float_or_tpsa_const_part(eta))) {
        xt_float_or_tpsa ratio=0.;
        if (!xt_float_or_tpsa_is_zero(eta)) ratio=sqrt(eta);
        forward_weight=1./(1.+ratio);
        backward_weight=ratio/(1.+ratio);
    }
    const xt_float_or_tpsa slow=2.*H*forward_weight
        *CavityModel_static_sinc(a-H*source->carrier);
    const xt_float_or_tpsa fast=2.*H*backward_weight
        *CavityModel_static_sinc(a+H*source->carrier);
    const xt_float_or_tpsa integral_real=slow+fast*cos(counter_phase);
    const xt_float_or_tpsa integral_imag=fast*sin(counter_phase);
    const xt_float_or_tpsa square=integral_real*integral_real+integral_imag*integral_imag;
    const double scale=fabs(xt_float_or_tpsa_const_part(slow))
                       +fabs(xt_float_or_tpsa_const_part(fast));
    const double threshold=64.*2.2204460492503131e-16*scale;
    if (!CavityModel_scalar_finite(square)
        || xt_float_or_tpsa_const_part(square)<=threshold*threshold)
        return CAVITY_MODEL_ARGUMENT;
    // The same complex crest integral controls amplitude AND oscillator phase.
    const xt_float_or_tpsa real=integral_real*cos(a)-integral_imag*sin(a);
    const xt_float_or_tpsa imag=integral_real*sin(a)+integral_imag*cos(a);
    const xt_float_or_tpsa amplitude=voltage/(mass0*sqrt(square));
    source->forward=amplitude*forward_weight;
    source->backward=amplitude*backward_weight;
    source->voltage_norm=voltage/mass0;
    source->clock_origin-=atan2(imag,real);
    return CAVITY_MODEL_OK;
}

GPUFUN
int CavityQuadratic_flow(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z,const CavityQuadraticSource *source,
        int boundary,xt_float_or_tpsa out[6]) {
    if (!CavityModel_vector_finite(q,6)
        || xt_float_or_tpsa_const_part(q[5])<=1.) return CAVITY_MODEL_DOMAIN;
    xt_float_or_tpsa value[7],second[7];
    if (boundary) {
        CavityQuadratic_first_boundary(q,z,source->fast,source->carrier,source->counter_phase,value);
        if (source->boundary_order==2) {
            CavityQuadratic_second_boundary(q,z,source->slow,source->fast,
                source->carrier,source->counter_phase,second);
            for (int i=0;i<7;++i) value[i]+=second[i];
        }
    } else {
        CavityQuadratic_body(q,z,source->slow,source->fast,source->carrier,value);
    }
    out[0]=value[4]; out[1]=value[5]; out[2]=-value[6];
    out[3]=-value[1];out[4]=-value[2];out[5]=value[3];
    for (int i=0;i<6;++i)
        if (source->frozen_coordinates & ((int64_t)1<<i)) out[i]=0.;
    if (!boundary && (source->frozen_coordinates & ((int64_t)1<<2)))
        out[2]=1./source->reference_beta;
    return CavityModel_vector_finite(out,6)
        ? CAVITY_MODEL_OK : CAVITY_MODEL_NONFINITE;
}

// The retained uniform facade uses implicit MIDPOINT, not a renamed G2 map.
// Physical body and auxiliary boundary meshes are independent controls.
GPUFUN
int CavityQuadratic_integrate(xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg a,xt_float_or_tpsa_arg b,
        xt_float_or_tpsa_arg physical_z,int boundary,int64_t steps,
        const CavityQuadraticSource *source) {
    if (steps<=0 || source->tolerance<=0.) return CAVITY_MODEL_ARGUMENT;
    const xt_float_or_tpsa h=(b-a)/steps;
    for (int64_t cell=0;cell<steps;++cell) {
        xt_float_or_tpsa z=physical_z;
        if (!boundary) z=a+(cell+.5)*h;
        xt_float_or_tpsa stage[6],flow[6],full[6];
        int status=CavityQuadratic_flow(q,z,source,boundary,flow);
        if (status!=CAVITY_MODEL_OK) return status;
        for (int i=0;i<6;++i) stage[i]=.5*h*flow[i];
        int converged=0;
        for (int iteration=0;iteration<80;++iteration) {
            for (int i=0;i<6;++i) full[i]=q[i]+stage[i];
            status=CavityQuadratic_flow(full,z,source,boundary,flow);
            if (status!=CAVITY_MODEL_OK) return status;
            double defect=0.;
            for (int i=0;i<6;++i) {
                const xt_float_or_tpsa next=.5*h*flow[i];
                defect=fmax(defect,CavityModel_scalar_residual(stage[i]-next,q[i]+next,1));
                stage[i]=next;
            }
            if (defect<=source->tolerance) {converged=1;break;}
        }
        if (!converged) return CAVITY_MODEL_ITERATIONS;
        for (int i=0;i<6;++i) full[i]=q[i]+stage[i];
        status=CavityQuadratic_flow(full,z,source,boundary,flow);
        if (status!=CAVITY_MODEL_OK) return status;
        for (int i=0;i<6;++i)
            if (CavityModel_scalar_residual(stage[i]-.5*h*flow[i],q[i]+stage[i],1)
                    >source->tolerance) return CAVITY_MODEL_ITERATIONS;
        for (int i=0;i<6;++i) q[i]+=h*flow[i];
    }
    return CAVITY_MODEL_OK;
}

GPUFUN
int CavityQuadratic_prefix(xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg end,const CavityQuadraticSource *source,int reverse) {
    const xt_float_or_tpsa start=-source->half_span;
    if (xt_float_or_tpsa_is_zero(end-start)) return CAVITY_MODEL_OK;
    int status;
    if (!reverse) {
        status=CavityQuadratic_integrate(q,0.,1.,start,1,source->boundary_steps,source);
        if (status==CAVITY_MODEL_OK)
            status=CavityQuadratic_integrate(q,start,end,0.,0,source->body_steps,source);
        if (status==CAVITY_MODEL_OK)
            status=CavityQuadratic_integrate(q,0.,-1.,end,1,source->boundary_steps,source);
    } else {
        status=CavityQuadratic_integrate(q,0.,1.,end,1,source->boundary_steps,source);
        if (status==CAVITY_MODEL_OK)
            status=CavityQuadratic_integrate(q,end,start,0.,0,source->body_steps,source);
        if (status==CAVITY_MODEL_OK)
            status=CavityQuadratic_integrate(q,0.,-1.,start,1,source->boundary_steps,source);
    }
    return status;
}

GPUFUN
int CavityQuadratic_interval(xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg a,xt_float_or_tpsa_arg b,
        const CavityQuadraticSource *source,int reverse) {
    xt_float_or_tpsa candidate[6];
    for (int i=0;i<6;++i) candidate[i]=q[i];
    int status=CavityQuadratic_prefix(candidate,reverse ? b : a,source,1);
    if (status==CAVITY_MODEL_OK)
        status=CavityQuadratic_prefix(candidate,reverse ? a : b,source,0);
    if (status==CAVITY_MODEL_OK)
        for (int i=0;i<6;++i) q[i]=candidate[i];
    return status;
}

// Actual retained quadratic uniform physical potentials, WITHOUT adding the
// radial-cubic A_perp or quartic Az terms of the P16/M2 field implementation.
GPUFUN
void CavityQuadratic_potential(const xt_float_or_tpsa q[6],
        xt_float_or_tpsa_arg z,const CavityModelSource *source,
        xt_float_or_tpsa potential[4]) {
    const xt_float_or_tpsa nu=source->carrier;
    const xt_float_or_tpsa chi=q[2]-nu*z;
    const xt_float_or_tpsa psi=q[2]+nu*z+source->counter_phase;
    const xt_float_or_tpsa radial=.5*nu*(-source->forward*cos(chi)
                                       +source->backward*cos(psi));
    potential[0]=q[0]*radial;potential[1]=q[1]*radial;
    potential[2]=-(source->forward*sin(chi)+source->backward*sin(psi))
        *(1.+(nu*nu-1.)*(q[0]*q[0]+q[1]*q[1])*.25);
    potential[3]=0.;
}

// Centred Gaussian magnetic compact source and finite product.
#include <float.h>
enum CavityMagPhysicalModel {CAVITY_MODEL_GAUSSIAN_MAGNETIC=20};

// Direct consumer of magnetic_compact.GaussianCenteredMagnetic: joint RF
// order two, transverse quadratic, exact uniform solenoid/normal-quadrupole
// dependence within that scalar projection. The anchored 286 COMPLEX source
// coefficients use the owner's fixed RK4, followed by the same midpoint
// entry(+W), body, exit(-W) product. This is not the full-root model17.
// Units c=m=omega=1. Signed slow/fast,b,g already contain actual species Q.
// Native source/controls remain generic; no fitted/frozen source table.
typedef struct {xt_float_or_tpsa re,im;} CavityMagComplex;
typedef struct {
    xt_float_or_tpsa anchor,width,centre,nu,slow,fast,b,g,phase;
    int64_t source_steps,body_steps,boundary_steps,frozen_coordinates;
    double tolerance,reference_beta;
} CavityMagSource;
GPUFUN
int CavityMag_validate(const CavityMagSource *source) {
    xt_float_or_tpsa controls[9]={source->anchor,source->width,source->centre,
        source->nu,source->slow,source->fast,source->b,source->g,source->phase};
    if(!CavityModel_vector_finite(controls,9)||source->source_steps<=0
        ||xt_float_or_tpsa_const_part(source->width)<=0.
        ||xt_float_or_tpsa_const_part(source->nu)<=0.)return CAVITY_MODEL_ARGUMENT;
    return CAVITY_MODEL_OK;
}
GPUFUN
CavityMagComplex CavityMag_value(xt_float_or_tpsa_arg re,xt_float_or_tpsa_arg im) {
    CavityMagComplex out;out.re=re;out.im=im;return out;
}
GPUFUN
CavityMagComplex CavityMag_add(CavityMagComplex a,CavityMagComplex b) {
    return CavityMag_value(a.re+b.re,a.im+b.im);
}
GPUFUN
CavityMagComplex CavityMag_scale(CavityMagComplex a,xt_float_or_tpsa_arg b) {
    return CavityMag_value(a.re*b,a.im*b);
}
GPUFUN
CavityMagComplex CavityMag_mul(CavityMagComplex a,CavityMagComplex b) {
    return CavityMag_value(a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re);
}
GPUFUN
CavityMagComplex CavityMag_i(CavityMagComplex a) {return CavityMag_value(-a.im,a.re);}
GPUFUN
void CavityMag_acc(CavityMagComplex *a,CavityMagComplex b,xt_float_or_tpsa_arg scale) {
    a->re+=b.re*scale;a->im+=b.im*scale;
}
GPUFUN
int CavityMag_first_mode(int i,int axis) {
    const int modes[4][2]={{-1,0},{0,-1},{0,1},{1,0}};return modes[i][axis];
}
GPUFUN
int CavityMag_second_mode(int i,int axis) {
    const int modes[9][2]={{-2,0},{-1,-1},{-1,1},{0,-2},{0,0},{0,2},{1,-1},{1,1},{2,0}};
    return modes[i][axis];
}
GPUFUN
int CavityMag_second_index(int m,int n) {
    for(int i=0;i<9;++i)if(CavityMag_second_mode(i,0)==m&&CavityMag_second_mode(i,1)==n)return i;
    return -1;
}
GPUFUN
int CavityMag_second_state(int i) {
    const int offsets[6]={1,2,3,5,6,7};return offsets[i];
}
GPUFUN
int CavityMag_channel(int i,int j) {
    if(i>j){const int saved=i;i=j;j=saved;}
    return 1+4*i-i*(i-1)/2+j-i;
}
GPUFUN
int CavityMag_J_index(int i){return i<2?i+2:i-2;}
GPUFUN
double CavityMag_J_sign(int i){return i<2?1.:-1.;}

// Divided E coefficients through four from the positive p²=(E−1)(E+1)
// root, then the same reciprocal recurrence as inverse_momentum_coefficients.
GPUFUN
int CavityMag_magnet(xt_float_or_tpsa_arg energy,const CavityMagSource *source,
        xt_float_or_tpsa K[5][4][4],xt_float_or_tpsa KE[4][4][4],
        xt_float_or_tpsa A[4][4][4],xt_float_or_tpsa clock[4]) {
    if(xt_float_or_tpsa_const_part(energy)<=1.)return CAVITY_MODEL_DOMAIN;
    xt_float_or_tpsa root[5],inv[5],square[5]={(energy-1.)*(energy+1.),2.*energy,1.,0.,0.};
    root[0]=sqrt(square[0]);inv[0]=1./root[0];
    for(int n=1;n<5;++n) {
        xt_float_or_tpsa known=0.;for(int j=1;j<n;++j)known+=root[j]*root[n-j];
        root[n]=(square[n]-known)/(2.*root[0]);known=0.;
        for(int j=1;j<=n;++j)known+=root[j]*inv[n-j];inv[n]=-known/root[0];
    }
    xt_float_or_tpsa shape[4][4];for(int i=0;i<4;++i)for(int j=0;j<4;++j)shape[i][j]=0.;
    shape[0][0]=source->b*source->b/4.;shape[1][1]=shape[0][0];
    shape[0][3]=-source->b/2.;shape[3][0]=shape[0][3];
    shape[1][2]=source->b/2.;shape[2][1]=shape[1][2];shape[2][2]=1.;shape[3][3]=1.;
    for(int n=0;n<5;++n)for(int i=0;i<4;++i)for(int j=0;j<4;++j)K[n][i][j]=inv[n]*shape[i][j];
    K[0][0][0]+=source->g;K[0][1][1]-=source->g;
    for(int n=0;n<4;++n) {
        clock[n]=energy*inv[n];if(n)clock[n]+=inv[n-1];
        for(int i=0;i<4;++i)for(int j=0;j<4;++j) {
            KE[n][i][j]=(n+1.)*K[n+1][i][j];
            A[n][i][j]=CavityMag_J_sign(i)*K[n][CavityMag_J_index(i)][j];
        }
    }
    return CAVITY_MODEL_OK;
}
GPUFUN
int CavityMag_physical(xt_float_or_tpsa_arg energy,xt_float_or_tpsa_arg position,
        const CavityMagSource *source,CavityMagComplex first[4][4][11],
        CavityMagComplex second[9][3][11]) {
    xt_float_or_tpsa K[5][4][4],KE[4][4][4],A[4][4][4],clock[4];
    int status=CavityMag_magnet(energy,source,K,KE,A,clock);
    if(status!=CAVITY_MODEL_OK)return status;
    xt_float_or_tpsa inv[4];for(int n=0;n<4;++n)inv[n]=K[n][2][2];
    xt_float_or_tpsa distance=(position-source->centre)/source->width;
    xt_float_or_tpsa envelope=exp(-distance*distance/2.);
    xt_float_or_tpsa slope=-distance*envelope/source->width;
    xt_float_or_tpsa curvature=(distance*distance-1.)*envelope/(source->width*source->width);
    CavityMagComplex qz[4];
    for(int mode=0;mode<4;++mode) {
        const int m=CavityMag_first_mode(mode,0),n=CavityMag_first_mode(mode,1),ell=m+n;
        xt_float_or_tpsa strength=n?source->fast:source->slow;
        xt_float_or_tpsa axial=(n-m)*source->nu,angle=axial*position;
        CavityMagComplex amplitude=CavityMag_value(ell*strength*sin(angle)/2.,-ell*strength*cos(angle)/2.);
        CavityMagComplex q=CavityMag_scale(amplitude,envelope);
        qz[mode]=CavityMag_mul(amplitude,CavityMag_value(slope,axial*envelope));
        CavityMagComplex radial=CavityMag_add(q,CavityMag_mul(amplitude,
            CavityMag_value(curvature-axial*axial*envelope,2.*axial*slope)));
        for(int j=0;j<4;++j)for(int channel=0;channel<11;++channel)first[mode][j][channel]=CavityMag_value(0.,0.);
        first[mode][0][0]=q;
        first[mode][0][CavityMag_channel(0,0)]=CavityMag_scale(radial,-.25);
        first[mode][0][CavityMag_channel(1,1)]=CavityMag_scale(radial,-.25);
        for(int j=0;j<4;++j) {
            first[mode][j][CavityMag_channel(0,2)]=CavityMag_scale(qz[mode],-.25*inv[j]);
            first[mode][j][CavityMag_channel(1,3)]=CavityMag_scale(qz[mode],-.25*inv[j]);
        }
    }
    for(int mode=0;mode<9;++mode)for(int j=0;j<3;++j)for(int c=0;c<11;++c)second[mode][j][c]=CavityMag_value(0.,0.);
    for(int left=0;left<4;++left)for(int right=0;right<4;++right) {
        int index=CavityMag_second_index(CavityMag_first_mode(left,0)+CavityMag_first_mode(right,0),
            CavityMag_first_mode(left,1)+CavityMag_first_mode(right,1));
        if(index<0)return CAVITY_MODEL_ARGUMENT;
        CavityMagComplex product=CavityMag_mul(qz[left],qz[right]);
        for(int j=0;j<3;++j) {
            CavityMag_acc(&second[index][j][CavityMag_channel(0,0)],product,inv[j]/8.);
            CavityMag_acc(&second[index][j][CavityMag_channel(1,1)],product,inv[j]/8.);
        }
    }
    return CAVITY_MODEL_OK;
}

// Exact quadratic packet bracket, including the retained T--E clock terms.
// A packet is scalar + z^T M z, not scalar + z^T M z/2.
GPUFUN
void CavityMag_bracket(const CavityMagComplex left[4][11],
        const CavityMagComplex right[4][11],int left_ell,int right_ell,
        CavityMagComplex out[3][11]) {
    for(int n=0;n<3;++n) {
        out[n][0]=CavityMag_value(0.,0.);
        for(int e=0;e<=n;++e) {
            CavityMag_acc(&out[n][0],CavityMag_i(CavityMag_mul(left[e+1][0],right[n-e][0])),right_ell*(e+1.));
            CavityMag_acc(&out[n][0],CavityMag_i(CavityMag_mul(left[e][0],right[n-e+1][0])),-left_ell*(n-e+1.));
        }
        for(int i=0;i<4;++i)for(int j=i;j<4;++j) {
            CavityMagComplex result=CavityMag_value(0.,0.);
            for(int e=0;e<=n;++e) {
                for(int k=0;k<4;++k) {
                    int jk=CavityMag_J_index(k);double sign=CavityMag_J_sign(k);
                    CavityMag_acc(&result,CavityMag_mul(left[e][CavityMag_channel(i,k)],right[n-e][CavityMag_channel(jk,j)]),2.*sign);
                    CavityMag_acc(&result,CavityMag_mul(right[e][CavityMag_channel(i,k)],left[n-e][CavityMag_channel(jk,j)]),-2.*sign);
                }
                CavityMag_acc(&result,CavityMag_i(CavityMag_mul(left[e+1][0],right[n-e][CavityMag_channel(i,j)])),right_ell*(e+1.));
                CavityMag_acc(&result,CavityMag_i(CavityMag_mul(right[e][0],left[n-e+1][CavityMag_channel(i,j)])),right_ell*(n-e+1.));
                CavityMag_acc(&result,CavityMag_i(CavityMag_mul(left[e][0],right[n-e+1][CavityMag_channel(i,j)])),-left_ell*(n-e+1.));
                CavityMag_acc(&result,CavityMag_i(CavityMag_mul(right[e+1][0],left[n-e][CavityMag_channel(i,j)])),-left_ell*(e+1.));
            }
            out[n][CavityMag_channel(i,j)]=result;
        }
    }
}
GPUFUN
void CavityMag_second_source(const CavityMagComplex first[4][4][11],
        const CavityMagComplex direct[9][3][11],const CavityMagComplex state[286],
        CavityMagComplex out[9][3][11]) {
    for(int mode=0;mode<9;++mode)for(int e=0;e<3;++e)for(int c=0;c<11;++c)out[mode][e][c]=direct[mode][e][c];
    for(int left=0;left<2;++left)for(int right=0;right<4;++right) {
        int left_n=left?1:-1;
        int right_m=CavityMag_first_mode(right,0),right_n=CavityMag_first_mode(right,1);
        int mode=CavityMag_second_index(right_m,left_n+right_n);
        CavityMagComplex generator[4][11],bracket[3][11];
        for(int e=0;e<4;++e)for(int c=0;c<11;++c)generator[e][c]=state[44*left+11*e+c];
        CavityMag_bracket(generator,first[right],left_n,right_m+right_n,bracket);
        double weight=right_n?.5:1.;
        for(int e=0;e<3;++e)for(int c=0;c<11;++c)CavityMag_acc(&out[mode][e][c],bracket[e][c],weight);
    }
}
GPUFUN
void CavityMag_homology(const CavityMagComplex *state,const CavityMagComplex *forcing,
        int rank,int ell,const xt_float_or_tpsa KE[4][4][4],
        const xt_float_or_tpsa A[4][4][4],const xt_float_or_tpsa clock[4],
        CavityMagComplex *out) {
    for(int n=0;n<=rank;++n) {
        CavityMagComplex scalar=CavityMag_scale(forcing[11*n],-1.);
        for(int e=0;e<=n;++e)CavityMag_acc(&scalar,CavityMag_i(state[11*(n-e)]),-ell*clock[e]);
        out[11*n]=scalar;
        for(int i=0;i<4;++i)for(int j=i;j<4;++j) {
            CavityMagComplex result=CavityMag_scale(forcing[11*n+CavityMag_channel(i,j)],-1.);
            for(int e=0;e<=n;++e) {
                for(int k=0;k<4;++k) {
                    CavityMag_acc(&result,state[11*(n-e)+CavityMag_channel(k,j)],-A[e][k][i]);
                    CavityMag_acc(&result,state[11*e+CavityMag_channel(i,k)],-A[n-e][k][j]);
                }
                CavityMag_acc(&result,CavityMag_i(state[11*(n-e)+CavityMag_channel(i,j)]),-ell*clock[e]);
                CavityMag_acc(&result,CavityMag_i(state[11*e]),ell*KE[n-e][i][j]/2.);
            }
            out[11*n+CavityMag_channel(i,j)]=result;
        }
    }
}
GPUFUN
int CavityMag_source_rhs(const CavityMagComplex state[286],xt_float_or_tpsa_arg energy,
        xt_float_or_tpsa_arg position,const CavityMagSource *source,CavityMagComplex out[286]) {
    CavityMagComplex first[4][4][11],direct[9][3][11],second[9][3][11];
    int status=CavityMag_physical(energy,position,source,first,direct);
    if(status!=CAVITY_MODEL_OK)return status;
    CavityMag_second_source(first,direct,state,second);
    xt_float_or_tpsa K[5][4][4],KE[4][4][4],A[4][4][4],clock[4];
    status=CavityMag_magnet(energy,source,K,KE,A,clock);if(status!=CAVITY_MODEL_OK)return status;
    for(int mode=0;mode<2;++mode)CavityMag_homology(state+44*mode,&first[mode+1][0][0],3,mode?1:-1,KE,A,clock,out+44*mode);
    for(int mode=0;mode<6;++mode) {
        int index=CavityMag_second_state(mode),ell=CavityMag_second_mode(index,0)+CavityMag_second_mode(index,1);
        CavityMag_homology(state+88+33*mode,&second[index][0][0],2,ell,KE,A,clock,out+88+33*mode);
    }
    return CAVITY_MODEL_OK;
}
GPUFUN
int CavityMag_source_state(xt_float_or_tpsa_arg energy,xt_float_or_tpsa_arg position,
        const CavityMagSource *source,CavityMagComplex out[286]) {
    int valid=CavityMag_validate(source);if(valid!=CAVITY_MODEL_OK)return valid;
    if(!CavityModel_scalar_finite(energy)||!CavityModel_scalar_finite(position))return CAVITY_MODEL_ARGUMENT;
    for(int i=0;i<286;++i)out[i]=CavityMag_value(0.,0.);
    xt_float_or_tpsa h=(position-source->anchor)/source->source_steps;
    CavityMagComplex k1[286],k2[286],k3[286],k4[286],stage[286];
    for(int64_t cell=0;cell<source->source_steps;++cell) {
        xt_float_or_tpsa z=source->anchor+cell*h;
        int status=CavityMag_source_rhs(out,energy,z,source,k1);if(status!=CAVITY_MODEL_OK)return status;
        for(int i=0;i<286;++i)stage[i]=CavityMag_add(out[i],CavityMag_scale(k1[i],h/2.));
        status=CavityMag_source_rhs(stage,energy,z+h/2.,source,k2);if(status!=CAVITY_MODEL_OK)return status;
        for(int i=0;i<286;++i)stage[i]=CavityMag_add(out[i],CavityMag_scale(k2[i],h/2.));
        status=CavityMag_source_rhs(stage,energy,z+h/2.,source,k3);if(status!=CAVITY_MODEL_OK)return status;
        for(int i=0;i<286;++i)stage[i]=CavityMag_add(out[i],CavityMag_scale(k3[i],h));
        status=CavityMag_source_rhs(stage,energy,z+h,source,k4);if(status!=CAVITY_MODEL_OK)return status;
        for(int i=0;i<286;++i) {
            CavityMagComplex sum=CavityMag_add(CavityMag_add(k1[i],CavityMag_scale(k2[i],2.)),CavityMag_add(CavityMag_scale(k3[i],2.),k4[i]));
            CavityMag_acc(&out[i],sum,h/6.);
        }
    }
    for(int i=0;i<286;++i)if(!CavityModel_scalar_finite(out[i].re)||!CavityModel_scalar_finite(out[i].im))return CAVITY_MODEL_NONFINITE;
    return CAVITY_MODEL_OK;
}
GPUFUN
void CavityMag_scalar_packet(const CavityMagComplex packet[3][11],int m,int n,
        const xt_float_or_tpsa q[6],const CavityMagSource *source,CavityMagComplex out[43]) {
    const int slots[4]={0,1,3,4};const int ell=m+n;
    xt_float_or_tpsa angle=ell*q[2]+n*source->phase;
    CavityMagComplex wave=CavityMag_value(cos(angle),sin(angle)),amp[3],gz[2][4];
    for(int e=0;e<3;++e) {
        amp[e]=packet[e][0];
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)
            CavityMag_acc(&amp[e],packet[e][CavityMag_channel(i,j)],q[slots[i]]*q[slots[j]]);
        if(e<2)for(int i=0;i<4;++i) {
            gz[e][i]=CavityMag_value(0.,0.);
            for(int j=0;j<4;++j)CavityMag_acc(&gz[e][i],packet[e][CavityMag_channel(i,j)],2.*q[slots[j]]);
        }
    }
    CavityMag_acc(&out[0],CavityMag_mul(wave,amp[0]),1.);
    CavityMag_acc(&out[3],CavityMag_i(CavityMag_mul(wave,amp[0])),ell);
    CavityMag_acc(&out[6],CavityMag_mul(wave,amp[1]),1.);
    CavityMag_acc(&out[7+6*2+2],CavityMag_mul(wave,amp[0]),-ell*ell);
    CavityMagComplex te=CavityMag_scale(CavityMag_i(CavityMag_mul(wave,amp[1])),ell);
    CavityMag_acc(&out[7+6*2+5],te,1.);CavityMag_acc(&out[7+6*5+2],te,1.);
    CavityMag_acc(&out[7+6*5+5],CavityMag_mul(wave,amp[2]),2.);
    for(int i=0;i<4;++i) {
        int slot=slots[i];CavityMag_acc(&out[1+slot],CavityMag_mul(wave,gz[0][i]),1.);
        CavityMagComplex time=CavityMag_scale(CavityMag_i(CavityMag_mul(wave,gz[0][i])),ell);
        CavityMagComplex energy=CavityMag_mul(wave,gz[1][i]);
        CavityMag_acc(&out[7+6*slot+2],time,1.);CavityMag_acc(&out[7+6*2+slot],time,1.);
        CavityMag_acc(&out[7+6*slot+5],energy,1.);CavityMag_acc(&out[7+6*5+slot],energy,1.);
        for(int j=0;j<4;++j)CavityMag_acc(&out[7+6*slot+slots[j]],CavityMag_mul(wave,packet[0][CavityMag_channel(i,j)]),2.);
    }
}
GPUFUN
int CavityMag_scalar(const xt_float_or_tpsa q[6],xt_float_or_tpsa_arg position,
        const CavityMagSource *source,int boundary,xt_float_or_tpsa out[43]) {
    if(!CavityModel_vector_finite(q,6))return CAVITY_MODEL_NONFINITE;
    CavityMagComplex state[286],combined[43];
    int status=CavityMag_source_state(q[5],position,source,state);
    if(status!=CAVITY_MODEL_OK)return status;
    for(int i=0;i<43;++i)combined[i]=CavityMag_value(0.,0.);
    if(boundary) {
        CavityMagComplex packet[3][11];
        for(int mode=0;mode<2;++mode) {
            for(int e=0;e<3;++e)for(int c=0;c<11;++c)packet[e][c]=state[44*mode+11*e+c];
            CavityMag_scalar_packet(packet,0,mode?1:-1,q,source,combined);
        }
        for(int mode=0;mode<6;++mode) {
            int index=CavityMag_second_state(mode);
            for(int e=0;e<3;++e)for(int c=0;c<11;++c)packet[e][c]=state[88+33*mode+11*e+c];
            CavityMag_scalar_packet(packet,CavityMag_second_mode(index,0),CavityMag_second_mode(index,1),q,source,combined);
        }
    } else {
        CavityMagComplex first[4][4][11],direct[9][3][11],second[9][3][11];
        status=CavityMag_physical(q[5],position,source,first,direct);if(status!=CAVITY_MODEL_OK)return status;
        CavityMag_second_source(first,direct,state,second);
        CavityMag_scalar_packet(first[0],-1,0,q,source,combined);
        CavityMag_scalar_packet(first[3],1,0,q,source,combined);
        for(int mode=0;mode<9;++mode)if(CavityMag_second_mode(mode,1)==0)
            CavityMag_scalar_packet(second[mode],CavityMag_second_mode(mode,0),0,q,source,combined);
        xt_float_or_tpsa K[5][4][4],KE[4][4][4],A[4][4][4],clock[4];
        status=CavityMag_magnet(q[5],source,K,KE,A,clock);if(status!=CAVITY_MODEL_OK)return status;
        const int slots[4]={0,1,3,4};xt_float_or_tpsa momentum=sqrt((q[5]-1.)*(q[5]+1.));
        combined[0].re-=momentum;combined[6].re-=q[5]/momentum;
        combined[7+6*5+5].re+=1./(momentum*momentum*momentum);
        for(int i=0;i<4;++i)for(int j=0;j<4;++j) {
            xt_float_or_tpsa product=q[slots[i]]*q[slots[j]];
            combined[0].re+=product*K[0][i][j]/2.;combined[6].re+=product*KE[0][i][j]/2.;
            combined[1+slots[i]].re+=K[0][i][j]*q[slots[j]];
            combined[7+6*slots[i]+slots[j]].re+=K[0][i][j];
            combined[7+6*slots[i]+5].re+=KE[0][i][j]*q[slots[j]];
            combined[7+6*5+slots[i]].re+=KE[0][i][j]*q[slots[j]];
            combined[7+6*5+5].re+=product*K[2][i][j];
        }
    }
    // Same real-conjugacy check as the owner, extended to represented outer
    // coefficients. Real projection never repairs inconsistent source data.
    double scale=1.,imaginary=0.;
    for(int i=0;i<43;++i) {
        double real=CavityModel_scalar_norm(combined[i].re),imag=CavityModel_scalar_norm(combined[i].im);
        if(!isfinite(real)||!isfinite(imag))return CAVITY_MODEL_NONFINITE;
        if(real>scale)scale=real;if(imag>scale)scale=imag;if(imag>imaginary)imaginary=imag;
    }
    if(imaginary>256.*DBL_EPSILON*scale)return CAVITY_MODEL_NONFINITE;
    for(int i=0;i<43;++i)out[i]=combined[i].re;
    return CAVITY_MODEL_OK;
}
GPUFUN
int CavityMag_flow(const xt_float_or_tpsa q[6],xt_float_or_tpsa_arg position,
        const CavityMagSource *source,int boundary,xt_float_or_tpsa out[6]) {
    xt_float_or_tpsa scalar[43];int status=CavityMag_scalar(q,position,source,boundary,scalar);
    if(status!=CAVITY_MODEL_OK)return status;
    out[0]=scalar[4];out[1]=scalar[5];out[2]=-scalar[6];
    out[3]=-scalar[1];out[4]=-scalar[2];out[5]=scalar[3];
    for(int i=0;i<6;++i)if(source->frozen_coordinates&((int64_t)1<<i))out[i]=0.;
    if(!boundary&&(source->frozen_coordinates&((int64_t)1<<2)))out[2]=1./source->reference_beta;
    return CAVITY_MODEL_OK;
}
GPUFUN
int CavityMag_midpoint(xt_float_or_tpsa q[6],xt_float_or_tpsa_arg start,
        xt_float_or_tpsa_arg end,const CavityMagSource *source,int boundary,
        xt_float_or_tpsa_arg physical_position,int64_t steps) {
    if(steps<=0||!isfinite(source->tolerance)||source->tolerance<=0.)return CAVITY_MODEL_ARGUMENT;
    xt_float_or_tpsa h=(end-start)/steps;
    for(int64_t cell=0;cell<steps;++cell) {
        xt_float_or_tpsa position=physical_position;
        if(!boundary)position=start+(cell+.5)*h;
        xt_float_or_tpsa increment[6],stage[6],flow[6];
        int status=CavityMag_flow(q,position,source,boundary,flow);if(status!=CAVITY_MODEL_OK)return status;
        for(int i=0;i<6;++i)increment[i]=h*flow[i]/2.;int converged=0;
        for(int iteration=0;iteration<80;++iteration) {
            for(int i=0;i<6;++i)stage[i]=q[i]+increment[i];
            status=CavityMag_flow(stage,position,source,boundary,flow);if(status!=CAVITY_MODEL_OK)return status;
            double defect=0.;
            for(int i=0;i<6;++i) {
                xt_float_or_tpsa next=h*flow[i]/2.;
                double residual=CavityModel_scalar_residual(increment[i]-next,q[i]+next,1);
                if(residual>defect)defect=residual;increment[i]=next;
            }
            if(defect<=source->tolerance){converged=1;break;}
        }
        if(!converged)return CAVITY_MODEL_ITERATIONS;
        for(int i=0;i<6;++i)stage[i]=q[i]+increment[i];
        status=CavityMag_flow(stage,position,source,boundary,flow);if(status!=CAVITY_MODEL_OK)return status;
        for(int i=0;i<6;++i)if(CavityModel_scalar_residual(increment[i]-h*flow[i]/2.,stage[i],1)>source->tolerance)return CAVITY_MODEL_ITERATIONS;
        for(int i=0;i<6;++i)q[i]+=h*flow[i];
    }
    return CAVITY_MODEL_OK;
}
GPUFUN
int CavityMag_map(xt_float_or_tpsa q[6],xt_float_or_tpsa_arg start,
        xt_float_or_tpsa_arg end,const CavityMagSource *source) {
    int valid=CavityMag_validate(source);if(valid!=CAVITY_MODEL_OK)return valid;
    if(source->body_steps<=0||source->boundary_steps<=0||!isfinite(source->tolerance)
        ||source->tolerance<=0.||!CavityModel_scalar_finite(start)
        ||!CavityModel_scalar_finite(end))return CAVITY_MODEL_ARGUMENT;
    if((source->frozen_coordinates&((int64_t)1<<2))&&!(source->reference_beta>0.&&source->reference_beta<=1.))return CAVITY_MODEL_ARGUMENT;
    if(!CavityModel_vector_finite(q,6))return CAVITY_MODEL_NONFINITE;
    if(xt_float_or_tpsa_const_part(q[5])<=1.)return CAVITY_MODEL_DOMAIN;
    xt_float_or_tpsa value[6];for(int i=0;i<6;++i)value[i]=q[i];
    if(xt_float_or_tpsa_is_zero(end-start))return CAVITY_MODEL_OK;
    int status=CavityMag_midpoint(value,0.,1.,source,1,start,source->boundary_steps);
    if(status==CAVITY_MODEL_OK)status=CavityMag_midpoint(value,start,end,source,0,0.,source->body_steps);
    if(status==CAVITY_MODEL_OK)status=CavityMag_midpoint(value,1.,0.,source,1,end,source->boundary_steps);
    if(status==CAVITY_MODEL_OK)for(int i=0;i<6;++i)q[i]=value[i];return status;
}
// Parent-anchored finite family. In particular a slice never re-anchors the
// RK4 preparation or independently normalizes the physical Gaussian field.
GPUFUN
int CavityMag_interval(xt_float_or_tpsa q[6],xt_float_or_tpsa_arg a,
        xt_float_or_tpsa_arg b,const CavityMagSource *source,int reverse) {
    if(xt_float_or_tpsa_is_zero(b-a))return CAVITY_MODEL_OK;
    xt_float_or_tpsa value[6];for(int i=0;i<6;++i)value[i]=q[i];int status=CAVITY_MODEL_OK;
    if(reverse) {
        if(!xt_float_or_tpsa_is_zero(b-source->anchor))status=CavityMag_map(value,b,source->anchor,source);
        if(status==CAVITY_MODEL_OK&&!xt_float_or_tpsa_is_zero(a-source->anchor))status=CavityMag_map(value,source->anchor,a,source);
    } else {
        if(!xt_float_or_tpsa_is_zero(a-source->anchor))status=CavityMag_map(value,a,source->anchor,source);
        if(status==CAVITY_MODEL_OK&&!xt_float_or_tpsa_is_zero(b-source->anchor))status=CavityMag_map(value,source->anchor,b,source);
    }
    if(status==CAVITY_MODEL_OK)for(int i=0;i<6;++i)q[i]=value[i];return status;
}
// Actual retained M1 physical lift, as MagneticPointField. The projected
// compact scalar must not inherit M2's cubic Ar or quartic Az at its faces.
GPUFUN
void CavityMag_potential(const xt_float_or_tpsa q[6],xt_float_or_tpsa_arg z,
        const CavityModelSource *source,xt_float_or_tpsa out[4]) {
    xt_float_or_tpsa real[5],imag[5],S[3];CavityModel_profile(z,source,real,imag);
    xt_float_or_tpsa sine=sin(q[2]),cosine=cos(q[2]);
    for(int i=0;i<3;++i)S[i]=real[i]*sine-imag[i]*cosine;
    CavityModelPotential potential;
    for(int i=0;i<4;++i){potential.a[i]=0.;for(int j=0;j<3;++j)potential.d[i][j]=0.;}
    potential.a[0]=q[0]*S[1]/2.;potential.a[1]=q[1]*S[1]/2.;
    potential.a[2]=-S[0]+(q[0]*q[0]+q[1]*q[1])*(S[0]+S[2])/4.;
    CavityModel_add_static(q,source,&potential);for(int i=0;i<4;++i)out[i]=potential.a[i];
}
GPUFUN
int CavityMag_bind(const CavityModelSource *source,xt_float_or_tpsa_arg Q,
        int64_t source_steps,int64_t factor_steps,int64_t frozen_coordinates,
        CavityMagSource *out) {
    if(source->rf_profile!=2||xt_float_or_tpsa_const_part(source->profile_sigma)<=0.
        ||!xt_float_or_tpsa_is_zero(source->dc_gradient))return CAVITY_MODEL_ARGUMENT;
    for(int64_t i=0;i<=source->order;++i)
        if((i!=1&&!xt_float_or_tpsa_is_zero(CavityModel_static_coefficient(source,i,0)))
            ||!xt_float_or_tpsa_is_zero(CavityModel_static_coefficient(source,i,1)))return CAVITY_MODEL_ARGUMENT;
    out->anchor=-source->half_span;out->width=source->profile_sigma;out->centre=source->profile_center;
    out->nu=source->carrier;out->slow=Q*source->forward;out->fast=Q*source->backward;
    // With exactly zero RF this is only a coordinate-unit chart. The unused
    // wave carrier cannot obstruct the well-defined zero-frequency magnet.
    if(xt_float_or_tpsa_is_zero(out->slow)&&xt_float_or_tpsa_is_zero(out->fast)
        &&xt_float_or_tpsa_const_part(out->nu)<=0.)out->nu=1.;
    out->b=Q*source->magnetic_scale*source->ks/source->k;out->g=0.;
    if(source->order>=1)out->g=Q*source->magnetic_scale*CavityModel_static_coefficient(source,1,0)
        /(source->length*source->k*source->k);
    out->phase=source->counter_phase;out->source_steps=source_steps;
    out->body_steps=factor_steps;out->boundary_steps=factor_steps;out->frozen_coordinates=frozen_coordinates;
    out->reference_beta=source->reference_beta;out->tolerance=2e-13;
    return CavityMag_validate(out);
}


// Prepared RF2/transverse2 envelope scalar tables. Fixed hardware, live amplitudes/phases.
typedef struct {
    double energy_min,energy_max,position_min,position_max;
    double nu,anchor,profile_width,profile_centre;
    int64_t n_modes,n_energy,n_position;
    GPUGLMEM const int64_t *modes_m,*modes_n;
    GPUGLMEM const double *body_real,*body_imag;
    GPUGLMEM const double *boundary_real,*boundary_imag;
} XTCavityEnvelopeTable;

/* Bounded recurrence avoids a temporary polynomial-basis array. */
GPUFUN void xt_cavity_envelope_cheb(
        int64_t n,xt_float_or_tpsa_arg x,
        xt_float_or_tpsa *value,xt_float_or_tpsa *derivative){
    xt_float_or_tpsa previous=1.,current=x,dprevious=0.,dcurrent=1.;
    if(n==0){*value=previous;*derivative=dprevious;return;}
    for(int64_t j=1;j<n;++j){
        xt_float_or_tpsa next=2.*x*current-previous;
        xt_float_or_tpsa dnext=2.*current+2.*x*dcurrent-dprevious;
        previous=current;current=next;dprevious=dcurrent;dcurrent=dnext;
    }
    *value=current;*derivative=dcurrent;
}

/* Return zero without touching outputs for an invalid interpolation point. */
GPUFUN int xt_cavity_envelope_scalar(
        const XTCavityEnvelopeTable *table,const xt_float_or_tpsa q[6],
        double position,xt_float_or_tpsa_arg slow,xt_float_or_tpsa_arg fast,
        xt_float_or_tpsa_arg phase_s,xt_float_or_tpsa_arg phase_f,
        int boundary,xt_float_or_tpsa *value,xt_float_or_tpsa gradient[6]){
    const double energy=xt_float_or_tpsa_const_part(q[5]);
    for(int i=0;i<6;++i)if(!CavityModel_scalar_finite(q[i]))return 0;
    if(!CavityModel_scalar_finite(slow)||!CavityModel_scalar_finite(fast)||
       !CavityModel_scalar_finite(phase_s)||!CavityModel_scalar_finite(phase_f))return 0;
    const double etol=8.*2.22044604925031308085e-16*
        fmax(1.,fmax(fabs(table->energy_min),fabs(table->energy_max)));
    const double ztol=8.*2.22044604925031308085e-16*
        fmax(1.,fmax(fabs(table->position_min),fabs(table->position_max)));
    if(!isfinite(energy)||!isfinite(position)||energy<=1.||
       energy<table->energy_min-etol||energy>table->energy_max+etol||
       position<table->position_min-ztol||position>table->position_max+ztol)
        return 0;
    if(table->n_modes<1||table->n_energy<1||table->n_position<1)return 0;
    const double clipped_z=fmin(table->position_max,fmax(table->position_min,position));
    const double z=2.*(clipped_z-table->position_min)/
        (table->position_max-table->position_min)-1.;
    const double escale=2./(table->energy_max-table->energy_min);
    /* Endpoint roundoff adjusts the primal only, preserving its derivatives. */
    xt_float_or_tpsa e=q[5]+(fmin(table->energy_max,
        fmax(table->energy_min,energy))-energy);
    e=(e-table->energy_min)*escale-1.;
    xt_float_or_tpsa powers[5]={slow,fast,slow*slow,slow*fast,fast*fast};
    xt_float_or_tpsa c[4]={0.,0.,0.,0.},ct[4]={0.,0.,0.,0.},ce[4]={0.,0.,0.,0.};
    GPUGLMEM const double *real=boundary?table->boundary_real:table->body_real;
    GPUGLMEM const double *imag=boundary?table->boundary_imag:table->body_imag;
    for(int64_t a=0;a<5;++a)for(int64_t mode=0;mode<table->n_modes;++mode){
        int64_t m=table->modes_m[mode],n=table->modes_n[mode];
        xt_float_or_tpsa phase=(m+n)*q[2]+m*phase_s+n*phase_f+
            (n-m)*table->nu*position;
        xt_float_or_tpsa cs=cos(phase),sn=sin(phase);
        for(int64_t invariant=0;invariant<4;++invariant){
            xt_float_or_tpsa re=0.,im=0.,dre=0.,dim=0.;
            for(int64_t ie=0;ie<table->n_energy;++ie){
                xt_float_or_tpsa te,dte;
                xt_cavity_envelope_cheb(ie,e,&te,&dte);
                for(int64_t iz=0;iz<table->n_position;++iz){
                    xt_float_or_tpsa tz,dtz;
                    xt_cavity_envelope_cheb(iz,z,&tz,&dtz);
                    int64_t index=((((a*table->n_modes+mode)*4+invariant)*
                        table->n_energy+ie)*table->n_position+iz);
                    re+=real[index]*te*tz;im+=imag[index]*te*tz;
                    dre+=real[index]*dte*tz*escale;
                    dim+=imag[index]*dte*tz*escale;
                }
            }
            c[invariant]+=powers[a]*(re*cs-im*sn);
            ct[invariant]-=powers[a]*(m+n)*(re*sn+im*cs);
            ce[invariant]+=powers[a]*(dre*cs-dim*sn);
        }
    }
    xt_float_or_tpsa invariants[4]={1.,q[0]*q[0]+q[1]*q[1],
        q[0]*q[3]+q[1]*q[4],q[3]*q[3]+q[4]*q[4]};
    xt_float_or_tpsa result=0.,g[6]={0.,0.,0.,0.,0.,0.};
    for(int i=0;i<4;++i){result+=c[i]*invariants[i];
        g[2]+=ct[i]*invariants[i];g[5]+=ce[i]*invariants[i];}
    g[0]=2.*c[1]*q[0]+c[2]*q[3];g[1]=2.*c[1]*q[1]+c[2]*q[4];
    g[3]=c[2]*q[0]+2.*c[3]*q[3];g[4]=c[2]*q[1]+2.*c[3]*q[4];
    if(!boundary){
        xt_float_or_tpsa p=sqrt((q[5]-1.)*(q[5]+1.));
        result+=-p+invariants[3]/(2.*p);
        g[3]+=q[3]/p;g[4]+=q[4]/p;
        g[5]-=q[5]/p+invariants[3]*q[5]/(2.*p*p*p);
    }
    if(!CavityModel_scalar_finite(result))return 0;
    for(int i=0;i<6;++i)if(!CavityModel_scalar_finite(g[i]))return 0;
    *value=result;for(int i=0;i<6;++i)gradient[i]=g[i];return 1;
}


// Finite map of one fixed prepared RF2/transverse2 scalar table. State,
// signed strengths and clock phases remain live. Coefficient-defining
// hardware is fixed independently of this solver and parent interval.
#define CAVITY_MODEL_PREPARED_ENVELOPE 21

typedef struct {
    const XTCavityEnvelopeTable *table;
    xt_float_or_tpsa slow,fast,phase_s,phase_f;
    int64_t body_steps,boundary_steps,frozen_coordinates;
    double tolerance,reference_beta;
} CavityEnvelopeSource;

GPUFUN
int CavityEnvelope_flow(const xt_float_or_tpsa q[6],double position,
        const CavityEnvelopeSource *source,int boundary,xt_float_or_tpsa out[6]) {
    xt_float_or_tpsa value,gradient[6];
    if(!xt_cavity_envelope_scalar(source->table,q,position,source->slow,
            source->fast,source->phase_s,source->phase_f,boundary,&value,gradient))
        return CAVITY_MODEL_DOMAIN;
    CavityAveraged_J(gradient,out);
    for(int i=0;i<6;++i)
        if(source->frozen_coordinates&((int64_t)1<<i))out[i]=0.;
    if(!boundary&&(source->frozen_coordinates&((int64_t)1<<2)))
        out[2]=1./source->reference_beta;
    return CavityModel_vector_finite(out,6)?CAVITY_MODEL_OK:CAVITY_MODEL_NONFINITE;
}

// Identical increment/fresh-defining-equation midpoint acceptance as the
// other compact averaged maps. q-primal remains an absolute residual.
GPUFUN
int CavityEnvelope_integrate(xt_float_or_tpsa q[6],double a,double b,
        double physical_z,int boundary,int64_t steps,const CavityEnvelopeSource *source) {
    if(steps<=0||source->tolerance<=0.)return CAVITY_MODEL_ARGUMENT;
    const double h=(b-a)/steps;
    for(int64_t cell=0;cell<steps;++cell) {
        const double z=boundary?physical_z:a+(cell+.5)*h;
        xt_float_or_tpsa stage[6],flow[6],full[6];
        int status=CavityEnvelope_flow(q,z,source,boundary,flow);
        if(status!=CAVITY_MODEL_OK)return status;
        for(int i=0;i<6;++i)stage[i]=.5*h*flow[i];
        int converged=0;
        for(int iteration=0;iteration<80;++iteration) {
            for(int i=0;i<6;++i)full[i]=q[i]+stage[i];
            status=CavityEnvelope_flow(full,z,source,boundary,flow);
            if(status!=CAVITY_MODEL_OK)return status;
            double defect=0.;
            for(int i=0;i<6;++i) {
                const xt_float_or_tpsa next=.5*h*flow[i];
                defect=fmax(defect,CavityModel_scalar_residual(stage[i]-next,q[i]+next,1));
                stage[i]=next;
            }
            if(defect<=source->tolerance){converged=1;break;}
        }
        if(!converged)return CAVITY_MODEL_ITERATIONS;
        for(int i=0;i<6;++i)full[i]=q[i]+stage[i];
        status=CavityEnvelope_flow(full,z,source,boundary,flow);
        if(status!=CAVITY_MODEL_OK)return status;
        for(int i=0;i<6;++i)
            if(CavityModel_scalar_residual(stage[i]-.5*h*flow[i],q[i]+stage[i],1)
                    >source->tolerance)return CAVITY_MODEL_ITERATIONS;
        for(int i=0;i<6;++i)q[i]+=h*flow[i];
    }
    return CAVITY_MODEL_OK;
}

GPUFUN
int CavityEnvelope_prefix(xt_float_or_tpsa q[6],double end,
        const CavityEnvelopeSource *source,int reverse) {
    const double start=source->table->position_min;
    if(end==start)return CAVITY_MODEL_OK;
    int status;
    if(!reverse) {
        status=CavityEnvelope_integrate(q,0.,1.,start,1,source->boundary_steps,source);
        if(status==CAVITY_MODEL_OK)
            status=CavityEnvelope_integrate(q,start,end,0.,0,source->body_steps,source);
        if(status==CAVITY_MODEL_OK)
            status=CavityEnvelope_integrate(q,0.,-1.,end,1,source->boundary_steps,source);
    } else {
        status=CavityEnvelope_integrate(q,0.,1.,end,1,source->boundary_steps,source);
        if(status==CAVITY_MODEL_OK)
            status=CavityEnvelope_integrate(q,end,start,0.,0,source->body_steps,source);
        if(status==CAVITY_MODEL_OK)
            status=CavityEnvelope_integrate(q,0.,-1.,start,1,source->boundary_steps,source);
    }
    return status;
}

GPUFUN
int CavityEnvelope_interval(xt_float_or_tpsa q[6],double a,double b,
        const CavityEnvelopeSource *source,int reverse) {
    xt_float_or_tpsa candidate[6],check[6];
    for(int i=0;i<6;++i)candidate[i]=q[i];
    // The owner validates both physical planes even for a zero-span call.
    int status=CavityEnvelope_flow(q,a,source,0,check);
    if(status==CAVITY_MODEL_OK)status=CavityEnvelope_flow(q,b,source,0,check);
    if(status!=CAVITY_MODEL_OK||a==b)return status;
    status=CavityEnvelope_prefix(candidate,reverse?b:a,source,1);
    if(status==CAVITY_MODEL_OK)
        status=CavityEnvelope_prefix(candidate,reverse?a:b,source,0);
    if(status==CAVITY_MODEL_OK)for(int i=0;i<6;++i)q[i]=candidate[i];
    return status;
}

#endif
