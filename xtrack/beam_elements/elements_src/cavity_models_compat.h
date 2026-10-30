// Scalar operations shared by ordinary and TPSA cavity tracking.
// The TPSA norm is only an implicit-stage stopping diagnostic. It is not
// an observable error bound or a change to the selected physical map.
#ifndef XTRACK_CAVITY_MODELS_COMPAT_H
#define XTRACK_CAVITY_MODELS_COMPAT_H

#include <math.h>
#ifdef CAVITY_MODEL_DEBUG
#include <stdio.h>
#endif
#include "xtrack/headers/track.h"

GPUFUN
double CavityModel_scalar_norm(xt_float_or_tpsa_arg value) {
#ifdef XTRACK_TPSA_TRACK
    // Include every represented coefficient: primal convergence alone can
    // leave state/parameter derivatives of an implicit stage unconverged.
    return mad_tpsa_nrm(value.ptr());
#else
    return fabs(value);
#endif
}

GPUFUN
int CavityModel_scalar_finite(xt_float_or_tpsa_arg value) {
    return isfinite(CavityModel_scalar_norm(value));
}


GPUFUN
int CavityModel_scalar_constant(xt_float_or_tpsa_arg value) {
#ifdef XTRACK_TPSA_TRACK
    // In particular this is well defined for a constant infinity: subtracting
    // its primal would manufacture NaN instead of testing Taylor dependence.
    return mad_tpsa_isval(value.ptr());
#else
    return 1;
#endif
}

// Implicit-stage residual recipe, separate from source/domain norms. Physical
// canonical state primals retain an absolute test. Taylor coefficients and
// transported auxiliary jets use coefficientwise absolute/relative scaling;
// their units and magnitudes can differ by many orders. This controls the
// finite defining equations, not an observable or derivative error bound.
GPUFUN
double CavityModel_scalar_residual(xt_float_or_tpsa_arg residual,
        xt_float_or_tpsa_arg stage,int physical_state) {
#ifdef XTRACK_TPSA_TRACK
    const int64_t count=mad_tpsa_len(residual.ptr(),0);
    double maximum=0.;
    for (int64_t i=0;i<count;++i) {
        const double delta=mad_tpsa_geti(residual.ptr(),i);
        const double value=mad_tpsa_geti(stage.ptr(),i);
        if (!isfinite(delta) || !isfinite(value)) return INFINITY;
        const double scale=(physical_state && i==0) ? 1. : 1.+fabs(value);
        maximum=fmax(maximum,fabs(delta)/scale);
    }
    return maximum;
#else
    if (!isfinite(residual) || !isfinite(stage)) return INFINITY;
    return fabs(residual)/(physical_state ? 1. : 1.+fabs(stage));
#endif
}

// Optional CPU diagnostics; absent from ordinary builds. This reports the
// actual residual without changing the solver or its stopping criterion.
GPUFUN
void CavityModel_debug_residual(const char *tag, int row, int stage,
        xt_float_or_tpsa_arg residual, xt_float_or_tpsa_arg scale) {
#ifdef CAVITY_MODEL_DEBUG
    double maximum=0.,magnitude=0.;
    int64_t index=0;
#ifdef XTRACK_TPSA_TRACK
    const int64_t count=mad_tpsa_len(residual.ptr(),0);
    for (int64_t i=0;i<count;++i) {
        const double coefficient=fabs(mad_tpsa_geti(residual.ptr(),i));
        if (coefficient>maximum) {
            maximum=coefficient;index=i;
            magnitude=fabs(mad_tpsa_geti(scale.ptr(),i));
        }
    }
#else
    maximum=fabs(residual);magnitude=fabs(scale);
#endif
    const double spacing=nextafter(magnitude,INFINITY)-magnitude;
    fprintf(stderr,"CAVITY_RESIDUAL %s row=%d stage=%d L1=%.17g "
        "max=%.17g coefficient=%lld primal=%.17g scale=%.17g "
        "ULP=%.17g max_over_ULP=%.17g\n",tag,row,stage,
        CavityModel_scalar_norm(residual),maximum,(long long)index,
        xt_float_or_tpsa_const_part(residual),magnitude,spacing,
        spacing>0. ? maximum/spacing : INFINITY);
#else
    (void)tag;(void)row;(void)stage;(void)residual;(void)scale;
#endif
}


// u,v here are stage increments. Optional diagnostics also show the naive
// full-stage subtraction at the same solution, exposing its large-base floor.
GPUFUN
void CavityModel_debug_stages(const char *tag,
        const xt_float_or_tpsa *value,const xt_float_or_tpsa *u,
        const xt_float_or_tpsa *v,const xt_float_or_tpsa *f,
        const xt_float_or_tpsa *g,xt_float_or_tpsa_arg h,int count) {
#ifdef CAVITY_MODEL_DEBUG
    const double a01=.25-sqrt(3.)/6.,a10=.25+sqrt(3.)/6.;
    double worst=-1.,physical=-1.;
    int row=0,stage=0,physical_row=0,physical_stage=0;
    for (int i=0;i<count;++i) {
        xt_float_or_tpsa du=u[i]-h*(.25*f[i]+a01*g[i]);
        xt_float_or_tpsa dv=v[i]-h*(a10*f[i]+.25*g[i]);
        const double nu=CavityModel_scalar_norm(du),nv=CavityModel_scalar_norm(dv);
        if (nu>worst) {worst=nu;row=i;stage=0;}
        if (nv>worst) {worst=nv;row=i;stage=1;}
        if (i<6) {
            const double pu=fabs(xt_float_or_tpsa_const_part(du));
            const double pv=fabs(xt_float_or_tpsa_const_part(dv));
            if (pu>physical) {physical=pu;physical_row=i;physical_stage=0;}
            if (pv>physical) {physical=pv;physical_row=i;physical_stage=1;}
        }
    }
    xt_float_or_tpsa residual=stage ? v[row]-h*(a10*f[row]+.25*g[row])
                                  : u[row]-h*(.25*f[row]+a01*g[row]);
    xt_float_or_tpsa full=value[row]+(stage ? v[row] : u[row]);
    CavityModel_debug_residual(tag,row,stage,residual,full);
    row=physical_row;stage=physical_stage;
    residual=stage ? v[row]-h*(a10*f[row]+.25*g[row])
                   : u[row]-h*(.25*f[row]+a01*g[row]);
    full=value[row]+(stage ? v[row] : u[row]);
    CavityModel_debug_residual("physical-increment-primal",row,stage,residual,full);
    residual=stage ? full-value[row]-h*(a10*f[row]+.25*g[row])
                   : full-value[row]-h*(.25*f[row]+a01*g[row]);
    CavityModel_debug_residual("naive-full-stage-subtraction",row,stage,residual,full);
#else
    (void)tag;(void)value;(void)u;(void)v;(void)f;(void)g;(void)h;(void)count;
#endif
}

#endif
