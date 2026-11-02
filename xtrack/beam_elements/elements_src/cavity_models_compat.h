// Scalar operations shared by ordinary and TPSA cavity tracking.
#ifndef XTRACK_CAVITY_MODELS_COMPAT_H
#define XTRACK_CAVITY_MODELS_COMPAT_H

#include <math.h>
#include "xtrack/headers/track.h"

GPUFUN
double CavityModel_scalar_norm(xt_float_or_tpsa_arg value) {
#ifdef XTRACK_TPSA_TRACK
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

// Convergence test on the primal (constant part) only; Taylor coefficients
// are settled by the fixed sweeps that follow primal convergence.
GPUFUN
double CavityModel_primal_residual(xt_float_or_tpsa_arg residual) {
#ifdef XTRACK_TPSA_TRACK
    const int64_t count=mad_tpsa_len(residual.ptr(),0);
    for (int64_t i=0;i<count;++i)
        if (!isfinite(mad_tpsa_geti(residual.ptr(),i))) return INFINITY;
#endif
    const double value=xt_float_or_tpsa_const_part(residual);
    return isfinite(value) ? fabs(value) : INFINITY;
}


#endif
