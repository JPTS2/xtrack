// Scalar operations shared by ordinary and TPSA cavity tracking.
// The TPSA norm is only an implicit-stage stopping diagnostic. It is not
// an observable error bound or a change to the selected physical map.
#ifndef XTRACK_CAVITY_MODELS_COMPAT_H
#define XTRACK_CAVITY_MODELS_COMPAT_H

#include <math.h>
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

#endif
