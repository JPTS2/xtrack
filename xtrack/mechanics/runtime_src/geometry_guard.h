// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_MECHANICS_GEOMETRY_GUARD_H
#define XTRACK_MECHANICS_GEOMETRY_GUARD_H
#include "xobjects/headers/common.h"
#ifdef XO_CONTEXT_CPU
    #include <stdint.h>
#endif

/* Gather storage bytes without assumptions about alignment or scalar type.
 * All supported Xsuite CPU/CUDA/OpenCL targets use little-endian storage.
 */
/*gpukern*/ void mechanics_guard_gather(
    const int64_t n,
    /*gpuglmem*/ const int64_t* offsets,
    /*gpuglmem*/ const int8_t* sizes,
    /*gpuglmem*/ const int8_t* buffer,
    /*gpuglmem*/ uint64_t* out) {
    VECTORIZE_OVER(i, n)
        uint64_t word = 0;
        for (int j = 0; j < sizes[i]; j++) {
            word |= ((uint64_t)(uint8_t)buffer[offsets[i]+j]) << (8*j);
        }
        out[i] = word;
    END_VECTORIZE
}
#endif
