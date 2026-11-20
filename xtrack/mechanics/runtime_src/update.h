// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_MECHANICS_UPDATE_H
#define XTRACK_MECHANICS_UPDATE_H

#include "xobjects/headers/common.h"
#ifdef XO_CONTEXT_CPU
    #include <math.h>
    #include <stdint.h>
#endif
#ifdef XO_CONTEXT_CL
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/* Publish the shared reference clock at a kernel boundary. No particle thread
 * writes shared clock fields while another particle may still read them. */
/*gpukern*/ void mechanics_publish_clock(
        /*gpuglmem*/ int8_t* buffer,
        /*gpuglmem*/ const int64_t* offsets,
        const double time_s, const double reference_turns, const double t_sim) {
    VECTORIZE_OVER(thread, 1)
        *((/*gpuglmem*/ double*)(buffer+offsets[0])) = time_s;
        *((/*gpuglmem*/ double*)(buffer+offsets[1])) = reference_turns;
        *((/*gpuglmem*/ double*)(buffer+offsets[2])) = t_sim;
    END_VECTORIZE
}

/* A local affine deformation is stored as [d(3), E(9)], E = R - I.
 * Composition uses E_left + E_right + E_left E_right, avoiding the
 * subtraction of nearly equal world matrices for nanometre motion.
 */
GPUFUN void mechanics_compose(const double left[12], const double right[12],
                             double out[12]) {
    double result[12];
    for (int i = 0; i < 3; i++) {
        result[i] = left[i] + right[i];
        for (int j = 0; j < 3; j++)
            result[i] += left[3 + 3*i + j] * right[j];
        for (int j = 0; j < 3; j++) {
            double value = left[3 + 3*i + j] + right[3 + 3*i + j];
            for (int k = 0; k < 3; k++)
                value += left[3 + 3*i + k] * right[3 + 3*k + j];
            result[3 + 3*i + j] = value;
        }
    }
    for (int i = 0; i < 12; i++) out[i] = result[i];
}

/* relation = [C(9), r(3)], C=Rp^T Rc, r=Rp^T(pc-pp). */
GPUFUN void mechanics_inherit(const double parent[12], const double relation[12],
                             double out[12]) {
    double intermediate[9], translated[3];
    for (int i = 0; i < 3; i++) {
        translated[i] = parent[i];
        for (int k = 0; k < 3; k++)
            translated[i] += parent[3 + 3*i + k] * relation[9+k];
        for (int j = 0; j < 3; j++) {
            double value = 0.;
            for (int k = 0; k < 3; k++)
                value += parent[3 + 3*i + k] * relation[3*k+j];
            intermediate[3*i+j] = value;
        }
    }
    for (int i = 0; i < 3; i++) {
        out[i] = 0.;
        for (int k = 0; k < 3; k++) out[i] += relation[3*k+i] * translated[k];
        for (int j = 0; j < 3; j++) {
            double value = 0.;
            for (int k = 0; k < 3; k++)
                value += relation[3*k+i] * intermediate[3*k+j];
            out[3 + 3*i+j] = value;
        }
    }
}

GPUFUN void mechanics_rotation_product(const double left[9], const double right[9],
                                      double out[9]) {
    double result[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double value = left[3*i+j] + right[3*i+j];
            for (int k = 0; k < 3; k++) value += left[3*i+k] * right[3*k+j];
            result[3*i+j] = value;
        }
    }
    for (int i = 0; i < 9; i++) out[i] = result[i];
}

GPUFUN void mechanics_rotation_increment(const double phi, const double theta,
                                        const double psi, double out[9]) {
    double pitch[9] = {0.}, yaw[9] = {0.}, roll[9] = {0.}, intermediate[9];
    double half_sine = sin(phi/2.);
    pitch[4] = pitch[8] = -2.*half_sine*half_sine;
    pitch[5] = sin(phi);
    pitch[7] = -pitch[5];
    half_sine = sin(theta/2.);
    yaw[0] = yaw[8] = -2.*half_sine*half_sine;
    yaw[2] = sin(theta);
    yaw[6] = -yaw[2];
    half_sine = sin(psi/2.);
    roll[0] = roll[4] = -2.*half_sine*half_sine;
    roll[3] = sin(psi);
    roll[1] = -roll[3];
    mechanics_rotation_product(yaw, pitch, intermediate);
    mechanics_rotation_product(intermediate, roll, out);
}

/* General source values are packed once per freeze, never per particle.
 * CSR source_rows preserves insertion order while the value rows retain their
 * source/target layout. Values contain local XYZ and survey pitch/yaw/roll.
 * This bounded input also supports sampled PSD/response records without an
 * implicit GPU history or per-body host matrix construction.
 */
/*gpukern*/ void mechanics_source_values(
        const int64_t n,
        /*gpuglmem*/ const int64_t* source_starts,
        /*gpuglmem*/ const int64_t* source_rows,
        /*gpuglmem*/ const double* values,
        /*gpuglmem*/ double* dynamic_local) {
    VECTORIZE_OVER(body, n)
    double result[12] = {0.};
    for (int64_t source = source_starts[body]; source < source_starts[body+1]; source++) {
        const int64_t row = source_rows[source];
        for (int i = 0; i < 3; i++) result[i] += values[6*row+i];
        double phi = values[6*row+3], theta = values[6*row+4], psi = values[6*row+5];
        if (phi != 0. || theta != 0. || psi != 0.) {
            double left[9], right[9], combined[9];
            mechanics_rotation_increment(phi, theta, psi, left);
            for (int i = 0; i < 9; i++) right[i] = result[3+i];
            mechanics_rotation_product(left, right, combined);
            for (int i = 0; i < 9; i++) result[3+i] = combined[i];
        }
    }
    for (int i = 0; i < 12; i++) dynamic_local[12*body+i] = result[i];
    END_VECTORIZE
}

/* Analytic sources use body-local CSR rows, in source insertion order.
 * Each source_data row is [component, amplitude, omega, phase, px, py, pz].
 * component 0..2 drives a local translation; 3..5 drives survey pitch/yaw/roll;
 * -1 drives translation with a preprojected local polarization. For plane waves
 * the prepared phase includes the nominal-position propagation delay.
 * No device buffers, body dictionaries or waveform arrays return to the host.
 */
/*gpukern*/ void mechanics_sources(
        const int64_t n,
        /*gpuglmem*/ const int64_t* source_starts,
        /*gpuglmem*/ const double* source_data,
        const double time_s,
        /*gpuglmem*/ double* dynamic_local) {
    VECTORIZE_OVER(body, n)
    double result[12] = {0.};
    for (int64_t source = source_starts[body]; source < source_starts[body+1]; source++) {
        int component = (int)source_data[7*source];
        double value = source_data[7*source+1] * sin(
            source_data[7*source+2] * time_s + source_data[7*source+3]);
        if (component == -1) {
            for (int i = 0; i < 3; i++)
                result[i] += value * source_data[7*source+4+i];
        } else if (component < 3) {
            result[component] += value;
        } else {
            double rotation[9] = {0.};
            double sine = sin(value), half_sine = sin(value/2.);
            double cosine_minus_one = -2.*half_sine*half_sine;
            if (component == 3) {
                rotation[4] = cosine_minus_one;
                rotation[8] = cosine_minus_one;
                rotation[5] = sine;
                rotation[7] = -sine;
            } else if (component == 4) {
                rotation[0] = cosine_minus_one;
                rotation[8] = cosine_minus_one;
                rotation[2] = sine;
                rotation[6] = -sine;
            } else {
                rotation[0] = cosine_minus_one;
                rotation[4] = cosine_minus_one;
                rotation[1] = -sine;
                rotation[3] = sine;
            }
            double combined[9];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    double current = rotation[3*i+j] + result[3 + 3*i+j];
                    for (int k = 0; k < 3; k++)
                        current += rotation[3*i+k] * result[3 + 3*k+j];
                    combined[3*i+j] = current;
                }
            }
            for (int i = 0; i < 9; i++) result[3+i] = combined[i];
        }
    }
    for (int i = 0; i < 12; i++) dynamic_local[12*body+i] = result[i];
    END_VECTORIZE
}

/* Each body thread traverses immutable root-to-body indices independently.
 * No thread reads a parent's output, so no in-kernel global barrier is needed.
 * Absolute motion retains inherited static placement but replaces inherited
 * vibration, exactly as the public reference model specifies.
 */
/*gpukern*/ void mechanics_body_update(
        const int64_t n,
        /*gpuglmem*/ const int64_t* paths_offsets,
        /*gpuglmem*/ const int64_t* paths,
        /*gpuglmem*/ const double* relations,
        /*gpuglmem*/ const double* static_local,
        /*gpuglmem*/ const double* dynamic_local,
        /*gpuglmem*/ const int64_t* absolute,
        /*gpuglmem*/ double* out) {
    VECTORIZE_OVER(body, n)
    double current[12] = {0.}, fixed[12] = {0.};
    for (int64_t position = paths_offsets[body];
         position < paths_offsets[body+1]; position++) {
        int64_t node = paths[position];
        double relation[12], local_static_transform[12], dynamic[12], inherited[12], inherited_fixed[12];
        for (int i = 0; i < 12; i++) {
            relation[i] = relations[12*node+i];
            local_static_transform[i] = static_local[12*node+i];
            dynamic[i] = dynamic_local[12*node+i];
        }
        mechanics_inherit(fixed, relation, inherited_fixed);
        mechanics_compose(inherited_fixed, local_static_transform, fixed);
        if (absolute[node]) {
            // Dynamic translation is added in nominal body axes; its rotation
            // acts on the static orientation, without rotating its translation.
            for (int i = 0; i < 3; i++) current[i] = fixed[i] + dynamic[i];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    double value = fixed[3 + 3*i+j] + dynamic[3 + 3*i+j];
                    for (int k = 0; k < 3; k++)
                        value += dynamic[3 + 3*i+k] * fixed[3 + 3*k+j];
                    current[3 + 3*i+j] = value;
                }
            }
        } else {
            mechanics_inherit(current, relation, inherited);
            // Relative translation adds to local static translation; dynamic
            // rotation left-multiplies the local static orientation only.
            double combined[12];
            for (int i = 0; i < 3; i++) combined[i] = local_static_transform[i] + dynamic[i];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    double value = local_static_transform[3 + 3*i+j] + dynamic[3 + 3*i+j];
                    for (int k = 0; k < 3; k++)
                        value += dynamic[3 + 3*i+k] * local_static_transform[3 + 3*k+j];
                    combined[3 + 3*i+j] = value;
                }
            }
            mechanics_compose(inherited, combined, current);
        }
    }
    for (int i = 0; i < 12; i++) out[12*body+i] = current[i];
    END_VECTORIZE
}

/* geometry = [A(9), lever(3), entry_translation(3), entry_rotation(9)].
 * Offsets address prepared numeric alignment scalar fields in one buffer.
 * A distinct output row is retained for bounded guard/bookkeeping reads.
 */
/*gpukern*/ void mechanics_lower(
        const int64_t n,
        /*gpuglmem*/ const int64_t* body_indices,
        /*gpuglmem*/ const double* geometry,
        /*gpuglmem*/ const double* baseline,
        /*gpuglmem*/ const int64_t* offsets,
        /*gpuglmem*/ const double* bodyout,
        /*gpuglmem*/ int8_t* buffer,
        /*gpuglmem*/ double* values) {
    VECTORIZE_OVER(attachment, n)
    int64_t body = body_indices[attachment];
    double deformation[12], geom[24], result[7];
    int nonzero = 0;
    for (int i = 0; i < 12; i++) {
        deformation[i] = bodyout[12*body+i];
        nonzero |= deformation[i] != 0.;
    }
    if (!nonzero) {
        // The exact adopted deliberate values, including the anchor, survive.
        for (int i = 0; i < 7; i++) result[i] = baseline[7*attachment+i];
    } else {
        for (int i = 0; i < 24; i++) geom[i] = geometry[24*attachment+i];
        double local_shift[3], ae[9], increment[9], rotation[9];
        for (int i = 0; i < 3; i++) {
            local_shift[i] = deformation[i];
            for (int k = 0; k < 3; k++)
                local_shift[i] += deformation[3 + 3*i+k] * geom[9+k];
            for (int j = 0; j < 3; j++) {
                double value = 0.;
                for (int k = 0; k < 3; k++)
                    value += geom[3*i+k] * deformation[3 + 3*k+j];
                ae[3*i+j] = value;
            }
        }
        for (int i = 0; i < 3; i++) {
            result[i] = geom[12+i];
            for (int k = 0; k < 3; k++) result[i] += geom[3*i+k] * local_shift[k];
            for (int j = 0; j < 3; j++) {
                double value = 0.;
                for (int k = 0; k < 3; k++) value += ae[3*i+k] * geom[3*j+k];
                increment[3*i+j] = value;
            }
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double value = geom[15 + 3*i+j];
                for (int k = 0; k < 3; k++)
                    value += increment[3*i+k] * geom[15 + 3*k+j];
                rotation[3*i+j] = value;
            }
        }
        double cosine_pitch = hypot(rotation[3], rotation[4]);
        result[3] = atan2(rotation[5], cosine_pitch);
        result[5] = atan2(rotation[3], rotation[4]);
        result[4] = atan2(rotation[2], rotation[8]);
        if (cosine_pitch < .01) {
            // Near pitch lock, theta/psi are ill-conditioned separately. Their
            // observable combination is recovered from the well-scaled x/z
            // entries, retaining a rotation equivalent to the input matrix.
            result[4] = atan2(-rotation[6], rotation[0])
                - atan2(sin(result[3])*sin(result[5]), cos(result[5]));
            result[4] = atan2(sin(result[4]), cos(result[4]));
        }
        result[6] = 0.;
    }
    for (int i = 0; i < 7; i++) {
        values[7*attachment+i] = result[i];
        /*gpuglmem*/ double* destination = (/*gpuglmem*/ double*)(buffer + offsets[7*attachment+i]);
        *destination = result[i];
    }
    END_VECTORIZE
}

#endif
