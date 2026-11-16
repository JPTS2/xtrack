// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
// Slices of the physical cavity models, tracked as intervals of the parent.
#ifndef XTRACK_CAVITY_MODELS_SLICE_H
#define XTRACK_CAVITY_MODELS_SLICE_H

#include "xtrack/beam_elements/elements_src/cavity_models_adapter.h"
#include "xtrack/beam_elements/elements_src/track_misalignments.h"
#include "xtrack/beam_elements/elements_src/track_drift.h"

// A drift slice of a physical model, or of a MagnetOverlappedCavity, carries the
// parent's fields over its interval, in the parent's frame. A Cavity's drift slice
// of a longitudinal model is a drift.
GPUFUN
void CavityModels_track_drift_slice(
    const CavityPlan* prepared,
    LocalParticle* part0,
    double weight,
    double start_fraction
) {
    const CavityPlan plan = *prepared;
    const RFModelFields* parent = &plan.fields;
    const int8_t reverse = LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK);
    const int static_fields = plan.model < 10 && parent->knl;
    if (plan.model < 10 && !static_fields) {
        const double drift_length = (reverse ? -weight : weight) * parent->length;
        START_PER_PARTICLE_BLOCK(part0, part)
            ;
            Drift_single_particle(part, drift_length);
        END_PER_PARTICLE_BLOCK;
        return;
    }
    const double length = parent->length * weight;
    const double anchor = parent->rot_shift_anchor - parent->length * start_fraction;
    const double dx = parent->shift_x;
    const double dy = parent->shift_y;
    const double ds = parent->shift_s;
    const double rx = parent->rot_x_rad;
    const double ry = parent->rot_y_rad;
    const double rz = parent->rot_s_rad;
    const double rz0 = parent->rot_s_rad_no_frame;
    const int transformed =
        (dx != 0. || dy != 0. || ds != 0. || rx != 0. || ry != 0. || rz != 0.
         || rz0 != 0.);
    if (transformed) {
        if (!reverse)
            track_misalignment_entry_straight(
                part0, dx, dy, ds, ry, rx, rz0, anchor, length, rz, reverse
            );
        else
            track_misalignment_exit_straight(
                part0, dx, dy, ds, ry, rx, rz0, anchor, length, rz, reverse
            );
    }
    if (static_fields) {
        CavityModels_track_longitudinal_magnet(
            part0, &plan, NULL, weight, CAVITY_MAGNET_DRIFT
        );
    }
    else {
        CavityMagnusCache magnus_cache;
        CavityMagnusCache_init(&magnus_cache);
        START_PER_PARTICLE_BLOCK(part0, part)
            ;
            track_cavity_single_particle(&plan, plan.model, part, &magnus_cache);
        END_PER_PARTICLE_BLOCK;
        CavityMagnusCache_free(&magnus_cache);
    }
    if (transformed) {
        if (!reverse)
            track_misalignment_exit_straight(
                part0, dx, dy, ds, ry, rx, rz0, anchor, length, rz, reverse
            );
        else
            track_misalignment_entry_straight(
                part0, dx, dy, ds, ry, rx, rz0, anchor, length, rz, reverse
            );
    }
}

#endif
