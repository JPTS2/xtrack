// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_TRACK_MAGNET_TRANSFORM_H
#define XTRACK_TRACK_MAGNET_TRANSFORM_H

#include "xtrack/headers/track.h"
#include "xtrack/headers/particle_states.h"
#include "xtrack/beam_elements/elements_src/track_magnet_structs.h"
#include "xtrack/beam_elements/elements_src/track_misalignments.h"

// Misalignment on one side of a plan-tracked element: before it (at_exit = 0) or after it
// (at_exit = 1), in the direction of travel. Returns 0 if the particles were
// lost (and the element must not be tracked).
GPUFUN
int8_t magnet_transform(LocalParticle* part0, const MagnetTransform* tr, int8_t at_exit) {
    if (!at_exit && tr->invalid) {
        START_PER_PARTICLE_BLOCK(part0, part);
            LocalParticle_set_state(part, XT_INVALID_THIN_SLICE_TRANSFORM);
        END_PER_PARTICLE_BLOCK;
        return 0;
    }
    int8_t const backtrack = LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK);
    // Backtracking crosses the exit side first
    uint8_t const entry_side = at_exit == backtrack;
    if (tr->curved) {
        if (entry_side) {
            track_misalignment_entry_curved(
                part0, tr->shift_x, tr->shift_y, tr->shift_s, tr->rot_y_rad, tr->rot_x_rad,
                tr->rot_s_rad_no_frame, tr->anchor, tr->length, tr->angle, tr->h,
                tr->rot_s_rad, backtrack
            );
        }
        else {
            track_misalignment_exit_curved(
                part0, tr->shift_x, tr->shift_y, tr->shift_s, tr->rot_y_rad, tr->rot_x_rad,
                tr->rot_s_rad_no_frame, tr->anchor, tr->length, tr->angle, tr->h,
                tr->rot_s_rad, backtrack
            );
        }
    }
    else {
        if (entry_side) {
            track_misalignment_entry_straight(
                part0, tr->shift_x, tr->shift_y, tr->shift_s, tr->rot_y_rad, tr->rot_x_rad,
                tr->rot_s_rad_no_frame, tr->anchor, tr->length, tr->rot_s_rad, backtrack
            );
        }
        else {
            track_misalignment_exit_straight(
                part0, tr->shift_x, tr->shift_y, tr->shift_s, tr->rot_y_rad, tr->rot_x_rad,
                tr->rot_s_rad_no_frame, tr->anchor, tr->length, tr->rot_s_rad, backtrack
            );
        }
    }
    return 1;
}

#endif
