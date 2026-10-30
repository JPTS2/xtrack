// Parent-frame handling for source-bearing cavity layout intervals.
#ifndef XTRACK_CAVITY_MODELS_SLICE_H
#define XTRACK_CAVITY_MODELS_SLICE_H

#include "xtrack/beam_elements/elements_src/cavity_models_adapter.h"
#include "xtrack/beam_elements/elements_src/track_misalignments.h"

// Ordinary DriftSliceCavity deliberately has no automatic parent transform:
// its legacy path must remain an untransformed drift. New continuous models
// instead integrate the physical parent source on this interval. Mirror the
// standard straight-element wrapper only for that branch.
GPUFUN
void CavityModels_track_source_interval(CavityData parent,
        LocalParticle* part0, double weight, double start_fraction) {
    // Protect callers of this interval helper too, before its manual frame.
    if(!CavityModels_slice_preflight(parent,part0))return;
    const double length=CavityData_get_length(parent)*weight;
    const double anchor=CavityData_get_rot_shift_anchor(parent)
        -CavityData_get_length(parent)*start_fraction;
    const double dx=CavityData_get_shift_x(parent);
    const double dy=CavityData_get_shift_y(parent);
    const double ds=CavityData_get_shift_s(parent);
    const double rx=CavityData_get_rot_x_rad(parent);
    const double ry=CavityData_get_rot_y_rad(parent);
    const double rz=CavityData_get_rot_s_rad(parent);
    const double rz0=CavityData_get_rot_s_rad_no_frame(parent);
    const int8_t reverse=LocalParticle_check_track_flag(part0,XS_FLAG_BACKTRACK);
    const int transformed=dx!=0. || dy!=0. || ds!=0. || rx!=0. || ry!=0.
        || rz!=0. || rz0!=0.;
    if (transformed) {
        if (!reverse)
            track_misalignment_entry_straight(part0,dx,dy,ds,ry,rx,rz0,
                                              anchor,length,rz,reverse);
        else
            track_misalignment_exit_straight(part0,dx,dy,ds,ry,rx,rz0,
                                             anchor,length,rz,reverse);
    }
    CavityModels_track_local_particle(parent,part0,weight,start_fraction);
    if (transformed) {
        if (!reverse)
            track_misalignment_exit_straight(part0,dx,dy,ds,ry,rx,rz0,
                                             anchor,length,rz,reverse);
        else
            track_misalignment_entry_straight(part0,dx,dy,ds,ry,rx,rz0,
                                              anchor,length,rz,reverse);
    }
}

#endif
