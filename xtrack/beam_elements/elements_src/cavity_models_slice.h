// Parent-frame handling for source-bearing cavity layout intervals.
#ifndef XTRACK_CAVITY_MODELS_SLICE_H
#define XTRACK_CAVITY_MODELS_SLICE_H

#include "xtrack/beam_elements/elements_src/cavity_models_adapter.h"
#include "xtrack/beam_elements/elements_src/track_misalignments.h"
#include "xtrack/beam_elements/elements_src/track_drift.h"

// Ordinary DriftSliceCavity deliberately has no automatic parent transform:
// its legacy path must remain an untransformed drift. New continuous models
// instead integrate the physical parent source on this interval, with the
// standard straight-element wrapper. When "adaptive" picks both in one call,
// the particles on the longitudinal map drift in the same transformed frame,
// as they do in a thick slice.
GPUFUN
void CavityModels_track_drift_slice(CavityData parent, LocalParticle* part0,
        double weight, double start_fraction) {
    CavityPlan plan;
    CavityPlan_prepare(parent,part0,weight,start_fraction,&plan);
    const int8_t reverse=LocalParticle_check_track_flag(part0,XS_FLAG_BACKTRACK);
    const double drift_length=(reverse ? -weight : weight)*CavityData_get_length(parent);
    if (!plan.adaptive && plan.model<10) {
        START_PER_PARTICLE_BLOCK(part0,part);
            Drift_single_particle(part,drift_length);
        END_PER_PARTICLE_BLOCK;
        return;
    }
    int any_physical=0;
    START_PER_PARTICLE_BLOCK(part0,part);
        if (CavityPlan_choose(&plan,parent,part)>=10) any_physical=1;
    END_PER_PARTICLE_BLOCK;
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
    const int transformed=any_physical && (dx!=0. || dy!=0. || ds!=0. || rx!=0. || ry!=0.
        || rz!=0. || rz0!=0.);
    if (transformed) {
        if (!reverse)
            track_misalignment_entry_straight(part0,dx,dy,ds,ry,rx,rz0,
                                              anchor,length,rz,reverse);
        else
            track_misalignment_exit_straight(part0,dx,dy,ds,ry,rx,rz0,
                                             anchor,length,rz,reverse);
    }
    CavityMagnusCache magnus_cache;
    magnus_cache.valid=0;
    START_PER_PARTICLE_BLOCK(part0,part);
        const int model=CavityPlan_choose(&plan,parent,part);
        if (model<10) Drift_single_particle(part,drift_length);
        else track_cavity_single_particle(parent,&plan,model,part,&magnus_cache);
    END_PER_PARTICLE_BLOCK;
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
