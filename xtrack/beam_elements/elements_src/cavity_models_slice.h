// Parent-frame handling for source-bearing cavity layout intervals.
#ifndef XTRACK_CAVITY_MODELS_SLICE_H
#define XTRACK_CAVITY_MODELS_SLICE_H

#include "xtrack/beam_elements/elements_src/cavity_models_adapter.h"
#include "xtrack/beam_elements/elements_src/track_misalignments.h"
#include "xtrack/beam_elements/elements_src/track_drift.h"

// Ordinary DriftSliceCavity deliberately has no automatic parent transform:
// its legacy path must remain an untransformed drift. New continuous models
// instead integrate the physical parent source on this interval, with the
// standard straight-element wrapper.
GPUFUN
void CavityModels_track_drift_slice(const CavityPlan* prepared, LocalParticle* part0,
        double weight, double start_fraction) {
    const CavityPlan plan=*prepared;
    const RFModelFields* parent=&plan.fields;
    const int8_t reverse=LocalParticle_check_track_flag(part0,XS_FLAG_BACKTRACK);
    const double drift_length=(reverse ? -weight : weight)*parent->length;
    if (plan.model<10) {
        START_PER_PARTICLE_BLOCK(part0,part);
            Drift_single_particle(part,drift_length);
        END_PER_PARTICLE_BLOCK;
        return;
    }
    const double length=parent->length*weight;
    const double anchor=parent->rot_shift_anchor
        -parent->length*start_fraction;
    const double dx=parent->shift_x;
    const double dy=parent->shift_y;
    const double ds=parent->shift_s;
    const double rx=parent->rot_x_rad;
    const double ry=parent->rot_y_rad;
    const double rz=parent->rot_s_rad;
    const double rz0=parent->rot_s_rad_no_frame;
    const int transformed=(dx!=0. || dy!=0. || ds!=0. || rx!=0. || ry!=0.
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
        const int model=plan.model;
        if (model<10) Drift_single_particle(part,drift_length);
        else track_cavity_single_particle(&plan,model,part,&magnus_cache);
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
