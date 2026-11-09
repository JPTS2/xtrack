// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_MAGNET_OVERLAPPED_CAVITY_PLAN_H
#define XTRACK_MAGNET_OVERLAPPED_CAVITY_PLAN_H

#include "xtrack/beam_elements/elements_src/cavity_models_adapter.h"

// Slices call this with their parent, so it is not copied into slice headers.
GPUFUN
void MagnetOverlappedCavity_prepare_cavity_plan(MagnetOverlappedCavityData el, LocalParticle* part0,
                                double weight, double slice_start, CavityPlan* plan) {
    RF_MODEL_FIELDS_FILL_RF(MagnetOverlappedCavityData, el, &plan->fields);
    RF_MODEL_FIELDS_FILL_WAVES(MagnetOverlappedCavityData, el, &plan->fields);
    RF_MODEL_FIELDS_FILL_STATIC(MagnetOverlappedCavityData, el, &plan->fields);
    CavityPlan_prepare(part0, weight, slice_start, plan);
}

#endif
