// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_MAGNET_OVERLAPPED_CAVITY_H
#define XTRACK_MAGNET_OVERLAPPED_CAVITY_H

#include "xtrack/beam_elements/elements_src/track_rf.h"
#include "xtrack/beam_elements/elements_src/magnet_overlapped_cavity_plan.h"

GPUFUN
void MagnetOverlappedCavity_prepare_rf(MagnetOverlappedCavityData el, RFPlan* plan, LocalParticle* part0)
{
    // Settings aligned by hand: not formatted
    // clang-format off
    RFSettings rf;
    RFSettings_init_default(&rf);
    rf.weight                 = 1.;
    rf.length                 = MagnetOverlappedCavityData_get_length(el);
    rf.voltage                = MagnetOverlappedCavityData_get_voltage(el);
    rf.frequency              = MagnetOverlappedCavityData_get_frequency(el);
    rf.harmonic               = MagnetOverlappedCavityData_get_harmonic(el);
    rf.lag                    = MagnetOverlappedCavityData_get_lag(el);
    rf.phase                  = MagnetOverlappedCavityData_get_phase(el);
    rf.lag_taper              = MagnetOverlappedCavityData_get_lag_taper(el);
    rf.phase_taper            = MagnetOverlappedCavityData_get_phase_taper(el);
    rf.absolute_time          = MagnetOverlappedCavityData_get_absolute_time(el);
    rf.num_kicks              = MagnetOverlappedCavityData_get_num_kicks(el);
    rf.model                  = MagnetOverlappedCavityData_get_model(el) >= 10 ? MODEL_ADAPTIVE : MagnetOverlappedCavityData_get_model(el);
    rf.integrator             = MagnetOverlappedCavityData_get_integrator(el);
    // clang-format on
    RFPlan_prepare(plan, &rf, part0);
    MagnetOverlappedCavity_prepare_cavity_plan(el, part0, 1., 0., &plan->cavity);
}

#endif  // XTRACK_MAGNET_OVERLAPPED_CAVITY_H
