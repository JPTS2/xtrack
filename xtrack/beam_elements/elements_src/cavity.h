// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_CAVITY_H
#define XTRACK_CAVITY_H

#include "xtrack/beam_elements/elements_src/track_rf.h"
#include "xtrack/beam_elements/elements_src/cavity_plan.h"

GPUFUN
void Cavity_prepare_rf(CavityData el, RFPlan* plan, LocalParticle* part0)
{
    // Settings aligned by hand: not formatted
    // clang-format off
    RFSettings rf;
    RFSettings_init_default(&rf);
    rf.weight                 = 1.;
    rf.length                 = CavityData_get_length(el);
    rf.voltage                = CavityData_get_voltage(el);
    rf.frequency              = CavityData_get_frequency(el);
    rf.harmonic               = CavityData_get_harmonic(el);
    rf.lag                    = CavityData_get_lag(el);
    rf.phase                  = CavityData_get_phase(el);
    rf.lag_taper              = CavityData_get_lag_taper(el);
    rf.phase_taper            = CavityData_get_phase_taper(el);
    rf.absolute_time          = CavityData_get_absolute_time(el);
    rf.num_kicks              = CavityData_get_num_kicks(el);
    rf.model                  = CavityData_get_model(el) >= 10 ? MODEL_ADAPTIVE : CavityData_get_model(el);
    rf.integrator             = CavityData_get_integrator(el);
    // clang-format on
    RFPlan_prepare(plan, &rf, part0);
    Cavity_prepare_cavity_plan(el, part0, 1., 0., &plan->cavity);
}

#endif  // XTRACK_CAVITY_H
