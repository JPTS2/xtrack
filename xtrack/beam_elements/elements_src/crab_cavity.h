// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_CRABCAVITY_H
#define XTRACK_CRABCAVITY_H

#include "xtrack/beam_elements/elements_src/track_rf.h"

GPUFUN
void CrabCavity_prepare_rf(CrabCavityData el, RFPlan* plan, LocalParticle* part0)
{
    // Settings aligned by hand: not formatted
    // clang-format off
    RFSettings rf;
    RFSettings_init_default(&rf);
    rf.weight                 = 1.;
    rf.length                 = CrabCavityData_get_length(el);
    rf.frequency              = CrabCavityData_get_frequency(el);
    rf.transverse_voltage     = CrabCavityData_get_crab_voltage(el);
    rf.transverse_lag         = CrabCavityData_get_lag(el);
    rf.transverse_phase       = CrabCavityData_get_phase(el);
    rf.lag_taper              = CrabCavityData_get_lag_taper(el);
    rf.phase_taper            = CrabCavityData_get_phase_taper(el);
    rf.absolute_time          = CrabCavityData_get_absolute_time(el);
    rf.num_kicks              = CrabCavityData_get_num_kicks(el);
    rf.model                  = CrabCavityData_get_model(el);
    rf.integrator             = CrabCavityData_get_integrator(el);
    // clang-format on
    RFPlan_prepare(plan, &rf, part0);
}

#endif  // XTRACK_CRABCAVITY_H
