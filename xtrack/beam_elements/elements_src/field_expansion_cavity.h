// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_FIELD_EXPANSION_CAVITY_H
#define XTRACK_FIELD_EXPANSION_CAVITY_H

#include "xtrack/beam_elements/elements_src/track_rf.h"
#include "xtrack/beam_elements/elements_src/field_expansion_cavity_plan.h"

GPUFUN
void FieldExpansionCavity_prepare_rf(FieldExpansionCavityData el, RFPlan* plan, LocalParticle* part0)
{
    // Settings aligned by hand: not formatted
    // clang-format off
    RFSettings rf;
    RFSettings_init_default(&rf);
    rf.weight                 = 1.;
    rf.length                 = FieldExpansionCavityData_get_length(el);
    rf.voltage                = FieldExpansionCavityData_get_voltage(el);
    rf.frequency              = FieldExpansionCavityData_get_frequency(el);
    rf.harmonic               = FieldExpansionCavityData_get_harmonic(el);
    rf.lag                    = FieldExpansionCavityData_get_lag(el);
    rf.phase                  = FieldExpansionCavityData_get_phase(el);
    rf.lag_taper              = FieldExpansionCavityData_get_lag_taper(el);
    rf.phase_taper            = FieldExpansionCavityData_get_phase_taper(el);
    rf.absolute_time          = FieldExpansionCavityData_get_absolute_time(el);
    rf.num_kicks              = FieldExpansionCavityData_get_num_kicks(el);
    rf.model                  = FieldExpansionCavityData_get_model(el) >= 10 ? MODEL_ADAPTIVE : FieldExpansionCavityData_get_model(el);
    rf.integrator             = FieldExpansionCavityData_get_integrator(el);
    // clang-format on
    RFPlan_prepare(plan, &rf, part0);
    FieldExpansionCavity_prepare_cavity_plan(el, part0, 1., 0., &plan->cavity);
}

#endif  // XTRACK_FIELD_EXPANSION_CAVITY_H
