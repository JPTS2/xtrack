// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_RFMULTIPOLE_H
#define XTRACK_RFMULTIPOLE_H

#include "xtrack/beam_elements/elements_src/track_rf.h"

GPUFUN
void RFMultipole_prepare_rf(RFMultipoleData el, RFPlan* plan, LocalParticle* part0)
{
    // Settings aligned by hand: not formatted
    // clang-format off
    RFSettings rf;
    RFSettings_init_default(&rf);
    rf.weight                 = 1.;
    rf.length                 = RFMultipoleData_get_length(el);
    rf.voltage                = RFMultipoleData_get_voltage(el);
    rf.frequency              = RFMultipoleData_get_frequency(el);
    rf.lag                    = RFMultipoleData_get_lag(el);
    rf.phase                  = RFMultipoleData_get_phase(el);
    rf.absolute_time          = RFMultipoleData_get_absolute_time(el);
    rf.order                  = RFMultipoleData_get_order(el);
    rf.knl                    = RFMultipoleData_getp1_knl(el, 0);
    rf.ksl                    = RFMultipoleData_getp1_ksl(el, 0);
    rf.pn                     = RFMultipoleData_getp1_pn(el, 0);
    rf.ps                     = RFMultipoleData_getp1_ps(el, 0);
    rf.phase_n                = RFMultipoleData_getp1_phase_n(el, 0);
    rf.phase_s                = RFMultipoleData_getp1_phase_s(el, 0);
    rf.num_kicks              = RFMultipoleData_get_num_kicks(el);
    rf.model                  = RFMultipoleData_get_length(el) == 0. ? MODEL_KICK_ONLY : RFMultipoleData_get_model(el);
    rf.integrator             = RFMultipoleData_get_integrator(el);
    // clang-format on
    RFPlan_prepare(plan, &rf, part0);
    plan->is_thick = RFMultipoleData_get_length(el) != 0.;
}

#endif
