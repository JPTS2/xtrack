// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_CAVITY_H
#define XTRACK_CAVITY_H

#include "xtrack/beam_elements/elements_src/track_rf.h"
#include "xtrack/beam_elements/elements_src/cavity_models_adapter.h"

GPUFUN
void Cavity_track_local_particle(CavityData el, LocalParticle* part0)
{
    if (CavityData_get_model(el) >= 10) {
        CavityModels_track_local_particle(el, part0, 1., 0.);
        return;
    }
    // Legacy RF recipes have no simultaneous static-field Hamiltonian.
    // Do not silently ignore newly exposed magnetic strengths after Line.set
    // or an indexed expression update. Zero fields leave the legacy path exact.
    int static_field=!xt_float_or_tpsa_is_zero(CavityData_get_ks(el));
    for (int64_t i=0; i<CavityData_len_knl(el); ++i)
        static_field |= CavityData_get_knl(el,i)!=0.;
    for (int64_t i=0; i<CavityData_len_ksl(el); ++i)
        static_field |= CavityData_get_ksl(el,i)!=0.;
    // A zero constant with a nonzero static-field TPSA jet is still an
    // unsupported simultaneous field in the legacy RF recipes.
    for (int64_t i=0; i<CavityData_len_knl_tpsa(el); ++i)
        static_field |= CavityData_get_knl_tpsa(el,i)!=0;
    for (int64_t i=0; i<CavityData_len_ksl_tpsa(el); ++i)
        static_field |= CavityData_get_ksl_tpsa(el,i)!=0;
    // Legacy voltage-only projections support their existing P16/uniform
    // reductions, but cannot silently consume a requested Gaussian source.
    if (static_field || CavityData_get_rf_profile(el)<0
                     || CavityData_get_rf_profile(el)>1) {
        START_PER_PARTICLE_BLOCK(part0,part)
            LocalParticle_set_state(part,-99744);
        END_PER_PARTICLE_BLOCK;
        return;
    }
    track_rf_particles(
        /*weight*/                1.,
        /*part0*/                 part0,
        /*length*/                CavityData_get_length(el),
        /*voltage*/               CavityData_get_voltage(el),
        /*frequency*/             CavityData_get_frequency(el),
        /*harmonic*/              CavityData_get_harmonic(el),
        /*lag*/                   CavityData_get_lag(el),
        /*phase*/                 CavityData_get_phase(el),
        /*transverse_voltage*/    0.,
        /*transverse_lag*/        0.,
        /*transverse_phase*/      0.,
        /*absolute_time*/         CavityData_get_absolute_time(el),
        /*order*/                 -1, // not used here
        /*knl*/                   NULL,
        /*ksl*/                   NULL,
        /*pn*/                    NULL,
        /*ps*/                    NULL,
        /*phase_n*/               NULL,
        /*phase_s*/               NULL,
        /*num_kicks*/             CavityData_get_num_kicks(el),
        /*model*/                 CavityData_get_model(el),
        /*default_model*/         6, // drift-kick-drift-expanded
        /*integrator*/            CavityData_get_integrator(el),
        /*default_integrator*/    3, // Uniform
        /*radiation_flag*/        0, // not used here
        /*radiation_flag_parent*/ 0, // not used here
        /*lag_taper*/             CavityData_get_lag_taper(el),
        /*phase_taper*/           CavityData_get_phase_taper(el),
        /*body_active*/           1,
        /*edge_entry_active*/     0, // not used here
        /*edge_exit_active*/      0  // not used here
    );

}

#endif  // XTRACK_CAVITY_H
