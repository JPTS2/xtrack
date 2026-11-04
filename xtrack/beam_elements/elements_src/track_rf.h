// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_TRACK_RF_H
#define XTRACK_TRACK_RF_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_magnet_drift.h"
#include "xtrack/beam_elements/elements_src/track_magnet_configure.h"
#include "xtrack/beam_elements/elements_src/integrator.h"
#include "xtrack/beam_elements/elements_src/track_magnet_transform.h"

// What a Cavity, CrabCavity or RFMultipole sets for its plan; the rest keeps
// RFSettings_init_default.
typedef struct RFSettings_s {
    double weight, length;
    xt_float_or_tpsa voltage, frequency, harmonic, lag, phase;
    xt_float_or_tpsa lag_taper, phase_taper;
    double transverse_voltage, transverse_lag, transverse_phase;
    int64_t absolute_time, order, num_kicks;
    GPUGLMEM const double* knl;
    GPUGLMEM const double* ksl;
    GPUGLMEM const double* pn;
    GPUGLMEM const double* ps;
    GPUGLMEM const double* phase_n;
    GPUGLMEM const double* phase_s;
    int8_t model, integrator;
} RFSettings;

// The kick-drift map of one particle, for one direction of travel.
typedef struct RFBody_s {
    double length;
    xt_float_or_tpsa voltage, frequency, harmonic, lag, phase;
    double transverse_voltage, transverse_lag, transverse_phase;
    int64_t absolute_time, order, num_kicks;
    double factor_knl_ksl;
    GPUGLMEM const double* knl;
    GPUGLMEM const double* ksl;
    GPUGLMEM const double* pn;
    GPUGLMEM const double* ps;
    GPUGLMEM const double* phase_n;
    GPUGLMEM const double* phase_s;
    int8_t drift_model, integrator;
    uint8_t kill_energy_kick;
} RFBody;

// One physical model of a cavity: "adaptive" keeps one per model it can pick.
typedef struct CavityChoice_s {
    int controls;
    int64_t cells;
} CavityChoice;

// The physical models of a Cavity or a slice of it (cavity_models_adapter.h).
// "adaptive" keeps the parts of its choice that depend on the element only;
// each particle's reference is compared with them by CavityPlan_choose.
typedef struct CavityPlan_s {
    GPUGLMEM void* el;
    int model, adaptive, thin_slice;
    double adaptive_length, adaptive_frequency, adaptive_harmonic;
    double adaptive_voltage, adaptive_gain;
    int adaptive_needs_beat, adaptive_beat_small;
    CavityChoice choice[4];
    int integrator, profile, magnetic_profile, spin_tracking;
    int64_t requested;
    double eta, envelope_width, length, weight;
    double left_fraction, right_fraction;
    double start_fraction, end_fraction, direction;
    int reverse, clear_face, physical_outer;
} CavityPlan;

// An RF element, or a slice of it, prepared once per element for one direction
// of travel. The pointers in it point into the element.
typedef struct RFPlan_s {
    RFBody body;
    uint8_t is_thick;   // the global aperture is checked after it
    MagnetTransform transform;
    CavityPlan cavity;
} RFPlan;


GPUFUN
void track_rf_kick_single_particle(
    LocalParticle* part,
    xt_float_or_tpsa voltage,
    xt_float_or_tpsa frequency,
    xt_float_or_tpsa harmonic,
    xt_float_or_tpsa lag,
    xt_float_or_tpsa phase,
    double transverse_voltage,
    double transverse_lag,
    double transverse_phase,
    int64_t absolute_time,
    int64_t order,
    double factor_knl_ksl,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    GPUGLMEM const double* pn,
    GPUGLMEM const double* ps,
    GPUGLMEM const double* phase_n,
    GPUGLMEM const double* phase_s,
    const uint8_t kill_energy_kick
){

    xt_float_or_tpsa phase0 = 0.;

    double const beta0 = LocalParticle_get_beta0(part);
    xt_float_or_tpsa const set_frequency = frequency;
    int const harmonic_locked = !xt_float_or_tpsa_is_zero(harmonic);

    if (harmonic_locked) {
        double const line_length = part->line_length;
        double const t_rev0 = line_length / (beta0 * C_LIGHT);
        frequency += (harmonic / t_rev0);
    }

    // Arrival time t = time_s + (s - zeta) / (beta0 c) and phase
    // Phi = Phi_clock + 2 pi f (t - time_s); the zeta term is added below.
    // Phi_clock is the RF phase accumulated at time_s: f time_s cycles for a
    // set frequency, one revolution's h cycles per reference turn when locked.
    // The clock is scalar metadata; s and the RF controls may carry a TPSA
    // dependence. Use the same phase in both tracking modes.
    if (absolute_time == 1) {
        double const time_s = LocalParticle_get_time_s(part);
        xt_float_or_tpsa const s = LocalParticle_get_s(part);
        if (harmonic_locked) {
            phase0 += 2 * PI * (set_frequency * time_s
                + harmonic * LocalParticle_get_reference_turns(part)
                + frequency * s / (beta0 * C_LIGHT));
        } else {
            phase0 += 2 * PI * frequency * (time_s + s / (beta0 * C_LIGHT));
        }
    }

    xt_float_or_tpsa const zeta  = LocalParticle_get_zeta(part);
    double const q = fabs(LocalParticle_get_q0(part)) * LocalParticle_get_charge_ratio(part);
    xt_float_or_tpsa const tau = zeta / beta0;

    xt_float_or_tpsa const energy_kick = q * voltage
        * sin(phase0 + DEG2RAD * lag + phase
              - (2.0 * PI) / C_LIGHT * frequency * tau);

    double rfmultipole_energy_kick = 0;
#ifndef XTRACK_TPSA_TRACK
    // rfmultipole kick reads x,y as doubles; only RFMultipole/CrabCavity use order>=0.
    // A plain Cavity passes order=-1 (inactive), so the TPSA flavor omits this block.
    if (order >= 0) {

        double dpx = 0.0;
        double dpy = 0.0;
        double dptr = 0.0;
        double zre = 1.0;
        double zim = 0.0;
        double factorial = 1.0;

        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double const p0c = LocalParticle_get_p0c(part);

        for (int64_t kk = 0; kk <= order; kk++)
        {

            if (kk>0){
                factorial *= kk;
            }

            double const pn_kk = phase0 + DEG2RAD * pn[kk] + phase_n[kk] - (2.0 * PI) / C_LIGHT * frequency * tau;
            double const ps_kk = phase0 + DEG2RAD * ps[kk] + phase_s[kk] - (2.0 * PI) / C_LIGHT * frequency * tau;

            double bal_n_kk = factor_knl_ksl * knl[kk]/factorial;
            double bal_s_kk = factor_knl_ksl * ksl[kk]/factorial;

            double const cn = cos(pn_kk);
            double const cs = cos(ps_kk);
            double const sn = sin(pn_kk);
            double const ss = sin(ps_kk);

            dpx += cn * (bal_n_kk * zre) - cs * (bal_s_kk * zim);
            dpy += cs * (bal_s_kk * zre) + cn * (bal_n_kk * zim);

            double const zret = zre * x - zim * y;
            zim = zim * x + zre * y;
            zre = zret;

            // Integrate the transverse multipole: the potential has (kk+1)!.
            dptr += (sn * (bal_n_kk * zre) - ss * (bal_s_kk * zim)) / (kk + 1);
        }

        // knl/ksl already include the reference charge through the rigidity.
        rfmultipole_energy_kick = - LocalParticle_get_charge_ratio(part)
            * (frequency * (2.0 * PI / C_LIGHT) * p0c) * dptr;
        double const chi    = LocalParticle_get_chi(part);

        double const px_kick = - chi * dpx;
        double const py_kick =   chi * dpy;

        LocalParticle_add_to_px(part, px_kick);
        LocalParticle_add_to_py(part, py_kick);

    }

    if (transverse_voltage != 0) {
        double dpx = 0.0;
        double dpy = 0.0;
        double dptr = 0.0;
        double zre = 1.0;
        double zim = 0.0;

        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double const p0c = LocalParticle_get_p0c(part);

        double const pn_kk = phase0 + DEG2RAD * (transverse_lag + 90.)
                           + transverse_phase
                           - (2.0 * PI) / C_LIGHT * frequency * tau;
        double const k0l = fabs(LocalParticle_get_q0(part)) * transverse_voltage / p0c;

        double bal_n_kk = k0l;

        double const cn = cos(pn_kk);
        double const sn = sin(pn_kk);

        dpx += cn * (bal_n_kk * zre);
        dpy += cn * (bal_n_kk * zim);

        double const zret = zre * x - zim * y;
        zim = zim * x + zre * y;
        zre = zret;

        dptr += sn * (bal_n_kk * zre);

        // The reference charge is already included in k0l.
        rfmultipole_energy_kick += - LocalParticle_get_charge_ratio(part)
            * (frequency * (2.0 * PI / C_LIGHT) * p0c) * dptr;
        double const chi    = LocalParticle_get_chi(part);

        double const px_kick = - chi * dpx;
        double const py_kick =   chi * dpy;

        LocalParticle_add_to_px(part, px_kick);
        LocalParticle_add_to_py(part, py_kick);

    }
#endif  // XTRACK_TPSA_TRACK (rfmultipole + transverse-kick blocks)


    if (!kill_energy_kick) {
        #ifdef XTRACK_CAVITY_PRESERVE_ANGLE
        LocalParticle_add_to_energy(part, energy_kick + rfmultipole_energy_kick, 0);
        #else
        LocalParticle_add_to_energy(part, energy_kick + rfmultipole_energy_kick, 1);
        #endif
    }

}


GPUFUN
void track_rf_body_single_particle(LocalParticle* part, const RFBody* body) {

    const double length = body->length;
    const int64_t num_kicks = body->num_kicks;
    const int8_t integrator = body->integrator;

    #define RF_KICK(part, kick_weight) \
        track_rf_kick_single_particle(\
            part, body->voltage * (kick_weight), body->frequency, body->harmonic,\
            body->lag, body->phase,\
            body->transverse_voltage * (kick_weight), body->transverse_lag,\
            body->transverse_phase, body->absolute_time, body->order, \
            body->factor_knl_ksl * (kick_weight), body->knl, body->ksl,\
            body->pn, body->ps, body->phase_n, body->phase_s,\
            body->kill_energy_kick\
        )

    #define RF_DRIFT(part, dlength) \
        track_magnet_drift_single_particle(\
            part, (dlength), 0., 0., 0., 0., 0., 0., body->drift_model\
        )

    // No radiation implemented for RF elements for now
    #define WITH_RF_RADIATION(ll, code)\
        {\
            code;\
        }

    RUN_INTEGRATOR(
        integrator, length, num_kicks, part,
        RF_DRIFT, RF_KICK, WITH_RF_RADIATION
    );


    #undef RF_KICK
    #undef RF_DRIFT
    #undef WITH_RF_RADIATION

}

GPUFUN
void RFSettings_init_default(RFSettings* rf) {
    rf->weight = 1.;
    rf->length = 0.;
    rf->voltage = 0.;
    rf->frequency = 0.;
    rf->harmonic = 0.;
    rf->lag = 0.;
    rf->phase = 0.;
    rf->lag_taper = 0.;
    rf->phase_taper = 0.;
    rf->transverse_voltage = 0.;
    rf->transverse_lag = 0.;
    rf->transverse_phase = 0.;
    rf->absolute_time = 0;
    rf->order = -1;
    rf->num_kicks = 0;
    rf->knl = NULL;
    rf->ksl = NULL;
    rf->pn = NULL;
    rf->ps = NULL;
    rf->phase_n = NULL;
    rf->phase_s = NULL;
    rf->model = MODEL_ADAPTIVE;
    rf->integrator = INTEGRATOR_ADAPTIVE;
}

// Reads only the track flags of the particle (direction of travel, energy kick).
GPUFUN
void RFPlan_prepare(RFPlan* plan, const RFSettings* rf, LocalParticle* part0) {
    RFBody* body = &plan->body;
    const int backtrack = LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK);
    const double direction = backtrack ? -1. : 1.;
    const double weight = rf->weight;

    int8_t model = rf->model == MODEL_ADAPTIVE ? MODEL_DRIFT_KICK_DRIFT_EXPANDED : rf->model;
    int8_t integrator = rf->integrator == INTEGRATOR_ADAPTIVE ? INTEGRATOR_UNIFORM : rf->integrator;
    int64_t num_kicks = rf->num_kicks;
    if (model == MODEL_KICK_ONLY) {
        integrator = INTEGRATOR_UNIFORM;
        num_kicks = 1;
    }
    if (weight != 1.0 && num_kicks > 0) {
        num_kicks = (int64_t) ceil(num_kicks * weight);
    }
    if (num_kicks == 0) { // num_kicks = 0 means auto mode
        num_kicks = 1;
    }

    xt_float_or_tpsa lag = rf->lag;
    xt_float_or_tpsa phase = rf->phase;
    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        lag += rf->lag_taper;
        phase += rf->phase_taper;
    #endif

    body->length = direction * rf->length * weight;
    body->voltage = direction * rf->voltage * weight;
    body->frequency = rf->frequency;
    body->harmonic = rf->harmonic;
    body->lag = lag;
    body->phase = phase;
    body->transverse_voltage = direction * rf->transverse_voltage * weight;
    body->transverse_lag = rf->transverse_lag;
    body->transverse_phase = rf->transverse_phase;
    body->absolute_time = rf->absolute_time;
    body->order = rf->order;
    body->num_kicks = num_kicks;
    body->factor_knl_ksl = direction * weight;
    body->knl = rf->knl;
    body->ksl = rf->ksl;
    body->pn = rf->pn;
    body->ps = rf->ps;
    body->phase_n = rf->phase_n;
    body->phase_s = rf->phase_s;
    // Only the drift model is used: no field
    body->drift_model = configure_tracking_model(
        model, /*k0*/ 0., /*k1*/ 0., /*h*/ 0., /*ks*/ 0.
    ).drift.model;
    body->integrator = integrator;
    body->kill_energy_kick = LocalParticle_check_track_flag(part0, XS_FLAG_KILL_CAVITY_KICK);

    plan->is_thick = 0;
    plan->transform.active = 0;
}

GPUFUN
void track_rf_body_particles(LocalParticle* part0, const RFBody* body) {
    START_PER_PARTICLE_BLOCK(part0, part);
        track_rf_body_single_particle(part, body);
    END_PER_PARTICLE_BLOCK;
}

// CrabCavity, RFMultipole and their slices.
GPUFUN
void track_rf_plan(LocalParticle* part0, const RFPlan* plan) {
    const MagnetTransform* transform = &plan->transform;
    if (transform->active && !magnet_transform(part0, transform, 0)) {
        return;
    }
    track_rf_body_particles(part0, &plan->body);
    if (transform->active) {
        magnet_transform(part0, transform, 1);
    }
}

#endif
