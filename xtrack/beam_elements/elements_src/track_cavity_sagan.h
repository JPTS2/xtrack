// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_TRACK_CAVITY_SAGAN_H
#define XTRACK_TRACK_CAVITY_SAGAN_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_drift.h"


GPUFUN
double cavity_sagan_reference_phase(
        LocalParticle* part,
        double frequency,
        double lag,
        double phase,
        int64_t absolute_time
) {
    double phase0 = 0.0;
    if (absolute_time == 1) {
        double const t_sim = LocalParticle_get_t_sim(part);
        int64_t const at_turn = LocalParticle_get_at_turn(part);
        phase0 = 2.0 * PI * at_turn * frequency * t_sim;
    }
    return phase0 + DEG2RAD * lag + phase;
}


GPUFUN
double cavity_sagan_phase(
        LocalParticle* part,
        double frequency,
        double lag,
        double phase,
        int64_t absolute_time
) {
    double const reference_phase = cavity_sagan_reference_phase(
        part, frequency, lag, phase, absolute_time);
    double const beta0 = LocalParticle_get_beta0(part);
    double const tau = LocalParticle_get_zeta(part) / beta0;
    return reference_phase - 2.0 * PI * frequency * tau / C_LIGHT;
}


GPUFUN
void cavity_sagan_energy_kick(
        LocalParticle* part,
        double voltage,
        double frequency,
        double lag,
        double phase,
        int64_t absolute_time
) {
    double const rf_phase = cavity_sagan_phase(
        part, frequency, lag, phase, absolute_time);
    double const q0_abs = fabs(LocalParticle_get_q0(part));
    double const charge_ratio = LocalParticle_get_charge_ratio(part);
    double const particle_energy_gain = (
        q0_abs * charge_ratio * voltage * sin(rf_phase));
    LocalParticle_add_to_energy(part, particle_energy_gain, 1);
}


GPUFUN
void cavity_sagan_fringe_kick(
        LocalParticle* part,
        double direction,
        double edge,
        double gradient,
        double frequency,
        double lag,
        double phase,
        int64_t absolute_time
) {
    double const rf_phase = cavity_sagan_phase(
        part, frequency, lag, phase, absolute_time);
    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);
    double const p0c = LocalParticle_get_p0c(part);
    double const q0_abs = fabs(LocalParticle_get_q0(part));
    double const charge_ratio = LocalParticle_get_charge_ratio(part);
    double const chi = LocalParticle_get_chi(part);
    double const normalized_charge = q0_abs * chi;
    double const transverse_strength = (
        direction * edge * 0.5 * normalized_charge * gradient
        * sin(rf_phase) / p0c);

    LocalParticle_add_to_px(part, -transverse_strength * x);
    LocalParticle_add_to_py(part, -transverse_strength * y);

    double const wave_number = 2.0 * PI * frequency / C_LIGHT;
    double const energy_companion = (
        direction * edge * 0.25 * q0_abs * charge_ratio
        * wave_number * gradient * cos(rf_phase) * (x * x + y * y));
    LocalParticle_add_to_energy(part, energy_companion, 1);
}


GPUFUN
void cavity_sagan_ponderomotive_kick(
        LocalParticle* part,
        double direction,
        double step_length,
        double gradient
) {
    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);
    double const p0c = LocalParticle_get_p0c(part);
    double const one_plus_delta = 1.0 + LocalParticle_get_delta(part);
    double const charge_ratio = LocalParticle_get_charge_ratio(part);
    double const chi = LocalParticle_get_chi(part);
    double const mass_ratio = charge_ratio / chi;
    double const physical_momentum = p0c * one_plus_delta * mass_ratio;
    double const beta = (
        LocalParticle_get_beta0(part) * LocalParticle_get_rvv(part));
    double const physical_energy = physical_momentum / beta;
    double const q0_abs = fabs(LocalParticle_get_q0(part));
    double const charged_gradient = q0_abs * charge_ratio * gradient;
    double const hamiltonian_coefficient = (
        step_length * charged_gradient * charged_gradient / 32.0);
    double const physical_kick = (
        direction * 2.0 * hamiltonian_coefficient / physical_momentum);

    LocalParticle_add_to_px(
        part, -physical_kick * x / (p0c * mass_ratio));
    LocalParticle_add_to_py(
        part, -physical_kick * y / (p0c * mass_ratio));

    double const delta_tau = (
        -direction * hamiltonian_coefficient * physical_energy
        / (physical_momentum * physical_momentum * physical_momentum)
        * (x * x + y * y));
    LocalParticle_add_to_zeta(
        part, LocalParticle_get_beta0(part) * delta_tau);
}


GPUFUN
void cavity_sagan_track_single_particle(
        LocalParticle* part,
        double length,
        double voltage,
        double frequency,
        double harmonic,
        double lag,
        double phase,
        int64_t absolute_time,
        int64_t num_steps,
        int64_t fringe_model,
        int64_t cavity_type,
        int64_t edge_entry_active,
        int64_t edge_exit_active,
        int64_t backtrack,
        int64_t kill_energy_kick
) {
    if (harmonic != 0.0) {
        double const line_length = part->line_length;
        double const beta0 = LocalParticle_get_beta0(part);
        frequency += harmonic * beta0 * C_LIGHT / line_length;
    }
    if (num_steps < 1) {
        num_steps = 1;
    }

    if (kill_energy_kick) {
        Drift_single_particle_exact(part, backtrack ? -length : length);
        return;
    }

    if (length == 0.0) {
        cavity_sagan_energy_kick(
            part, backtrack ? -voltage : voltage,
            frequency, lag, phase, absolute_time);
        return;
    }

    double const step_length = length / num_steps;
    double const gradient = voltage / length;
    int64_t const with_fringe = fringe_model != -1;
    int64_t const with_ponderomotive = cavity_type == 0;

    if (!backtrack) {
        if (with_fringe && edge_entry_active) {
            cavity_sagan_fringe_kick(
                part, 1.0, 1.0, gradient,
                frequency, lag, phase, absolute_time);
        }
        for (int64_t step = 0; step <= num_steps; step++) {
            if (with_ponderomotive && step > 0) {
                cavity_sagan_ponderomotive_kick(
                    part, 1.0, step_length, gradient);
            }
            double const kick_weight = (
                (step == 0 || step == num_steps) ? 0.5 : 1.0);
            cavity_sagan_energy_kick(
                part, kick_weight * voltage / num_steps,
                frequency, lag, phase, absolute_time);
            if (step < num_steps) {
                if (with_ponderomotive) {
                    cavity_sagan_ponderomotive_kick(
                        part, 1.0, step_length, gradient);
                }
                Drift_single_particle_exact(part, step_length);
            }
        }
        if (with_fringe && edge_exit_active) {
            cavity_sagan_fringe_kick(
                part, 1.0, -1.0, gradient,
                frequency, lag, phase, absolute_time);
        }
    }
    else {
        if (with_fringe && edge_exit_active) {
            cavity_sagan_fringe_kick(
                part, -1.0, -1.0, gradient,
                frequency, lag, phase, absolute_time);
        }
        for (int64_t step = num_steps; step >= 0; step--) {
            if (step < num_steps) {
                Drift_single_particle_exact(part, -step_length);
                if (with_ponderomotive) {
                    cavity_sagan_ponderomotive_kick(
                        part, -1.0, step_length, gradient);
                }
            }
            double const kick_weight = (
                (step == 0 || step == num_steps) ? 0.5 : 1.0);
            cavity_sagan_energy_kick(
                part, -kick_weight * voltage / num_steps,
                frequency, lag, phase, absolute_time);
            if (with_ponderomotive && step > 0) {
                cavity_sagan_ponderomotive_kick(
                    part, -1.0, step_length, gradient);
            }
        }
        if (with_fringe && edge_entry_active) {
            cavity_sagan_fringe_kick(
                part, -1.0, 1.0, gradient,
                frequency, lag, phase, absolute_time);
        }
    }
}


GPUFUN
void track_cavity_sagan_particles(
        double weight,
        LocalParticle* part0,
        double length,
        double voltage,
        double frequency,
        double harmonic,
        double lag,
        double phase,
        int64_t absolute_time,
        int64_t num_steps,
        int64_t fringe_model,
        int64_t cavity_type,
        int64_t edge_entry_active,
        int64_t edge_exit_active,
        double lag_taper,
        double phase_taper
) {
    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        lag += lag_taper;
        phase += phase_taper;
    #endif

    int64_t const backtrack = LocalParticle_check_track_flag(
        part0, XS_FLAG_BACKTRACK);
    int64_t const kill_energy_kick = LocalParticle_check_track_flag(
        part0, XS_FLAG_KILL_CAVITY_KICK);

    length *= weight;
    voltage *= weight;
    if (weight != 1.0 && num_steps > 0) {
        num_steps = (int64_t) ceil(num_steps * weight);
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        cavity_sagan_track_single_particle(
            part, length, voltage, frequency, harmonic, lag, phase,
            absolute_time, num_steps, fringe_model, cavity_type,
            edge_entry_active, edge_exit_active,
            backtrack, kill_energy_kick);
    END_PER_PARTICLE_BLOCK;
}

#endif
