// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_TRACK_CAVITY_SAD_TWISS_TRPT_H
#define XTRACK_TRACK_CAVITY_SAD_TWISS_TRPT_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_cavity_sagan.h"

GPUFUN
double cavity_sad_twiss_trpt_reference_dv(
        double delta,
        double normalized_p0,
        double normalized_energy0
) {
    double const momentum_ratio = 1.0 + delta;
    double const momentum = normalized_p0 * momentum_ratio;
    double const energy = sqrt(1.0 + momentum * momentum);
    return -(momentum_ratio + 1.0) / energy
        / (energy + momentum_ratio * normalized_energy0) * delta;
}

GPUFUN
double cavity_sad_twiss_trpt_reference_dvh(
        double energy_gain,
        double normalized_p0,
        double normalized_energy0
) {
    if (energy_gain == 0.0) {
        return 0.0;
    }
    double const energy = normalized_energy0 + energy_gain;
    double const momentum = sqrt(energy * energy - 1.0);
    return -(normalized_p0 + momentum)
        / (momentum * normalized_energy0 + normalized_p0 * energy)
        / momentum / normalized_p0 * energy_gain;
}

GPUFUN
void cavity_sad_twiss_trpt_drift(
        LocalParticle* part,
        double length,
        double normalized_p0,
        double normalized_energy0
) {
    double const px = LocalParticle_get_px(part);
    double const py = LocalParticle_get_py(part);
    double const delta = LocalParticle_get_delta(part);
    double const momentum_ratio = 1.0 + delta;
    double const transverse_momentum2 = px * px + py * py;
    double const longitudinal_momentum = sqrt(
        momentum_ratio * momentum_ratio - transverse_momentum2);
    double const inverse_longitudinal_momentum = 1.0 / longitudinal_momentum;
    double const dv = cavity_sad_twiss_trpt_reference_dv(
        delta, normalized_p0, normalized_energy0);

    LocalParticle_add_to_x(
        part, px * inverse_longitudinal_momentum * length);
    LocalParticle_add_to_y(
        part, py * inverse_longitudinal_momentum * length);
    LocalParticle_add_to_zeta(part, -(
        transverse_momentum2
            / (momentum_ratio + longitudinal_momentum)
            * inverse_longitudinal_momentum
        + dv) * length);
}

GPUFUN
void cavity_sad_twiss_trpt_fringe_forward(
        LocalParticle* part,
        double length,
        double normalized_signed_voltage,
        double wave_number,
        double sad_phase,
        double normalized_p0
) {
    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);
    double const delta_before = LocalParticle_get_delta(part);
    double const momentum_ratio_before = 1.0 + delta_before;
    double const momentum_before = normalized_p0 * momentum_ratio_before;
    double const energy_before = sqrt(
        1.0 + momentum_before * momentum_before);
    double const velocity_before = momentum_before / energy_before;
    double const particle_time = -LocalParticle_get_zeta(part)
        / velocity_before;
    double const fringe_phase = wave_number * particle_time - sad_phase;
    double const fringe_voltage = 0.5 * normalized_signed_voltage
        / length / normalized_p0;
    double const transverse_kick = fringe_voltage * sin(fringe_phase);

    LocalParticle_add_to_px(part, x * transverse_kick);
    LocalParticle_add_to_py(part, y * transverse_kick);

    double const energy_gain = -0.5 * wave_number * fringe_voltage
        * normalized_p0 * (x * x + y * y) * cos(fringe_phase);
    double const energy_after = energy_before + energy_gain;
    double const momentum_after = sqrt(energy_after * energy_after - 1.0);
    double const momentum_ratio_after = momentum_after / normalized_p0;
    LocalParticle_update_delta(part, momentum_ratio_after - 1.0);
    LocalParticle_scale_zeta(part,
        momentum_ratio_after / momentum_ratio_before
        * energy_before / energy_after);
}

GPUFUN
void cavity_sad_twiss_trpt_fringe_inverse(
        LocalParticle* part,
        double length,
        double normalized_signed_voltage,
        double wave_number,
        double sad_phase,
        double normalized_p0
) {
    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);
    double const momentum_ratio_after = 1.0
        + LocalParticle_get_delta(part);
    double const momentum_after = normalized_p0 * momentum_ratio_after;
    double const energy_after = sqrt(1.0 + momentum_after * momentum_after);
    double const velocity_after = momentum_after / energy_after;
    double const particle_time = -LocalParticle_get_zeta(part)
        / velocity_after;
    double const fringe_phase = wave_number * particle_time - sad_phase;
    double const fringe_voltage = 0.5 * normalized_signed_voltage
        / length / normalized_p0;
    double const energy_gain = -0.5 * wave_number * fringe_voltage
        * normalized_p0 * (x * x + y * y) * cos(fringe_phase);
    double const energy_before = energy_after - energy_gain;
    double const momentum_before = sqrt(energy_before * energy_before - 1.0);
    double const momentum_ratio_before = momentum_before / normalized_p0;
    double const transverse_kick = fringe_voltage * sin(fringe_phase);

    LocalParticle_add_to_px(part, -x * transverse_kick);
    LocalParticle_add_to_py(part, -y * transverse_kick);
    LocalParticle_update_delta(part, momentum_ratio_before - 1.0);
    LocalParticle_set_zeta(
        part, -particle_time * momentum_before / energy_before);
}

GPUFUN
void cavity_sad_twiss_trpt_kick_forward(
        LocalParticle* part,
        double normalized_step_voltage,
        double transverse_voltage,
        double wave_number,
        double sad_phase,
        double normalized_p0
) {
    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);
    double const momentum_before = normalized_p0
        * (1.0 + LocalParticle_get_delta(part));
    double const energy_before = sqrt(
        1.0 + momentum_before * momentum_before);
    double const velocity_before = momentum_before / energy_before;
    double const particle_time = -LocalParticle_get_zeta(part)
        / velocity_before;
    double const rf_phase = sad_phase - wave_number * particle_time;
    double const local_voltage = normalized_step_voltage
        + transverse_voltage * (x * x + y * y);
    double const energy_gain = local_voltage * sin(rf_phase);
    double const energy_after = fmax(
        1.0 + 3.83e-12, energy_before + energy_gain);
    double const momentum_after = sqrt(energy_after * energy_after - 1.0);
    double transverse_kick_scale = 0.0;
    if (wave_number != 0.0) {
        transverse_kick_scale = -cos(rf_phase)
            / wave_number / normalized_p0;
    }

    LocalParticle_add_to_px(
        part, 2.0 * transverse_voltage * x * transverse_kick_scale);
    LocalParticle_add_to_py(
        part, 2.0 * transverse_voltage * y * transverse_kick_scale);
    LocalParticle_update_delta(part, momentum_after / normalized_p0 - 1.0);
    LocalParticle_set_zeta(
        part, -particle_time * momentum_after / energy_after);
}

GPUFUN
void cavity_sad_twiss_trpt_kick_inverse(
        LocalParticle* part,
        double normalized_step_voltage,
        double transverse_voltage,
        double wave_number,
        double sad_phase,
        double normalized_p0
) {
    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);
    double const momentum_after = normalized_p0
        * (1.0 + LocalParticle_get_delta(part));
    double const energy_after = sqrt(1.0 + momentum_after * momentum_after);
    double const velocity_after = momentum_after / energy_after;
    double const particle_time = -LocalParticle_get_zeta(part)
        / velocity_after;
    double const rf_phase = sad_phase - wave_number * particle_time;
    double const local_voltage = normalized_step_voltage
        + transverse_voltage * (x * x + y * y);
    double const energy_gain = local_voltage * sin(rf_phase);
    double const energy_before = energy_after - energy_gain;
    double const momentum_before = sqrt(energy_before * energy_before - 1.0);
    double transverse_kick_scale = 0.0;
    if (wave_number != 0.0) {
        transverse_kick_scale = -cos(rf_phase)
            / wave_number / normalized_p0;
    }

    LocalParticle_add_to_px(
        part, -2.0 * transverse_voltage * x * transverse_kick_scale);
    LocalParticle_add_to_py(
        part, -2.0 * transverse_voltage * y * transverse_kick_scale);
    LocalParticle_update_delta(part, momentum_before / normalized_p0 - 1.0);
    LocalParticle_set_zeta(
        part, -particle_time * momentum_before / energy_before);
}

GPUFUN
void cavity_sad_twiss_trpt_track_single_particle(
        LocalParticle* part,
        double length,
        double voltage,
        double nominal_voltage,
        double nominal_length,
        double length_offset,
        double frequency,
        double harmonic,
        double lag,
        double phase,
        int64_t absolute_time,
        int64_t num_steps,
        int64_t fringe_model,
        int64_t edge_entry_active,
        int64_t edge_exit_active,
        int64_t backtrack,
        int64_t kill_energy_kick
) {
    if (harmonic != 0.0) {
        frequency += harmonic * LocalParticle_get_beta0(part) * C_LIGHT
            / part->line_length;
    }

    double const p0c = LocalParticle_get_p0c(part);
    double const mass0 = LocalParticle_get_mass0(part);
    double const q0_abs = fabs(LocalParticle_get_q0(part));
    double const normalized_p0 = p0c / mass0;
    double const normalized_energy0 = sqrt(
        1.0 + normalized_p0 * normalized_p0);
    if (kill_energy_kick || voltage == 0.0) {
        cavity_sad_twiss_trpt_drift(
            part, backtrack ? -length : length,
            normalized_p0, normalized_energy0);
        LocalParticle_add_to_s(part, backtrack ? -length : length);
        return;
    }

    double const reference_phase = cavity_sagan_reference_phase(
        part, frequency, lag, phase, absolute_time);
    double const sad_phase = PI - reference_phase;
    double const normalized_voltage = q0_abs * voltage / mass0;
    double const reference_gain = normalized_voltage * sin(reference_phase);
    double const normalized_reference_energy_out =
        normalized_energy0 + reference_gain;
    if (normalized_reference_energy_out <= 1.0) {
        LocalParticle_kill_particle(part, -20);
        return;
    }
    double const normalized_reference_momentum_out = sqrt(
        normalized_reference_energy_out * normalized_reference_energy_out
        - 1.0);
    double const wave_number = 2.0 * PI * frequency / C_LIGHT;
    if (num_steps < 1) {
        double const gain_ratio = fabs(normalized_voltage * (
            1.0 / normalized_energy0
            + 1.0 / normalized_reference_energy_out));
        num_steps = 1 + (int64_t) fmin(
            fabs(wave_number * length),
            sqrt(gain_ratio * gain_ratio / 0.03));
    }

    double const step_length = length / num_steps;
    double const normalized_step_voltage = normalized_voltage / num_steps;
    // Both the focusing coefficient and the reference-phase progression are
    // properties of the whole (unsliced) cavity, mirroring
    // track_cavity_sad_trpt.h: SAD's own vcorr uses vnominal (never the
    // local vc used for the actual kick strength), and the reference
    // particle's own clock does not reset at an externally-imposed slice
    // boundary. nominal_voltage/nominal_length equal voltage/length
    // themselves when untracked (length_offset=0).
    double const normalized_nominal_voltage = q0_abs * nominal_voltage
        / mass0;
    double const nominal_reference_gain = normalized_nominal_voltage
        * sin(reference_phase);
    double const nominal_normalized_reference_energy_out =
        normalized_energy0 + nominal_reference_gain;
    double const nominal_normalized_reference_momentum_out = sqrt(
        nominal_normalized_reference_energy_out
        * nominal_normalized_reference_energy_out - 1.0);
    double const inverse_reference_momentum_average =
        0.5 / normalized_p0
        + 0.5 / nominal_normalized_reference_momentum_out;
    double const transverse_voltage = normalized_step_voltage
        * 0.25 * wave_number * wave_number
        * inverse_reference_momentum_average
        * inverse_reference_momentum_average;
    double const nominal_energy_step =
        (normalized_reference_momentum_out - normalized_p0)
        * (normalized_reference_momentum_out + normalized_p0)
        / (normalized_reference_energy_out + normalized_energy0)
        / num_steps;
    // Reference gain is linear in length under this model's own uniform-
    // gradient assumption, so the fraction already applied by the time this
    // slice begins is exactly length_offset/nominal_length -- no dependence
    // on runtime p0c or any external reference-energy bookkeeping.
    double const accumulated_gain_offset = (nominal_length > 0.0)
        ? (length_offset / nominal_length) * nominal_reference_gain
        : 0.0;
    int64_t const with_fringe = fringe_model != -1 && length != 0.0;
    double const beta_in = normalized_p0 / normalized_energy0;
    double const beta_out = normalized_reference_momentum_out
        / normalized_reference_energy_out;

    if (backtrack) {
        LocalParticle_scale_zeta(part, beta_out / beta_in);
        if (with_fringe && edge_exit_active) {
            cavity_sad_twiss_trpt_fringe_inverse(
                part, length, -normalized_voltage, wave_number,
                sad_phase, normalized_p0);
        }

        LocalParticle_add_to_zeta(part,
            -cavity_sad_twiss_trpt_reference_dvh(
                accumulated_gain_offset + nominal_energy_step * num_steps,
                normalized_p0, normalized_energy0)
            * 0.5 * step_length);
        cavity_sad_twiss_trpt_drift(
            part, -0.5 * step_length,
            normalized_p0, normalized_energy0);

        for (int64_t step = num_steps - 1; step >= 0; step--) {
            cavity_sad_twiss_trpt_kick_inverse(
                part, normalized_step_voltage, transverse_voltage,
                wave_number, sad_phase, normalized_p0);
            double const drift_length = step == 0
                ? 0.5 * step_length : step_length;
            LocalParticle_add_to_zeta(part,
                -cavity_sad_twiss_trpt_reference_dvh(
                    accumulated_gain_offset + nominal_energy_step * step,
                    normalized_p0, normalized_energy0)
                * drift_length);
            cavity_sad_twiss_trpt_drift(
                part, -drift_length,
                normalized_p0, normalized_energy0);
        }

        if (with_fringe && edge_entry_active) {
            cavity_sad_twiss_trpt_fringe_inverse(
                part, length, normalized_voltage, wave_number,
                sad_phase, normalized_p0);
        }
        LocalParticle_add_to_s(part, -length);
        return;
    }

    if (with_fringe && edge_entry_active) {
        cavity_sad_twiss_trpt_fringe_forward(
            part, length, normalized_voltage, wave_number,
            sad_phase, normalized_p0);
    }

    double accumulated_nominal_gain = accumulated_gain_offset;
    for (int64_t step = 0; step < num_steps; step++) {
        double const drift_length = step == 0
            ? 0.5 * step_length : step_length;
        cavity_sad_twiss_trpt_drift(
            part, drift_length, normalized_p0, normalized_energy0);
        LocalParticle_add_to_zeta(part,
            cavity_sad_twiss_trpt_reference_dvh(
                accumulated_nominal_gain,
                normalized_p0, normalized_energy0)
            * drift_length);
        cavity_sad_twiss_trpt_kick_forward(
            part, normalized_step_voltage, transverse_voltage,
            wave_number, sad_phase, normalized_p0);
        accumulated_nominal_gain += nominal_energy_step;
    }

    cavity_sad_twiss_trpt_drift(
        part, 0.5 * step_length, normalized_p0, normalized_energy0);
    LocalParticle_add_to_zeta(part,
        cavity_sad_twiss_trpt_reference_dvh(
            accumulated_nominal_gain, normalized_p0, normalized_energy0)
        * 0.5 * step_length);

    if (with_fringe && edge_exit_active) {
        cavity_sad_twiss_trpt_fringe_forward(
            part, length, -normalized_voltage, wave_number,
            sad_phase, normalized_p0);
    }
    LocalParticle_scale_zeta(part, beta_in / beta_out);
    LocalParticle_add_to_s(part, length);
}

GPUFUN
void track_cavity_sad_twiss_trpt_particles(
        double weight,
        LocalParticle* part0,
        double length,
        double voltage,
        double length_offset,
        double frequency,
        double harmonic,
        double lag,
        double phase,
        int64_t absolute_time,
        int64_t num_steps,
        int64_t fringe_model,
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
    // nominal_voltage/nominal_length are the whole (unsliced) cavity's
    // values, captured before the local/per-slice scaling below -- see the
    // comment at their use in cavity_sad_twiss_trpt_track_single_particle.
    double const nominal_voltage = voltage;
    double const nominal_length = length;
    length *= weight;
    voltage *= weight;
    if (weight != 1.0 && num_steps > 0) {
        num_steps = (int64_t) ceil(num_steps * weight);
    }
    START_PER_PARTICLE_BLOCK(part0, part);
        cavity_sad_twiss_trpt_track_single_particle(
            part, length, voltage, nominal_voltage, nominal_length,
            length_offset, frequency, harmonic, lag, phase, absolute_time,
            num_steps, fringe_model, edge_entry_active, edge_exit_active,
            backtrack, kill_energy_kick);
    END_PER_PARTICLE_BLOCK;
}

#endif
