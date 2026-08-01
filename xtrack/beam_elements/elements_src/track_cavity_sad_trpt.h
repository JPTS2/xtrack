// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_TRACK_CAVITY_SAD_TRPT_H
#define XTRACK_TRACK_CAVITY_SAD_TRPT_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_drift.h"
#include "xtrack/beam_elements/elements_src/track_cavity_sagan.h"

GPUFUN
double cavity_sad_trpt_phase(
        LocalParticle* part,
        double frequency,
        double lag,
        double phase,
        int64_t absolute_time
) {
    double const reference_phase = cavity_sagan_reference_phase(
        part, frequency, lag, phase, absolute_time);
    double const p0c = LocalParticle_get_p0c(part);
    double const mass0 = LocalParticle_get_mass0(part);
    double const charge_ratio = LocalParticle_get_charge_ratio(part);
    double const chi = LocalParticle_get_chi(part);
    double const physical_momentum = (
        p0c * (1.0 + LocalParticle_get_delta(part)) * charge_ratio / chi);
    double const physical_energy = sqrt(
        physical_momentum * physical_momentum + mass0 * mass0);
    double const arrival_time = (
        LocalParticle_get_zeta(part) * physical_energy / physical_momentum);
    return reference_phase - 2.0 * PI * frequency * arrival_time / C_LIGHT;
}

GPUFUN
double cavity_sad_trpt_fringe_forward(
        LocalParticle* part,
        double length,
        double signed_voltage,
        double frequency,
        double reference_phase,
        double dv
) {
    if (length == 0.0 || frequency == 0.0) {
        return dv;
    }

    double const wave_number = 2.0 * PI * frequency / C_LIGHT;
    double const p0c = LocalParticle_get_p0c(part);
    double const mass0 = LocalParticle_get_mass0(part);
    double const q0_abs = fabs(LocalParticle_get_q0(part));
    double const normalized_p0 = p0c / mass0;
    double const normalized_h0 = sqrt(1.0 + normalized_p0 * normalized_p0);
    double const normalized_voltage = q0_abs * signed_voltage / mass0;
    double const fringe_voltage = 0.5 * normalized_voltage
        / length / normalized_p0;
    double const momentum_ratio_before = 1.0
        + LocalParticle_get_delta(part);
    double const momentum_before = normalized_p0 * momentum_ratio_before;
    double const energy_before = dv > 0.1
        ? sqrt(1.0 + momentum_before * momentum_before)
        : momentum_ratio_before * normalized_h0 / (1.0 - dv);
    double const zeta_before = LocalParticle_get_zeta(part);
    double const particle_time = -zeta_before * energy_before
        / momentum_before;
    double const dphis = PI - reference_phase;
    double const fringe_phase = wave_number * particle_time;
    double const transverse_kick = fringe_voltage
        * sin(fringe_phase - dphis) / momentum_ratio_before;
    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);
    LocalParticle_add_to_px(part, x * transverse_kick);
    LocalParticle_add_to_py(part, y * transverse_kick);

    double const energy_gain = -0.5 * wave_number * normalized_p0
        * fringe_voltage * (x * x + y * y)
        * cos(fringe_phase - dphis);
    double const energy_after = energy_before + energy_gain;
    double const relative_momentum_square_change = fmax(
        energy_gain * (energy_before + energy_after)
            / (momentum_before * momentum_before),
        -1.0);
    double const relative_momentum_change = relative_momentum_square_change
        / (1.0 + sqrt(1.0 + relative_momentum_square_change));
    double const momentum_ratio_after = momentum_ratio_before
        * (1.0 + relative_momentum_change);
    double const delta_after = momentum_ratio_after - 1.0;
    LocalParticle_update_delta(part, delta_after);
    LocalParticle_set_zeta(part, zeta_before
        * momentum_ratio_after / momentum_ratio_before
        * energy_before / energy_after);
    return -(1.0 + momentum_ratio_after) / energy_after
        / (energy_after + momentum_ratio_after * normalized_h0)
        * delta_after;
}

GPUFUN
double cavity_sad_trpt_fringe_inverse(
        LocalParticle* part,
        double length,
        double signed_voltage,
        double frequency,
        double reference_phase
) {
    if (length == 0.0 || frequency == 0.0) {
        double const normalized_p0 = LocalParticle_get_p0c(part)
            / LocalParticle_get_mass0(part);
        double const normalized_h0 = sqrt(
            1.0 + normalized_p0 * normalized_p0);
        double const momentum_ratio = 1.0
            + LocalParticle_get_delta(part);
        double const energy = sqrt(1.0
            + normalized_p0 * normalized_p0
                * momentum_ratio * momentum_ratio);
        return -(1.0 + momentum_ratio) / energy
            / (energy + momentum_ratio * normalized_h0)
            * LocalParticle_get_delta(part);
    }

    double const wave_number = 2.0 * PI * frequency / C_LIGHT;
    double const p0c = LocalParticle_get_p0c(part);
    double const mass0 = LocalParticle_get_mass0(part);
    double const q0_abs = fabs(LocalParticle_get_q0(part));
    double const normalized_p0 = p0c / mass0;
    double const normalized_h0 = sqrt(1.0 + normalized_p0 * normalized_p0);
    double const normalized_voltage = q0_abs * signed_voltage / mass0;
    double const fringe_voltage = 0.5 * normalized_voltage
        / length / normalized_p0;
    double const momentum_ratio_after = 1.0
        + LocalParticle_get_delta(part);
    double const momentum_after = normalized_p0 * momentum_ratio_after;
    double const energy_after = sqrt(
        1.0 + momentum_after * momentum_after);
    double const zeta_after = LocalParticle_get_zeta(part);
    double const particle_time = -zeta_after * energy_after
        / momentum_after;
    double const dphis = PI - reference_phase;
    double const fringe_phase = wave_number * particle_time;
    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);
    double const energy_gain = -0.5 * wave_number * normalized_p0
        * fringe_voltage * (x * x + y * y)
        * cos(fringe_phase - dphis);
    double const energy_before = energy_after - energy_gain;
    double const momentum_before = sqrt(
        energy_before * energy_before - 1.0);
    double const momentum_ratio_before = momentum_before / normalized_p0;
    double const delta_before = momentum_ratio_before - 1.0;
    double const transverse_kick = fringe_voltage
        * sin(fringe_phase - dphis) / momentum_ratio_before;

    LocalParticle_add_to_px(part, -x * transverse_kick);
    LocalParticle_add_to_py(part, -y * transverse_kick);
    LocalParticle_update_delta(part, delta_before);
    LocalParticle_set_zeta(part, -particle_time
        * momentum_before / energy_before);
    return -(1.0 + momentum_ratio_before) / energy_before
        / (energy_before + momentum_ratio_before * normalized_h0)
        * delta_before;
}

GPUFUN
void cavity_sad_trpt_track_single_particle(
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
    if (kill_energy_kick || voltage == 0.0) {
        Drift_single_particle_exact(part, backtrack ? -length : length);
        return;
    }
    if (length == 0.0) {
        cavity_sagan_energy_kick(
            part, backtrack ? -voltage : voltage,
            frequency, lag, phase, absolute_time);
        return;
    }

    int64_t const with_fringe = fringe_model != -1;

    double const reference_phase = cavity_sagan_reference_phase(
        part, frequency, lag, phase, absolute_time);
    double const q0_abs = fabs(LocalParticle_get_q0(part));
    double const reference_gain = q0_abs * voltage * sin(reference_phase);
    double const p0c = LocalParticle_get_p0c(part);
    double const mass0 = LocalParticle_get_mass0(part);
    double const reference_energy_in = sqrt(p0c * p0c + mass0 * mass0);
    double const reference_energy_out = reference_energy_in + reference_gain;
    double const wave_number = 2.0 * PI * frequency / C_LIGHT;
    if (reference_energy_out <= mass0) {
        LocalParticle_kill_particle(part, -20);
        return;
    }

    double const gain_ratio = fabs(reference_gain * (
        1.0 / reference_energy_in + 1.0 / reference_energy_out));
    if (num_steps < 1) {
        num_steps = 1 + (int64_t) fmin(
            fabs(wave_number * length),
            sqrt(gain_ratio * gain_ratio / 0.003));
        if (gain_ratio > 1e-12) {
            double const log_momentum_ratio = asinh(
                gain_ratio * (2.0 + gain_ratio)
                / (2.0 * (1.0 + gain_ratio)));
            num_steps = (int64_t) fmax(
                (double) num_steps,
                floor(10.0 * log_momentum_ratio + 0.5));
        }
    }

    double weight = 1.0 / num_steps;
    double weight_multiplier = 1.0;
    if (gain_ratio > 1e-12) {
        double const log_momentum_ratio = asinh(
            gain_ratio * (2.0 + gain_ratio)
            / (2.0 * (1.0 + gain_ratio)));
        double const half_log_step = log_momentum_ratio / (2.0 * num_steps);
        double const first_increment = 2.0 * exp(half_log_step)
            * sinh(half_log_step);
        weight = first_increment / gain_ratio;
        weight_multiplier = 1.0 + first_increment;
    }
    double const first_weight = weight;

    double previous_weight = 0.0;
    double const normalized_voltage = q0_abs * voltage / mass0;
    double const normalized_energy_in = reference_energy_in / mass0;
    double const normalized_energy_out = reference_energy_out / mass0;
    double const normalized_momentum_in = p0c / mass0;
    double const normalized_momentum_out = sqrt(
        normalized_energy_out * normalized_energy_out - 1.0);
    // Both the focusing coefficient and the reference-phase progression are
    // properties of the whole (unsliced) cavity: SAD's own vcorr uses
    // vnominal (never the local vc used for the actual kick strength
    // below), and the reference particle's own clock does not reset at an
    // externally-imposed slice boundary. nominal_voltage/nominal_length are
    // the parent cavity's true, un-weight-scaled values (equal to
    // voltage/length themselves when untracked, i.e. length_offset=0).
    double const nominal_reference_gain = q0_abs * nominal_voltage
        * sin(reference_phase);
    double const nominal_normalized_energy_out = (
        reference_energy_in + nominal_reference_gain) / mass0;
    double const nominal_normalized_momentum_out = sqrt(
        nominal_normalized_energy_out * nominal_normalized_energy_out - 1.0);
    // Reference gain is linear in length under this model's own uniform-
    // gradient assumption, so the fraction already applied by the time this
    // slice begins is exactly length_offset/nominal_length -- no dependence
    // on runtime p0c or any external reference-energy bookkeeping.
    double const reference_sum_offset = (nominal_length > 0.0)
        ? (length_offset / nominal_length) * nominal_reference_gain / mass0
        : 0.0;
    double reference_sum = reference_sum_offset;
    double const transverse_voltage = normalized_voltage * 0.25
        * wave_number * wave_number
        * (0.5 / normalized_momentum_in
            + 0.5 / nominal_normalized_momentum_out)
        * (0.5 / normalized_momentum_in
            + 0.5 / nominal_normalized_momentum_out);
    double const sad_phase = PI - reference_phase;

    if (backtrack) {
        double momentum_ratio_after = 1.0
            + LocalParticle_get_delta(part);
        LocalParticle_set_px(part, LocalParticle_get_px(part)
            / momentum_ratio_after);
        LocalParticle_set_py(part, LocalParticle_get_py(part)
            / momentum_ratio_after);
        double const beta_in = normalized_momentum_in / normalized_energy_in;
        double const beta_out = normalized_momentum_out / normalized_energy_out;
        LocalParticle_scale_zeta(part, beta_out / beta_in);

        double momentum_after = normalized_momentum_in * momentum_ratio_after;
        double energy_after = sqrt(1.0 + momentum_after * momentum_after);
        double dv = -(1.0 + momentum_ratio_after) / energy_after
            / (energy_after
                + momentum_ratio_after * normalized_energy_in)
            * LocalParticle_get_delta(part);

        if (with_fringe && edge_exit_active) {
            dv = cavity_sad_trpt_fringe_inverse(
                part, length, -voltage, frequency, reference_phase);
        }

        double reverse_weight = first_weight
            * pow(weight_multiplier, num_steps - 1);
        double drift_length = 0.5 * length * reverse_weight;
        double px = LocalParticle_get_px(part);
        double py = LocalParticle_get_py(part);
        double transverse_square = px * px + py * py;
        double longitudinal_change = sqrt(1.0 - transverse_square) - 1.0;
        double effective_length = drift_length
            / (1.0 + longitudinal_change);
        LocalParticle_add_to_x(part, -px * effective_length);
        LocalParticle_add_to_y(part, -py * effective_length);
        LocalParticle_add_to_zeta(part,
            -longitudinal_change * effective_length + dv * drift_length);

        double reverse_reference_sum = reference_sum_offset;
        double reference_weight = first_weight;
        for (int64_t step = 0; step < num_steps; step++) {
            reverse_reference_sum += reference_gain * reference_weight / mass0;
            reference_weight *= weight_multiplier;
        }

        for (int64_t step = num_steps - 1; step >= 0; step--) {
            double const reference_energy = normalized_energy_in
                + reverse_reference_sum;
            double const reference_momentum = sqrt(
                reference_energy * reference_energy - 1.0);
            double const reference_delta = (
                reference_momentum - normalized_momentum_in)
                / normalized_momentum_in;
            double const reference_ratio = 1.0 + reference_delta;
            double const reference_dv = -reference_delta
                * (1.0 + reference_ratio) / reference_energy
                / (reference_energy
                    + reference_ratio * normalized_energy_in);
            double const reference_zeta = -reference_dv * length
                * reverse_weight * 0.5;

            momentum_ratio_after = 1.0
                + LocalParticle_get_delta(part);
            momentum_after = normalized_momentum_in * momentum_ratio_after;
            energy_after = sqrt(1.0 + momentum_after * momentum_after);
            double const particle_time = -(
                LocalParticle_get_zeta(part) + reference_zeta)
                * energy_after / momentum_after;
            double const rf_phase = sad_phase - wave_number * particle_time;
            double const x = LocalParticle_get_x(part);
            double const y = LocalParticle_get_y(part);
            double const energy_gain = (normalized_voltage
                + transverse_voltage * (x * x + y * y))
                * sin(rf_phase) * reverse_weight;
            double const energy_before = energy_after - energy_gain;
            double const momentum_before = sqrt(
                energy_before * energy_before - 1.0);
            double const momentum_ratio_before = momentum_before
                / normalized_momentum_in;
            double const delta_before = momentum_ratio_before - 1.0;
            double const kick_scale = -cos(
                wave_number * particle_time - sad_phase)
                * reverse_weight / (wave_number * normalized_momentum_in);
            double const px_after = LocalParticle_get_px(part);
            double const py_after = LocalParticle_get_py(part);
            LocalParticle_set_px(part,
                (px_after * momentum_ratio_after
                    - 2.0 * transverse_voltage * x * kick_scale)
                / momentum_ratio_before);
            LocalParticle_set_py(part,
                (py_after * momentum_ratio_after
                    - 2.0 * transverse_voltage * y * kick_scale)
                / momentum_ratio_before);
            LocalParticle_update_delta(part, delta_before);
            LocalParticle_set_zeta(part, -particle_time
                * momentum_before / energy_before + reference_zeta);
            dv = -(1.0 + momentum_ratio_before) / energy_before
                / (energy_before
                    + momentum_ratio_before * normalized_energy_in)
                * delta_before;

            reverse_reference_sum -= reference_gain
                * reverse_weight / mass0;
            double const previous_weight = step == 0
                ? 0.0 : reverse_weight / weight_multiplier;
            drift_length = step == 0
                ? 0.5 * length * reverse_weight
                : 0.5 * length * (previous_weight + reverse_weight);
            px = LocalParticle_get_px(part);
            py = LocalParticle_get_py(part);
            transverse_square = px * px + py * py;
            longitudinal_change = sqrt(1.0 - transverse_square) - 1.0;
            effective_length = drift_length
                / (1.0 + longitudinal_change);
            LocalParticle_add_to_x(part, -px * effective_length);
            LocalParticle_add_to_y(part, -py * effective_length);
            LocalParticle_add_to_zeta(part,
                -longitudinal_change * effective_length + dv * drift_length);
            reverse_weight = previous_weight;
        }

        if (with_fringe && edge_entry_active) {
            dv = cavity_sad_trpt_fringe_inverse(
                part, length, voltage, frequency, reference_phase);
        }

        double const momentum_ratio_in = 1.0
            + LocalParticle_get_delta(part);
        LocalParticle_set_px(part, LocalParticle_get_px(part)
            * momentum_ratio_in);
        LocalParticle_set_py(part, LocalParticle_get_py(part)
            * momentum_ratio_in);
        LocalParticle_add_to_s(part, -length);
        return;
    }

    double const initial_momentum_ratio = (
        1.0 + LocalParticle_get_delta(part));
    double const initial_momentum = (
        normalized_momentum_in * initial_momentum_ratio);
    double const initial_energy = sqrt(
        1.0 + initial_momentum * initial_momentum);
    double dv = -(initial_momentum_ratio - 1.0)
        * (1.0 + initial_momentum_ratio) / initial_energy
        / (initial_energy + initial_momentum_ratio * normalized_energy_in);
    LocalParticle_set_px(part,
        LocalParticle_get_px(part) / initial_momentum_ratio);
    LocalParticle_set_py(part,
        LocalParticle_get_py(part) / initial_momentum_ratio);

    if (!backtrack && with_fringe && edge_entry_active) {
        dv = cavity_sad_trpt_fringe_forward(
            part, length, voltage, frequency, reference_phase, dv);
    }

    for (int64_t step = 0; step < num_steps; step++) {
        double const drift_length = (step == 0)
            ? 0.5 * length * weight
            : 0.5 * length * (previous_weight + weight);
        double const x_before = LocalParticle_get_x(part);
        double const y_before = LocalParticle_get_y(part);
        double const px_before = LocalParticle_get_px(part);
        double const py_before = LocalParticle_get_py(part);
        double const transverse_square = px_before * px_before
            + py_before * py_before;
        double const longitudinal_change = sqrt(1.0 - transverse_square) - 1.0;
        double const effective_length = drift_length / (1.0 + longitudinal_change);
        LocalParticle_set_x(part, x_before + px_before * effective_length);
        LocalParticle_set_y(part, y_before + py_before * effective_length);
        LocalParticle_add_to_zeta(part, longitudinal_change * effective_length
            - dv * drift_length);

        reference_sum += reference_gain * weight / mass0;
        double const reference_energy = normalized_energy_in + reference_sum;
        double const reference_momentum = sqrt(
            reference_energy * reference_energy - 1.0);
        double const reference_delta = (
            reference_momentum - normalized_momentum_in)
            / normalized_momentum_in;
        double const reference_ratio = 1.0 + reference_delta;
        double const reference_dv = -reference_delta * (1.0 + reference_ratio)
            / reference_energy / (reference_energy
                + reference_ratio * normalized_energy_in);
        double const reference_zeta = -reference_dv * length * weight * 0.5;

        double const momentum_ratio_before = (
            1.0 + LocalParticle_get_delta(part));
        double const momentum_before = normalized_momentum_in
            * momentum_ratio_before;
        double const energy_before = momentum_ratio_before
            * normalized_energy_in / (1.0 - dv);
        LocalParticle_add_to_zeta(part, -reference_zeta);
        double const particle_time = -LocalParticle_get_zeta(part)
            * energy_before / momentum_before;
        double const rf_phase = sad_phase - wave_number * particle_time;
        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double const energy_gain = (normalized_voltage
            + transverse_voltage * (x * x + y * y))
            * sin(rf_phase) * weight;
        double const energy_after = energy_before + energy_gain;
        double const momentum_change = sqrt(1.0 + energy_gain
            * (energy_before + energy_after)
            / (momentum_before * momentum_before)) - 1.0;
        double const momentum_ratio_after = momentum_ratio_before
            * (1.0 + momentum_change);
        dv = -(1.0 + momentum_ratio_after) / energy_after
            / (energy_after + momentum_ratio_after * normalized_energy_in)
            * (momentum_ratio_after - 1.0);
        double const kick_scale = -cos(2.0 * 0.5 * wave_number
            * particle_time - sad_phase) * weight
            / (wave_number * normalized_momentum_in);
        LocalParticle_update_delta(part, momentum_ratio_after - 1.0);
        LocalParticle_set_px(part, (px_before * momentum_ratio_before
            + 2.0 * transverse_voltage * x * kick_scale)
            / momentum_ratio_after);
        LocalParticle_set_py(part, (py_before * momentum_ratio_before
            + 2.0 * transverse_voltage * y * kick_scale)
            / momentum_ratio_after);
        LocalParticle_set_zeta(part, -particle_time * momentum_ratio_after
            * normalized_momentum_in / energy_after - reference_zeta);

        previous_weight = weight;
        weight *= weight_multiplier;
    }
    double const drift_length = 0.5 * length * previous_weight;
    double const px = LocalParticle_get_px(part);
    double const py = LocalParticle_get_py(part);
    double const transverse_square = px * px + py * py;
    double const longitudinal_change = sqrt(1.0 - transverse_square) - 1.0;
    double const effective_length = drift_length / (1.0 + longitudinal_change);
    LocalParticle_add_to_x(part, px * effective_length);
    LocalParticle_add_to_y(part, py * effective_length);
    LocalParticle_add_to_zeta(part, longitudinal_change * effective_length
        - dv * drift_length);

    if (!backtrack && with_fringe && edge_exit_active) {
        dv = cavity_sad_trpt_fringe_forward(
            part, length, -voltage, frequency, reference_phase, dv);
    }

    double const sad_to_xtrack_exit_scale = (
        1.0 + LocalParticle_get_delta(part));
    LocalParticle_set_px(part, LocalParticle_get_px(part)
        * sad_to_xtrack_exit_scale);
    LocalParticle_set_py(part, LocalParticle_get_py(part)
        * sad_to_xtrack_exit_scale);
    double const beta_in = (
        normalized_momentum_in / normalized_energy_in);
    double const beta_out = (
        normalized_momentum_out / normalized_energy_out);
    LocalParticle_scale_zeta(part, beta_in / beta_out);
    LocalParticle_add_to_s(part, backtrack ? -length : length);

}

GPUFUN
void track_cavity_sad_trpt_particles(
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
    // comment at their use in cavity_sad_trpt_track_single_particle.
    double const nominal_voltage = voltage;
    double const nominal_length = length;
    length *= weight;
    voltage *= weight;
    if (weight != 1.0 && num_steps > 0) {
        num_steps = (int64_t) ceil(num_steps * weight);
    }
    START_PER_PARTICLE_BLOCK(part0, part);
        cavity_sad_trpt_track_single_particle(
            part, length, voltage, nominal_voltage, nominal_length,
            length_offset, frequency, harmonic, lag, phase, absolute_time,
            num_steps, fringe_model, edge_entry_active, edge_exit_active,
            backtrack, kill_energy_kick);
    END_PER_PARTICLE_BLOCK;
}

#endif
