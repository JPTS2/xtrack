// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_TRACK_CAVITY_RS_H
#define XTRACK_TRACK_CAVITY_RS_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_drift.h"
#include "xtrack/beam_elements/elements_src/track_cavity_sagan.h"


GPUFUN
void cavity_rs_track_single_particle(
        LocalParticle* part,
        double length,
        double voltage,
        double frequency,
        double harmonic,
        double lag,
        double phase,
        int64_t absolute_time,
        int64_t fringe_model,
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

    double const reference_phase = cavity_sagan_reference_phase(
        part, frequency, lag, phase, absolute_time);
    double const phase_factor = sin(reference_phase);
    double const q0_abs = fabs(LocalParticle_get_q0(part));
    double const reference_gain = q0_abs * voltage * phase_factor;

    double const zeta_before = LocalParticle_get_zeta(part);
    double const delta_before = LocalParticle_get_delta(part);

    double const p0c = LocalParticle_get_p0c(part);
    double const mass0 = LocalParticle_get_mass0(part);
    double const energy_in = sqrt(p0c * p0c + mass0 * mass0);
    double const energy_out = energy_in + reference_gain;

    if (energy_in <= mass0 || energy_out <= mass0) {
        LocalParticle_kill_particle(part, -20);
        return;
    }

    double const voltage_scale = q0_abs * voltage;
    double const log_energy_ratio = log(energy_out / energy_in);
    double const alpha = (
        fabs(phase_factor) > 1e-14
        ? sqrt(1.0 / 8.0) * log_energy_ratio / phase_factor
        : sqrt(1.0 / 8.0) * voltage_scale / energy_in);
    double const sin_alpha = sin(alpha);
    double const cos_alpha = cos(alpha);
    double const r12 = (
        fabs(voltage_scale) > 1e-14
        ? sqrt(8.0) * energy_in * length / voltage_scale * sin_alpha
        : length);
    double const r21 = (
        -sqrt(1.0 / 8.0) * voltage_scale
        / (length * energy_out) * sin_alpha);
    double const r22 = energy_in / energy_out * cos_alpha;
    double const gradient_on_phase = reference_gain / length;
    double const entry_strength = -gradient_on_phase / (2.0 * energy_in);
    double const exit_strength = gradient_on_phase / (2.0 * energy_out);
    int64_t const with_fringe = fringe_model != -1;

    double m11 = cos_alpha;
    double m12 = r12;
    double m21 = r21;
    double m22 = r22;
    if (with_fringe && edge_entry_active) {
        m21 += m22 * entry_strength;
        m11 += m12 * entry_strength;
    }
    if (with_fringe && edge_exit_active) {
        m21 += exit_strength * m11;
        m22 += exit_strength * m12;
    }

    double const p_in = sqrt(energy_in * energy_in - mass0 * mass0);
    double const p_out = sqrt(energy_out * energy_out - mass0 * mass0);
    double const momentum_ratio = p_out / p_in;

    m21 *= momentum_ratio;
    m22 *= momentum_ratio;

    // Linear zeta/delta companion (R&S itself has none); backtrack uses
    // exit-frame delta as an approximation, exact only for weak fields.
    double const wave_number = 2.0 * PI * frequency / C_LIGHT;
    double const gamma_in = energy_in / mass0;
    double const gamma_out = energy_out / mass0;
    double const beta_in = p_in / energy_in;
    double const beta_out = p_out / energy_out;
    double const gamma_diff = gamma_in - gamma_out;

    double const r56 = -length / (gamma_out * gamma_out * gamma_in * beta_out)
        * (gamma_out + gamma_in) / (beta_out + beta_in);

    double r55_cor = 0.0;
    if (fabs(gamma_diff) > 1e-8 * gamma_in) {
        double const sin_delta_phi = cos(reference_phase);
        r55_cor = wave_number * length * beta_in * (voltage_scale / mass0)
            * sin_delta_phi
            * (gamma_in * gamma_out * (beta_in * beta_out - 1.0) + 1.0)
            / (beta_out * gamma_out * gamma_diff * gamma_diff);
    }

    if (backtrack) {
        double const determinant = m11 * m22 - m12 * m21;
        double const inverse_m11 = m22 / determinant;
        double const inverse_m12 = -m12 / determinant;
        double const inverse_m21 = -m21 / determinant;
        double const inverse_m22 = m11 / determinant;
        m11 = inverse_m11;
        m12 = inverse_m12;
        m21 = inverse_m21;
        m22 = inverse_m22;
    }

    double const x = LocalParticle_get_x(part);
    double const px = LocalParticle_get_px(part);
    double const y = LocalParticle_get_y(part);
    double const py = LocalParticle_get_py(part);
    double const new_x = m11 * x + m12 * px;
    double const new_px = m21 * x + m22 * px;
    double const new_y = m11 * y + m12 * py;
    double const new_py = m21 * y + m22 * py;

    double const rf_phase = cavity_sagan_phase(
        part, frequency, lag, phase, absolute_time);
    double const charge_ratio = LocalParticle_get_charge_ratio(part);
    double const particle_gain = q0_abs * charge_ratio * voltage * sin(rf_phase);
    LocalParticle_add_to_energy(
        part, backtrack ? -particle_gain : particle_gain, 1);

    LocalParticle_set_x(part, new_x);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_y(part, new_y);
    LocalParticle_set_py(part, new_py);

    if (backtrack) {
        LocalParticle_set_zeta(
            part, (zeta_before + r56 * delta_before) / (1.0 + r55_cor));
    } else {
        LocalParticle_set_zeta(
            part, (1.0 + r55_cor) * zeta_before - r56 * delta_before);
    }

    LocalParticle_add_to_s(part, backtrack ? -length : length);
}


GPUFUN
void track_cavity_rs_particles(
        double weight,
        LocalParticle* part0,
        double length,
        double voltage,
        double frequency,
        double harmonic,
        double lag,
        double phase,
        int64_t absolute_time,
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

    length *= weight;
    voltage *= weight;
    int64_t const backtrack = LocalParticle_check_track_flag(
        part0, XS_FLAG_BACKTRACK);
    int64_t const kill_energy_kick = LocalParticle_check_track_flag(
        part0, XS_FLAG_KILL_CAVITY_KICK);

    START_PER_PARTICLE_BLOCK(part0, part);
        cavity_rs_track_single_particle(
            part, length, voltage, frequency, harmonic, lag, phase,
            absolute_time, fringe_model, edge_entry_active,
            edge_exit_active, backtrack, kill_energy_kick);
    END_PER_PARTICLE_BLOCK;
}

#endif
