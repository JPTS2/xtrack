// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_ADAPTER_H
#define XTRACK_CAVITY_MODELS_ADAPTER_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_rf.h"
#include "xtrack/beam_elements/elements_src/track_magnet.h"
#include "xtrack/beam_elements/elements_src/cavity_models.h"
#include "xtrack/beam_elements/elements_src/cavity_models_frozen.h"
#include "xtrack/beam_elements/elements_src/cavity_models_averaged.h"
#include "xtrack/beam_elements/elements_src/cavity_models_paraxial.h"

GPUFUN
int CavityModels_physical_shell(
    const xt_float_or_tpsa q[6],
    xt_float_or_tpsa_arg charge,
    const xt_float_or_tpsa potential[4]
) {
    if (!CavityModel_vector_finite(q, 6) || !CavityModel_vector_finite(potential, 4))
        return CAVITY_MODEL_NONFINITE;
    const xt_float_or_tpsa energy = q[5] - charge * potential[3];
    const xt_float_or_tpsa px = q[3] - charge * potential[0];
    const xt_float_or_tpsa py = q[4] - charge * potential[1];
    const xt_float_or_tpsa square = (energy - 1.) * (energy + 1.) - px * px - py * py;
    if (!CavityModel_scalar_finite(square)) return CAVITY_MODEL_NONFINITE;
    return xt_float_or_tpsa_const_part(energy) > 1.
                   && xt_float_or_tpsa_const_part(square) > 0.
               ? CAVITY_MODEL_OK
               : CAVITY_MODEL_DOMAIN;
}

GPUFUN
int64_t CavityModels_freeze_mask(const CavityModelSource* source) {
    int64_t mask = 0;
    if (source->freeze_x) mask |= (int64_t)1 << 0;
    if (source->freeze_y) mask |= (int64_t)1 << 1;
    if (source->freeze_zeta) mask |= (int64_t)1 << 2;
    if (source->freeze_px) mask |= (int64_t)1 << 3;
    if (source->freeze_py) mask |= (int64_t)1 << 4;
    if (source->freeze_energy) mask |= (int64_t)1 << 5;
    return mask;
}

// Triple-jump compositions of a symmetric fourth-order step: w1 = 1/(2-2^(1/5))
// and w0 = 1-2*w1 give order 6; nesting with 2^(1/7) gives order 8.
#define CAVITY_W6_OUTER 1.1746717580893635
#define CAVITY_W6_INNER (-1.349343516178727)
#define CAVITY_W8_OUTER 1.1161829393253857
#define CAVITY_W8_INNER (-1.2323658786507714)

// The averaged map takes two steps per half-wavelength of the uniform field.
// The paraxial map integrates its generator through each step: two steps are
// enough for a travelling wave, four once a counter wave turns its phase by 2k. Its
// kick error is then about (dE/E per step)^2, so it takes a step for each 2 % of
// energy change, from V/E, the crest voltage over the total energy, and at least 8 to
// resolve a short cavity: the step error of the kick is then below 1e-4, at any
// frequency, where 4 steps give about 1e-3 and one step 10%.
// The full map takes 16 steps per half-wavelength, as 8 lose particles an accelerating
// orbit keeps, and at least 64 for a short cavity, where the wave alone would ask for
// one to three. These minima were measured on smooth envelopes.
GPUFUN
int64_t CavityPlan_cells(
    int model,
    int64_t requested,
    double frequency,
    double length,
    double spatial_half_waves,
    double counter_wave_amplitude,
    double weight,
    int controls,
    double voltage_over_energy
) {
    int64_t parent_cells = requested > 0 ? requested : 64;
    if (requested == 0) {
        const double half_wave = frequency > 0. ? .5 * C_LIGHT / frequency : 0.;
        // Cells shorter than the light-line half wave turn the phase faster.
        const double half_cells =
            fmax(frequency > 0. ? length / half_wave : 0., spatial_half_waves);
        if (model == CAVITY_MODEL_AVERAGED || model == CAVITY_MODEL_PARAXIAL) {
            double per_half = 2.;
            if (model == CAVITY_MODEL_PARAXIAL && counter_wave_amplitude > 0.)
                per_half = 4.;
            parent_cells = half_cells > .5 ? (int64_t)ceil(per_half * half_cells) : 1;
            if (model == CAVITY_MODEL_PARAXIAL)
                parent_cells = (int64_t)fmax(
                    fmax((double)parent_cells, 8.), ceil(voltage_over_energy / .02)
                );
        }
        else if (model == CAVITY_MODEL_FULL) {
            parent_cells = (int64_t)fmax(64., ceil(16. * half_cells));
        }
    }
    return controls == CAVITY_MODEL_OK ? (int64_t)ceil(parent_cells * weight) : 1;
}

GPUFUN
int CavityPlan_controls(
    const RFModelFields* el,
    LocalParticle* part0,
    int model,
    int integrator,
    int64_t requested,
    double weight,
    double slice_start
) {
    const double length = el->length;
    const int profile = el->rf_profile;
    int controls = CAVITY_MODEL_OK;
    if (!(isfinite(length) && length > 0. && isfinite(weight) && weight > 0.
          && isfinite(slice_start) && slice_start >= 0. && weight <= 1.
          && slice_start + weight <= 1. + 8e-15)
        || el->n_cells < 0 || requested < 0
        || (model != CAVITY_MODEL_FULL && model != CAVITY_MODEL_FROZEN
            && model != CAVITY_MODEL_AVERAGED && model != CAVITY_MODEL_PARAXIAL)
        || profile < 0 || profile > 1
        || (profile == 1 && (model != CAVITY_MODEL_FULL || el->num_harmonics <= 0))
        || (el->knl_len > 0
            && (el->order < 0 || el->knl_len <= el->order || el->ksl_len <= el->order)))
        controls = CAVITY_MODEL_ARGUMENT;
    if (model == CAVITY_MODEL_FROZEN
        && (!xt_float_or_tpsa_is_zero(el->counter_wave_amplitude) || el->n_cells != 0))
        controls = CAVITY_MODEL_ARGUMENT;
    if (
        integrator != INTEGRATOR_ADAPTIVE && integrator != INTEGRATOR_UNIFORM
        && integrator != INTEGRATOR_YOSHIDA4 && integrator != INTEGRATOR_YOSHIDA6
        && integrator != INTEGRATOR_YOSHIDA8 && integrator != CAVITY_INTEGRATOR_GEOMETRIC
    ) {
        controls = CAVITY_MODEL_ARGUMENT;
    }
    // The whole-cavity taper pass takes the longitudinal map (track_cavity_plan);
    // tapering refuses cavity slices.
    if (LocalParticle_check_track_flag(part0, XS_FLAG_SR_TAPER))
        controls = CAVITY_MODEL_ARGUMENT;
    return controls;
}

// Spin follows the magnets: it is transported when spin or radiation is compiled in,
// for particles whose spin is not zero, in every model and slice.
GPUFUN
int CavityPlan_spin_supported(const RFModelFields* el) {
    return !el->tpsa_enabled;
}

// The half wavelengths of the field's fastest spatial variation along the element.
GPUFUN
double CavityPlan_spatial_half_waves(const RFModelFields* el) {
    double half_waves = (double)el->n_cells;
    for (int64_t m = 0; m < el->num_harmonics; ++m)
        half_waves =
            fmax(half_waves, fabs(el->harmonic_wave_number[m]) * el->length / PI);
    return half_waves;
}

// Reads only the track flags of the particle, as RFPlan_prepare does.
GPUFUN
void CavityPlan_prepare(
    LocalParticle* part0,
    double weight,
    double slice_start,
    CavityPlan* plan
) {
    const RFModelFields* el = &plan->fields;
    const int model = el->model;
    plan->model = model;
    plan->thin_slice = 0;
    // Slice ends within a few ulps of the parent's ends are taken as those ends.
    const double left_fraction = fabs(slice_start) < 8e-15 ? 0. : slice_start;
    const double raw_right = slice_start + weight;
    const double right_fraction = fabs(raw_right - 1.) < 8e-15 ? 1. : raw_right;
    plan->left_fraction = left_fraction;
    plan->right_fraction = right_fraction;
    if (model < 10) return;

    const double length = el->length;
    const double frequency = xt_float_or_tpsa_const_part(el->frequency);
    const double counter_wave_amplitude =
        xt_float_or_tpsa_const_part(el->counter_wave_amplitude);
    const int integrator = el->integrator;
    const int64_t requested = el->num_kicks;
    plan->integrator = integrator;
    plan->requested = requested;
    plan->counter_wave_amplitude = counter_wave_amplitude;
    plan->profile = el->rf_profile;
    plan->spin_supported =
        CavityPlan_spin_supported(el);
    const int controls =
        CavityPlan_controls(el, part0, model, integrator, requested, weight, slice_start);
    plan->choice.controls = controls;
    plan->choice.cells = CavityPlan_cells(
        model,
        requested,
        frequency,
        length,
        CavityPlan_spatial_half_waves(el),
        counter_wave_amplitude,
        weight,
        controls,
        0.
    );

    const int reverse = LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK);
    plan->length = length;
    plan->weight = weight;
    plan->start_fraction = reverse ? right_fraction : left_fraction;
    plan->end_fraction = reverse ? left_fraction : right_fraction;
    plan->direction = reverse ? -1. : 1.;
    plan->reverse = reverse;
    plan->clear_face = reverse ? el->edge_entry_active : el->edge_exit_active;
    plan->physical_outer = reverse ? left_fraction == 0. : right_fraction == 1.;
}


// Gauge transformation between the phase-resolved temporal RF potential and
// the field-free-outside gauge of the end-plane impulse. The gauge function
// Lambda = sum_{n<M} (-1/4)^n/(4 n!(n+1)!(n+1)) r^(2n+2) P_n', with
// P_n = (1 + d_Z^2)^n S (at M = 2, (r^2/4) S1 - (r^4/64)(S1+S3)), gives the
// temporal A_perp = grad Lambda and the outside gauge's phi = d_T Lambda;
// to_temporal applies P += Q grad Lambda, E -= Q d_T Lambda, otherwise the
// inverse. The energy change is the face's r^2 impulse of E_z, which turns
// the spin through the transverse velocity.
GPUFUN
void CavityModels_rf_gauge(
    xt_float_or_tpsa q[6],
    const CavityModelSource* source,
    xt_float_or_tpsa_arg Q,
    xt_float_or_tpsa_arg z,
    int to_temporal,
    const CavityEffects* spin
) {
    const int M = (int)(source->radial_order / 2), jets = 2 * M;
    xt_float_or_tpsa real[CAVITY_MODEL_JETS], imag[CAVITY_MODEL_JETS];
    xt_float_or_tpsa S[CAVITY_MODEL_JETS], C[CAVITY_MODEL_JETS];
    CavityModel_profile(z, source, real, imag, jets);
    const xt_float_or_tpsa c = cos(q[2]), s = sin(q[2]);
    for (int j = 0; j < jets; ++j) {
        S[j] = real[j] * s - imag[j] * c;
        C[j] = real[j] * c + imag[j] * s;
    }
    const xt_float_or_tpsa r2 = q[0] * q[0] + q[1] * q[1];
    xt_float_or_tpsa radial = 0., scalar = 0., power = 1.;
    double coefficient = .5;
    for (int n = 0; n < M; ++n) {
        radial += coefficient * power * S[1];
        power = power * r2;
        scalar += coefficient / (2. * (n + 1.)) * power * C[1];
        for (int j = 0; j + 2 < jets - 2 * n; ++j) {
            S[j] += S[j + 2];
            C[j] += C[j + 2];
        }
        coefficient *= -.25 / ((n + 1.) * (n + 2.));
    }
    xt_float_or_tpsa charge = Q;
    if (!to_temporal) charge = -Q;
    double kinetic[2] = {
        xt_float_or_tpsa_const_part(q[3]), xt_float_or_tpsa_const_part(q[4])
    };
    q[3] += charge * q[0] * radial;
    q[4] += charge * q[1] * radial;
    if (!to_temporal) {
        kinetic[0] = xt_float_or_tpsa_const_part(q[3]);
        kinetic[1] = xt_float_or_tpsa_const_part(q[4]);
    }
    if (!source->freeze_energy) {
        const double energy = xt_float_or_tpsa_const_part(q[5]);
        q[5] -= charge * scalar;
        CavitySpin_energy(
            spin,
            kinetic[0],
            kinetic[1],
            energy,
            xt_float_or_tpsa_const_part(q[5]) - energy
        );
    }
}

// The physical models for one particle, shared by the Cavity and its slices.
GPUFUN
void track_cavity_single_particle(
    const CavityPlan* plan,
    int model,
    LocalParticle* part,
    CavityMagnusCache* magnus_cache
) {
    const RFModelFields* el = &plan->fields;
    const int controls = plan->choice.controls;
    int64_t cells = plan->choice.cells;
    const int integrator = plan->integrator;
    const int profile = plan->profile;
    int spin_tracking = 0;
    const int64_t requested = plan->requested;
    const double length = plan->length;
    const double weight = plan->weight;
    const double left_fraction = plan->left_fraction;
    const double right_fraction = plan->right_fraction;
    const double start_fraction = plan->start_fraction;
    const double end_fraction = plan->end_fraction;
    const double direction = plan->direction;
    const int reverse = plan->reverse;
    const int clear_face = plan->clear_face;
    const int physical_outer = plan->physical_outer;

    int status = controls;
    double anomaly = 0., spin[3] = {0., 0., 0.};
    // Spin is set below once tracked; the static fields radiate going forward.
    CavityEffects effects = {NULL, 0., part, 0, 0., el->length, 0., -1, el->knl, el->ksl};
#if !defined(XTRACK_TPSA_TRACK) && !defined(XTRACK_MULTIPOLE_NO_SYNRAD)
    if (!reverse && LocalParticle_get_q0(part) != 0.) {
        effects.radiation_flag = el->radiation_flag;
        effects.rigidity = LocalParticle_get_p0c(part) / (C_LIGHT * LocalParticle_get_q0(part));
        effects.ks = xt_float_or_tpsa_const_part(el->ks);
        effects.order = el->knl_len > 0 ? el->order : -1;
    }
#endif
    const double mass0 = LocalParticle_get_mass0(part);
    const double q0 = LocalParticle_get_q0(part);
    const double kappa0 = LocalParticle_get_p0c(part) / mass0;
    const double beta0 = LocalParticle_get_beta0(part);
    const double chi = LocalParticle_get_chi(part);
    const double mass_ratio =
        chi != 0. ? LocalParticle_get_charge_ratio(part) / chi : -1.;
    const double Q = q0 * chi;
    if (!(isfinite(mass0) && mass0 > 0. && isfinite(q0) && q0 != 0. && isfinite(kappa0)
          && kappa0 > 0. && isfinite(beta0) && beta0 > 0. && beta0 < 1. && isfinite(Q)
          && isfinite(mass_ratio) && mass_ratio > 0.))
        status = CAVITY_MODEL_ARGUMENT;
    const xt_float_or_tpsa one_plus_delta = 1. + LocalParticle_get_delta(part);
    if (status == CAVITY_MODEL_OK && !CavityModel_scalar_finite(one_plus_delta))
        status = CAVITY_MODEL_NONFINITE;
    if (status == CAVITY_MODEL_OK && xt_float_or_tpsa_const_part(one_plus_delta) <= 0.)
        status = CAVITY_MODEL_DOMAIN;
    xt_float_or_tpsa frequency = el->frequency;
    const xt_float_or_tpsa harmonic = el->harmonic;
    if (!xt_float_or_tpsa_is_zero(harmonic)) {
        if (!(part->line_length > 0.))
            status = CAVITY_MODEL_ARGUMENT;
        else
            frequency += harmonic * beta0 * C_LIGHT / part->line_length;
    }
    // A harmonic frequency and the energy change follow this particle's reference.
    if (!xt_float_or_tpsa_is_zero(harmonic)
        || (model == CAVITY_MODEL_PARAXIAL && requested == 0))
        cells = CavityPlan_cells(
            model,
            requested,
            xt_float_or_tpsa_const_part(frequency),
            length,
            CavityPlan_spatial_half_waves(el),
            plan->counter_wave_amplitude,
            weight,
            controls,
            fabs(q0 * xt_float_or_tpsa_const_part(el->voltage))
                / sqrt(
                    mass0 * mass0
                    + LocalParticle_get_p0c(part) * LocalParticle_get_p0c(part)
                )
        );
    xt_float_or_tpsa phase =
        el->phase + DEG2RAD * el->lag + el->phase_taper + DEG2RAD * el->lag_taper;
    // The taper phases apply whether or not radiation is compiled, unlike the
    // longitudinal models (track_rf.h), which add them only with radiation.
    const xt_float_or_tpsa voltage = el->voltage;
    if (model == CAVITY_MODEL_FROZEN && !xt_float_or_tpsa_is_zero(voltage)
        && xt_float_or_tpsa_const_part(frequency) <= 0.)
        status = CAVITY_MODEL_ARGUMENT;
    // Cells are pi-mode cells of fixed length: they set the spatial wave number,
    // and voltage is the crest gain at their synchronous speed, at most c.
    xt_float_or_tpsa wave_number = (2. * PI / C_LIGHT) * frequency;
    xt_float_or_tpsa calibration_beta = el->calibration_beta;
    if (el->n_cells > 0) {
        wave_number = PI * (double)el->n_cells / length;
        const xt_float_or_tpsa synchronous =
            frequency * (2. * length / (C_LIGHT * (double)el->n_cells));
        if (xt_float_or_tpsa_const_part(synchronous) < 1.) calibration_beta = synchronous;
    }
    CavityModelSource source;
    if (status == CAVITY_MODEL_OK && profile == 1) {
        status = CavityModel_prepare_harmonics(
            frequency,
            voltage,
            length,
            phase,
            calibration_beta,
            mass0,
            q0,
            el->num_harmonics,
            el->harmonic_wave_number,
            el->harmonic_re,
            el->harmonic_im,
            &source
        );
        if (status == CAVITY_MODEL_OK && el->table) {
            // The table is used only for the harmonics and length it was built
            // from (FieldExpansionCavity._field_table_key); else the direct sum.
            double key = 11. * el->length;
            for (int64_t m = 0; m < el->num_harmonics; ++m)
                key += (double)(m + 1)
                       * (el->harmonic_wave_number[m] + 3. * el->harmonic_re[m]
                          + 7. * el->harmonic_im[m]);
            if (fabs(key - el->table_key) <= 1e-12 * (fabs(key) + 1.)) {
                source.table = el->table;
                source.table_points = el->table_points;
                source.table_z0 = el->table_z0;
                source.table_dz = el->table_dz;
            }
        }
    }
    else if (status == CAVITY_MODEL_OK) {
        status = CavityModel_prepare(
            frequency,
            model == CAVITY_MODEL_FROZEN ? 0. : voltage,
            length,
            phase,
            el->counter_wave_amplitude,
            el->counter_wave_phase,
            calibration_beta,
            wave_number,
            mass0,
            q0,
            &source
        );
    }
    xt_float_or_tpsa q[6], clock_base = 0.;
    xt_float_or_tpsa z = 0., h = 0.;
    if (status == CAVITY_MODEL_OK) {
        source.ks = el->ks;
        source.radial_order = el->radial_order;
        if (model == CAVITY_MODEL_FROZEN) source.voltage_norm = voltage / mass0;
        source.magnetic_scale = LocalParticle_get_p0c(part) / (q0 * mass0);
        source.order = el->order;
        source.knl = el->knl;
        source.ksl = el->ksl;
        source.knl_tpsa = el->knl_tpsa;
        source.ksl_tpsa = el->ksl_tpsa;
        source.reference_beta = beta0;
        // The reduced models compose their symmetric step; the frozen model's
        // compositions are of the uniform mesh.
        source.composition = model == CAVITY_MODEL_FULL ? 0
                             : integrator == INTEGRATOR_YOSHIDA4 ? 1
                             : integrator == INTEGRATOR_YOSHIDA6 ? 2
                             : integrator == INTEGRATOR_YOSHIDA8 ? 3 : 0;
        source.uniform_mesh = integrator == INTEGRATOR_UNIFORM || source.composition > 0;
        source.freeze_energy =
            LocalParticle_check_track_flag(part, XS_FLAG_KILL_CAVITY_KICK)
            || XT_LP_IS_FROZEN(delta) || XT_LP_IS_FROZEN(pzeta);
        source.freeze_zeta = XT_LP_IS_FROZEN(zeta);
        source.freeze_x = XT_LP_IS_FROZEN(x);
        source.freeze_px = XT_LP_IS_FROZEN(px);
        source.freeze_y = XT_LP_IS_FROZEN(y);
        source.freeze_py = XT_LP_IS_FROZEN(py);
#if !defined(XTRACK_TPSA_TRACK) && !defined(XTRACK_MULTIPOLE_NO_SYNRAD)
        if (LocalParticle_get_spin_x(part) != 0. || LocalParticle_get_spin_y(part) != 0.
            || LocalParticle_get_spin_z(part) != 0.) {
            if (plan->spin_supported)
                spin_tracking = 1;
            else
                status = CAVITY_MODEL_ARGUMENT;
        }
#endif
        if (spin_tracking) {
            if (XT_LP_IS_FROZEN(s)
                || XT_LP_IS_FROZEN(rpp) || XT_LP_IS_FROZEN(rvv) || XT_LP_IS_FROZEN(ax)
                || XT_LP_IS_FROZEN(ay) || XT_LP_IS_FROZEN(spin_x)
                || XT_LP_IS_FROZEN(spin_y) || XT_LP_IS_FROZEN(spin_z))
                status = CAVITY_MODEL_ARGUMENT;
#ifndef XTRACK_TPSA_TRACK
            if (status == CAVITY_MODEL_OK) {
                anomaly = LocalParticle_get_anomalous_magnetic_moment(part);
                spin[0] = LocalParticle_get_spin_x(part);
                spin[1] = LocalParticle_get_spin_y(part);
                spin[2] = LocalParticle_get_spin_z(part);
                if (!isfinite(anomaly)) status = CAVITY_MODEL_ARGUMENT;
                for (int j = 0; j < 3; ++j)
                    if (!isfinite(spin[j])) status = CAVITY_MODEL_NONFINITE;
                const int static_start = reverse ? right_fraction == 1. : left_fraction == 0.;
                const int start_face = reverse ? el->edge_exit_active : el->edge_entry_active;
                effects.spin = spin;
                effects.anomaly = anomaly;
                if (status == CAVITY_MODEL_OK && static_start && start_face) {
                    CavitySpin_solenoid_face(
                        &effects,
                        xt_float_or_tpsa_const_part(Q * source.magnetic_scale * el->ks),
                        kappa0 * (1. + LocalParticle_get_delta(part)),
                        LocalParticle_get_x(part),
                        LocalParticle_get_y(part),
                        1.
                    );
                }
            }
#endif
        }
        if (model == CAVITY_MODEL_FROZEN
            && (source.freeze_x || source.freeze_px || source.freeze_y
                || source.freeze_py))
            status = CAVITY_MODEL_ARGUMENT;
        z = source.k * length * (start_fraction - .5);
        h = direction * source.k * length * weight / cells;
        clock_base = source.clock_origin;
        if (el->absolute_time) {
            const xt_float_or_tpsa parent_s =
                LocalParticle_get_s(part) - length * start_fraction;
            // The RF phase accumulated at time_s, as in track_rf.
            if (!xt_float_or_tpsa_is_zero(harmonic))
                clock_base += 2. * PI
                              * (el->frequency * LocalParticle_get_time_s(part)
                                 + harmonic * LocalParticle_get_reference_turns(part));
            else
                clock_base += source.omega * LocalParticle_get_time_s(part);
            clock_base += source.k * parent_s / beta0;
        }
        q[0] = source.k * LocalParticle_get_x(part);
        q[1] = source.k * LocalParticle_get_y(part);
        q[2] = clock_base + (z + source.half_span) / beta0
               - source.k * LocalParticle_get_zeta(part) / beta0;
        q[3] = kappa0 * LocalParticle_get_px(part);
        q[4] = kappa0 * LocalParticle_get_py(part);
        const xt_float_or_tpsa momentum = kappa0 * one_plus_delta;
        q[5] = sqrt(1. + momentum * momentum);
    }
    if (status == CAVITY_MODEL_OK
        && (model == CAVITY_MODEL_AVERAGED || model == CAVITY_MODEL_PARAXIAL)) {
        xt_float_or_tpsa next[6];
        if (source.freeze_x || source.freeze_px || source.freeze_y || source.freeze_py)
            status = CAVITY_MODEL_ARGUMENT;
        else {
            const xt_float_or_tpsa gain =
                fabs(q0 * xt_float_or_tpsa_const_part(el->voltage)) * weight / mass0;
            const xt_float_or_tpsa entry = sqrt(1. + kappa0 * kappa0);
            const xt_float_or_tpsa grading = gain * (1. / entry + 1. / (entry + gain));
            if (model == CAVITY_MODEL_AVERAGED)
                status = CavityModel_averaged_interval(
                    q,
                    Q,
                    &source,
                    cells,
                    left_fraction,
                    right_fraction,
                    reverse,
                    el->edge_entry_active,
                    el->edge_exit_active,
                    el->rf_edge_entry_active,
                    el->rf_edge_exit_active,
                    kappa0,
                    integrator,
                    grading,
                    &effects,
                    next
                );
            else
                status = CavityModel_paraxial_interval(
                    q,
                    Q,
                    &source,
                    cells,
                    left_fraction,
                    right_fraction,
                    reverse,
                    el->edge_entry_active,
                    el->edge_exit_active,
                    el->rf_edge_entry_active,
                    el->rf_edge_exit_active,
                    kappa0,
                    integrator,
                    grading,
                    &effects,
                    magnus_cache,
                    next
                );
        }
        if (status == CAVITY_MODEL_OK) {
            for (int j = 0; j < 6; ++j) q[j] = next[j];
            z = source.k * length * (end_fraction - .5);
        }
    }
    if (status == CAVITY_MODEL_OK && model == CAVITY_MODEL_FROZEN) {
        xt_float_or_tpsa next[6];
        // The nominal programme sees the parent-reference arrival phase,
        // including the laboratory anchor, never the particle's zeta or
        // evolving delta. Actual species charge enters the word via Q.
        const xt_float_or_tpsa nominal_gain =
            el->nominal_energy_gain_from_rf
                ? fabs(q0) * source.voltage_norm
                      * sin(source.phase + clock_base - source.clock_origin)
                : el->nominal_energy_gain / mass0;
        // Explicit eV/reference-particle programme already includes the
        // reference charge. Actual-species chi never retunes this input.
        if (!CavityModel_scalar_finite(nominal_gain))
            status = CAVITY_MODEL_NONFINITE;
        else
            status = CavityModel_frozen_interval(
                q,
                Q,
                &source,
                kappa0,
                nominal_gain,
                requested,
                el->edge_entry_active,
                el->edge_exit_active,
                el->rf_edge_entry_active,
                el->rf_edge_exit_active,
                left_fraction,
                right_fraction,
                reverse,
                &effects,
                next
            );
        if (status == CAVITY_MODEL_OK) {
            for (int j = 0; j < 6; ++j) q[j] = next[j];
            z = source.k * length * (end_fraction - .5);
        }
    }
    // A whole cavity's end planes are hard edges of both waves: the impulse in
    // the field-free-outside gauge, then the temporal gauge. With an RF face
    // off the field continues past it: the gauge change alone.
    const int physical_start = reverse ? right_fraction == 1. : left_fraction == 0.;
    const int rf_start = reverse ? el->rf_edge_exit_active : el->rf_edge_entry_active;
    if (status == CAVITY_MODEL_OK && model == CAVITY_MODEL_FULL && physical_start) {
        xt_float_or_tpsa envelope[5];
        CavityModel_source_envelope(z, &source, envelope);
        if (rf_start)
            status = CavityMagnus_edge(
                q, &source, Q, z, direction * envelope[0], direction, &effects
            );
        if (status == CAVITY_MODEL_OK)
            CavityModels_rf_gauge(q, &source, Q, z, 1, &effects);
    }
    // Composition of the fourth-order Gauss cell: yoshida-6 is one
    // triple jump, yoshida-8 two; geometric grades the cell lengths.
    const double order4[1] = {1.};
    const double order6[3] = {CAVITY_W6_OUTER, CAVITY_W6_INNER, CAVITY_W6_OUTER};
    const double order8[9] = {
        CAVITY_W8_OUTER * CAVITY_W6_OUTER,
        CAVITY_W8_OUTER * CAVITY_W6_INNER,
        CAVITY_W8_OUTER * CAVITY_W6_OUTER,
        CAVITY_W8_INNER * CAVITY_W6_OUTER,
        CAVITY_W8_INNER * CAVITY_W6_INNER,
        CAVITY_W8_INNER * CAVITY_W6_OUTER,
        CAVITY_W8_OUTER * CAVITY_W6_OUTER,
        CAVITY_W8_OUTER * CAVITY_W6_INNER,
        CAVITY_W8_OUTER * CAVITY_W6_OUTER
    };
    const double* substeps = order4;
    int num_substeps = 1;
    if (integrator == INTEGRATOR_YOSHIDA6) {
        substeps = order6;
        num_substeps = 3;
    }
    else if (integrator == INTEGRATOR_YOSHIDA8) {
        substeps = order8;
        num_substeps = 9;
    }
    xt_float_or_tpsa grading = 0., grading_log = 0.;
    if (model == CAVITY_MODEL_FULL && integrator == CAVITY_INTEGRATOR_GEOMETRIC
        && status == CAVITY_MODEL_OK) {
        const xt_float_or_tpsa entry = sqrt(1. + kappa0 * kappa0);
        const xt_float_or_tpsa gain =
            fabs(q0 * xt_float_or_tpsa_const_part(el->voltage)) * weight / mass0;
        grading = gain * (1. / entry + 1. / (entry + gain));
        grading_log = CavityModel_log1p(grading);
    }
    for (int64_t cell = 0;
         model == CAVITY_MODEL_FULL && status == CAVITY_MODEL_OK && cell < cells;
         ++cell) {
        xt_float_or_tpsa cell_h = h;
        if (integrator == CAVITY_INTEGRATOR_GEOMETRIC)
            cell_h = h * cells
                     * CavityModel_frozen_weight(
                         reverse ? cells - 1 - cell : cell, cells, grading, grading_log
                     );
        const double start_xy[2] = {xt_float_or_tpsa_const_part(q[0]),
                                    xt_float_or_tpsa_const_part(q[1])};
        for (int sub = 0; status == CAVITY_MODEL_OK && sub < num_substeps; ++sub) {
            const xt_float_or_tpsa step = cell_h * substeps[sub];
            xt_float_or_tpsa next[6];
#ifndef XTRACK_TPSA_TRACK
            if (spin_tracking) {
                double next_spin[3];
                status = CavityModel_spin_cell(
                    q, spin, z, step, Q, &source, anomaly, next, next_spin
                );
                if (status == CAVITY_MODEL_OK)
                    for (int j = 0; j < 3; ++j) spin[j] = next_spin[j];
            }
            else
#endif
                status = CavityModel_gauss_halving(
                    q, z, step, Q, &source, model, CAVITY_MODEL_HALVING_DEPTH, next
                );
            if (status == CAVITY_MODEL_OK) {
                for (int j = 0; j < 6; ++j) q[j] = next[j];
                z += step;
            }
        }
#ifndef XTRACK_TPSA_TRACK
        // The static fields radiate once per cell, at its end, from the kinetic
        // momenta and the energy outside the RF potential.
        if (status == CAVITY_MODEL_OK && effects.radiation_flag > 0) {
            double a[4];
            status = CavityModel_potential(q, z, &source, model, a);
            if (status == CAVITY_MODEL_OK) {
                double u[6] = {q[0], q[1], q[2], q[3] - Q * a[0], q[4] - Q * a[1],
                               q[5] - Q * a[3]};
                status = CavityEffects_radiate_state(
                    &effects, u, start_xy, source.k, 0., cell_h / source.k
                );
                q[3] = u[3] + Q * a[0];
                q[4] = u[4] + Q * a[1];
                q[5] = u[5] + Q * a[3];
            }
        }
#endif
    }
    xt_float_or_tpsa result[6], potential[4];
    CavityModelSource field_source = source;
    if (status == CAVITY_MODEL_OK) {
        // The potential is read at the slice's end plane, not at the summed
        // steps, which can stop a few ulps inside a face.
        if (model == CAVITY_MODEL_FULL) z = source.k * length * (end_fraction - .5);
        // The exit plane of a whole cavity: back to the outside gauge and
        // the closing impulse, after which only static fields remain.
        const int rf_end = reverse ? el->rf_edge_entry_active : el->rf_edge_exit_active;
        if (model == CAVITY_MODEL_FULL && physical_outer) {
            xt_float_or_tpsa envelope[5];
            CavityModel_source_envelope(z, &source, envelope);
                CavityModels_rf_gauge(q, &source, Q, z, 0, &effects);
            if (rf_end)
                status = CavityMagnus_edge(
                    q, &source, Q, z, -direction * envelope[0], direction, &effects
                );
            field_source.forward = 0.;
            field_source.backward = 0.;
        }
        // Every model reads the full field's potential, for the kinetic momenta.
        if (status == CAVITY_MODEL_OK)
            status = CavityModel_potential(q, z, &field_source, model, potential);
    }
    if (status == CAVITY_MODEL_OK) {
        const xt_float_or_tpsa energy = q[5] - Q * potential[3];
        const xt_float_or_tpsa total2 = (energy - 1.) * (energy + 1.);
        status = CavityModels_physical_shell(q, Q, potential);
        if (status == CAVITY_MODEL_OK) {
            result[0] = q[0] / source.k;
            result[1] = q[3] / kappa0;
            result[2] = q[1] / source.k;
            result[3] = q[4] / kappa0;
            result[4] =
                beta0 / source.k
                * (clock_base
                   + (source.k * length * (end_fraction - .5) + source.half_span) / beta0
                   - q[2]);
            result[5] = sqrt(total2) / kappa0 - 1.;
            if (!CavityModel_vector_finite(result, 6)) status = CAVITY_MODEL_NONFINITE;
            if (status == CAVITY_MODEL_OK
                && xt_float_or_tpsa_const_part(1. + result[5]) <= 0.)
                status = CAVITY_MODEL_DOMAIN;
        }
    }
    // Beyond an active face of a whole element only the RF potential remains:
    // the static fields end there. Inner slice planes keep both.
    xt_float_or_tpsa readback[4];
    if (status == CAVITY_MODEL_OK) {
        for (int i = 0; i < 4; ++i) readback[i] = potential[i];
        if (clear_face && physical_outer) {
            CavityModelSource rf_only = field_source;
            rf_only.ks = 0.;
            rf_only.magnetic_scale = 0.;
            rf_only.order = -1;
            rf_only.knl = NULL;
            rf_only.ksl = NULL;
            rf_only.knl_tpsa = NULL;
            rf_only.ksl_tpsa = NULL;
            status = CavityModel_potential(q, z, &rf_only, model, readback);
            if (status == CAVITY_MODEL_OK)
                status = CavityModels_physical_shell(q, Q, readback);
        }
    }
    // Written once, so that a lost particle keeps its entry coordinates.
    if (status == CAVITY_MODEL_OK) {
        LocalParticle_set_x(part, result[0]);
        LocalParticle_set_px(part, result[1]);
        LocalParticle_set_y(part, result[2]);
        LocalParticle_set_py(part, result[3]);
        LocalParticle_set_zeta(part, result[4]);
        LocalParticle_update_delta(part, result[5]);
        LocalParticle_add_to_s(part, direction * length * weight);
        LocalParticle_set_ax(part, Q * readback[0] / kappa0);
        LocalParticle_set_ay(part, Q * readback[1] / kappa0);
#ifndef XTRACK_TPSA_TRACK
        if (spin_tracking) {
            const int end_face = reverse ? el->edge_entry_active : el->edge_exit_active;
            if (physical_outer && end_face) {
                CavitySpin_solenoid_face(
                    &effects,
                    xt_float_or_tpsa_const_part(Q * source.magnetic_scale * el->ks),
                    kappa0 * (1. + xt_float_or_tpsa_const_part(result[5])),
                    xt_float_or_tpsa_const_part(result[0]),
                    xt_float_or_tpsa_const_part(result[2]),
                    -1.
                );
            }
            LocalParticle_set_spin_x(part, spin[0]);
            LocalParticle_set_spin_y(part, spin[1]);
            LocalParticle_set_spin_z(part, spin[2]);
        }
#endif
    }
    else if (status != CAVITY_MODEL_LOST) {
        // -99741 domain, -99742 stage exhaustion, -99743 nonfinite,
        // -99744 invalid/unfinished source or numerical configuration.
        LocalParticle_set_state(part, -99740 + status);
    }
}

// Longitudinal-kick model of a MagnetOverlappedCavity: the magnet's integrator,
// with the RF energy kick added to each magnet kick, and the faces of a straight
// magnet where the piece meets the parent's ends.
#define CAVITY_MAGNET_BODY  0
#define CAVITY_MAGNET_KICK  1
#define CAVITY_MAGNET_DRIFT 2
GPUFUN
void CavityModels_track_longitudinal_magnet(
    LocalParticle* part0,
    const CavityPlan* cavity,
    const RFBody* rf,
    double weight,
    int part_of_body
) {
    const RFModelFields* fields = &cavity->fields;
    const int with_kicks = part_of_body != CAVITY_MAGNET_DRIFT;
    const int with_faces = part_of_body != CAVITY_MAGNET_KICK;
    // The faces of a Quadrupole or UniformSolenoid: "full", with no soft fringe.
    const int64_t face_flags = EDGE_ACTIVE | EDGE_BEND | EDGE_MULT | EDGE_QUAD_LW;
    MagnetFace entry_face = MagnetFace_empty(0);
    MagnetFace exit_face = MagnetFace_empty(1);
    if (with_faces && fields->edge_entry_active && cavity->left_fraction == 0.)
        entry_face.flags = face_flags;
    if (with_faces && fields->edge_exit_active && cavity->right_fraction == 1.)
        exit_face.flags = face_flags;
    MagnetSettings magnet;
    MagnetSettings_init_default(&magnet);
    magnet.length = fields->length;
    magnet.weight = weight;
    magnet.order = fields->order;
    magnet.inv_factorial_order = fields->inv_factorial_order;
    magnet.knl = fields->knl;
    magnet.ksl = fields->ksl;
    magnet.ks = fields->ks;
    // The parent's count, scaled to the slice by the magnet; 0 is one kick, as
    // for Cavity, so the RF kick is never dropped.
    const int64_t num_kicks = fields->num_kicks > 0 ? fields->num_kicks : 1;
    magnet.num_multipole_kicks = with_kicks ? num_kicks : 0;
    magnet.integrator = with_kicks ? rf->integrator : INTEGRATOR_UNIFORM;
    magnet.radiation_flag = cavity->reverse ? 0 : fields->radiation_flag;
    if (part_of_body == CAVITY_MAGNET_KICK)
        magnet.model = MODEL_KICK_ONLY;
    else if (xt_float_or_tpsa_const_part(fields->ks) != 0.)
        magnet.model = MODEL_SOL_KICK_SOL;
    else
        magnet.model = MODEL_DRIFT_KICK_DRIFT_EXACT;
    MagnetPlan plan;
    MagnetPlan_prepare(&plan, &magnet, &entry_face, &exit_face, part0);
    // A drift piece keeps the multipoles for its faces only.
    if (!with_kicks) plan.body.drift_only = 1;
    const MagnetBody* body = &plan.body;
    const uint8_t first_active = plan.first_active;
    const uint8_t second_active = plan.second_active;

#define LONGITUDINAL_MAGNET_DRIFT(part, dlength) \
    MagnetDrift_apply(part, &body->drift, (dlength))
#define LONGITUDINAL_MAGNET_KICK(part, kick_weight)                                  \
    do {                                                                             \
        if (!body->kick_is_noop) MagnetKick_apply(part, &body->kick, (kick_weight)); \
        if (rf)                                                                      \
            track_rf_kick_single_particle(                                           \
                part,                                                                \
                rf->voltage * (kick_weight),                                         \
                rf->frequency,                                                       \
                rf->harmonic,                                                        \
                rf->lag,                                                             \
                rf->phase,                                                           \
                0.,                                                                  \
                0.,                                                                  \
                0.,                                                                  \
                rf->absolute_time,                                                   \
                -1,                                                                  \
                0.,                                                                  \
                NULL,                                                                \
                NULL,                                                                \
                NULL,                                                                \
                NULL,                                                                \
                NULL,                                                                \
                NULL,                                                                \
                rf->kill_energy_kick                                                 \
            );                                                                       \
    } while (0)
#if defined(XTRACK_TPSA_TRACK) || defined(XTRACK_MULTIPOLE_NO_SYNRAD)
#define LONGITUDINAL_MAGNET_STEP(ll, code) \
    {                                      \
        code;                              \
    }
#else
    // The magnet body's spin and radiation after each step, as in track_magnet.h.
    const double length = body->length;
    double dp_record_exit = 0., dpx_record_exit = 0., dpy_record_exit = 0.;
#define LONGITUDINAL_MAGNET_STEP(ll, code)                                                \
    {                                                                                     \
        double const old_x = LocalParticle_get_x(part);                                   \
        double const old_y = LocalParticle_get_y(part);                                   \
        double const old_zeta = LocalParticle_get_zeta(part);                             \
        double const old_kin_px = LocalParticle_get_px(part) - LocalParticle_get_ax(part); \
        double const old_kin_py = LocalParticle_get_py(part) - LocalParticle_get_ay(part); \
        code;                                                                             \
        if (length > 0)                                                                   \
            magnet_body_spin_radiation(                                                   \
                part, body->field, length, body->inv_factorial_order,                     \
                body->factor_knl_ksl, body->drift.ks, body->dks_ds,                       \
                body->drift.x0_solenoid, body->drift.y0_solenoid, body->h_radiation,      \
                body->radiation_flag, body->radiation_record, (ll), old_x, old_y,         \
                old_zeta, old_kin_px, old_kin_py, &dp_record_exit, &dpx_record_exit,      \
                &dpy_record_exit                                                          \
            );                                                                            \
    }
#endif

    START_PER_PARTICLE_BLOCK(part0, part)
        ;
        if (first_active) MagnetFace_enter_single_particle(part, &plan.first_face);
        if (body->drift_only) {
            LONGITUDINAL_MAGNET_STEP(body->length,
                                     LONGITUDINAL_MAGNET_DRIFT(part, body->length));
        }
        else
            RUN_INTEGRATOR(
                body->integrator,
                body->length,
                body->num_kicks,
                part,
                LONGITUDINAL_MAGNET_DRIFT,
                LONGITUDINAL_MAGNET_KICK,
                LONGITUDINAL_MAGNET_STEP
            );
        if (second_active) MagnetFace_leave_single_particle(part, &plan.second_face);
    END_PER_PARTICLE_BLOCK;

#undef LONGITUDINAL_MAGNET_DRIFT
#undef LONGITUDINAL_MAGNET_KICK
#undef LONGITUDINAL_MAGNET_STEP
}

// Cavity and its thick and thin slices. In a thin layout the physical models
// are transported on the source-bearing drift slices: the thin slices are
// identity partition markers there, not RF kicks.
GPUFUN
void track_cavity_plan(LocalParticle* part0, const RFPlan* plan) {
    const MagnetTransform* transform = &plan->transform;
    if (transform->active && !magnet_transform(part0, transform, 0)) return;
    const CavityPlan* cavity = &plan->cavity;
    // The taper pass (tapering.py) only shares the energy loss among the
    // cavities at zero frequency: every model uses the longitudinal map there.
    const int taper = LocalParticle_check_track_flag(part0, XS_FLAG_SR_TAPER);
    // Only a MagnetOverlappedCavity carries static fields.
    if (!taper && cavity->model < 10 && cavity->fields.knl) {
        CavityModels_track_longitudinal_magnet(
            part0,
            cavity,
            &plan->body,
            fabs(plan->body.factor_knl_ksl),
            cavity->thin_slice ? CAVITY_MAGNET_KICK : CAVITY_MAGNET_BODY
        );
    }
    else if (taper || cavity->model < 10) {
        track_rf_body_particles(part0, &plan->body);
    }
    else {
        // Reference-only paraxial step generators, reused across the call.
        CavityMagnusCache magnus_cache;
        CavityMagnusCache_init(&magnus_cache);
        START_PER_PARTICLE_BLOCK(part0, part)
            ;
            const int model = cavity->model;
            if (model < 10)
                track_rf_body_single_particle(part, &plan->body);
            else if (!cavity->thin_slice)
                track_cavity_single_particle(cavity, model, part, &magnus_cache);
        END_PER_PARTICLE_BLOCK;
        CavityMagnusCache_free(&magnus_cache);
    }
    if (transform->active) magnet_transform(part0, transform, 1);
}

#endif
