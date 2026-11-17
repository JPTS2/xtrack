// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_RADIATION_H
#define XTRACK_CAVITY_MODELS_RADIATION_H

// Radiation of the static fields in the physical models, as the magnets radiate
// (synrad_spectrum.h): mean, quantum and quantum-kick, from the field across the
// motion over a forward step, applied as a collinear recoil. Energies are in m c^2
// and momenta in m c, as in the models; fields in tesla.
#include <math.h>
#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/cavity_models.h"
#include "xtrack/beam_elements/elements_src/cavity_models_spin_rules.h"

#ifndef XTRACK_TPSA_TRACK
#include "xtrack/headers/synrad_spectrum.h"

// The static field at (x, y) in metres, from the strengths normalised to the
// reference rigidity, as the magnets evaluate theirs.
GPUFUN
void CavityEffects_static_field(const CavityEffects* effects, double x, double y,
                                double B[3]) {
    const double scale = effects->rigidity / effects->length;
    double re = 1., im = 0., by = 0., bx = 0.;
    for (int64_t n = 0; n <= effects->order; ++n) {
        by += scale * (effects->knl[n] * re - effects->ksl[n] * im);
        bx += scale * (effects->knl[n] * im + effects->ksl[n] * re);
        const double next = (re * x - im * y) / (n + 1.);
        im = (re * y + im * x) / (n + 1.);
        re = next;
    }
    B[0] = bx;
    B[1] = by;
    B[2] = effects->rigidity * effects->ks;
}

// Radiation over a path of `path` metres at (x, y) with kinetic momenta (px, py)
// and energy E, which it updates; a particle that radiates all its energy is lost
// as by the magnets.
GPUFUN
int CavityEffects_radiate(
    const CavityEffects* effects,
    double x,
    double y,
    double path,
    double* px,
    double* py,
    double* energy
) {
    if (!effects || effects->radiation_flag <= 0 || !(path > 0.)) return CAVITY_MODEL_OK;
    LocalParticle* part = effects->part;
    double* first = effects->first_kick;
    if (first && part->ipart != 0) {
        *energy += first[0];
        *px += first[1];
        *py += first[2];
        return CAVITY_MODEL_OK;
    }
    const double energy0 = *energy, px0 = *px, py0 = *py;
    double B[3];
    CavityEffects_static_field(effects, x, y, B);
    const double gamma = *energy, p = sqrt(gamma * gamma - 1.);
    const double pz2 = p * p - *px * *px - *py * *py;
    if (first) first[0] = first[1] = first[2] = 0.;
    if (!(pz2 > 0.)) return CAVITY_MODEL_OK;
    const double n[3] = {*px / p, *py / p, sqrt(pz2) / p};
    const double along = B[0] * n[0] + B[1] * n[1] + B[2] * n[2];
    double across2 = 0.;
    for (int j = 0; j < 3; ++j) across2 += (B[j] - along * n[j]) * (B[j] - along * n[j]);
    const double B_perp = sqrt(across2);
    if (B_perp < 1e-4) return CAVITY_MODEL_OK;  // as the magnets
    const double mass0 = LocalParticle_get_mass0(part);
    const double q0 = LocalParticle_get_q0(part);
    const double curvature = B_perp * C_LIGHT * fabs(q0) / (p * mass0);
    double loss;  // eV
    if (effects->radiation_flag == 1) {
        const double mass0_kg = mass0 * QELEM / (C_LIGHT * C_LIGHT);
        const double r0 = (q0 * QELEM) * (q0 * QELEM)
                          / (4. * PI * EPSILON_0 * mass0_kg * C_LIGHT * C_LIGHT);
        // dE/ds = (2/3) r0 beta^3 gamma^4 kappa^2 m c^2 (Lienard, exact beta)
        loss = 2. / 3. * r0 * p * p * p * gamma * curvature * curvature * path * mass0;
    }
    else {
        const double critical = 1.5 * 1.973269804593025e-07 * gamma * gamma * gamma
                                * curvature;
        const int64_t count = synrad_gen_photon_count(
            part, synrad_average_number_of_photons(mass0, q0, p, B_perp, path)
        );
        if (effects->radiation_flag == 3)
            loss = synrad_gen_total_energy_normalized_quantum_kick(part, count) * critical;
        else {
            loss = 0.;
            for (int64_t i = 0; i < count; ++i)
                loss += synrad_gen_photon_energy_normalized(part) * critical;
        }
    }
    const double after = gamma - loss / mass0;
    if (!(after > 1.)) {
        LocalParticle_set_state(part, XT_LOST_ALL_E_IN_SYNRAD);
        return CAVITY_MODEL_LOST;
    }
    const double scale = sqrt(after * after - 1.) / p;
    *px *= scale;
    *py *= scale;
    *energy = after;
    if (first) {
        first[0] = *energy - energy0;
        first[1] = *px - px0;
        first[2] = *py - py0;
    }
    return CAVITY_MODEL_OK;
}

// Radiation of the RF field: each wave, or space harmonic, radiates its own
// Lienard power, gamma^2 [(E + beta x B)^2 - (beta.E)^2] per unit charge, into
// the weak-dipole line at its seen frequency nu = omega |1 - kappa beta_z|:
// photon numbers (3/2)(1 - 2x + 2x^2) on x = eps / eps_max up to
// eps_max = hbar nu / (1 - beta), each photon's energy raised with the quiver
// strength K = |F| / nu by 1 + 0.6 K^2 / (1 + 0.3 K^2) (the harmonics outweigh
// the fundamental's red shift). Cross terms between waves beat and average
// over a cell. A hard edge has no width and does not radiate.
#define CAVITY_RADIATION_TUKEY_MAX 12
// Tukey-lambda quantiles of the energy of n photons, n = 2 to 12: lambda with the
// excess kurtosis -1.39/n and the scale with the variance n/10 of the maximum.
GPUFUN
double CavityEffects_photons_energy(LocalParticle* part, int64_t count) {
    const double lambda[11] = {0.348807, 0.251225, 0.214949, 0.195949, 0.184248, 0.176314,
                               0.170580, 0.166242, 0.162844, 0.160112, 0.157867};
    const double scale[11] = {0.405959, 0.439723, 0.483773, 0.526997, 0.567994, 0.606718,
                              0.643381, 0.678221, 0.711458, 0.743277, 0.773835};
    if (count <= 0) return 0.;
    const double v = RandomUniform_generate(part);
    if (count == 1) {
        // The line's cumulative distribution is a cubic with an explicit root.
        const double h = v - .5, d = sqrt(.25 * h * h + 1. / 64.);
        return cbrt(.5 * h + d) + cbrt(.5 * h - d) + .5;
    }
    double total;
    if (count <= CAVITY_RADIATION_TUKEY_MAX) {
        const double L = lambda[count - 2];
        const double w = fmin(fmax(v, 1e-300), 1. - 1e-16);
        total = .5 * count + scale[count - 2] * (pow(w, L) - pow(1. - w, L)) / L;
    }
    else {
        const double u = fmax(RandomUniform_generate(part), 1e-300);
        total = .5 * count + sqrt(.1 * count) * sqrt(-2. * log(u)) * cos(2. * PI * v);
    }
    return fmin(fmax(total, 0.), (double)count);
}

// The phase-averaged Lienard term of one wave at (X, Y, T, Z): the fields at T
// and a quarter period later are its quadrature pair.
GPUFUN
int CavityEffects_wave_power(
    const CavityModelSource* wave,
    double X,
    double Y,
    double T,
    double Z,
    const double beta[3],
    int averaged,
    double* power
) {
    *power = 0.;
    for (int quarter = 0; quarter <= averaged; ++quarter) {
        const double q[6] = {X, Y, T + quarter * .5 * PI, 0., 0., 0.};
        CavityModelEffectsField field;
        const int status = CavityModel_effects_fields(q, Z, wave, &field);
        if (status != CAVITY_MODEL_OK) return status;
        double cross[3], force2 = 0., along = 0.;
        CavityModel_effects_cross(beta, field.B, cross);
        for (int j = 0; j < 3; ++j) {
            const double f = field.E[j] + cross[j];
            force2 += f * f;
            along += beta[j] * field.E[j];
        }
        *power += (force2 - along * along) / (1. + averaged);
    }
    return CAVITY_MODEL_OK;
}

// RF radiation over `length` metres at the scaled point (X, Y, T, Z), with the
// kinetic momenta (px, py) and energy E in m c and m c^2, which it updates.
GPUFUN
int CavityEffects_radiate_rf(
    const CavityEffects* effects,
    const CavityModelSource* source,
    double Q,
    double X,
    double Y,
    double T,
    double Z,
    double length,
    double* px,
    double* py,
    double* energy
) {
    if (!effects || effects->radiation_flag <= 0 || !source || !(length > 0.))
        return CAVITY_MODEL_OK;
    LocalParticle* part = effects->part;
    double* first = effects->first_kick;
    if (first && part->ipart != 0) return CAVITY_MODEL_OK;  // in the static kick
    const double gamma = *energy, p = sqrt(gamma * gamma - 1.);
    const double pz2 = p * p - *px * *px - *py * *py;
    if (!(pz2 > 0.)) return CAVITY_MODEL_DOMAIN;
    const double beta[3] = {*px / gamma, *py / gamma, sqrt(pz2) / gamma};
    const double k = xt_float_or_tpsa_const_part(source->k);
    const double mass0 = LocalParticle_get_mass0(part);
    const double q0 = LocalParticle_get_q0(part);
    const double mass0_kg = mass0 * QELEM / (C_LIGHT * C_LIGHT);
    const double r0 = (q0 * QELEM) * (q0 * QELEM)
                      / (4. * PI * EPSILON_0 * mass0_kg * C_LIGHT * C_LIGHT);
    // dE/dz = tau0 omega Q^2 gamma^2 W dT/dz, tau0 omega = (2/3) r0 k, dT/dz = k / beta_z
    const double per_power = 2. / 3. * r0 * k * k * gamma * gamma * length / beta[2];
    const double photon_unit = 1.054571817e-34 * C_LIGHT * k / (mass0_kg * C_LIGHT * C_LIGHT);
    const double one_minus_beta = 1. / (gamma * (gamma + p));
    // Waves: the forward and backward waves, or each space harmonic.
    const int64_t waves = source->rf_profile == 1 ? source->num_harmonics : 2;
    double loss = 0.;
    for (int64_t m = 0; m < waves; ++m) {
        CavityModelSource wave = *source;
        double kappa;
        if (source->rf_profile == 1) {
            wave.table = NULL;
            wave.table_points = 0;
            wave.num_harmonics = 1;
            wave.harmonic_wave_number = source->harmonic_wave_number + m;
            wave.harmonic_re = source->harmonic_re + m;
            wave.harmonic_im = source->harmonic_im + m;
            kappa = source->harmonic_wave_number[m] / k;
        }
        else {
            if (m == 0) wave.backward = 0.;
            else wave.forward = 0.;
            kappa = (m == 0 ? 1. : -1.) * xt_float_or_tpsa_const_part(source->carrier);
            if (xt_float_or_tpsa_is_zero(m == 0 ? wave.forward : wave.backward)) continue;
        }
        const double nu = fabs(1. - kappa * beta[2]);
        // Resolved by the cell when its seen phase advance is under a quarter turn.
        const int averaged = nu * k * length / beta[2] > .5 * PI;
        double W;
        const int status =
            CavityEffects_wave_power(&wave, X, Y, T, Z, beta, averaged, &W);
        if (status != CAVITY_MODEL_OK) return status;
        const double mean = per_power * Q * Q * W;
        if (!(mean > 0.)) continue;
        if (effects->radiation_flag == 1) {
            loss += mean;
            continue;
        }
        const double strength = fabs(Q) * sqrt(W) / fmax(nu, 1e-300);
        const double top = photon_unit * nu / one_minus_beta
                           * (1. + .6 * strength * strength / (1. + .3 * strength * strength));
        const double photons = 2. * mean / top;  // the line's mean photon is top / 2
        if (photons > 32.) {
            // Many small photons: their sum is Gaussian, variance photons * 0.35 top^2.
            const double u = fmax(RandomUniform_generate(part), 1e-300);
            const double z = sqrt(-2. * log(u)) * cos(2. * PI * RandomUniform_generate(part));
            loss += fmax(mean + sqrt(.35 * photons) * top * z, 0.);
            continue;
        }
        const int64_t count = synrad_gen_photon_count(part, photons);
        if (effects->radiation_flag == 3)
            loss += CavityEffects_photons_energy(part, count) * top;
        else
            for (int64_t i = 0; i < count; ++i)
                loss += CavityEffects_photons_energy(part, 1) * top;
    }
    if (loss == 0.) return CAVITY_MODEL_OK;
    const double after = gamma - loss;
    if (!(after > 1.)) {
        LocalParticle_set_state(part, XT_LOST_ALL_E_IN_SYNRAD);
        return CAVITY_MODEL_LOST;
    }
    const double scale = sqrt(after * after - 1.) / p;
    *px *= scale;
    *py *= scale;
    *energy = after;
    if (first) {
        // The first particle's RF loss joins its static kick for the others.
        first[0] += after - gamma;
        first[1] += *px * (1. - 1. / scale);
        first[2] += *py * (1. - 1. / scale);
    }
    return CAVITY_MODEL_OK;
}

// The static and RF radiation after a forward step of `length` metres of the
// models' state, at the step's mean position: positions times `k` (k = 1 for
// metres), canonical momenta in the solenoid's symmetric gauge, S = Q B_z per
// unit m c; the RF field at the scaled time T and position Z (no source: none).
GPUFUN
int CavityEffects_radiate_state(
    const CavityEffects* effects,
    double q[6],
    const double start[2],
    double k,
    double solenoid,
    double length,
    const CavityModelSource* source,
    double Q,
    double T,
    double Z
) {
    if (!effects || effects->radiation_flag <= 0) return CAVITY_MODEL_OK;
    const double x = .5 * (q[0] + start[0]) / k, y = .5 * (q[1] + start[1]) / k;
    double px = q[3] + .5 * solenoid * q[1] / k, py = q[4] - .5 * solenoid * q[0] / k;
    double energy = q[5];
    const double pz2 = energy * energy - 1. - px * px - py * py;
    if (!(pz2 > 0.)) return CAVITY_MODEL_DOMAIN;
    const double path = length * sqrt(energy * energy - 1.) / sqrt(pz2);
    int status = CavityEffects_radiate(effects, x, y, path, &px, &py, &energy);
    if (status == CAVITY_MODEL_OK && source) {
        const double scale = xt_float_or_tpsa_const_part(source->k);
        status = CavityEffects_radiate_rf(
            effects, source, Q, scale * x, scale * y, T, Z, length, &px, &py, &energy
        );
    }
    q[3] = px - .5 * solenoid * q[1] / k;
    q[4] = py + .5 * solenoid * q[0] / k;
    q[5] = energy;
    return status;
}

#else
#define CavityEffects_radiate_state(effects, q, start, k, solenoid, length, source, Q, T, Z) \
    CAVITY_MODEL_OK
#endif

#endif
