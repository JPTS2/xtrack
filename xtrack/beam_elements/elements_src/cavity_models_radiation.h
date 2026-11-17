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
    double B[3];
    CavityEffects_static_field(effects, x, y, B);
    const double gamma = *energy, p = sqrt(gamma * gamma - 1.);
    const double pz2 = p * p - *px * *px - *py * *py;
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
    return CAVITY_MODEL_OK;
}

// The same after a forward step of `length` metres of the reduced models' state,
// with the field at the step's mean position: positions times `k` (k = 1 for
// metres), canonical momenta in the solenoid's symmetric gauge, S = Q B_z per
// unit m c.
GPUFUN
int CavityEffects_radiate_state(
    const CavityEffects* effects,
    double q[6],
    const double start[2],
    double k,
    double solenoid,
    double length
) {
    if (!effects || effects->radiation_flag <= 0) return CAVITY_MODEL_OK;
    const double x = q[0] / k, y = q[1] / k;
    double px = q[3] + .5 * solenoid * y, py = q[4] - .5 * solenoid * x, energy = q[5];
    const double pz2 = energy * energy - 1. - px * px - py * py;
    if (!(pz2 > 0.)) return CAVITY_MODEL_DOMAIN;
    const int status = CavityEffects_radiate(
        effects, .5 * (x + start[0] / k), .5 * (y + start[1] / k),
        length * sqrt(energy * energy - 1.) / sqrt(pz2), &px, &py, &energy
    );
    q[3] = px - .5 * solenoid * y;
    q[4] = py + .5 * solenoid * x;
    q[5] = energy;
    return status;
}

#else
#define CavityEffects_radiate_state(effects, q, start, k, solenoid, length) CAVITY_MODEL_OK
#endif

#endif
