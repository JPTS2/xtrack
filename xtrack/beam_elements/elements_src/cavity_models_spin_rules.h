// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_SPIN_RULES_H
#define XTRACK_CAVITY_MODELS_SPIN_RULES_H

// Thomas-BMT rotations of the compact cavity factors and of the static
// magnetic fields, shared by the phase-averaged, paraxial and frozen models. Momenta and energies are in units
// of m c and m c^2. CPU scalar only; spin is NULL when not tracked.
#include <math.h>

typedef struct {
    double* spin;
    double anomaly;
} CavitySpin;

GPUFUN
void CavitySpin_rotate(double spin[3], double ax, double ay, double az) {
    const double angle = sqrt(ax * ax + ay * ay + az * az);
    if (angle == 0.) return;
    const double n[3] = {ax / angle, ay / angle, az / angle};
    const double c = cos(angle), s = sin(angle);
    const double dot = n[0] * spin[0] + n[1] * spin[1] + n[2] * spin[2];
    const double cross[3] = {
        n[1] * spin[2] - n[2] * spin[1],
        n[2] * spin[0] - n[0] * spin[2],
        n[0] * spin[1] - n[1] * spin[0]
    };
    for (int i = 0; i < 3; ++i)
        spin[i] = spin[i] * c + cross[i] * s + n[i] * dot * (1. - c);
}

// Rotation with a transverse kick (dpx, dpy) at energy E; kind 0 is an edge
// (radial electric field), kind 1 the ponderomotive kick, kind 2 a static
// magnetic field across the motion.
GPUFUN
void CavitySpin_kick(
    const CavitySpin* spin,
    double dpx,
    double dpy,
    double energy,
    int kind
) {
    if (!spin || !spin->spin) return;
    const double a = spin->anomaly, gamma = energy,
                 beta = sqrt(1. - 1. / (gamma * gamma));
    const double p = beta * gamma;
    double factor;
    if (kind == 2) {
        factor = 1. + a * gamma;
    }
    else if (kind == 1) {
        const double radial =
            ((a + 1. / gamma) + beta * (a + 1. / (gamma + 1.))) / (1. + beta);
        factor = beta * gamma * (radial - 2. * (a + 1. / (gamma + 1.))) + 2.;
    }
    else {
        factor = gamma * beta * beta * (a + 1. / (gamma + 1.));
    }
    CavitySpin_rotate(spin->spin, -factor * dpy / p, factor * dpx / p, 0.);
}

// Precession in the co-propagating wave's radial fields over one step,
// strength = (energy gain per length) k sin(theta) length, with the offset
// (x, y) in the same length unit and nu = k_z / k; it vanishes on crest.
GPUFUN
void CavitySpin_carrier(
    const CavitySpin* spin,
    double strength,
    double nu,
    double energy,
    double x,
    double y
) {
    if (!spin || !spin->spin) return;
    const double a = spin->anomaly, beta = sqrt(1. - 1. / (energy * energy));
    const double factor = strength / (2. * beta)
                          * ((a + 1. / energy) - nu * beta * (a + 1. / (energy + 1.)));
    CavitySpin_rotate(spin->spin, -factor * y, factor * x, 0.);
}

// Rotation from the longitudinal field of an energy kick d_energy, from the
// energy before it and the transverse momenta during it, taken at the kick's
// mean energy so that a step and its inverse cancel.
GPUFUN
void CavitySpin_energy(
    const CavitySpin* spin,
    double px,
    double py,
    double energy,
    double d_energy
) {
    if (!spin || !spin->spin) return;
    const double E = energy + .5 * d_energy;
    const double pz = sqrt(E * E - 1. - px * px - py * py);
    const double factor = (spin->anomaly + 1. / (E + 1.)) * d_energy / (pz / E);
    CavitySpin_rotate(spin->spin, factor * py / E, -factor * px / E, 0.);
}

// A uniform solenoid body of strength S = Q B_z / (m c) over a length ell,
// from the kinetic momenta before and after: (1 + a gamma) across the field
// and (1 + a) along it, at the mean velocity.
GPUFUN
void CavitySpin_solenoid(
    const CavitySpin* spin,
    double strength,
    double ell,
    double px0,
    double py0,
    double px1,
    double py1,
    double energy
) {
    if (!spin || !spin->spin) return;
    const double a = spin->anomaly, gamma = energy;
    const double p = sqrt(gamma * gamma - 1.);
    const double vx = .5 * (px0 + px1) / p, vy = .5 * (py0 + py1) / p;
    const double vz = sqrt(fmax(1. - vx * vx - vy * vy, 0.));
    const double angle = strength * ell / (p * vz);
    const double across = 1. + a * gamma, along = 1. + a;
    CavitySpin_rotate(
        spin->spin,
        -angle * (along - across) * vz * vx,
        -angle * (along - across) * vz * vy,
        -angle * (across * (1. - vz * vz) + along * vz * vz)
    );
}

// A solenoid's hard face, the zero-length limit of its fringe: the radial
// field integrates to -sign S r / 2, sign +1 entering and -1 leaving.
GPUFUN
void CavitySpin_solenoid_face(
    const CavitySpin* spin,
    double strength,
    double momentum,
    double x,
    double y,
    double sign
) {
    if (!spin || !spin->spin || strength == 0.) return;
    const double gamma = sqrt(1. + momentum * momentum);
    const double f = sign * (1. + spin->anomaly * gamma) * strength / (2. * momentum);
    CavitySpin_rotate(spin->spin, f * x, f * y, 0.);
}

#endif
