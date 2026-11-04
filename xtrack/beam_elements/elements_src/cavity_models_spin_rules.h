// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_CAVITY_MODELS_SPIN_RULES_H
#define XTRACK_CAVITY_MODELS_SPIN_RULES_H

// Thomas-BMT rotations of the compact cavity factors, shared
// by the phase-averaged and frozen models. Momenta and energies are in units
// of m c and m c^2. CPU scalar only; spin is NULL when not tracked.
#include <math.h>

typedef struct {
    double *spin;
    double anomaly;
} CavitySpin;

GPUFUN
void CavitySpin_rotate(double spin[3], double ax, double ay, double az) {
    const double angle=sqrt(ax*ax+ay*ay+az*az);
    if (angle==0.) return;
    const double n[3]={ax/angle,ay/angle,az/angle};
    const double c=cos(angle), s=sin(angle);
    const double dot=n[0]*spin[0]+n[1]*spin[1]+n[2]*spin[2];
    const double cross[3]={n[1]*spin[2]-n[2]*spin[1],n[2]*spin[0]-n[0]*spin[2],
                           n[0]*spin[1]-n[1]*spin[0]};
    for (int i=0;i<3;++i) spin[i]=spin[i]*c+cross[i]*s+n[i]*dot*(1.-c);
}

// Rotation with a transverse kick (dpx, dpy) at energy E; kind 0 is an edge
// (radial electric field), kind 1 the ponderomotive kick.
GPUFUN
void CavitySpin_kick(const CavitySpin *spin, double dpx, double dpy,
                     double energy, int kind) {
    if (!spin || !spin->spin) return;
    const double a=spin->anomaly, gamma=energy, beta=sqrt(1.-1./(gamma*gamma));
    const double p=beta*gamma;
    double factor;
    if (kind==1) {
        const double radial=((a+1./gamma)+beta*(a+1./(gamma+1.)))/(1.+beta);
        factor=beta*gamma*(radial-2.*(a+1./(gamma+1.)))+2.;
    } else {
        factor=gamma*beta*beta*(a+1./(gamma+1.));
    }
    CavitySpin_rotate(spin->spin, -factor*dpy/p, factor*dpx/p, 0.);
}

// Precession in the co-propagating wave's radial fields over one step,
// strength = (energy gain per length) k sin(theta) length, with the offset
// (x, y) in the same length unit; it vanishes on crest.
GPUFUN
void CavitySpin_carrier(const CavitySpin *spin, double strength, double energy,
                        double x, double y) {
    if (!spin || !spin->spin) return;
    const double a=spin->anomaly, beta=sqrt(1.-1./(energy*energy));
    const double factor=strength/(2.*beta)*((a+1./energy)-beta*(a+1./(energy+1.)));
    CavitySpin_rotate(spin->spin, -factor*y, factor*x, 0.);
}

// Rotation from the longitudinal field of an energy kick d_energy, taken at
// the state (px, py, E) before the kick.
GPUFUN
void CavitySpin_energy(const CavitySpin *spin, double px, double py,
                       double energy, double d_energy) {
    if (!spin || !spin->spin) return;
    const double pz=sqrt(energy*energy-1.-px*px-py*py);
    const double factor=(spin->anomaly+1./(energy+1.))*d_energy/(pz/energy);
    CavitySpin_rotate(spin->spin, factor*py/energy, -factor*px/energy, 0.);
}

#endif
