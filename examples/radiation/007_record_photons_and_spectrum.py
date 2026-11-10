# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, elementary_charge, hbar
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Electron probes and a thin radiating dipole kick
########################################
P0C                 = 10e9              # Reference momentum [eV/c]
BEND_LENGTH         = 1.0               # [m]
BEND_ANGLE          = 0.05              # [rad]

########################################
# Photon acquisition
########################################
N_PARTICLES         = 4096
RECORD_CAPACITY     = 100000
N_BINS              = 60
SPECTRUM_RANGE      = (1e-6, 10.0)       # Photon energy / critical energy

################################################################################
# Record individual photons from the native quantum-radiation model
################################################################################

########################################
# Give the thin kick a physical bending length for its radiation calculation
########################################
line    = xt.Line(elements = {
    'bend': xt.Multipole(
        length  = BEND_LENGTH,
        knl     = [BEND_ANGLE],
        hxl     = BEND_ANGLE)})
line.set_particle_ref('electron', p0c = P0C)
line.configure_radiation(model = 'quantum')
particles   = line.build_particles(x = np.zeros(N_PARTICLES))
initial     = particles.copy()
# This is one thin radiating kick, not finite-length bend transport.

########################################
# Allocate a finite photon buffer and stop acquisition after one passage
########################################
record  = line.start_internal_logging_for_elements_of_type(
    xt.Multipole, capacity = RECORD_CAPACITY)
line.track(particles, num_turns = 1)
line.stop_internal_logging_for_elements_of_type(xt.Multipole)

# Native tracking initializes particle RNG streams; realizations vary.
# quantum-kick has no individual photons to record: use quantum here.

################################################################################
# Inspect recorded events and account for the emitted energy
################################################################################

########################################
# Read only occupied slots and detect a possibly truncated acquisition
########################################
n_recorded      = int(record._index.num_recorded)
if n_recorded >= RECORD_CAPACITY:
    raise RuntimeError('Photon buffer filled; increase RECORD_CAPACITY.')
photon_energy   = np.asarray(record.photon_energy[:n_recorded])
photon_ids      = np.asarray(record.particle_id[:n_recorded])
print('Recorded photons / capacity:', n_recorded, RECORD_CAPACITY)
n_preview       = min(5, n_recorded)
xt.Table(
    data    = {
        'slot':         np.arange(n_preview),
        'particle_id':  photon_ids[:n_preview],
        'element':      record.at_element[:n_preview],
        'passage':      record.at_turn[:n_preview],
        'energy_keV':   photon_energy[:n_preview] / 1e3},
    index   = 'slot').show(maxwidth = 'full')

########################################
# Compare particle energy loss with the sum of its recorded photons
########################################
recorded_loss   = np.bincount(photon_ids, weights = photon_energy,
                              minlength = N_PARTICLES)
particle_loss   = initial.energy - particles.energy
balance_error   = particle_loss - recorded_loss[particles.particle_id]
print('Total particle loss / recorded photon energy [MeV]:',
      np.sum(particle_loss) / 1e6, np.sum(photon_energy) / 1e6)
print('Maximum per-particle energy-balance residual [eV]:',
      np.max(np.abs(balance_error)))

# A tiny residual remains from the native ultrarelativistic recoil approximation
# and floating-point arithmetic. Finite-mass energy accounting is approximate.

################################################################################
# Plot the photon spectrum and mark its characteristic energy scale
################################################################################

########################################
# Use the entrance ultrarelativistic critical energy for this uniform bend
########################################
rho             = BEND_LENGTH / abs(BEND_ANGLE)
gamma0          = initial.gamma0[0]
critical_energy = 1.5 * hbar * c / elementary_charge * gamma0**3 / rho
print('Entrance critical photon energy [keV]:', critical_energy / 1e3)
# Radiation slightly changes the energy during the kick. This entrance scale
# is an ultrarelativistic reference, not a fitted spectrum or an exact cutoff.

########################################
# Count photons in logarithmic energy bins
########################################
bins        = np.geomspace(*SPECTRUM_RANGE, N_BINS + 1)
scaled      = photon_energy / critical_energy
outside     = (scaled < bins[0]) | (scaled > bins[-1])
print('Photons outside the plotted energy window:', np.count_nonzero(outside))
# The energy balance above includes every recorded photon, including soft
# photons outside this display window. Counts are per bin, not dN/dE.
fig, ax     = plt.subplots(figsize = (8, 4))
ax.hist(
    scaled,
    bins        = bins,
    histtype    = 'step',
    label       = 'Recorded photons')
ax.axvline(
    1.0,
    color       = 'k',
    linestyle   = '--',
    label       = 'Entrance critical energy')
ax.set_xscale('log')
ax.set_xlabel('Photon energy / entrance critical energy')
ax.set_ylabel('Photons per logarithmic energy bin')
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()
