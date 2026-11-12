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
from scipy.integrate import quad
from scipy.special import gamma, kv
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
# Compare the recorded photon spectrum with its analytical reference
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

########################################
# Integrate photon-number probabilities over the actual logarithmic bins
########################################
# For x=E_gamma/E_c, p(x)=3/(5*pi)*integral_x^infinity K_(5/3)(t) dt.
# This angle-integrated number density differs from energy-weighted x*p(x).
# The factor normalizes the full interval [0,infinity), not the display window.
# See Hofmann, The Physics of Synchrotron Radiation, and the Geant4 manual.
probability = []
for low, high in zip(bins[:-1], bins[1:]):
    # Interchange integrals: integrate (t-low)*K(t) up to high, then
    # (high-low)*K(t) above high. Log(t) resolves the soft end; the omitted
    # tail beyond high+100 is exponentially negligible.
    within  = quad(lambda t: (t - low) * kv(5 / 3, t), low, high)[0]
    tail    = quad(lambda u: np.exp(u) * kv(5 / 3, np.exp(u)),
                   np.log(high), np.log(high + 100.0))[0]
    probability.append(3 / (5 * np.pi) * (within + (high - low) * tail))
probability     = np.array(probability)
expected_counts = n_recorded * probability
count_spread    = np.sqrt(n_recorded * probability * (1 - probability))
print('Analytical probability inside / outside the plotted window:',
      np.sum(probability), 1 - np.sum(probability))
print('Observed / expected photons outside the window:',
      np.count_nonzero(outside), n_recorded * (1 - np.sum(probability)))

# The comparison conditions on the total recorded photon count. Pointwise
# binomial count spreads are not independent or a simultaneous confidence band.
fig, ax     = plt.subplots(figsize = (8, 4))
ax.hist(
    scaled,
    bins        = bins,
    histtype    = 'step',
    label       = 'Recorded photons')
ax.stairs(expected_counts, bins, color = 'C1', label = 'Analytical bin counts')
ax.stairs(expected_counts + count_spread, bins,
          baseline = np.maximum(0.0, expected_counts - count_spread),
          fill = True, alpha = 0.2, color = 'C1',
          label = 'Pointwise 1-sigma spread')
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

################################################################################
# Compare the first two photon-energy moments and their sampling precision
################################################################################

########################################
# Use full-support moments, including the fourth for the second-moment error
########################################
# Integrating x^n*p(x) gives the standard K-Bessel integral below:
# <x>=8/(15*sqrt(3)) and <x^2>=11/27. The fourth moment supplies Var(x^2).
orders      = np.array([1, 2, 4])
moments     = (3 / (5 * np.pi) * 2.0**orders / (orders + 1)
           * gamma(orders / 2 + 11 / 6) * gamma(orders / 2 + 1 / 6))

########################################
# Report finite-acquisition differences in units of the expected standard error
########################################
measured    = np.array([np.mean(scaled), np.mean(scaled**2)])
expected    = moments[:2]
std_error   = np.sqrt((moments[[1, 2]] - expected**2) / n_recorded)
xt.Table(
    data    = {
        'moment':           np.array(['<E_gamma/E_c>', '<(E_gamma/E_c)^2>']),
        'measured':         measured,
        'entrance_theory':  expected,
        'expected_SE':      std_error,
        'difference_in_SE': (measured - expected) / std_error},
    index   = 'moment').show(maxwidth = 'full')
print('Largest particle fractional energy loss:',
      np.max(particle_loss / initial.energy))

# These standard errors assume independent photons at the fixed entrance
# spectrum, conditional on the acquired count. Actual recoil changes that
# scale slightly; acquisition uncertainty does not include this model mismatch.
# No independent photon sampler or normalization to the visible window is used.
plt.show()
