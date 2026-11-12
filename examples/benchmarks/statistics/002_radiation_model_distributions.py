# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Same single-bend setting for both native stochastic models
########################################
P0C             = 10e9                  # Reference p0*c [eV]
BEND_LENGTH     = 1.0                   # [m]
BEND_ANGLE      = 0.005                 # [rad]
N_KICKS         = 8

########################################
# Finite distribution and tail resolution
########################################
N_PARTICLES     = 65536
N_GRID          = 160
TAIL_MULTIPLES  = np.array([1, 2, 4, 8, 16, 32])
CONFIDENCE      = 0.95

################################################################################
# Prepare the fixed bend and a deterministic scale for tail thresholds
################################################################################

########################################
# Keep geometry, reference, initial coordinates and integration identical
########################################
line    = xt.Line(elements = {
    'bend': xt.Bend(
        length                  = BEND_LENGTH,
        angle                   = BEND_ANGLE,
        model                   = 'bend-kick-bend',
        num_multipole_kicks     = N_KICKS)})
line.set_particle_ref('electron', p0c = P0C)
line.configure_radiation(model = 'mean')
probe       = line.build_particles(x = 0.0)
energy_in   = probe.energy.copy()
line.track(probe)
mean_loss   = float((energy_in - probe.energy)[0]) / 1e3
thresholds  = TAIL_MULTIPLES * mean_loss
# Tail thresholds are fixed before sampling, rather than fitted quantiles.

########################################
# Sample total energy loss with native automatically initialized RNG streams
########################################
losses  = {}
for model in ['quantum', 'quantum-kick']:
    line.configure_radiation(model = model)
    particles       = line.build_particles(x = np.zeros(N_PARTICLES))
    energy_in       = particles.energy.copy()
    line.track(particles)
    losses[model]   = np.asarray(energy_in - particles.energy) / 1e3
# Fresh native streams give independent realizations; no NumPy seed is claimed.
# quantum resolves individual emissions; quantum-kick samples their total kick.

################################################################################
# Resolve the zero-loss atom and fixed-threshold tails with finite uncertainty
################################################################################
probabilities   = {}
intervals       = {}
normal_z        = NormalDist().inv_cdf((1 + CONFIDENCE) / 2)
for model, loss in losses.items():
    counts      = np.array([np.count_nonzero(loss == 0)]
        + [np.count_nonzero(loss > threshold) for threshold in thresholds])
    frequency   = counts / N_PARTICLES
    denominator = 1 + normal_z**2 / N_PARTICLES
    center      = (frequency + normal_z**2 / (2 * N_PARTICLES)) / denominator
    half_width  = normal_z * np.sqrt(
        frequency * (1 - frequency) / N_PARTICLES
        + normal_z**2 / (4 * N_PARTICLES**2)) / denominator
    probabilities[model]    = frequency
    intervals[model]        = np.array([
        center - half_width, center + half_width])
    print('\nModel:', model)
    print('Mean / rms spread [keV]:', np.mean(loss), np.std(loss, ddof = 1))
    xt.Table(
        data    = {
            'event':        np.array(['Zero loss']
                + [f'Loss > {value:g} * mean' for value in TAIL_MULTIPLES]),
            'count':        counts,
            'probability':  frequency,
            'lower_bound':  center - half_width,
            'upper_bound':  center + half_width},
        index   = 'event').show(maxwidth = 'full')
# Zero loss resolves the no-emission atom to the precision of stored energies;
# this is not an individual photon counter for the total-kick model.
# Wilson intervals are pointwise approximations, not simultaneous tail bounds.
print('\nOne observed event corresponds to probability:', 1 / N_PARTICLES)
print('No observed events does not establish zero tail probability.')

################################################################################
# Compare cumulative distributions and the tails that this sample can resolve
################################################################################

########################################
# Plot complementary CDFs with distribution-free simultaneous bands
########################################
grid        = np.linspace(0, max(np.max(v) for v in losses.values()), N_GRID)
# DKW bands apply to each full CDF, including the atom at zero. Splitting the
# error probability across two samples gives joint coverage at least CONFIDENCE.
epsilon     = np.sqrt(np.log(4 / (1 - CONFIDENCE)) / (2 * N_PARTICLES))
fig, axes   = plt.subplots(1, 2, figsize = (12, 4))
for model, loss in losses.items():
    cdf         = np.searchsorted(np.sort(loss), grid, side = 'right')
    survival    = 1 - cdf / N_PARTICLES
    axes[0].plot(grid, survival, label = model)
    axes[0].fill_between(grid,
        np.maximum(0, survival - epsilon),
        np.minimum(1, survival + epsilon), alpha = 0.18)
axes[0].set_xlabel('Energy loss [keV]')
axes[0].set_ylabel('Probability of loss greater than threshold')
axes[0].set_title(f'CDF bands: joint coverage at least {CONFIDENCE:.0%}')
axes[0].legend()

########################################
# Keep low-count and zero-count tail estimates visible
########################################
for index, (model, frequency) in enumerate(probabilities.items()):
    bounds      = intervals[model][:, 1:]
    positions   = TAIL_MULTIPLES * (1 + 0.04 * (2 * index - 1))
    axes[1].errorbar(positions, frequency[1:],
        yerr    = np.maximum(0, np.array([
            frequency[1:] - bounds[0], bounds[1] - frequency[1:]])),
        fmt     = 'o',
        capsize = 4,
        label   = model)
axes[1].set_xscale('log', base = 2)
axes[1].set_yscale('symlog', linthresh = 1 / N_PARTICLES)
axes[1].set_ylim(0, 1)
axes[1].set_xlabel('Threshold / deterministic mean loss')
axes[1].set_ylabel(f'Tail probability; pointwise {CONFIDENCE:.0%} intervals')
axes[1].legend()
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
