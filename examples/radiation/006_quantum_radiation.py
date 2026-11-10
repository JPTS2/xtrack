# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Electron probes and a single sector bend
########################################
P0C             = 10e9                  # Reference momentum [eV/c]
BEND_LENGTH     = 1.0                   # [m]
BEND_ANGLE      = 0.05                  # [rad]
N_KICKS         = 8

########################################
# Finite stochastic sample
########################################
N_PARTICLES     = 4096
N_BINS          = 60

################################################################################
# Track identical probes with three native radiation models
################################################################################

########################################
# Construct the same finite-length bend for every model
########################################
line    = xt.Line(elements = {
    'bend': xt.Bend(
        length                  = BEND_LENGTH,
        angle                   = BEND_ANGLE,
        model                   = 'bend-kick-bend',
        num_multipole_kicks     = N_KICKS)})
line.set_particle_ref('electron', p0c = P0C)
initial = line.build_particles(x = np.zeros(N_PARTICLES))

########################################
# Keep deterministic loss distinct from fluctuations in the total emitted energy
########################################
losses  = {}
for model in ['mean', 'quantum', 'quantum-kick']:
    line.configure_radiation(model = model)
    particles       = initial.copy()
    line.track(particles, num_turns = 1)
    losses[model]   = initial.energy - particles.energy

# Line tracking initializes native particle RNG streams automatically.
# Realizations vary. quantum samples individual photons; quantum-kick samples
# a stochastic total kick and cannot provide individual photon records.

################################################################################
# Compare the finite-sample loss distributions
################################################################################

########################################
# Report means, rms spreads and statistical precision of each sample mean
########################################
models      = np.array(list(losses))
means       = np.array([np.mean(losses[model]) for model in models])
spreads     = np.array([np.std(losses[model]) for model in models])
xt.Table(
    data    = {
        'model':            models,
        'mean_loss_keV':    means / 1e3,
        'rms_loss_keV':     spreads / 1e3,
        'mean_stderr_keV':  spreads / np.sqrt(N_PARTICLES) / 1e3},
    index   = 'model').show(maxwidth = 'full')
print('Final particle states:', np.unique(particles.state))

########################################
# Plot stochastic totals alongside the deterministic mean loss
########################################
bins        = np.linspace(0, max(np.max(loss) for loss in losses.values()),
                          N_BINS + 1)
fig, ax     = plt.subplots(figsize = (8, 4))
for model in ['quantum', 'quantum-kick']:
    ax.hist(
        losses[model] / 1e3,
        bins        = bins / 1e3,
        histtype    = 'step',
        label       = model)
ax.axvline(means[0] / 1e3, color = 'k', linestyle = '--', label = 'Mean model')
ax.set_xlabel('Energy lost in one passage [keV]')
ax.set_ylabel('Particles per energy bin')
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()

# This illustrates fluctuations, not a precision comparison of rare tails or
# ring equilibrium. Optics and matching use mean radiation, not random kicks.
