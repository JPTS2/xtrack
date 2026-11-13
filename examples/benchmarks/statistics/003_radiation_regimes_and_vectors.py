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
P0C         = 10e9                  # Electron reference p0*c [eV]
BEND_LENGTH = 1.0                   # [m]
BEND_ANGLES = [0.005, 0.05]         # Low/high mean emission occurrence
N_KICKS     = 8
N_PARTICLES = 8192
N_BINS      = 70

################################################################################
# Compare native stochastic models in two bending regimes
################################################################################
fig, axes = plt.subplots(2, 2, figsize = (11, 8), layout = 'constrained')
for row, angle in enumerate(BEND_ANGLES):
    line = xt.Line(elements = {'bend': xt.Bend(
        length = BEND_LENGTH, angle = angle, model = 'bend-kick-bend',
        num_multipole_kicks = N_KICKS)})
    line.set_particle_ref('electron', p0c = P0C)
    line.configure_radiation(model = 'mean')
    probe       = line.build_particles()
    energy_in   = probe.energy.copy()
    line.track(probe)
    mean_loss = float((energy_in - probe.energy)[0])
    print('Bend angle [rad], deterministic loss [keV]:', angle, mean_loss / 1e3)
    for model in ['quantum', 'quantum-kick']:
        line.configure_radiation(model = model)
        particles   = line.build_particles(x = np.zeros(N_PARTICLES))
        energy_in   = particles.energy.copy()
        line.track(particles)
        loss = np.asarray(energy_in - particles.energy)
        print(model, 'zero-loss fraction / mean loss +- SE [keV]:',
              np.mean(loss == 0), np.mean(loss) / 1e3,
              np.std(loss, ddof = 1) / np.sqrt(N_PARTICLES) / 1e3)
        print('  std(x) [m], std(px), corr(x,delta):',
              np.std(particles.x), np.std(particles.px),
              np.corrcoef(particles.x, particles.delta)[0, 1])
        print('  Max |py| (planar control), surviving fraction:',
              np.max(np.abs(particles.py)), np.mean(particles.state > 0))
        axes[row, 0].hist(loss / mean_loss, bins = N_BINS,
                          histtype = 'step', density = True, label = model)
        axes[row, 1].scatter(particles.x * 1e6, particles.px * 1e6,
                             s = 2, alpha = 0.15, label = model)
    axes[row, 0].set(title = f'Angle {angle:g} rad',
                     xlabel = 'Loss / deterministic mean', ylabel = 'Density')
    axes[row, 1].set(xlabel = 'Exit x [µm]', ylabel = 'Exit px [10^-6]')
for axis in axes.ravel():
    axis.grid()
    axis.legend()
# The stronger bend raises emission occurrence as well as photon-energy scale.
# Fresh native streams are independent; these finite samples do not certify
# distribution equality, rare-tail accuracy or a stored-ring equilibrium.
# Transverse exit correlations probe information absent from scalar-loss means.
plt.show()
