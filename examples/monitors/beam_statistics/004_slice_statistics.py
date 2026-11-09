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
# Two selected bunches and local half-open slice window
########################################
P0C             = 1e9                  # Electron reference p0*c [eV]
SPACING         = 10.0                 # [m], not an RF-bucket definition
SLOTS           = [0, 2]
ZETA_RANGE      = (-1.0, 1.0)           # Local bunch coordinate [m]
N_SLICES        = 8

########################################
# Prepared sample and transverse variation along each bunch
########################################
N_PER_BUNCH     = 96
SEED            = 123
AMPLITUDE       = 1e-3                 # Horizontal pattern amplitude [m]

################################################################################
# Prepare and acquire one snapshot, without an RF-equilibrium assumption
################################################################################

########################################
# Construct absolute zeta from the slot and its local coordinate
########################################
rng         = np.random.default_rng(SEED)
slot        = np.repeat(SLOTS, N_PER_BUNCH)
zeta_local  = rng.uniform(-1.2, 1.2, len(slot))
weights     = rng.uniform(1, 3, len(slot))
x           = AMPLITUDE * np.sin(np.pi * zeta_local)
particles   = xt.Particles(
    mass0   = xt.ELECTRON_MASS_EV,
    q0      = -1,
    p0c     = P0C,
    x       = x,
    zeta    = zeta_local - slot * SPACING,
    weight  = weights,
)

########################################
# Store slice moments on a specified longitudinal window
########################################
monitor = xt.BeamStatsMonitor(
    start_at_turn       = 0,
    stop_at_turn        = 1,
    filled_slots        = SLOTS,
    bunch_spacing_zeta  = SPACING,
    zeta_range          = ZETA_RANGE,
    num_slices          = N_SLICES,
    stats               = ['num_particles', 'mean_x', 'sigma_x', 'mean_zeta'],
)
monitor.track(particles)
print('Levels/default/shape:', monitor.available_levels,
      monitor.default_level, monitor.mean_x.shape)
print('Slice-center shape:', monitor.zeta_centers.shape)

################################################################################
# Compare recorded-window reductions with the accepted population
################################################################################

########################################
# Select one physical slot and one slice through the public API
########################################
print('Slot 2 slice centroids [m]:', monitor.get('mean_x', turn = 0, slot = 2))
print('First slice of slot 2:',
      monitor.get('mean_x', turn = 0, slot = 2, slice_index = 0))

########################################
# Exclude out-of-window particles from bunch and beam reductions
########################################
accepted    = (zeta_local >= ZETA_RANGE[0]) & (zeta_local < ZETA_RANGE[1])
print('All / accepted weight:', np.sum(weights), np.sum(weights[accepted]))
print('Recorded accepted weight:',
      monitor.get('num_particles', level = 'beam', turn = 0))
print('Accepted weighted mean x: monitor / NumPy [m]:',
      monitor.get('mean_x', level = 'beam', turn = 0),
      np.average(x[accepted], weights = weights[accepted]))

# Reductions pool the recorded primitive moments, not all particles originally
# assigned to each bunch. The range is half-open at its upper edge.

################################################################################
# Plot populated slices on the absolute-zeta grid
################################################################################
fig, ax = plt.subplots(figsize = (8, 4))
for index, physical_slot in enumerate(monitor.selected_slots):
    counts  = monitor.num_particles[0, index]
    valid   = counts > 0
    ax.plot(monitor.zeta_centers[index, valid],
            1e3 * monitor.mean_x[0, index, valid], 'o-',
            label   = f'slot {physical_slot}')
ax.set_xlabel('Absolute zeta [m]')
ax.set_ylabel('Weighted slice mean x [mm]')
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()
