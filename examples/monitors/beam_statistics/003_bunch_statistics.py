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
# Slot grid and selected physical slot order
########################################
P0C             = 1e9                  # Electron reference p0*c [eV]
SPACING         = 10.0                 # Zeta distance between slots [m]
FILLING         = [1, 0, 1, 1]
SELECTED        = [3, 0, 2]

########################################
# Explicit weighted population
########################################
SLOTS           = np.array([0, 0, 1, 3, 3])
X               = np.array([-1, 1, 99, 2, 4]) * 1e-3  # [m]
WEIGHTS         = np.array([1, 3, 100, 2, 4])
LOCAL_ZETA      = np.array([-0.2, 0.2, 0.0, -0.1, 0.1])  # [m]

################################################################################
# Acquire one prepared bunch-pattern snapshot without transporting particles
################################################################################
particles   = xt.Particles(
    mass0   = xt.ELECTRON_MASS_EV,
    q0      = -1,
    p0c     = P0C,
    x       = X,
    zeta    = LOCAL_ZETA - SLOTS * SPACING,
    weight  = WEIGHTS,
)
monitor = xt.BeamStatsMonitor(
    start_at_turn       = 0,
    stop_at_turn        = 1,
    filling_scheme      = FILLING,
    selected_slots      = SELECTED,
    bunch_spacing_zeta  = SPACING,
    stats               = ['num_particles', 'mean_x', 'sigma_x'],
)
monitor.track(particles)

# Slot j is centered at zeta=-j*SPACING. This grid does not define RF buckets.
# Slot 1 is excluded by the pattern. Slot 2 is selected but has no particles.

################################################################################
# Inspect the slot axis and pooled population
################################################################################

########################################
# Read by physical slot number rather than array index
########################################
print('Levels/default:', monitor.available_levels, monitor.default_level)
print('Selected slots and array shape:', monitor.selected_slots,
      monitor.mean_x.shape)
print('slot; represented population; weighted mean x [m]; rms x [m]')
for slot in monitor.selected_slots:
    print(slot, monitor.get('num_particles', turn = 0, slot = slot),
          monitor.get('mean_x', turn = 0, slot = slot),
          monitor.get('sigma_x', turn = 0, slot = slot))
# Empty-bin means and widths are returned as zero; zero is not a measurement.

########################################
# Pool primitive weighted moments over accepted slots
########################################
accepted    = np.isin(SLOTS, SELECTED)
mean_x      = np.average(X[accepted], weights = WEIGHTS[accepted])
sigma_x     = np.sqrt(np.average(
    (X[accepted] - mean_x)**2, weights = WEIGHTS[accepted]))
print('Accepted-beam mean: monitor / NumPy [m]:',
      monitor.get('mean_x', level = 'beam', turn = 0), mean_x)
print('Accepted-beam rms: monitor / NumPy [m]:',
      monitor.get('sigma_x', level = 'beam', turn = 0), sigma_x)

# The pooled width includes differences between bunch centroids. It is not an
# average of bunch widths. Beam reductions exclude unselected/unfilled slots.

################################################################################
# Plot populated slots and display their represented populations
################################################################################
counts  = monitor.num_particles[0]
valid   = counts > 0
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
axes[0].plot(monitor.selected_slots[valid], 1e3 * monitor.mean_x[0, valid], 'o')
axes[1].bar(monitor.selected_slots, counts)
axes[0].set_ylabel('Weighted mean x [mm]')
axes[1].set_ylabel('Sum of weights')
axes[1].set_xlabel('Physical slot')
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
