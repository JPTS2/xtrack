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
# Prepared population and profile ranges
########################################
P0C             = 1e9                  # Electron reference p0*c [eV]
N_PARTICLES     = 256
SEED            = 76
X_RANGE         = (-1e-3, 1e-3)         # Half-open profile range [m]
DELTA_RANGE     = (-1e-4, 1e-4)         # Relative momentum range
N_BINS          = 24

########################################
# Population scales
########################################
SIGMA_X         = 0.8e-3                # [m]; tails extend beyond X_RANGE
SIGMA_DELTA     = 2e-5                  # Prepared spread, without RF matching

########################################
# Consecutive drift passages
########################################
FIRST_TURN      = 10                   # Absolute acquisition label
N_ACQUISITIONS  = 3
DRIFT_LENGTH    = 0.5                  # Length per passage [m]
PX_HISTORY      = 5e-4                 # Canonical px = Px/p0

################################################################################
# Record weighted profiles and whole-population moments in one acquisition
################################################################################

########################################
# Prepare a weighted sample without lattice transport
########################################
rng         = np.random.default_rng(SEED)
x           = rng.normal(scale = SIGMA_X, size = N_PARTICLES)
delta       = rng.normal(scale = SIGMA_DELTA, size = N_PARTICLES)
weights     = rng.uniform(1, 3, N_PARTICLES)
particles   = xt.Particles(
    mass0   = xt.ELECTRON_MASS_EV,
    q0      = -1,
    p0c     = P0C,
    x       = x,
    delta   = delta,
    weight  = weights,
)

########################################
# Define separate coordinate histograms on the same accepted population
########################################
monitor = xt.BeamStatsMonitor(
    start_at_turn   = 0,
    stop_at_turn    = 1,
    stats           = ['num_particles', 'mean_x', 'sigma_x'],
    profiles        = {
        'x': {'range': X_RANGE, 'num_bins': N_BINS},
        'delta': {'range': DELTA_RANGE, 'num_bins': N_BINS},
    },
)
monitor.track(particles)
print('Profile coordinates/shapes:', monitor.profile_coordinates,
      {name: value.shape for name, value in monitor.profiles.items()})

################################################################################
# Compare weighted bin counts and identify profile truncation
################################################################################

########################################
# Recompute the horizontal histogram from the prepared sample
########################################
edges       = monitor.profile_bin_edges['x']
in_range    = (x >= edges[0]) & (x < edges[-1])
expected    = np.histogram(
    a       = x[in_range],
    bins    = edges,
    weights = weights[in_range],
)[0]
recorded    = monitor.profiles['x'][monitor.record_index(0)]
print('Maximum profile/NumPy bin difference:',
      np.max(np.abs(recorded - expected)))
print('Full population / in-range x profile weight:',
      monitor.num_particles[0], np.sum(recorded))
print('Weighted full-population mean x [m]:', monitor.mean_x[0])

# Profiles count represented population, not probability density. Histogram
# tails outside a coordinate's half-open range do not enter that profile, but
# they still contribute to scalar statistics and other in-range profiles.

########################################
# Normalize explicitly to the recorded in-range population and bin width
########################################
widths  = np.diff(edges)
density = recorded / (np.sum(recorded) * widths)
print('Integral of normalized in-range x density:', np.sum(density * widths))

# This conditional density integrates to one inside the displayed range. It
# neither reconstructs excluded tails nor defines an RF equilibrium.

################################################################################
# Plot the weighted profile and its conditional density
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
axes[0].stairs(recorded, 1e3 * edges)
axes[1].stairs(density / 1e3, 1e3 * edges)
axes[0].set_ylabel('Sum of particle weights')
axes[1].set_ylabel('Conditional density [mm^-1]')
axes[1].set_xlabel('x [mm]')
for axis in axes:
    axis.grid()
fig.tight_layout()

################################################################################
# Acquire consecutive profiles during transport
################################################################################

########################################
# Record three passages through a drift-only line
########################################
history = xt.BeamStatsMonitor(
    start_at_turn   = FIRST_TURN,
    stop_at_turn    = FIRST_TURN + N_ACQUISITIONS,
    stats           = ['num_particles'],
    profiles        = {
        'x': {'range': X_RANGE, 'num_bins': N_BINS},
        'delta': {'range': DELTA_RANGE, 'num_bins': N_BINS},
    },
)
history_line = xt.Line(
    elements        = [history, xt.Drift(length = DRIFT_LENGTH)],
    element_names   = ['profiles', 'drift'],
)
history_line.set_particle_ref('electron', p0c = P0C)
history_particles           = particles.copy()
history_particles.px        = PX_HISTORY
history_particles.at_turn   = FIRST_TURN
history_line.track(history_particles, num_turns = N_ACQUISITIONS)
# Each passage is recorded before its drift. The line's turn counter labels
# repeated transport steps; this is not a periodic ring or an equilibrium beam.

########################################
# Select profile rows using absolute turn labels
########################################
print('Recorded absolute turns:', history.turns)
print('Turn / row / scalar weight / x profile weight / delta profile weight:')
for turn in history.turns:
    row = history.record_index(turn)
    print(turn, row, history.num_particles[row],
          np.sum(history.profiles['x'][row]),
          np.sum(history.profiles['delta'][row]))
print('Maximum delta-profile change from the first acquisition:',
      np.max(np.abs(history.profiles['delta'] - history.profiles['delta'][0])))
# Drift changes x but preserves delta. Fewer particles inside X_RANGE means
# profile truncation, not beam loss; all live particles still enter the moments.

################################################################################
# Compare profile motion with the population inside each range
################################################################################

########################################
# Normalize each horizontal acquisition to its own in-range weight
########################################
history_edges   = history.profile_bin_edges['x']
x_weights       = np.sum(history.profiles['x'], axis = 1)
delta_weights   = np.sum(history.profiles['delta'], axis = 1)
fig, axes = plt.subplots(1, 2, figsize = (11, 4))
for turn in history.turns:
    row         = history.record_index(turn)
    row_density = history.profiles['x'][row] / (
        x_weights[row] * np.diff(history_edges))
    axes[0].stairs(row_density / 1e3, history_edges * 1e3,
                   label = f'Turn {turn}')
axes[0].set_xlabel('x [mm]')
axes[0].set_ylabel('Conditional density [mm^-1]')
axes[0].legend()

########################################
# Show the weight excluded by the horizontal range
########################################
axes[1].plot(history.turns, history.num_particles, 'k-',
              label = 'Whole population')
axes[1].plot(history.turns, delta_weights, 'x', label = 'Inside delta range')
axes[1].plot(history.turns, x_weights, 'o-', label = 'Inside x range')
axes[1].set_xlabel('Absolute turn label')
axes[1].set_xticks(history.turns)
axes[1].set_ylabel('Sum of particle weights')
axes[1].legend()
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
