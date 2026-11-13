# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Model and integration resolution
########################################
LATTICE_FILE    = 'test_data/example_lattices/elena_field/lattice.py'
STEPS           = [5, 10, 20]       # RK4 steps per fitted field segment
N_TURNS         = 64

########################################
# Common physical launches for every resolution
########################################
NEMITT          = 1e-9             # Normalized probe emittance [m]
X_NORMALIZED    = np.array([1.0, 2.0, 3.0])
Y_NORMALIZED    = np.array([0.5, 1.0, 1.5])

################################################################################
# Prepare the same launch from the finest-resolution periodic optics
################################################################################

########################################
# Change integration steps without changing the fitted field
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['elena']
field_names = line.get_table().rows.match(element_type = 'BFieldExpansion').name
for name in field_names:
    line[name].num_integration_steps = max(STEPS)
tw_ref  = line.twiss4d()

########################################
# Use the native coupled-optics launch transformation
########################################
initial = line.build_particles(
    method          = '4d',
    particle_on_co  = tw_ref.particle_on_co,
    W_matrix        = tw_ref.W_matrix[0],
    x_norm          = X_NORMALIZED,
    px_norm         = 0.0,
    y_norm          = Y_NORMALIZED,
    py_norm         = 0.0,
    nemitt_x        = NEMITT,
    nemitt_y        = NEMITT,
)
# Each resolution gets these identical physical coordinates. Rebuilding each
# launch from its own Twiss table would mix launch changes with tracking error.

################################################################################
# Follow the probes at three RK4 resolutions
################################################################################

########################################
# Record bounded-turn trajectories and final normalized coordinates
########################################
histories   = {}
endpoints   = {}
for steps in STEPS:
    for name in field_names:
        line[name].num_integration_steps = steps
    particles   = initial.copy()
    line.track(particles, num_turns = N_TURNS, turn_by_turn_monitor = True)
    record              = line.record_last_track
    histories[steps]    = (record.at_turn.copy(), record.x.copy())
    normalized          = tw_ref.get_normalized_coordinates(
        particles, nemitt_x = NEMITT, nemitt_y = NEMITT)
    endpoints[steps]    = np.array([
        normalized.x_norm, normalized.px_norm,
        normalized.y_norm, normalized.py_norm])
    print(f'{steps} steps: surviving probes =',
          np.count_nonzero(particles.state > 0))

########################################
# Compare endpoints against the finest computed result
########################################
errors  = []
for steps in STEPS:
    error   = np.linalg.norm(endpoints[steps] - endpoints[max(STEPS)], axis = 0)
    errors.append(np.max(error))
    print(f'{steps} steps: maximum normalized endpoint difference =',
          np.max(error))
# This compares integration resolutions of one fitted model. RK4 is not exactly
# symplectic; a 64-turn comparison does not establish long-term stability or
# field-fit accuracy. The finest result is a numerical reference, not truth.

################################################################################
# Display motion and the resolution dependence
################################################################################

########################################
# Display the finest-resolution trajectory of the largest launch
########################################
fig, axes = plt.subplots(1, 2, figsize = (11, 4), layout = 'constrained')
turns, positions = histories[max(STEPS)]
axes[0].plot(turns[-1], 1e3 * (positions[-1] - tw_ref.x[0]))
axes[0].set(xlabel = 'Turn', ylabel = 'x - closed orbit [mm]')

########################################
# Exclude the reference point from the logarithmic error plot
########################################
axes[1].loglog(STEPS[:-1], errors[:-1], 'o-')
axes[1].set(
    xlabel  = 'RK4 steps per fitted segment',
    ylabel  = 'Maximum normalized endpoint difference',
)
for ax in axes:
    ax.grid(alpha = 0.25)
plt.show()
