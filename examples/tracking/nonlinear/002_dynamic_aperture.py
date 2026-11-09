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
# Native ring and a physical nonlinear field
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
K3LF            = 3.0e4                 # Existing oct.f integrated k3 [m^-3]
K3LD            = 0.0                   # Existing oct.d integrated k3 [m^-3]
NEMITT_X        = 1e-6                  # Normalized rms emittance [m]
NEMITT_Y        = 1e-6                  # [m]
DELTA           = 0.0                   # (p-p0)/p0

########################################
# Fixed launch phases, radial/angle grid and finite horizon
########################################
RADII           = np.linspace(0, 12, 25)  # Total normalized amplitude [sigma]
ANGLES          = np.linspace(0.1, np.pi / 2 - 0.1, 7)  # x/y amplitude ratio
N_TURNS         = 256

################################################################################
# Build a launch grid using the unperturbed linear normalization
################################################################################

########################################
# Load existing octupole sites with their fields switched off
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
env['k3lf'] = 0.0
env['k3ld'] = 0.0

# The 1.5 km ring supplies a 1 GeV/c electron reference. RF voltage is zero;
# radiation is off and no physical aperture is added. At each octupole,
# Delta px=-knl[3]*(x^3-3*x*y^2)/6 for the reference species.

########################################
# Specify the SAME phase slice for every amplitude
########################################
radius, angle = np.meshgrid(RADII, ANGLES)
x_norm      = (radius * np.cos(angle)).ravel()
y_norm      = (radius * np.sin(angle)).ravel()

# px_norm=py_norm=0 fixes both betatron launch phases; this samples only one
# transverse phase slice. The polar angle changes the x/y amplitude ratio.
initial = line.build_particles(
    method      = '4d',
    x_norm      = x_norm,
    px_norm     = 0.0,
    y_norm      = y_norm,
    py_norm     = 0.0,
    nemitt_x    = NEMITT_X,
    nemitt_y    = NEMITT_Y,
    delta       = DELTA,
)

################################################################################
# Compare field-off control, nonlinear survival and a doubled horizon
################################################################################

########################################
# Track identical launches with and without the nonlinear field
########################################
control = initial.copy()
line.track(control, num_turns = N_TURNS)
env['k3lf'] = K3LF
env['k3ld'] = K3LD
particles = initial.copy()
line.track(particles, num_turns = N_TURNS)

########################################
# Continue the same nonlinear ensemble for another equal interval
########################################
extended = particles.copy()
line.track(extended, num_turns = N_TURNS)

# A surviving particle is not proven stable indefinitely. The second horizon
# probes delayed losses; even unchanged counts would not establish convergence.

################################################################################
# Restore launch identity and report loss classifications
################################################################################
# Tracking can reorder particles. Every mask is restored by particle_id before
# comparing it with the launch grid. at_turn of lost particles is a loss turn,
# whereas surviving particles carry the completed-turn count.
survival = {}
for label, tracked in [('Field off', control), ('Short', particles),
                       ('Doubled', extended)]:
    order           = np.argsort(tracked.particle_id)
    alive           = tracked.state[order] > 0
    survival[label] = alive
    states, counts  = np.unique(tracked.state, return_counts = True)
    print(label, 'survivors / launched:', np.count_nonzero(alive), len(alive))
    print(label, 'state codes / counts:', states, counts)
    if np.any(~alive):
        loss_turns = tracked.at_turn[tracked.state <= 0]
        print('First / last recorded loss turn:',
              np.min(loss_turns), np.max(loss_turns))
print('Numerical global x/y guard [m]:', line.config.XTRACK_GLOBAL_XY_LIMIT)

# Losses here are numerical escape/invalid-state guard events,
# not chamber impacts.

################################################################################
# Plot this sampled finite-turn survival region, not a universal aperture
################################################################################
fig, axes = plt.subplots(1, 2, sharex = True, sharey = True, figsize = (10, 5))
for axis, label, turns in zip(
    axes, ['Short', 'Doubled'], [N_TURNS, 2 * N_TURNS]
):
    alive           = survival[label]
    axis.scatter(x_norm[alive], y_norm[alive], s = 15, label = 'Survived')
    axis.scatter(
        x_norm[~alive], y_norm[~alive],
        s       = 18,
        marker  = 'x',
        label   = 'Lost',
    )
    axis.set_title(f'{turns} turns; fixed launch phases')
    axis.set_xlabel('Initial x_norm [sigma]')
    axis.grid()
    axis.legend()
axes[0].set_ylabel('Initial y_norm [sigma]')
fig.tight_layout()
plt.show()
