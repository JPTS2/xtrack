# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Model and tracking duration
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
N_TURNS         = 8
N_RESUME        = 2

########################################
# Physical launch coordinates at the entrance
########################################
X_START         = np.array([-1.0e-3, 0.0, 1.0e-3])     # [m]
Y_START         = 0.5e-3                              # [m]
DELTA_START     = np.array([-1.0e-4, 0.5e-4, 1.0e-4])  # (p - p0) / p0
ZETA_START      = np.array([0.0, 0.01, -0.01])          # [m]

################################################################################
# Load the static magnetic ring
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
# The model supplies a 1 GeV/c electron reference, with RF voltage zero and
# radiation off. zeta = s - beta0*c*t; delta is conserved without freezing.
# Freezing is a numerical constraint, not RF confinement. These small delta
# offsets qualify this sample near the bare lattice's integer vertical tune.

################################################################################
# Prepare a CPU context for the frozen tracking kernel
################################################################################
# Enable compilation only for this CPU context: a frozen tracking kernel may
# need just-in-time compilation, which requires a working local C compiler.
context = xo.ContextCpu()
context.allow_kernel_compilation = True
line.build_tracker(_context = context)

################################################################################
# Compare ordinary tracking and a frozen call
################################################################################

########################################
# Prepare identical launches
########################################
initial     = line.build_particles(
    x       = X_START,
    y       = Y_START,
    delta   = DELTA_START,
    zeta    = ZETA_START,
)
ordinary    = initial.copy()
frozen      = initial.copy()

########################################
# Track with and without freezing
########################################
line.track(ordinary, num_turns = N_TURNS)
line.track(
    frozen,
    num_turns           = N_TURNS,
    freeze_longitudinal = True,
)

# Save the frozen-call endpoint before continuing the same particles below.
frozen_end = frozen.copy()

################################################################################
# Inspect the coordinates and resume longitudinal motion
################################################################################

########################################
# Compare longitudinal coordinates
########################################
print('Initial coordinates:')
initial.get_table().cols[
    'particle_id zeta delta at_turn state'].show(maxwidth = 'full')
print('After ordinary tracking:')
ordinary.get_table().cols[
    'particle_id zeta delta at_turn state'].show(maxwidth = 'full')
print('After the frozen call:')
frozen_end.get_table().cols[
    'particle_id zeta delta at_turn state'].show(maxwidth = 'full')
for label, particles in [('Ordinary', ordinary), ('Frozen', frozen_end)]:
    delta_change    = np.max(np.abs(particles.delta - initial.delta))
    zeta_change     = np.max(np.abs(particles.zeta - initial.zeta))
    print(f'{label}: max |delta change| = {delta_change:.3e}; '
          f'max |zeta change| = {zeta_change:.3e} m')

########################################
# Compare transverse endpoints
########################################
# In this static lattice the transverse maps depend on delta, which is the
# same in both cases, and have no dependence on zeta. This is case-specific.
# px and py are momenta normalized to the reference momentum p0.
for coordinate, unit in [('x', 'm'), ('px', '1'), ('y', 'm'), ('py', '1')]:
    difference = np.max(np.abs(
        getattr(ordinary, coordinate) - getattr(frozen_end, coordinate)))
    print(f'Maximum ordinary/frozen difference in {coordinate}: '
          f'{difference:.3e} {unit}')

########################################
# Resume ordinary tracking from the frozen endpoint
########################################
# The flag acts only on its tracking call. The next call resumes zeta evolution
# from the frozen endpoint while retaining the particle state.
line.track(frozen, num_turns = N_RESUME)
print('The previously frozen particles after another ordinary tracking call:')
frozen.get_table().cols[
    'particle_id zeta delta at_turn state'].show(maxwidth = 'full')
print('Resumed zeta changes [m]:', frozen.zeta - frozen_end.zeta)
