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
RF_VOLTAGE      = 1e5                    # Later RF-on comparison [V]

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

################################################################################
# Choose persistent or temporary freezing
################################################################################

########################################
# Freeze the line until explicitly unfrozen
########################################
ordinary_optics = line.twiss4d()
persistent      = initial.copy()
line.freeze_longitudinal()
line.track(persistent, num_turns = N_TURNS)
persistent_end  = persistent.copy()
frozen_optics   = line.twiss4d()
line.freeze_longitudinal(False)
line.track(persistent, num_turns = N_RESUME)
print('Explicit freezing: maximum |zeta change| [m]:',
      np.max(np.abs(persistent_end.zeta - initial.zeta)))
print('After explicit unfreezing: zeta changes [m]:',
      persistent.zeta - persistent_end.zeta)
print('Slip factor of ordinary / frozen maps:',
      ordinary_optics.slip_factor, frozen_optics.slip_factor)
# Zero follows from suppressing longitudinal motion, not changing geometry.

########################################
# Restore the previous configuration on leaving a context
########################################
temporary       = initial.copy()
with xt.freeze_longitudinal(line):
    line.track(temporary, num_turns = N_TURNS)
temporary_end   = temporary.copy()
line.track(temporary, num_turns = N_RESUME)
print('Context freezing: maximum |zeta change| [m]:',
      np.max(np.abs(temporary_end.zeta - initial.zeta)))
print('After leaving the context: zeta changes [m]:',
      temporary.zeta - temporary_end.zeta)
# The context restores the preceding line configuration, including if an
# exception occurs. Here that preceding configuration permits zeta evolution.

################################################################################
# Separate longitudinal freezing from the choice of optics method
################################################################################

########################################
# Enable RF and compare 4D and 6D optics
########################################
env['voltage_rf'] = RF_VOLTAGE
twiss_4d = line.twiss4d()
twiss_6d = line.twiss6d()
print('RF-on 4D transverse tunes:', twiss_4d.qx, twiss_4d.qy)
print('RF-on 6D transverse / synchrotron tunes:',
      twiss_6d.qx, twiss_6d.qy, twiss_6d.qs)
# 4D optics conditions on fixed longitudinal coordinates; choosing that
# optics method does not change how subsequent particles are tracked.
# 6D optics includes the RF restoring map and its synchrotron mode.

########################################
# Compare RF tracking with a numerically frozen call
########################################
rf_ordinary = initial.copy()
rf_frozen   = initial.copy()
line.track(rf_ordinary, num_turns = N_TURNS)
line.track(
    rf_frozen,
    num_turns           = N_TURNS,
    freeze_longitudinal = True,
)
for label, particles in [('RF on', rf_ordinary), ('RF on, frozen', rf_frozen)]:
    print(label, 'maximum |zeta change| [m] and |delta change|:',
          np.max(np.abs(particles.zeta - initial.zeta)),
          np.max(np.abs(particles.delta - initial.delta)))
# Freezing now suppresses both RF energy kicks and longitudinal drift.
# A frozen longitudinal map needs 4D optics, not a singular 6D periodic solve.
