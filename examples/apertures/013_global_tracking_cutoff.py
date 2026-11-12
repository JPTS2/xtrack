# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Open drift and identical entrance probes
########################################
LENGTH          = 2.0                   # [m]
N_DRIFTS        = 16
P0C             = 1e9                   # Electron reference p0*c [eV]
X_INITIAL       = np.array([0.0, 0.005, -0.005])  # [m]
PX_INITIAL      = np.array([0.0, 0.006, -0.006])  # p_x/p0

########################################
# Numerical safety threshold
########################################
GLOBAL_LIMIT    = 0.010                 # Maximum absolute x or y [m]

################################################################################
# Track without any physical aperture elements
################################################################################
line    = xt.Line(elements = {
    f'drift.{index}': xt.Drift(length = LENGTH / N_DRIFTS, model = 'exact')
    for index in range(N_DRIFTS)
})
line.set_particle_ref('electron', p0c = P0C)
line.skip_end_turn_actions  = True
context     = xo.ContextCpu()
context.allow_kernel_compilation    = True
line.build_tracker(_context = context)
# Changing the compile-time guard requires a kernel for that configuration.
initial     = line.build_particles(x = X_INITIAL, px = PX_INITIAL)
print('Default global transverse limit [m]:',
      line.config.XTRACK_GLOBAL_XY_LIMIT)
standard    = initial.copy()
line.track(standard)

################################################################################
# Tighten the safety guard while keeping the same drift geometry
################################################################################
line.config.XTRACK_GLOBAL_XY_LIMIT  = GLOBAL_LIMIT
restricted  = initial.copy()
line.track(restricted)
# This is a rectangular numerical guard in reference x/y, not chamber geometry.
# For these static thick drifts it is checked after each drift map.

################################################################################
# Locate the discrete safety loss in particle-ID order
################################################################################

########################################
# Compare the two configurations on identical launches
########################################
for label, particles in [('Default guard', standard),
                         ('Tightened guard', restricted)]:
    particles.sort(interleave_lost_particles = True)
    print(label)
    particles.get_table().cols[
        'particle_id x px state s at_element'].show(maxwidth = 'full')

########################################
# Compare the first guard crossing with the drift-boundary station
########################################
slope   = PX_INITIAL / np.sqrt(1 - PX_INITIAL**2)
moving  = slope != 0
crossing_s  = (GLOBAL_LIMIT - np.abs(X_INITIAL[moving])) / np.abs(slope[moving])
print('Continuous |x| threshold crossings for the moving probes [m]:',
      crossing_s)
print('Drift-boundary spacing [m]:', LENGTH / N_DRIFTS)
print('Global guard loss code:', restricted.state[1:])
# The loss station is a drift boundary beyond the crossing, not a refined wall
# impact. Explicit Limit elements instead mark geometric aperture losses as 0.
