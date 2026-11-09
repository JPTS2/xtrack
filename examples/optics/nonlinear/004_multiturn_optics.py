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
# Thin octupole followed by a linear normal-form rotation
########################################
QX              = 0.255
QY              = 0.31
BETA            = 1.0                   # Both rotation-map beta functions [m]
K3L             = -1e5                  # Integrated normal octupole [m^-3]
P0C             = 7e12                  # Proton reference p0*c [eV]
MAP_LENGTH      = 1.0                   # Abstract turn length [m]

########################################
# Periodic orbit and tracking check
########################################
N_TURNS         = 4                     # Required by the period-four seed

################################################################################
# Construct a transparent nonlinear map with a period-four island
################################################################################
# This normal-form toy model has no RF, radiation or realistic ring geometry.
# The octupole kick is Delta px = -K3L*x^3/6 at y=0. A negative octupole
# shifts horizontal motion near the quarter-integer resonance into islands.
line = xt.Line(
    elements        = [
        xt.Multipole(knl = [0, 0, 0, K3L]),
        xt.LineSegmentMap(
            length  = MAP_LENGTH,
            qx      = QX,
            qy      = QY,
            betx    = BETA,
            bety    = BETA,
        ),
    ],
    element_names   = ['octupole', 'rotation'],
)
line.set_particle_ref('proton', p0c = P0C)

################################################################################
# Seed the nonzero orbit and solve its multi-turn periodic optics
################################################################################

########################################
# Derive the period-four seed
########################################
# For this kick-then-rotation map, a symmetric orbit has x=(a,a,-a,-a).
# The recurrence x[n+1]+x[n-1]=2*cos(mu)*x[n]-beta*sin(mu)*K3L*x[n]^3/6
# gives a^2=12*cos(mu)/(beta*sin(mu)*K3L), with px below at the kick entrance.
# These chosen parameters yield a stable island; arbitrary choices need not.
mu          = 2 * np.pi * QX
amplitude   = np.sqrt(12 * np.cos(mu) / (BETA * np.sin(mu) * K3L))
px_seed     = (1 + np.cos(mu)) * amplitude / (BETA * np.sin(mu))

########################################
# Compute and inspect the periodic optics
########################################
core        = line.twiss4d(chrom = False)
island      = line.twiss4d(
    num_turns   = N_TURNS,
    co_guess    = {'x': amplitude, 'px': px_seed},
    chrom       = False,
)

# num_turns searches an N-turn periodic orbit, not an amplitude scan or an open
# Twiss continuation. The seed selects this island instead of the zero core.
turn_starts = island.rows['_turn_.*']
turn_starts.cols['name s x px betx alfx mux'].show(maxwidth = 'full')
print('Multi-turn endpoint minus entrance (x [m], px normalized to p0):',
      island.x[-1] - island.x[0], island.px[-1] - island.px[0])
print('Core one-turn Qx:', core.qx)
print('Island cumulative local phase advance over four turns:', island.mux[-1])

# mux is the phase of linearized motion about the island orbit. It is not
# four times the core tune, nor an FFT tune of a finite-amplitude trajectory.

################################################################################
# Track the same periodic orbit and compare turn-start coordinates
################################################################################
particle = line.build_particles(x = island.x[0], px = island.px[0])
line.track(particle, num_turns = N_TURNS, turn_by_turn_monitor = True)
record = line.record_last_track
print('Maximum tracked-minus-Twiss turn-start residual (x [m], px):',
      np.max(np.abs(record.x[0] - turn_starts.x)),
      np.max(np.abs(record.px[0] - turn_starts.px)))
print('Final tracked state:', particle.state)

################################################################################
# Inspect the period-four orbit in phase space and at successive turns
################################################################################

########################################
# Compare the phase-space points
########################################
fig, axes = plt.subplots(1, 2, figsize = (11, 4))
axes[0].plot(1e3 * record.x[0], 1e3 * record.px[0], 'o', label = 'Tracked')
axes[0].plot(1e3 * turn_starts.x, 1e3 * turn_starts.px, 'x', label = 'Twiss')
axes[0].plot(1e3 * core.x[0], 1e3 * core.px[0], 'k+', label = 'Core orbit')
axes[0].set_xlabel('x [mm]')
axes[0].set_ylabel('px [1e-3]')
axes[0].legend()

########################################
# Show discrete turn-start samples and the return after four turns
########################################
turns   = np.arange(N_TURNS)
axes[1].plot(turns, 1e3 * record.x[0], 'o', label = 'Tracked turn starts')
axes[1].plot(N_TURNS, 1e3 * particle.x[0], 'x',
             label  = 'Return after four turns')
axes[1].set_xticks(np.arange(N_TURNS + 1))
axes[1].set_xlabel('Completed turns at the map entrance')
axes[1].legend()
axes[1].set_ylabel('Periodic orbit x [mm]')
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
