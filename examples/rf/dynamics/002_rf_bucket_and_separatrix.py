# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
import xpart as xp
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Ideal ring with a stationary single-harmonic RF system
########################################
P0C             = 1e9               # Electron reference p0*c [eV]
CIRCUMFERENCE   = 100.0             # [m]
ALPHA_C         = 0.02              # Constant momentum compaction
HARMONIC        = 20
RF_VOLTAGE      = 1e5               # [V]
RF_PHASE        = np.pi            # Above-transition stable phase [rad]

########################################
# Launches relative to the stationary bucket's momentum height
########################################
HEIGHT_FRACTIONS    = np.array([-0.8, 0.8, -1.2, 1.2])
N_TURNS             = 1024

################################################################################
# Define a native nonlinear RF map with independent transverse transport
################################################################################

########################################
# Specify the reference and its RF frequency
########################################
reference   = xt.Particles(pdg_id_0 = 'electron', p0c = P0C)
frequency   = HARMONIC * reference.beta0[0] * c / CIRCUMFERENCE

########################################
# Use sinusoidal longitudinal motion without transverse chromatic limitations
########################################
line    = xt.Line(
    elements        = [
        xt.LineSegmentMap(
            length                      = CIRCUMFERENCE,
            qx                          = 0.31,
            qy                          = 0.32,
            betx                        = 10.0,
            bety                        = 12.0,
            longitudinal_mode           = 'nonlinear',
            momentum_compaction_factor  = ALPHA_C,
            voltage_rf                  = RF_VOLTAGE,
            frequency_rf                = frequency,
            phase_rf                    = RF_PHASE,
        ),
    ],
    element_names   = ['turn_map'],
    particle_ref    = reference,
)

# This explicit ideal map isolates longitudinal dynamics. It does not inherit
# the shared demonstration ring's chromaticity or transverse acceptance.
# Reference energy is constant; acceleration, radiation and collective forces
# are absent. The single RF harmonic gives a stationary symmetric bucket.

################################################################################
# Obtain the native RF-bucket object and its separatrix
################################################################################

########################################
# Use Xpart's public bucket entry point
########################################
tw      = line.twiss6d()
bucket  = xp.longitudinal.get_bucket(line)
zeta    = np.linspace(bucket.z_left, bucket.z_right, 401)
upper   = bucket.separatrix(zeta)
lower   = bucket.separatrix(zeta, sgn = -1)
print('Selected bucket zeta interval [m]:', bucket.z_left, bucket.z_right)
print('Stable point zeta [m], delta:', bucket.z_sfp, bucket.dp0)

########################################
# Launch inside and outside the separatrix, avoiding the boundary itself
########################################
height      = bucket.separatrix(tw.zeta[0]) - bucket.dp0
zeta_start  = np.full(len(HEIGHT_FRACTIONS), tw.zeta[0])
delta_start = bucket.dp0 + HEIGHT_FRACTIONS * height
particles   = line.build_particles(zeta = zeta_start, delta = delta_start)
print('Initially inside the selected bucket:',
      bucket.is_in_separatrix(particles.zeta, particles.delta))

# delta=(p-p0)/p0; positive zeta means earlier arrival at a fixed position.
# The separatrix is centered on dp0. The bucket calculation retains leading-
# order momentum compaction; it is not a 6D acceptance model.

################################################################################
# Compare librating trajectories with circulating longitudinal motion
################################################################################

########################################
# Record the native map and retain physical particle states
########################################
line.track(particles, num_turns = N_TURNS, turn_by_turn_monitor = True)
record  = line.record_last_track
print('Final states:', particles.state)
print('Sampled zeta ranges [m]:', np.ptp(record.zeta, axis = 1))

# Outside-bucket particles slip in RF phase but remain live: this boundary is
# not a material aperture or a loss condition. The tracked map has discrete
# kicks; its trajectories need not exactly follow a continuous separatrix.

########################################
# Draw the separatrix and show unwrapped longitudinal slippage
########################################
fig, axes = plt.subplots(1, 2, figsize = (11, 4.5))
axes[0].plot(zeta, upper, 'k--', label = 'Native bucket separatrix')
axes[0].plot(zeta, lower, 'k--')
for slot, fraction in enumerate(HEIGHT_FRACTIONS):
    valid   = record.state[slot] > 0
    axes[0].plot(record.zeta[slot, valid], record.delta[slot, valid])
    axes[1].plot(record.at_turn[slot, valid], record.zeta[slot, valid],
                 label  = f'Initial height fraction {fraction:g}')
axes[0].set_xlim(bucket.z_left, bucket.z_right)
axes[0].set_xlabel('zeta [m]')
axes[0].set_ylabel('delta')
axes[0].legend()
axes[1].set_xlabel('Turn')
axes[1].set_ylabel('Unwrapped zeta [m]')
axes[1].legend()
for axis in axes:
    axis.grid()
fig.tight_layout()

plt.show()
