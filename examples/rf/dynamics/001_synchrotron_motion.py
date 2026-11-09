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
from scipy.signal import find_peaks
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Shared ring and stationary RF configuration
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
RF_VOLTAGE      = 1e5                    # [V]; harmonic 200, phase pi [rad]

########################################
# Small longitudinal launches at the ring entrance
########################################
ZETA_START      = np.array([0.01, 0.02])  # [m]; positive means earlier arrival
DELTA_START     = 0.0                    # (p-p0)/p0
N_TURNS         = 512

################################################################################
# Power the native cavity and find the synchronous reference
################################################################################

########################################
# Load a fresh 1 GeV/c electron ring
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
env['voltage_rf'] = RF_VOLTAGE
# Above transition, phase pi is stable in the public abs(q0)*V convention.
# Reference momentum is fixed; radiation and collective forces are off.

########################################
# Calculate the positive small-amplitude synchrotron tune
########################################
tw  = line.twiss6d()
print('Positive linear synchrotron tune:', tw.qs)
print('Linear oscillation period [turns]:', 1 / tw.qs)

################################################################################
# Track small offsets with the sinusoidal RF kick
################################################################################

########################################
# Prepare particles relative to the synchronous entrance coordinates
########################################
particles   = line.build_particles(
    zeta    = tw.zeta[0] + ZETA_START,
    delta   = tw.delta[0] + DELTA_START,
)

########################################
# Record the entrance and compare the observed period with 1/qs
########################################
line.track(particles, num_turns = N_TURNS, turn_by_turn_monitor = True)
record  = line.record_last_track
print('Final states:', particles.state)
print('Largest sampled |delta|:', np.max(np.abs(record.delta)))
peaks, _    = find_peaks(record.zeta[0])
print('First launch peak spacings [integer turns]:',
      np.diff(record.at_turn[0, peaks]))

# Peak positions are resolved only to whole turns, unlike the linear Twiss
# period. The finite record provides only a few observed oscillations.

# These small launches stay inside the qualified near-reference case. They
# establish neither RF-bucket acceptance nor transverse momentum acceptance.

################################################################################
# Inspect oscillation in time and in longitudinal phase space
################################################################################

########################################
# Follow the recorded longitudinal coordinate
########################################
fig, axes = plt.subplots(1, 2, figsize = (11, 4.5))
for slot, offset in enumerate(ZETA_START):
    valid   = record.state[slot] > 0
    turns   = record.at_turn[slot, valid]
    axes[0].plot(turns, 1e3 * record.zeta[slot, valid],
                 label  = f'Initial zeta offset {1e3 * offset:g} mm')
axes[0].set_xlabel('Turn at ring entrance')
axes[0].set_ylabel('zeta [mm]')
axes[0].legend()

########################################
# Inspect the longitudinal phase portrait
########################################
for slot in range(len(ZETA_START)):
    valid   = record.state[slot] > 0
    axes[1].plot(1e3 * record.zeta[slot, valid],
                 1e5 * record.delta[slot, valid])
axes[1].set_xlabel('zeta [mm]')
axes[1].set_ylabel('delta [1e-5]')
axes[1].set_title(f'Linear qs = {tw.qs:.6f}')
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
