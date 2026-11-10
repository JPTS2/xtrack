# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
from scipy.constants import c
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Electron reference and two observation positions
########################################
P0C         = 1e9                       # Reference p0*c [eV]
DISTANCE    = 0.75                      # Second position measured from s=0 [m]
CLOCK_START = 2e-9                      # Shared reference-clock epoch [s]
ZETA        = 0.02                      # Positive means earlier arrival [m]

########################################
# Identical thin-cavity voltage and frequency
########################################
VOLTAGE     = 1e5                       # [V]
FREQUENCY   = 100e6                     # [Hz]
PHASE       = 0.0                       # Local reference-arrival phase [rad]

################################################################################
# Express one chosen RF phase using the two timing conventions
################################################################################

########################################
# Set the local reference phase at s=0 and the chosen clock epoch
########################################
reference   = xt.Particles(pdg_id_0 = 'electron', p0c = P0C)
beta0       = float(reference.beta0[0])
relative    = xt.Cavity(voltage = VOLTAGE, frequency = FREQUENCY, phase = PHASE)

########################################
# Calibrate one laboratory-time origin, shared by both positions
########################################
absolute    = relative.copy()
absolute.absolute_time  = True
absolute.phase          = PHASE - 2 * np.pi * FREQUENCY * CLOCK_START
print('Relative phase / absolute phase at lab t=0 [rad]:',
      relative.phase, absolute.phase)

# With absolute_time=False, phase is relative to the local zeta=0 reference.
# With absolute_time=True, phase refers to lab t=0. Only the first sample is
# used for phase calibration; the downstream cavity keeps this same lab origin.

################################################################################
# Compare location and clock changes with native cavity tracking
################################################################################

########################################
# Choose three independent probes, including two equal arrival times
########################################
samples     = np.array(['Initial', 'Downstream', 'Later upstream'])
positions   = np.array([0.0, DISTANCE, 0.0])
clocks      = np.array([CLOCK_START, CLOCK_START,
                       CLOCK_START + DISTANCE / (beta0 * c)])
arrival     = clocks + (positions - ZETA) / (beta0 * c)
kicks       = []
for position, clock in zip(positions, clocks):
    initial = xt.Particles(
        pdg_id_0    = 'electron',
        p0c         = P0C,
        s           = position,
        time_s      = clock,
        zeta        = ZETA,
    )
    row     = []
    for cavity in [relative, absolute]:
        particles   = initial.copy()
        cavity.track(particles)
        row.append((particles.ptau[0] - initial.ptau[0]) * P0C)
    kicks.append(row)
kicks       = np.array(kicks)

# Arrival time is time_s+(s-zeta)/(beta0*c), not time_s alone. These are
# independent thin-element probes; no transport between samples is implied.

########################################
# Read the clock, position, arrival time and actual energy kicks together
########################################
xt.Table(
    data    = {
        'sample':               samples,
        's_m':                  positions,
        'time_s_ns':            clocks * 1e9,
        'arrival_ns':           arrival * 1e9,
        'relative_kick_keV':    kicks[:, 0] * 1e-3,
        'absolute_kick_keV':    kicks[:, 1] * 1e-3,
    },
    index   = 'sample',
).show(maxwidth = 'full')

# The last two samples arrive at the same lab time and receive the same
# absolute-time kick. The relative convention uses the same zeta in all three
# frames, so its kick is unchanged by their different s and time_s values.
