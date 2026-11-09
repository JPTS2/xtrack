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
from scipy.constants import c
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Ring and finite-duration horizontal drive
########################################
LATTICE_FILE        = 'test_data/example_lattices/demo_ring/lattice.py'
KICK_AMPLITUDE      = 1e-7              # Peak Delta px, normalized to p0
PHASE               = 0.0               # Cosine waveform phase [rad]
DRIVE_DETUNINGS     = [0.005, 0.10]      # Fractional drive detuning [turn^-1]

########################################
# Waveform clock and observation window
########################################
SAMPLES_PER_TURN    = 8
START_TURN          = 8
DRIVE_TURNS         = 192
RAMP_TURNS          = 32
N_TURNS             = 256

################################################################################
# Prepare the static ring and a sampled drive clock
################################################################################

########################################
# Calculate the unforced periodic optics
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
base        = env['demo']
reference   = base.twiss4d()
beta0       = float(base.particle_ref.beta0[0])
frev        = beta0 * c / base.get_length()
fs          = SAMPLES_PER_TURN * frev
natural_q   = reference.qx % 1

# The 1500 m demo ring has a 1 GeV/c electron reference; RF voltage and
# radiation are off. The drive addresses Qx modulo an integer: Qx%1~0.755.
# Exciter uses its at_turn/zeta clock, without enable_time_dependent_vars.

########################################
# Define a finite sampled envelope and the particle arrival offset
########################################
sample_turns    = np.arange(DRIVE_TURNS * SAMPLES_PER_TURN) / SAMPLES_PER_TURN
envelope        = np.minimum(
    np.clip(sample_turns / RAMP_TURNS, 0, 1),
    np.clip((DRIVE_TURNS - sample_turns) / RAMP_TURNS, 0, 1),
)

# Half a sample of arrival delay keeps this coasting particle away from sample
# edges. Here zeta=-C/(2*SAMPLES_PER_TURN)=-93.75 m is a clock offset,
# not a bunch length. It is allowed for this coasting particle with RF off.
zeta_start = -beta0 * c / (2 * fs)

################################################################################
# Compare a near-resonant drive with a more detuned drive
################################################################################
# For the reference species chi=1: Delta px=-knl[0]*sample.
# Sample clock: i=int(fs*((at_turn-start_turn)/frev-zeta/(beta0*c))).
results = []
for detuning in DRIVE_DETUNINGS:
    ########################################
    # Prepare the waveform and driven ring
    ########################################
    drive_q = natural_q + detuning
    samples = envelope * np.cos(2 * np.pi * drive_q * sample_turns + PHASE)
    exciter = xt.Exciter(
        knl                 = [-KICK_AMPLITUDE],
        samples             = samples,
        sampling_frequency  = fs,
        frev                = frev,
        start_turn          = START_TURN,
    )
    line = base.copy()
    line.insert('drive', exciter, at = 0.0)

    ########################################
    # Track the driven particle
    ########################################
    particle = line.build_particles(
        x       = reference.x[0],
        px      = reference.px[0],
        y       = reference.y[0],
        py      = reference.py[0],
        zeta    = zeta_start,
        delta   = 0.0,
    )
    line.track(particle, num_turns = N_TURNS, turn_by_turn_monitor = True)

    ########################################
    # Reconstruct kicks and measure the response
    ########################################
    record  = line.record_last_track
    turns   = np.arange(N_TURNS)
    # Reconstruct the actually addressed samples, including tracked zeta.
    indices = np.trunc(fs * (
        (turns - START_TURN) / frev - record.zeta[0] / (beta0 * c)
    )).astype(np.int64)
    kicks   = np.zeros(N_TURNS)
    valid   = (indices >= 0) & (indices < exciter.nduration)
    kicks[valid] = KICK_AMPLITUDE * np.asarray(exciter.samples)[indices[valid]]
    flat    = (turns >= START_TURN + RAMP_TURNS) & (
        turns < START_TURN + DRIVE_TURNS - RAMP_TURNS)
    x_trace = record.x[0] - reference.x[0]
    rms     = np.sqrt(np.mean(x_trace[flat]**2))
    print('Drive Qx / frequency [Hz]:', drive_q, drive_q * frev)
    print('Detuning / flattop x RMS [m]:', detuning, rms)
    print('Final state / delta:', particle.state, particle.delta)
    results.append((detuning, x_trace.copy(), kicks, rms, record, particle))

# A finite ramp is not an adiabaticity guarantee. Free oscillations may remain
# after switch-off. This finite-window comparison is not a stability threshold.

################################################################################
# Plot the sampled kicks and turn-by-turn transverse response
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (9, 6))
for detuning, x_trace, kicks, rms, record, particle in results:
    label = f'drive - natural tune = {detuning:+.3f}'
    axes[0].plot(turns, 1e6 * kicks, label = label)
    axes[1].plot(turns, 1e6 * x_trace, label = label)
axes[0].set_ylabel('Applied Delta px [1e-6]')
axes[1].set_ylabel('x - unforced orbit [um]')
axes[1].set_xlabel('Turn (recorded at entrance, before the drive kick)')
for axis in axes:
    axis.axvline(START_TURN, color = 'k', linestyle = ':')
    axis.axvline(START_TURN + DRIVE_TURNS, color = 'k', linestyle = ':')
    axis.grid()
    axis.legend()
fig.tight_layout()
plt.show()
