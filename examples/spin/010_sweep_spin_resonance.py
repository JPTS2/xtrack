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
# Shared electron ring, radiation and RF off
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
P0C             = 10e9                  # Reference p0*c [eV]
DELTA_MAX       = 1e-4                  # Uniform deterministic momentum sample
N_PARTICLES     = 21

########################################
# Turn-held radial magnetic field
########################################
DRIVE_S         = 2.0                   # Magnet entrance s [m]
DRIVE_LENGTH    = 0.1                   # [m]
K0S_AMPLITUDE   = 0.005                 # Radial field/(B*rho) [m^-1]
N_TURNS         = 1024
SAMPLE_EVERY    = 64                    # Choose a divisor of N_TURNS

########################################
# Crossing and detuned linear frequency sweeps
########################################
SWEEP_HALF_SPAN = 0.02                  # Around the spin alias [cycles/turn]
CONTROL_OFFSET  = -0.1                  # Same sweep, away from resonance

################################################################################
# Prepare a spin-capable magnetic drive and a common initial sample
################################################################################

########################################
# Calculate the undriven spin reference before enabling the field waveform
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.set_particle_ref('electron', p0c = P0C)
line.configure_spin('auto')
line.insert(
    'spin.drive',
    xt.Magnet(length = DRIVE_LENGTH, k0s = 0.0),
    at      = DRIVE_S,
    anchor  = 'start',
)
twiss       = line.twiss4d(polarization_analysis = True)
spin_tune   = twiss.spin_tune_fractional
period      = twiss.t_rev0
print('Reference fractional spin tune / revolution period [s]:',
      spin_tune, period)
print('Orbital tunes Qx / Qy:', twiss.qx, twiss.qy)
# A real turn-sampled cosine aliases spin frequencies modulo the revolution
# frequency. These sweeps use the native folded spin tune, not the full G*gamma.
# The radial field also kicks the vertical orbit; it is not an orbit-only drive.

########################################
# Launch identical vertically polarized probes at small momentum offsets
########################################
initial = line.build_particles(
    method      = '4d',
    delta       = np.linspace(-DELTA_MAX, DELTA_MAX, N_PARTICLES),
    x_norm      = 0.0,
    px_norm     = 0.0,
    y_norm      = 0.0,
    py_norm     = 0.0,
    spin_y      = 1.0,
)
# Zero normalized coordinates follow the linear dispersive closed orbit.
# The finite momentum range gives the probes different spin frequencies.
# The supplied settings keep all probes alive; the means use all launches.

################################################################################
# Compare a resonance crossing with an otherwise matched detuned sweep
################################################################################

########################################
# Prepare the common reference-clock and observation grids
########################################
turn_grid           = np.arange(N_TURNS + 1)
turns               = np.arange(0, N_TURNS + 1, SAMPLE_EVERY)
histories           = {}
frequencies         = {}
final_populations   = {}

########################################
# Integrate each frequency sweep and track its magnetic response
########################################
for label, offset in [('Crossing sweep', 0.0),
                       ('Detuned sweep', CONTROL_OFFSET)]:
    frequency_start = spin_tune - SWEEP_HALF_SPAN + offset
    frequency_step  = 2 * SWEEP_HALF_SPAN / N_TURNS
    phase           = 2 * np.pi * (
        frequency_start * turn_grid + 0.5 * frequency_step * turn_grid**2)
    frequencies[label]  = frequency_start + frequency_step * turns
    env.functions['spin_sweep'] = xt.FunctionPieceWiseLinear(
        x   = turn_grid * period,
        y   = K0S_AMPLITUDE * np.cos(phase),
    )
    env['drive_k0s']        = env.functions['spin_sweep'](env.ref['t_turn_s'])
    line['spin.drive'].k0s  = env.ref['drive_k0s']
    env['t_turn_s']         = 0.0
    particles               = initial.copy()
    samples                 = []
    peak_y                  = 0.0
    line.enable_time_dependent_vars         = True
    line.dt_update_time_dependent_vars      = None
    # Frequency is the phase derivative: phi(n)=2*pi*(nu_start*n+rate*n^2/2).
    # Substituting nu(n)*n would double the sweep contribution to frequency.

    for sample in range(len(turns)):
        if sample:
            line.track(particles, num_turns = SAMPLE_EVERY)
        spin    = np.array([particles.spin_x, particles.spin_y,
                            particles.spin_z])
        mean    = np.mean(spin, axis = 1)
        samples.append((mean[1], np.linalg.norm(mean)))
        peak_y  = max(peak_y, float(np.max(np.abs(particles.y))))
    histories[label]            = np.array(samples)
    final_populations[label]    = particles
    print(label, 'programmed start/end frequency [cycles/turn]:',
          frequencies[label][0], frequencies[label][-1])
    print(label, 'final <Sy> / |<S>|:', *histories[label][-1])
    print(label, 'final alive / launched:',
          np.count_nonzero(particles.state > 0), N_PARTICLES)
    print(label, 'maximum sampled entrance |y| [m]:', peak_y)
    print(label, 'maximum individual unit-spin error:',
          np.max(np.abs(np.linalg.norm(spin, axis = 0) - 1)))
line.enable_time_dependent_vars = False

# Both cases have the same amplitude, initial phase, sweep rate and duration.
# Each field sample is applied at a reference-turn entrance and held through
# that turn, rather than following continuous laboratory-time RF phase.

################################################################################
# Relate the programmed crossing to spin rotation and ensemble dephasing
################################################################################

########################################
# Inspect the frequency programme relative to the reference spin alias
########################################
fig, axes = plt.subplots(3, 1, figsize = (8, 8), sharex = True)
for label, values in frequencies.items():
    axes[0].plot(turns, values, label = label)
axes[0].axhline(spin_tune, color = 'k', linestyle = '--',
                label = 'Undriven reference spin alias')
axes[0].set_ylabel('Drive frequency [cycles/turn]')

########################################
# Separate vertical projection from the magnitude of the mean spin
########################################
for label, history in histories.items():
    axes[1].plot(turns, history[:, 0], 'o-', label = label)
    axes[2].plot(turns, history[:, 1], 'o-', label = label)
axes[1].set_ylabel('Mean vertical spin <Sy>')
axes[2].set_ylabel('Magnitude of mean spin |<S>|')
axes[2].set_xlabel('Reference turns')
for axis in axes:
    axis.grid()
    axis.legend(fontsize = 8)
fig.tight_layout()
plt.show()

# The reference alias is crossed halfway through this programme. The finite
# momentum distribution and finite drive spread the response over frequency;
# this one sweep is not a precision resonance-frequency or energy calibration.
# A changing <Sy> includes coherent rotation; falling |<S>| is dephasing of
# this reversible finite sample, not a fitted depolarization lifetime.
