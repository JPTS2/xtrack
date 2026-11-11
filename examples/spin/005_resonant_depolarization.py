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
# Real radial-field drive, updated once at each reference-turn entrance
########################################
DRIVE_S         = 2.0                   # Magnet entrance s [m]
DRIVE_LENGTH    = 0.1                   # [m]
K0S_AMPLITUDE   = 0.005                 # Radial field/(B*rho) [m^-1]
DETUNING        = -0.1                  # Off-resonance control [cycles/turn]
N_TURNS         = 1024
SAMPLE_EVERY    = 64                    # Choose a divisor of N_TURNS

################################################################################
# Set up a native spin-capable magnet and a vertically polarized sample
################################################################################

########################################
# Obtain the reference spin tune before enabling time-dependent settings
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.set_particle_ref('electron', p0c = P0C)
line.configure_spin('auto')
line.insert(
    'spin.drive',
    xt.Magnet(
        length  = DRIVE_LENGTH,
        k0s     = 0.0,
    ),
    at      = DRIVE_S,
    anchor  = 'start',
)
twiss       = line.twiss4d(polarization_analysis = True)
spin_tune   = twiss.spin_tune_fractional
period      = twiss.t_rev0
print('Fractional spin tune / reference period [s]:', spin_tune, period)
print('Orbital tunes Qx / Qy:', twiss.qx, twiss.qy)

# A radial magnetic field rotates spin and kicks the vertical orbit.
# The detuned control stays away from both spin and betatron resonance.

########################################
# Prepare identical near-closed-orbit launches for the two drive frequencies
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
# The finite momentum spread gives the probes different spin frequencies.

################################################################################
# Compare resonant and detuned tracking with the same magnetic amplitude
################################################################################

########################################
# Prepare the common reference-clock sample grid
########################################
turn_grid   = np.arange(N_TURNS + 1)
turns       = np.arange(0, N_TURNS + 1, SAMPLE_EVERY)
histories   = {}

########################################
# Track each drive and record projection and ensemble-mean magnitude
########################################
for label, frequency in [('Resonant', spin_tune),
                          ('Detuned', spin_tune + DETUNING)]:
    env.functions['spin_drive'] = xt.FunctionPieceWiseLinear(
        x   = turn_grid * period,
        y   = K0S_AMPLITUDE * np.cos(2 * np.pi * frequency * turn_grid),
    )
    env['drive_k0s']            = env.functions['spin_drive'](
        env.ref['t_turn_s'])
    line['spin.drive'].k0s      = env.ref['drive_k0s']
    env['t_turn_s']             = 0.0
    particles   = initial.copy()
    samples     = []
    peak_y      = 0.0
    line.enable_time_dependent_vars         = True
    line.dt_update_time_dependent_vars      = None

    for sample in range(len(turns)):
        if sample:
            line.track(particles, num_turns = SAMPLE_EVERY)
        spin    = np.array([particles.spin_x, particles.spin_y,
                            particles.spin_z])
        mean    = np.mean(spin, axis = 1)
        samples.append((mean[1], np.linalg.norm(mean)))
        peak_y  = max(peak_y, float(np.max(np.abs(particles.y))))
    histories[label]    = np.array(samples)
    print(label, 'final <Sy> / |<S>|:', *histories[label][-1])
    print(label, 'final alive / launched:',
          np.count_nonzero(particles.state > 0), N_PARTICLES)
    print(label, 'maximum sampled entrance |y| [m]:', peak_y)
    print(label, 'maximum individual unit-spin error:',
          np.max(np.abs(np.linalg.norm(spin, axis = 0) - 1)))

# The field follows the reference clock and is held fixed within each turn.
# A real cosine drive can use the folded spin tune through frequency aliasing.

################################################################################
# Distinguish coherent rotation from finite-ensemble dephasing
################################################################################

########################################
# Vertical spin projection
########################################
fig, axes = plt.subplots(2, 1, figsize = (8, 6), sharex = True)
for label, history in histories.items():
    axes[0].plot(turns, history[:, 0], 'o-', label = label)
axes[0].set_ylabel('Mean vertical spin <Sy>')

########################################
# Magnitude of the ensemble polarization
########################################
for label, history in histories.items():
    axes[1].plot(turns, history[:, 1], 'o-', label = label)
axes[1].set_ylabel('Magnitude of mean spin |<S>|')
axes[1].set_xlabel('Turns')
for axis in axes:
    axis.grid()
    axis.legend()
fig.tight_layout()
plt.show()

# A changing <Sy> can be coherent rotation. A falling |<S>| instead measures
# dephasing across this momentum sample; individual spins retain unit norm.
# This reversible sample response does not give a depolarization lifetime.
