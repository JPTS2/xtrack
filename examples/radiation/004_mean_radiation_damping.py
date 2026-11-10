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
# Shared electron ring and compensated mean-radiation operating point
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
P0C             = 10e9                   # Electron reference p0*c [eV]
RF_VOLTAGE      = 50e6                   # [V]
HARMONIC        = 200

########################################
# Small horizontal-mode sample and short tracking horizon
########################################
GEMITT_START    = 1e-11                 # Initial mean reference-mode action [m]
N_PHASES        = 10
N_TURNS         = 2048
N_INTERVALS     = 16                    # Choose a divisor of N_TURNS

# Phase-uniform probes resolve deterministic mode action; this is not a Monte
# Carlo population measurement. Ten phases do not set the damping rate.

################################################################################
# Find a compensated periodic mean-radiation solution
################################################################################

########################################
# Power the existing cavity and compensate the native mean energy loss
########################################
repo_root           = Path(__file__).resolve().parents[2]
env                 = xt.load(repo_root / LATTICE_FILE)
line                = env['demo']
env['p0c']          = P0C
line.set_particle_ref('electron', p0c = env['p0c'])
env['frequency_rf'] = 0.0
env.set('rf.1', harmonic = HARMONIC)
env['voltage_rf']   = RF_VOLTAGE
line.configure_radiation(model = 'mean')
line.compensate_radiation_energy_loss()
# The geometry is unchanged. Fixed normalized magnet strengths mean physical
# fields scale with the new reference rigidity: this is an illustrative setting,
# not a hardware-qualified machine. Native harmonic RF follows the new beta.
# Compensation sets phase and taper in memory; input files stay unchanged.

########################################
# Extract the full damped-map eigenvalue rates
########################################
tw      = line.twiss(
    method              = '6d',
    chrom               = False,
    radiation_analysis  = True,
    radiation_method    = 'full',
)
rate    = tw.damping_constants_turns[0]
print('Mean radiation loss per turn [eV]:', tw.energy_loss)
print('Amplitude damping constants [1/turn]:', tw.damping_constants_turns)
print('Horizontal amplitude / action e-folding turns:',
      1 / rate, 1 / (2 * rate))
print('Tracking horizon in amplitude / action damping times:',
      N_TURNS * rate, 2 * N_TURNS * rate)
# The native constant is -log(abs(lambda)), an AMPLITUDE decrement per turn.
# A quadratic mode action therefore decays approximately as exp(-2*rate*turns).

################################################################################
# Track a small phase-uniform horizontal mode with deterministic mean loss
################################################################################

########################################
# Prepare the mode through the native optical matrix
########################################
phase       = 2 * np.pi * np.arange(N_PHASES) / N_PHASES
particles   = line.build_particles(
    particle_on_co  = tw.particle_on_co,
    W_matrix        = tw.W_matrix[0],
    x_norm          = np.sqrt(2) * np.cos(phase),
    px_norm         = np.sqrt(2) * np.sin(phase),
    y_norm          = 0.0,
    py_norm         = 0.0,
    zeta_norm       = 0.0,
    pzeta_norm      = 0.0,
    gemitt_x        = GEMITT_START,
)

########################################
# Measure action in the fixed entrance mode basis at bounded intervals
########################################
turns   = np.arange(N_INTERVALS + 1) * (N_TURNS // N_INTERVALS)
actions = []
for sample in range(N_INTERVALS + 1):
    if sample:
        line.track(particles, num_turns = N_TURNS // N_INTERVALS)
    normalized  = tw.get_normalized_coordinates(particles)
    actions.append(np.mean(
        (normalized.x_norm**2 + normalized.px_norm**2) / 2))
actions     = np.asarray(actions)
relative    = actions / actions[0]
predicted   = np.exp(-2 * rate * turns)
measured    = -np.log(relative[-1]) / (2 * turns[-1])
print('Measured / native amplitude decrement [1/turn]:', measured, rate)
print('Tracked / predicted final action ratio:', relative[-1], predicted[-1])
print('Final states:', particles.state)

# The default horizon spans about 2.08 amplitude damping times, leaving about
# 1.56% of the initial action. Small nonlinear and numerical terms remain. Mean
# tracking has no random photon excitation or nonzero quantum equilibrium.

################################################################################
# Compare the actual mean-map decay over two amplitude damping times
################################################################################
fig, ax = plt.subplots(figsize = (8, 4))
ax.plot(turns * rate, relative, 'o', label = 'Tracked mean map')
ax.plot(turns * rate, predicted, '--', label = 'Linear damping prediction')
ax.set_xlabel('Turns / horizontal amplitude damping time')
ax.set_ylabel('Mean reference-mode action / initial')
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()
