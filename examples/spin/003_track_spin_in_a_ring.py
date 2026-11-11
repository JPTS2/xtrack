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
# Shared ring at a fixed electron reference momentum
########################################
LATTICE_FILE        = 'test_data/example_lattices/demo_ring/lattice.py'
P0C                 = 10e9              # [eV/c]

########################################
# Deterministic momentum sample and observation horizon
########################################
DELTA_LIMIT         = 1e-4              # Samples range from -limit to +limit
N_PROBES            = 11
N_TURNS             = 512

################################################################################
# Prepare native closed-orbit probes with common longitudinal spin
################################################################################

########################################
# Enable spin with radiation and RF off
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.set_particle_ref('electron', p0c = P0C)
line.configure_spin('auto')
# Geometry and normalized strengths are fixed; fields scale with rigidity.

########################################
# Put each momentum on its own transverse closed orbit
########################################
deltas  = np.linspace(-DELTA_LIMIT, DELTA_LIMIT, N_PROBES)
orbits  = []
for delta in deltas:
    orbit           = line.twiss4d(delta0 = float(delta), chrom = False)
    particle        = orbit.particle_on_co.copy()
    particle.spin_z = 1.0
    orbits.append(particle)

# Add an on-momentum vertical-spin control. The native merge assigns unique IDs.
control         = line.twiss4d(chrom = False).particle_on_co.copy()
control.spin_y  = 1.0
particles       = xt.Particles.merge(orbits + [control])

################################################################################
# Track individual spins and the vector average of the momentum sample
################################################################################

########################################
# Sample the ring entrance before each turn
########################################
line.track(particles, num_turns = N_TURNS, turn_by_turn_monitor = True)
record          = line.record_last_track
spins           = np.stack(
    [record.spin_x, record.spin_y, record.spin_z], axis = -1)
norms           = np.linalg.norm(spins, axis = -1)
mean_spin       = np.mean(spins[:N_PROBES], axis = 0)
polarization    = np.linalg.norm(mean_spin, axis = 1)
turns           = np.arange(N_TURNS)
print('Individual spin norm range:', norms.min(), norms.max())
print('Initial / final sampled polarization magnitude:',
      polarization[0], polarization[-1])
print('Vertical control range:',
      record.spin_y[N_PROBES].min(), record.spin_y[N_PROBES].max())
print('Final particle states:', particles.state)

# Different momenta precess at different rates. Unit spins can average to a
# shorter vector. This finite sample can rephase: the signal is deterministic
# dephasing, with no radiation or spin flips and no polarization-lifetime fit.

########################################
# Inspect individual precession before averaging the vectors
########################################
fig, axes   = plt.subplots(2, 1, figsize = (8, 6))
for index in [0, N_PROBES // 2, N_PROBES - 1]:
    axes[0].plot(
        turns[:32], record.spin_z[index, :32],
        label   = f'delta = {deltas[index]:+.0e}',
    )
axes[0].set_ylabel('Individual Sz')
axes[0].set_xlabel('Turns at the ring entrance (first 32 samples)')
axes[0].legend()
axes[1].plot(turns, polarization, label = 'Magnitude of mean spin')
axes[1].plot(
    turns, record.spin_y[N_PROBES], '--', label = 'Vertical control Sy')
axes[1].set_ylabel('Polarization / control spin')
axes[1].set_xlabel('Turns at the ring entrance')
axes[1].legend()
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
