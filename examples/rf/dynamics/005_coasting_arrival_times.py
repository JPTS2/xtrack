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
# RF-free ring
########################################
# Lattice file relative to the repository root.
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'

########################################
# Unconfined probes and observation horizon
########################################
DELTA           = np.array([-1e-4, 0.0, 1e-4])
N_TURNS         = 128

################################################################################
# Prepare coasting probes on their transverse closed orbits
################################################################################

########################################
# Load the bare ring and its reference optics
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
tw          = line.twiss4d()
beta0       = line.particle_ref.beta0[0]

# The reference is a 1 GeV/c electron. RF voltage and radiation are zero.
# These are three unconfined probes, not a matched bunch or an acceptance scan.

########################################
# Use native closed-orbit solutions and merge the particle containers
########################################
orbits      = [line.twiss4d(delta0 = float(delta), chrom = False).particle_on_co
               for delta in DELTA]
particles   = xt.Particles.merge(orbits)

################################################################################
# Compare the shared reference clock with individual arrival times
################################################################################

########################################
# Observe each completed physical passage at the ring entrance
########################################
clock           = np.zeros(N_TURNS + 1)
arrival_delay   = np.zeros((N_TURNS + 1, len(DELTA)))
for turn in range(N_TURNS + 1):
    order   = np.argsort(particles.particle_id)
    arrival = particles.time_s + (particles.s - particles.zeta) / (beta0 * c)
    clock[turn] = particles.time_s
    arrival_delay[turn] = arrival[order] - particles.time_s
    if turn < N_TURNS:
        line.track(particles, num_turns = 1)

# time_s is the shared reference clock, not each particle's arrival time.
# At the entrance s=0, positive zeta means arrival before the reference.
# Ordinary turn tracking counts passages; it does not force equal arrival times.

########################################
# Compare accumulated delays with the first-order slip factor
########################################
# eta = alpha_c - 1/gamma0^2. Positive eta means positive delta arrives later.
prediction = clock[-1] * tw.slip_factor * DELTA
xt.Table(
    data    = {
        'delta':                DELTA,
        'arrival_delay_ns':     arrival_delay[-1] * 1e9,
        'linear_prediction_ns': prediction * 1e9},
    index   = 'delta').show(maxwidth = 'full')
print('Shared reference time [s]:', particles.time_s)
print('Individual passage counts:', particles.at_turn)
print('Particle states:', particles.state)

# The small offsets are qualified probes of this highly chromatic bare ring.
# The linear prediction is local; it is not an off-momentum acceptance claim.

################################################################################
# Plot the freely accumulating arrival-time differences
################################################################################
fig, ax = plt.subplots(figsize = (8, 4))
for index, delta in enumerate(DELTA):
    ax.plot(1e6 * clock, 1e9 * arrival_delay[:, index],
            label = f'delta = {delta:+.0e}')
ax.set_xlabel('Shared reference time [us]')
ax.set_ylabel('Arrival delay at entrance [ns]')
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()
