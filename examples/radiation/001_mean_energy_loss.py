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
# The lattice path is relative to the repository root.
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
P0C             = 10e9                  # Electron reference p0*c [eV]

################################################################################
# Launch the same electron through the RF-free ring with radiation off and on
################################################################################

########################################
# Select the 10 GeV/c electron reference and its bare closed orbit
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
env['p0c']  = P0C
line.set_particle_ref('electron', p0c = env['p0c'])
initial     = line.twiss4d(chrom = False).particle_on_co.copy()
# RF voltage starts at zero. No energy programme changes the reference.
# Fixed geometry and normalized strengths imply fields scaled with rigidity.

########################################
# Apply the native mean-radiation model for one physical passage
########################################
losses      = {}
summary     = []
for label, model in [('Radiation off', None), ('Mean radiation', 'mean')]:
    line.configure_radiation(model = model)
    particles   = initial.copy()
    line.track(particles, num_turns = 1, turn_by_turn_monitor = 'ONE_TURN_EBE')
    record      = line.record_last_track
    # At fixed p0c, the physical energy change is p0c times the ptau change.
    losses[label]   = -initial.p0c[0] * (record.ptau[0] - initial.ptau[0])
    summary.append((label, initial.energy[0] - particles.energy[0],
                    particles.delta[0], particles.state[0]))

################################################################################
# Inspect energy loss along the passage and at the exit
################################################################################

########################################
# Compare native physical particle energies
########################################
xt.Table(
    data    = {
        'model':            np.array([row[0] for row in summary]),
        'energy_loss_MeV':  np.array([row[1] for row in summary]) / 1e6,
        'exit_delta':       np.array([row[2] for row in summary]),
        'exit_state':       np.array([row[3] for row in summary])},
    index   = 'model').show(maxwidth = 'full')
print('Fixed reference p0*c [GeV]:', particles.p0c[0] / 1e9)

########################################
# Resolve the distributed loss using the native element-by-element record
########################################
table       = line.get_table()
fig, ax     = plt.subplots(figsize = (8, 4))
for label, loss in losses.items():
    ax.plot(table.s, loss / 1e6, label = label)
ax.set_xlabel('s [m]')
ax.set_ylabel('Accumulated energy loss [MeV]')
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()

# Mean radiation applies deterministic loss, without stochastic excitation.
# This uncompensated passage is not a periodic radiating closed orbit.
