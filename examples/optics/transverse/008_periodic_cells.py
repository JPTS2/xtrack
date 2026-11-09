# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import xtrack as xt

################################################################################
# User parameters
################################################################################
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAME       = 'ring'
CELL_START      = 'qf.1'
NEXT_CELL_START = 'qf.2'
NUM_CELLS       = 8

################################################################################
# Select one complete cell between equivalent entrances
################################################################################

########################################
# Read the ring sequence
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
table       = line.get_table()

########################################
# Build a subline excluding the next cell's first quadrupole
########################################
# Table slices include their final row. Omit that row to select [qf.1, qf.2).
# env_name identifies the actual elements, including shared drift occurrences.
cell_rows   = table.rows[CELL_START:NEXT_CELL_START]
cell        = env.new_line(components = list(cell_rows.env_name[:-1]))
cell.particle_ref = line.particle_ref.copy()
cell.get_table().cols['name s'].show(maxwidth = 'full')

################################################################################
# Solve a periodic cell and the full ring
################################################################################

########################################
# Impose periodicity on the selected cell
########################################
# This treats the cell as a repeated optical unit. Its reference geometry
# bends through 45 degrees and does not close by itself on the floor plan.
tw_cell = cell.twiss4d()
tw_cell.rows[[CELL_START, '_end_point']].cols[
    'name s betx alfx bety alfy dx dpx'].show(maxwidth = 'full')

########################################
# Compare with the ring made from eight identical cells
########################################
tw_ring = line.twiss4d()
print('Cell phase advances [turns]:', tw_cell.qx, tw_cell.qy)
print('Cell phase advances times cell count:',
      NUM_CELLS * tw_cell.qx, NUM_CELLS * tw_cell.qy)
print('Full ring tunes:', tw_ring.qx, tw_ring.qy)

################################################################################
# Compare the optical functions through the first cell
################################################################################

########################################
# Read the ring optics at the same entrances
########################################
ring_cell   = tw_ring.rows[CELL_START:NEXT_CELL_START]
print('Ring beta and dispersion at the cell entrance:')
tw_ring.rows[[CELL_START]].cols[
    'name betx alfx bety alfy dx dpx'].show(maxwidth = 'full')

########################################
# Overlay cell and ring beta functions
########################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for ax, column in zip(axes, ['betx', 'bety']):
    ax.plot(tw_cell.s, tw_cell[column], label = 'Periodic cell')
    ax.plot(
        ring_cell.s - ring_cell.s[0], ring_cell[column], 'o',
        label = 'Full-ring solution')
    ax.set_ylabel(f'{column} [m]')
    ax.legend()
axes[-1].set_xlabel('s from cell entrance [m]')
fig.tight_layout()
plt.show()
