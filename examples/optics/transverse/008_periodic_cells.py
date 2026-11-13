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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
CELL_NAME       = 'arc.cell'
CELL_START      = 'arc.s1.c1.qf.in'
NEXT_CELL_START = 'arc.s1.c2.qf.in'

################################################################################
# Load the reusable cell and the assembled ring
################################################################################

repo_root           = Path(__file__).resolve().parents[3]
env                 = xt.load(repo_root / LATTICE_FILE)
line                = env[LINE_NAME]
cell                = env[CELL_NAME]
cell.particle_ref   = line.particle_ref.copy()
cell.get_table().cols['name s'].show(maxwidth = 'full')

################################################################################
# Solve periodic optics for one cell
################################################################################

########################################
# Impose periodicity on the reusable optical unit
########################################
# The cell bends through 15 degrees and need not close on the floor plan.
# Its transverse phase advances are 90 degrees, or 0.25 turns, in both planes.
tw_cell = cell.twiss4d()
tw_cell.rows[[cell.element_names[0], '_end_point']].cols[
    'name s betx alfx bety alfy dx dpx'].show(maxwidth = 'full')

########################################
# Distinguish the cell phase from the full-ring tunes
########################################
tw_ring = line.twiss4d()
print('Periodic cell phase advances [turns]:', tw_cell.qx, tw_cell.qy)
print('Full ring tunes, including straight sections:', tw_ring.qx, tw_ring.qy)
# Four cells form each arc, but the insertion and utility sections contribute
# additional phase. Multiplying cell phases is not a calculation of ring tunes.
# The four-cell transverse identity alone does not select unique periodic beta.

################################################################################
# Impose periodicity only on a selected range of the assembled line
################################################################################
# The final boundary is a zero-length marker, so its entrance/exit coincide.
# init='periodic' solves this range as its own optical period rather than
# importing the full-ring dispersion or other boundary conditions.
tw_range = line.twiss4d(
    start   = CELL_START,
    end     = NEXT_CELL_START,
    init    = 'periodic',
)
print('Selected-range/cell periodic entrance betx,bety,dx:')
print(tw_range.betx[0], tw_cell.betx[0], tw_range.bety[0], tw_cell.bety[0],
      tw_range.dx[0], tw_cell.dx[0])

################################################################################
# Compare the first cell with its use in the ring
################################################################################

########################################
# Read the ring optics between equivalent cell entrances
########################################
# Table slicing includes the next cell's entrance as this cell's final plane.
ring_cell = tw_ring.rows[CELL_START:NEXT_CELL_START]
print('Ring beta and dispersion at the first cell entrance:')
tw_ring.rows[[CELL_START]].cols[
    'name betx alfx bety alfy dx dpx'].show(maxwidth = 'full')
# The straights match the periodic cell's beta/alpha, but impose zero dispersion
# at the arc ports. The ring dispersion is not the periodic-cell dispersion.

########################################
# Overlay beta functions with the two different dispersion solutions
########################################
fig, axes = plt.subplots(3, 1, sharex = True, figsize = (8, 8))
for ax, column in zip(axes, ['betx', 'bety', 'dx']):
    ax.plot(tw_cell.s, tw_cell[column], label = 'Periodic cell')
    ax.plot(
        ring_cell.s - ring_cell.s[0], ring_cell[column], 'o',
        label = 'Cell inside full ring')
    ax.set_ylabel(f'{column} [m]')
    ax.legend()
axes[-1].set_xlabel('s from cell entrance [m]')
fig.tight_layout()
plt.show()
