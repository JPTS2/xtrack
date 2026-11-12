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
# Model and selected range
########################################
LATTICE_FILE        = 'test_data/example_lattices/transport/lattice.py'
LINE_NAME           = 'transport'
START_ELEMENT       = 'b1.1'
END_ELEMENT         = 'optics_end'
INTERIOR_ELEMENT    = 'optics_middle'
INTERIOR_AT         = 'qd.1@start'
BOUNDARY_AT         = 'qf.2@start'

########################################
# Entrance optics
########################################
BETX_START      = 2.0                   # [m]
BETY_START      = 3.0                   # [m]
ALFX_START      = 0.2                   # Dimensionless
ALFY_START      = -0.1                  # Dimensionless

################################################################################
# Load the transport line
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
# A zero-length marker identifies the downstream optical plane unambiguously.
env.new(END_ELEMENT, xt.Marker)
line.insert(env.place(END_ELEMENT, at = BOUNDARY_AT))
# Interior initialization uses a marker plane, as required by this API.
env.new(INTERIOR_ELEMENT, xt.Marker)
line.insert(env.place(INTERIOR_ELEMENT, at = INTERIOR_AT))

################################################################################
# Calculate the full open-line solution
################################################################################
# The model supplies an electron reference particle with p0*c = 1 GeV.
# Unspecified entrance orbit and dispersion components are zero.
tw_full = line.twiss(
    method  = '4d',
    betx    = BETX_START,
    bety    = BETY_START,
    alfx    = ALFX_START,
    alfy    = ALFY_START,
)

################################################################################
# Reuse the boundary conditions for a subrange
################################################################################

########################################
# Extract the complete entrance state
########################################
# TwissInit carries the orbit, dispersion, optical matrix and phase origins.
# Named rows refer to element entrances; this is the entrance of b1.1.
init    = tw_full.get_twiss_init(START_ELEMENT)

########################################
# Propagate through the selected elements
########################################
# end is inclusive. Here the final marker has zero length, so its entrance
# and the final _end_point row describe the same physical plane.
tw_part = line.twiss(
    method  = '4d',
    start   = START_ELEMENT,
    end     = END_ELEMENT,
    init    = init,
)

########################################
# Reuse a downstream entrance state to propagate upstream
########################################
# The TwissInit records its own location; the solver transports it back to
# START_ELEMENT, then includes the exit of END_ELEMENT in the result.
init_downstream     = tw_full.get_twiss_init(END_ELEMENT)
tw_from_downstream  = line.twiss(
    method  = '4d',
    start   = START_ELEMENT,
    end     = END_ELEMENT,
    init    = init_downstream)
print('Subrange optics: s, beta and dx [m]; alpha dimensionless:')
tw_part.cols['name s betx bety alfx alfy dx'].show(maxwidth = 'full')

################################################################################
# Supply the optics at an interior element
################################################################################

########################################
# Propagate from the interior toward both boundaries
########################################
# A table plus init_at selects its complete state at the named entrance.
tw_from_interior = line.twiss(
    method  = '4d',
    start   = START_ELEMENT,
    end     = END_ELEMENT,
    init    = tw_full,
    init_at = INTERIOR_ELEMENT)

########################################
# Supply the particle and optical matrix explicitly
########################################
# This is an imposed open-line boundary, not a periodic normal form.
interior = tw_full.get_twiss_init(INTERIOR_ELEMENT)
explicit_init = xt.TwissInit(
    particle_on_co  = interior.particle_on_co.copy(),
    W_matrix        = interior.W_matrix.copy(),
    element_name    = INTERIOR_ELEMENT,
    mux             = interior.mux,
    muy             = interior.muy,
    muzeta          = interior.muzeta,
    reference_frame = interior.reference_frame)
tw_from_matrix = line.twiss(
    method  = '4d',
    start   = START_ELEMENT,
    end     = END_ELEMENT,
    init    = explicit_init)
print('Interior entrance and both range boundaries:')
tw_from_matrix.rows[[START_ELEMENT, INTERIOR_ELEMENT, END_ELEMENT]].cols[
    'name s betx bety alfx alfy dx'].show(maxwidth = 'full')

################################################################################
# Compare with the original solution
################################################################################

########################################
# Compare matching element entrances
########################################
# The subrange _end_point is not the full line's _end_point. Compare named
# entrance rows, including the boundary marker. The two _end_point rows
# describe different exits.
tw_reference    = tw_full.rows[tw_part.name[:-1]]
for column, unit in [
    ('betx', 'm'), ('bety', 'm'), ('alfx', '1'), ('alfy', '1'),
    ('dx', 'm'), ('dpx', '1'), ('mux', 'turns'), ('muy', 'turns'),
]:
    difference              = np.max(np.abs(
        tw_part[column][:-1] - tw_reference[column]))
    difference_downstream   = np.max(np.abs(
        tw_from_downstream[column][:-1] - tw_reference[column]))
    print(f'{column}: max differences from upstream/downstream init '
          f'{difference:.3e}, {difference_downstream:.3e} {unit}')
    for label, optics in [
        ('Interior table', tw_from_interior),
        ('Explicit matrix', tw_from_matrix),
    ]:
        residual = np.max(np.abs(
            optics[column][:-1] - tw_reference[column]))
        print(f'  {label}: {residual:.3e} {unit}')

########################################
# Overlay the full and propagated optics
########################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
axes[0].plot(tw_full.s, tw_full.betx, label = 'Full line')
axes[0].plot(tw_part.s, tw_part.betx, 'o', label = 'Reused TwissInit')
axes[0].set_ylabel(r'$\beta_x$ [m]')
axes[0].legend()
axes[1].plot(tw_full.s, tw_full.bety)
axes[1].plot(tw_part.s, tw_part.bety, 'o')
axes[1].set_ylabel(r'$\beta_y$ [m]')
axes[1].set_xlabel('s [m]')
fig.suptitle('Propagate the same boundary conditions through a subrange')
fig.tight_layout()
plt.show()
