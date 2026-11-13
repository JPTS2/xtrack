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
# Native ring at fixed reference momentum and magnet strengths
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
OBSERVATION     = 'arc.s1.c1.qf.in'
SECTION_END     = 'bpm.s4.f'

########################################
# Momentum offsets for derivatives and a small tune scan
########################################
CHROM_STEP  = 1e-5                    # Symmetric delta increment
DELTA_SCAN  = np.linspace(-5e-5, 5e-5, 9)
# The bare vertical tune is near an integer: this is a local optical sample,
# not a momentum-acceptance scan. Curvature is visible even in this window.

################################################################################
# Calculate first-order chromatic optics
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
nominal     = line.twiss4d(chrom = True, delta_chrom = CHROM_STEP)

################################################################################
# Inspect global and local chromatic quantities
################################################################################

########################################
# Tune derivatives with respect to delta
########################################
print('Linear chromaticities dQ/d(delta):', nominal.dqx, nominal.dqy)

########################################
# Local chromatic beta and alpha functions
########################################
# For each uncoupled plane: b = (1/beta)*d(beta)/d(delta),
# a = d(alpha)/d(delta) - alpha*b, and W = sqrt(a^2 + b^2).
# These are dimensionless derivatives at delta = 0, not beta functions.
nominal.rows[[OBSERVATION]].cols[
    'name bx_chrom ax_chrom wx_chrom by_chrom ay_chrom wy_chrom'].show(
        maxwidth = 'full')

################################################################################
# Compare with separately calculated off-momentum solutions
################################################################################

########################################
# Centered finite differences of tunes
########################################
plus    = line.twiss4d(delta0 = CHROM_STEP, chrom = False)
minus   = line.twiss4d(delta0 = -CHROM_STEP, chrom = False)
dqx_fd  = (plus.qx - minus.qx) / (2 * CHROM_STEP)
dqy_fd  = (plus.qy - minus.qy) / (2 * CHROM_STEP)
print('Centered tune derivatives:', dqx_fd, dqy_fd)

########################################
# Centered finite differences of local optics
########################################
bx_fd   = (plus.betx - minus.betx) / (2 * CHROM_STEP * nominal.betx)
ax_fd   = (plus.alfx - minus.alfx) / (2 * CHROM_STEP) - nominal.alfx * bx_fd
print('Largest x-plane derivative differences (b, a):',
      np.max(np.abs(bx_fd - nominal.bx_chrom)),
      np.max(np.abs(ax_fd - nominal.ax_chrom)))

# Finite-difference step size and numerical orbit/matrix precision limit this
# comparison; equality to machine precision is not expected.

################################################################################
# Scan tunes at fixed momentum offsets
################################################################################

########################################
# Recompute the periodic optics for each offset
########################################
# delta changes particle momentum, not the reference p0 or magnet strengths.
# RF voltage is zero; radiation is off. Each 4D solution fixes its delta.
qx_scan = np.zeros(len(DELTA_SCAN))
qy_scan = np.zeros(len(DELTA_SCAN))
for i, delta in enumerate(DELTA_SCAN):
    optics                  = line.twiss4d(delta0 = delta, chrom = False)
    qx_scan[i], qy_scan[i]  = optics.qx, optics.qy

########################################
# Compare with the first-order tune shifts
########################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for ax, values, tune0, chromaticity, plane in [
    (axes[0], qx_scan, nominal.qx, nominal.dqx, 'x'),
    (axes[1], qy_scan, nominal.qy, nominal.dqy, 'y'),
]:
    ax.plot(DELTA_SCAN * 1e3, values - tune0, 'o', label = 'Twiss scan')
    ax.plot(DELTA_SCAN * 1e3, chromaticity * DELTA_SCAN,
            label = 'Linear chromaticity')
    ax.set_ylabel(f'Q_{plane} - Q_{plane}(0)')
    ax.legend()
axes[-1].set_xlabel('delta [1e-3]')
fig.tight_layout()

################################################################################
# Propagate chromatic conditions through an open section
################################################################################

########################################
# Retain the periodic boundary response
########################################
periodic_init   = nominal.get_twiss_init(OBSERVATION)
inherited       = line.twiss4d(
    start       = OBSERVATION,
    end         = SECTION_END,
    init        = periodic_init,
    chrom       = True,
    delta_chrom = CHROM_STEP)

########################################
# Impose zero chromatic beta and alpha response at the entrance
########################################
# Keep the same nominal optics and dispersion, but change the chromatic
# boundary conditions. This open solution need not close around the ring.
zero_init           = nominal.get_twiss_init(OBSERVATION)
zero_init.ax_chrom  = 0.0
zero_init.bx_chrom  = 0.0
zero_init.ay_chrom  = 0.0
zero_init.by_chrom  = 0.0
zero_boundary       = line.twiss4d(
    start       = OBSERVATION,
    end         = SECTION_END,
    init        = zero_init,
    chrom       = True,
    delta_chrom = CHROM_STEP)

########################################
# Propagate zero chromatic response from the downstream boundary
########################################
end_init            = nominal.get_twiss_init(SECTION_END)
end_init.ax_chrom   = 0.0
end_init.bx_chrom   = 0.0
end_init.ay_chrom   = 0.0
end_init.by_chrom   = 0.0
zero_at_end         = line.twiss4d(
    start       = OBSERVATION,
    end         = SECTION_END,
    init        = end_init,
    chrom       = True,
    delta_chrom = CHROM_STEP)
for label, optics in [('Periodic boundary', inherited),
                       ('Zero entrance', zero_boundary),
                       ('Zero exit', zero_at_end)]:
    print(label, 'section phase derivatives d(mu_x,y)/d(delta):',
          optics.dmux[-1] - optics.dmux[0],
          optics.dmuy[-1] - optics.dmuy[0])
# These depend on the imposed boundary response; they are not an invariant
# partition of the ring chromaticity or a local matching/correction scheme.

########################################
# Check the differentiation step and display the boundary dependence
########################################
refined = line.twiss4d(
    start       = OBSERVATION,
    end         = SECTION_END,
    init        = zero_init,
    chrom       = True,
    delta_chrom = CHROM_STEP / 2)
print('Largest W_x,y change after halving delta_chrom:',
      np.max(np.abs(refined.wx_chrom - zero_boundary.wx_chrom)),
      np.max(np.abs(refined.wy_chrom - zero_boundary.wy_chrom)))
fig_open, axes_open = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for ax, column, plane in zip(axes_open, ['wx_chrom', 'wy_chrom'], ['x', 'y']):
    ax.plot(inherited.s, inherited[column], label = 'Periodic boundary')
    ax.plot(zero_boundary.s, zero_boundary[column], label = 'Zero entrance')
    ax.plot(zero_at_end.s, zero_at_end[column], label = 'Zero exit boundary')
    ax.set_ylabel(f'W_{plane}')
    ax.grid()
    ax.legend()
axes_open[-1].set_xlabel('s [m]')
fig_open.tight_layout()
plt.show()
