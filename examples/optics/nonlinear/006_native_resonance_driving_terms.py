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
# Native ring and one weak sextupole
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
SOURCE          = 'bpm.s4.f'             # Replace this zero-length device
OBSERVATION     = 'ip.1'                 # Compare away from the source
K2L             = 1e-3                   # Integrated normal strength [m^-2]
SHIFT_Y         = 0.5e-3                 # Sextupole displacement [m]
RDT_NAMES       = ['f3000', 'f1200', 'f1001', 'f1010']

################################################################################
# Prepare an unperturbed reference with the same element names
################################################################################

########################################
# Keep the nominal quadrupoles and switch off higher multipoles
########################################
repo_root = Path(__file__).resolve().parents[3]
env       = xt.load(repo_root / LATTICE_FILE)
line      = env[LINE_NAME]
for knob in ['k2lf', 'k2ld', 'k3lf', 'k3ld']:
    env[knob] = 0.0
# The ring retains its 1 GeV/c electron reference. RF and radiation are off.

########################################
# Register the source with zero strength for reference optics
########################################
env['rdt_k2l'] = 0.0
env.new(
    'rdt.sext', xt.Multipole,
    knl     = [0.0, 0.0, 'rdt_k2l'],
    shift_y = SHIFT_Y,
)
line.replace(SOURCE, 'rdt.sext')
reference = line.twiss(method = '4d')
print('Reference tunes:', reference.qx, reference.qy)

################################################################################
# Calculate RDTs using explicit optics, strengths and orbit
################################################################################

########################################
# Recompute the orbit with the actual perturbation
########################################
env['rdt_k2l'] = K2L
perturbed = line.twiss(method = '4d', coupling_edw_teng = True)
strengths = line.get_table(attr = True)
# All three tables must have the same named entrances in the same order.
# Feed-down uses the orbit relative to each displaced/rotated magnet.
rdts = xt.rdt_first_order_perturbation(
    rdt         = RDT_NAMES,
    twiss       = reference,
    strengths   = strengths,
    orbit       = perturbed,
    feed_down   = True,
)
rdts.rows[[OBSERVATION]].cols['name f3000 f1200 f1001 f1010'].show(
    maxwidth = 'full')
# Sextupole f3000/f1200 have units m^-1/2. Coupling f1001/f1010 are
# dimensionless.
# Their complex phases use the reference table's phase origin and tune branch.
# Near the corresponding resonance, the perturbative denominator becomes small.
print('f3000 horizontal sideband frequency [cycles/turn]:', rdts.f3000_freq_x)
print('f3000 horizontal frequency expression:', rdts.f3000_freq_x_expr)

########################################
# Isolate the coupling contribution from feed-down
########################################
without_feed_down = xt.rdt_first_order_perturbation(
    rdt         = ['f1001', 'f1010'],
    twiss       = reference,
    strengths   = strengths,
    orbit       = perturbed,
    feed_down   = False,
)
print('f1001 without feed-down:', without_feed_down['f1001', OBSERVATION])
print('f1001 including feed-down:', rdts['f1001', OBSERVATION])
# There is no installed skew quadrupole. Sampling the displaced sextupole
# away from its magnetic axis supplies the effective skew-quadrupole term.

################################################################################
# Compare weak-coupling optics and repeat at half strength
################################################################################

########################################
# Compare the same observation plane and phase convention
########################################
print('First-order f1001 at IP:', rdts['f1001', OBSERVATION])
print('Native Edwards-Teng f1001 at IP:', perturbed['f1001', OBSERVATION])
relative_difference = abs(
    (rdts['f1001', OBSERVATION] - perturbed['f1001', OBSERVATION])
    / perturbed['f1001', OBSERVATION])
print('Relative complex f1001 difference:', relative_difference)

########################################
# Retain reference optics while reducing the perturbation
########################################
env['rdt_k2l'] = K2L / 2
half_optics = line.twiss(method = '4d', coupling_edw_teng = True)
half_rdts = xt.rdt_first_order_perturbation(
    rdt         = RDT_NAMES,
    twiss       = reference,
    strengths   = line.get_table(attr = True),
    orbit       = half_optics,
    feed_down   = True,
)
print('Half-strength f3000 / full-strength f3000:',
      half_rdts['f3000', OBSERVATION] / rdts['f3000', OBSERVATION])
print('Half-strength f1001 / full-strength f1001:',
      half_rdts['f1001', OBSERVATION] / rdts['f1001', OBSERVATION])
print('Half-strength first-order/native relative f1001 difference:', abs(
    (half_rdts['f1001', OBSERVATION] - half_optics['f1001', OBSERVATION])
    / half_optics['f1001', OBSERVATION]))
# Finite-difference optics can limit this comparison at very weak coupling.
# First-order RDTs do not qualify finite-amplitude stability or higher orders.

################################################################################
# Inspect the full-strength RDTs around the ring
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (9, 6))
axes[0].plot(reference.s, rdts.f3000.real, label = 'Re(f3000)')
axes[0].plot(reference.s, rdts.f3000.imag, label = 'Im(f3000)')
axes[0].set_ylabel('Sextupole RDT [m^-1/2]')
axes[1].plot(reference.s, rdts.f1001.real, label = 'Re(f1001), first order')
axes[1].plot(reference.s, rdts.f1001.imag, label = 'Im(f1001), first order')
axes[1].plot(perturbed.s, perturbed.f1001.real, '--',
             label = 'Re(f1001), Edwards-Teng')
axes[1].plot(perturbed.s, perturbed.f1001.imag, '--',
             label = 'Im(f1001), Edwards-Teng')
axes[1].set_ylabel('Coupling RDT [dimensionless]')
axes[1].set_xlabel('Reference path length s [m]')
for axis in axes:
    axis.axvline(reference['s', 'rdt.sext'], color = 'grey', linestyle = ':',
                 label = 'Sextupole source')
    axis.grid()
    axis.legend()
# A thin source produces an RDT jump; its entrance/exit sampling convention
# matters in a pointwise comparison. The printed check uses the separate IP.
fig.tight_layout()
plt.show()
