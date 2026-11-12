# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Ring and reproducible relative field errors
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
SEED            = 29
NORMAL_RMS      = 1e-3                  # Relative quadrupole coefficient [1]
SKEW_RMS        = 1e-3                  # Skew/normal quadrupole ratio [1]
SEXT_RMS        = 0.02                  # Sextupole/quadrupole ratio [m^-1]

########################################
# Fixed absolute error and a main-family change
########################################
ABS_K2L         = 1e-3                  # Additive integrated sextupole [m^-2]
STRENGTH_FACTOR = 1.1                   # Main-strength and coefficient changes
SKEW_K1S        = 0.02                  # Isolated comparison magnet [m^-2]

################################################################################
# Select six quadrupoles and allocate their relative error arrays
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
quad_names  = [f'arc.s{sector}.c1.qd' for sector in range(1, 7)]
rng         = np.random.default_rng(SEED)
line.extend_knl_rel_ksl_rel(order = 2, element_names = quad_names)

# Quadrupoles use main_strength = k1*length by default. Setting main_is_skew
# would select k1s*length instead. The ring keeps the normal reference;
# an isolated copy below demonstrates that alternative.

################################################################################
# Add random relative errors and a fixed absolute sextupole component
################################################################################

########################################
# Define normal and skew errors relative to each signed main strength
########################################
# Index n denotes the 2*(n+1)-pole: n=1 quadrupole, n=2 sextupole.
# total_knl[n] includes knl[n] + main_strength*knl_rel[n], plus the body field.
# These ratios are not reference-radius b_n/a_n coefficients. Their units
# depend on order: knl_rel[1] is dimensionless, knl_rel[2] is in m^-1 here.
for name in quad_names:
    magnet              = line[name]
    magnet.knl_rel[1]   = rng.normal(scale = NORMAL_RMS)
    magnet.ksl_rel[1]   = rng.normal(scale = SKEW_RMS)
    magnet.knl_rel[2]   = rng.normal(scale = SEXT_RMS)

########################################
# Add the same absolute sextupole integral to every selected magnet
########################################
for name in quad_names:
    line[name].knl[2]   = ABS_K2L

# knl[n] and ksl[n] are integrated normalized strengths, without a 1/n!.
# The factorial enters the field polynomial, not the stored coefficient.
# These are field integrals; activation of fringe maps is a separate setting.
line.get_table(attr = True).rows[quad_names].cols[
    'name k1l k1sl k2l'].show(maxwidth = 'full')

################################################################################
# Change the powered family and inspect what follows its strength
################################################################################

########################################
# Keep one magnet's total integrals before and after the family change
########################################
magnet                      = line[quad_names[0]]
main_before                 = magnet.main_strength
knl_before, ksl_before      = magnet.get_total_knl_ksl()
env['kqd'] *= STRENGTH_FACTOR
knl_after, ksl_after        = magnet.get_total_knl_ksl()

########################################
# Separate the absolute contribution from the relative sextupole error
########################################
# No expression wiring is needed: relative errors follow main_strength at
# tracking time. Absolute knl[2] stays fixed when the main current changes.
print('Selected magnet:', quad_names[0])
print('Main integral before/after [m^-1]:',
      main_before, magnet.main_strength)
print('Fixed absolute sextupole integral [m^-2]:', magnet.knl[2])
print('Relative sextupole contribution before/after [m^-2]:',
      knl_before[2] - ABS_K2L, knl_after[2] - ABS_K2L)
print('Skew quadrupole integral before/after [m^-1]:',
      ksl_before[1], ksl_after[1])
line.get_table(attr = True).rows[quad_names].cols[
    'name k1l k1sl k2l'].show(maxwidth = 'full')

################################################################################
# Replace random arrays with a common expression-based error pattern
################################################################################

########################################
# Apply one live coefficient scale to all selected quadrupoles
########################################
# This alternative intentionally replaces the previous random relative arrays.
# Absolute knl remains separate. The arrays already have three entries from
# extend_knl_rel_ksl_rel above; env.set requires that matching array length.
env['systematic_scale'] = 1.0
env.set(
    quad_names,
    knl_rel = [0.0, env.ref['systematic_scale'] * NORMAL_RMS,
               env.ref['systematic_scale'] * SEXT_RMS],
    ksl_rel = [0.0, env.ref['systematic_scale'] * SKEW_RMS, 0.0],
)
print('Shared relative sextupole coefficient expression:',
      env.ref[quad_names[0]].knl_rel[2].xdeps.expr)
# Only the coefficient graph is wired here. Native main_strength supplies the
# signed main-field integral automatically, including after family changes.

########################################
# Change the common coefficient without changing any main magnet strength
########################################
bulk_before = line[quad_names[0]].get_total_knl_ksl()
env['systematic_scale'] = STRENGTH_FACTOR
bulk_after  = line[quad_names[0]].get_total_knl_ksl()
print('Relative sextupole integral before/after coefficient change [m^-2]:',
      bulk_before[0][2] - ABS_K2L, bulk_after[0][2] - ABS_K2L)
line.get_table(attr = True).rows[quad_names].cols[
    'name k1l k1sl k2l'].show(maxwidth = 'full')

################################################################################
# Choose the signed normal or skew main component as the error reference
################################################################################

########################################
# Compare both choices on one independent magnet outside the ring
########################################
# Keep both body components fixed while changing which one scales the errors.
# main_is_skew does not rotate the magnet or remove its normal body field.
# The copy shares coefficient knobs but is not inserted into the ring.
env['reference_k1s']    = SKEW_K1S
env.new('reference.quad', quad_names[0], k1s = 'reference_k1s')
reference_magnet    = env['reference.quad']
reference_totals    = {}
for label, use_skew in [('Normal main', False), ('Skew main', True)]:
    env.set('reference.quad', main_is_skew = use_skew)
    normal, skew    = reference_magnet.get_total_knl_ksl()
    reference_totals[label] = (normal, skew)
    print(label, 'signed main integral [m^-1]:', reference_magnet.main_strength)
    print('Body k1 / k1s [m^-2]:', reference_magnet.k1, reference_magnet.k1s)
    print('Relative sextupole integral [m^-2]:', normal[2] - ABS_K2L)

########################################
# Change the skew excitation with the skew reference selected
########################################
skew_main_before    = reference_magnet.main_strength
env['reference_k1s'] *= STRENGTH_FACTOR
skew_powered        = reference_magnet.get_total_knl_ksl()
print('Signed skew main integral before/after [m^-1]:',
      skew_main_before, reference_magnet.main_strength)
print('Relative sextupole integral before/after skew excitation [m^-2]:',
      reference_totals['Skew main'][0][2] - ABS_K2L,
      skew_powered[0][2] - ABS_K2L)
