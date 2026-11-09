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
# Model and later knob change
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
KQF_FACTOR      = 1.01                   # Applied AFTER tracking comparison

########################################
# Common entrance particles and propagation
########################################
N_TURNS         = 8
X_START         = np.array([0.1, 0.2, 0.3]) * 1e-3  # [m]
PX_START        = 2.0e-5                 # Normalized canonical momentum
Y_START         = -0.05e-3               # [m]
PY_START        = 0.0                    # Normalized canonical momentum
DELTA           = 1.0e-4                 # (p - p0) / p0

################################################################################
# Prepare an independent optimized snapshot
################################################################################

########################################
# Load the source and make a deep copy
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
source      = env['demo']

# Default line.copy() deep-copies elements and variables into a NEW environment.
# A shallow copy shares its environment and is unsuitable for this operation.
optimized = source.copy()
print('Independent environments:', source.env is not optimized.env)
# The model supplies a 1500 m electron ring at reference p0*c = 1 GeV.
# RF voltage and radiation are off. Both copies start with the same strengths.

########################################
# Optimize the copied configuration for tracking
########################################
# Optimization disables expressions for the ENTIRE environment owning its line.
# Keep the original environment independent so its strength knobs remain live.
# The public method requires an existing tracker and rebuilds it as necessary.
optimized.build_tracker()
optimized.optimize_for_tracking()
print('Expression managers active (source/optimized):',
      source.ref_manager is not None, optimized.ref_manager is not None)
print('Element counts before/after:',
      len(source.element_names), len(optimized.element_names))
print('IP marker in source/optimized element sequence:',
      'ip.1' in source.element_names, 'ip.1' in optimized.element_names)
print('Source BPM Device planes:')
source.get_table().rows['bpm[.].*'].cols['name element_type'].show(
    maxwidth = 'full')
print('Remaining optimized BPM names:',
      optimized.get_table().rows['bpm[.].*'].name)
print('Multipole counts before/after:',
      len(source.get_table().rows.match(element_type = 'Multipole')),
      len(optimized.get_table().rows.match(element_type = 'Multipole')))
print('First arc quadrupole index before/after:',
      source.element_names.index('arc.s1.c1.qf.in'),
      optimized.element_names.index('arc.s1.c1.qf.in'))
print('Circumferences [m]:', source.get_length(), optimized.get_length())

# Default optimization removes markers and inactive multipoles; it replaces
# passive Devices with drifts, which may merge. Named BPM observation planes
# and element indices need not survive.

# Device survey misalignments are discarded; this uses the nominal aligned
# model. Removed zero multipoles cannot subsequently be powered through their
# original circuits.

################################################################################
# Track identical particles through both configurations
################################################################################

########################################
# Track the source configuration
########################################
initial = source.build_particles(
    x       = X_START,
    px      = PX_START,
    y       = Y_START,
    py      = PY_START,
    delta   = DELTA,
)
source_particles = initial.copy()
source.track(source_particles, num_turns = N_TURNS)

########################################
# Track the optimized snapshot
########################################
optimized_particles = initial.copy()
optimized.track(optimized_particles, num_turns = N_TURNS)

################################################################################
# Inspect physical agreement and expression ownership
################################################################################

########################################
# Compare coordinates at the same ring entrance
########################################
print('Final particle states (source/optimized):')
print(source_particles.state, optimized_particles.state)
print('Coordinate    Maximum absolute difference    Unit')
for coordinate in ['x', 'px', 'y', 'py', 'zeta', 'delta']:
    difference  = getattr(optimized_particles, coordinate) - getattr(
        source_particles, coordinate)
    unit        = 'm' if coordinate in ['x', 'y', 'zeta'] else 'dimensionless'
    print(f'{coordinate:10s} {np.max(np.abs(difference)):27.12e}    {unit}')

# This checks this magnetic model and sample, not arbitrary devices, collective
# tracking, or a hardware performance claim. No timing benchmark is performed.

########################################
# Change only the original live strength knob
########################################
snapshot_k1     = optimized['arc.s1.c1.qf.in'].k1
original_kqf    = env['kqf']
env['kqf']      = KQF_FACTOR * original_kqf
print('Source first arc quadrupole k1 after source knob change [m^-2]:',
      source['arc.s1.c1.qf.in'].k1)
print('Optimized first arc quadrupole k1 before/after [m^-2]:',
      snapshot_k1, optimized['arc.s1.c1.qf.in'].k1)

# The original expressions still respond; the optimized environment contains a
# numeric snapshot. Its element fields remain editable; expression links are
# disabled. The tracking comparison was made BEFORE the source knob change.
