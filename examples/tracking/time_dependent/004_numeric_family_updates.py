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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
RIPPLE_FRACTION = 0.001                 # Relative amplitude
RIPPLE_SAMPLES  = 5                     # Both ends of one cycle

################################################################################
# Load the expression-linked ring
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
################################################################################
# Update a numeric family snapshot with MultiSetter
################################################################################

########################################
# Copy physical strengths without their expression dependencies
########################################
# MultiSetter writes element buffers directly; it does not update the knob
# graph. Use independent numeric copies here, leaving the live ring intact.
# For the expression-linked ring, change kqf or its time function instead.
qf_names    = list(line.get_table().rows['arc[.]s[1-6][.]c[1-4][.]qf.*'].name)
family      = xt.Line(
    elements        = {name: line.get(name).copy() for name in qf_names},
    particle_ref    = line.particle_ref.copy(),
)
family.build_tracker()
setter      = xt.MultiSetter(family, qf_names, field = 'k1')
baseline    = setter.get_values().copy()

########################################
# Apply one sampled ripple cycle and restore the baseline
########################################
for phase in np.linspace(0.0, 2 * np.pi, RIPPLE_SAMPLES):
    setter.set_values(baseline * (1 + RIPPLE_FRACTION * np.sin(phase)))
    values = setter.get_values()
    print(f'Ripple phase {phase:.3f}: {len(values)} strengths; '
          f'first k1 = {values[0]:.10f} m^-2')
setter.set_values(baseline)
print('Restored family maximum |Delta k1| [m^-2]:',
      np.max(np.abs(setter.get_values() - baseline)))
