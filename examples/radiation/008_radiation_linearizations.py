# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Shared geometry and reference
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
P0C             = 10e9                  # Reference p0*c [eV]

########################################
# Available RF voltage, above the energy loss per turn
########################################
RF_VOLTAGE  = 50e6                  # [V]
HARMONIC    = 200

################################################################################
# Configure one compensated operating point
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
env['p0c']  = P0C
line.set_particle_ref('electron', p0c = P0C)
env['frequency_rf'] = 0.0
env.set('rf.1', harmonic = HARMONIC)
env['voltage_rf']                   = RF_VOLTAGE
context                             = xo.ContextCpu()
context.allow_kernel_compilation    = True
line.build_tracker(_context = context)
line.configure_radiation(model = 'mean')
line.compensate_radiation_energy_loss()
twiss = line.twiss6d(
    chrom = False, radiation_analysis = True, radiation_method = 'full')

################################################################################
# Compare radiation linearization methods on the same compensated fields
################################################################################

########################################
# Recompute linear optics with three native treatments of test-particle kicks
########################################
# full differentiates the actual mean-radiation map. kick_as_co applies the
# closed-orbit kicks to its test particles; scale_as_co uses the closed-orbit
# momentum scaling. These approximations need not give identical optics.
print('Radiation method / Qx / Qy / Qs / maximum relative betx change:')
for method in ['full', 'kick_as_co', 'scale_as_co']:
    optics = line.twiss6d(chrom = False, radiation_method = method)
    print(method, optics.qx, optics.qy, optics.qs,
          np.max(np.abs(optics.betx / twiss.betx - 1)))
# This compares optics only. Equilibrium/damping analysis uses the full mean
# map in 005; selecting a method here does not change later physical tracking.
