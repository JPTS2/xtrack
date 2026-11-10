# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
from scipy.constants import c
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Shared electron ring and a small per-turn timing shift
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
RF_VOLTAGE      = 1e5                    # [V], harmonic 200, stable phase pi
SHIFT_ZETA      = 1e-3                   # Subtracted from zeta [m]

########################################
# Independent probes of the TimeDelay element itself
########################################
DELTA           = np.array([-1e-4, 0.0, 1e-4])  # (p-p0)/p0

################################################################################
# Inspect the native coordinate shift before changing the ring
################################################################################

########################################
# Load a fresh powered ring and its synchronous operating point
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
env['voltage_rf'] = RF_VOLTAGE
before      = line.twiss6d(chrom = False)
beta0       = float(line.particle_ref.beta0[0])
length      = line.get_length()

########################################
# Apply the same zeta shift to three different momenta
########################################
initial     = line.build_particles(delta = DELTA)
probes      = initial.copy()
delay       = xt.TimeDelay(shift_zeta = SHIFT_ZETA)
delay.track(probes)
arrival_change  = ((probes.s - probes.zeta) - (initial.s - initial.zeta)) / (
    beta0 * c)
xt.Table(
    data    = {
        'particle_id':          probes.particle_id,
        'delta':                probes.delta,
        'zeta_change_m':        probes.zeta - initial.zeta,
        'arrival_change_ps':    arrival_change * 1e12,
    },
    index   = 'particle_id',
).show(maxwidth = 'full')

# TimeDelay applies zeta_new=zeta_old-shift_zeta. It leaves s, time_s and
# momentum unchanged, giving the SAME arrival delay shift_zeta/(beta0*c) here.
# It is not a physical drift or a momentum-dependent coordinate kick.

################################################################################
# Find the off-momentum RF-synchronous point selected by a per-turn delay
################################################################################

########################################
# Append the zero-length delay and solve the native periodic 6D optics
########################################
line.append('timing.delay', delay)
after   = line.twiss6d(chrom = False)

# Before the extra delay, one-turn slip is approximately -C*eta*delta.
# Closure after subtracting SHIFT_ZETA needs delta=-SHIFT_ZETA/(C*eta).
# The delay changes longitudinal synchronization, without changing geometry.
prediction  = -SHIFT_ZETA / (length * before.slip_factor)
print('Circumference [m]:', line.get_length())
print('Synchronous delta: original / with delay / first-order estimate:',
      before.delta[0], after.delta[0], prediction)
print('Synchronous zeta at entrance / RF cavity [m]:',
      after.zeta[0], after['zeta', 'rf.1'])

########################################
# Follow the delayed synchronous particle for one turn
########################################
synchronous = after.particle_on_co.copy()
returned    = synchronous.copy()
line.track(returned, num_turns = 1)
print('One-turn return: zeta change [m], delta change:',
      returned.zeta[0] - synchronous.zeta[0],
      returned.delta[0] - synchronous.delta[0])
print('Particle state:', returned.state)

# RF voltage is 100 kV by default; radiation and collective forces are off.
# This small delay qualifies one off-momentum operating point, not the bare
# ring's momentum acceptance or a physical path-length modification.
