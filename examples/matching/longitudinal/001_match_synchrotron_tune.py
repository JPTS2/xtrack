# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Native RF model and target oscillation tune
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
RF_VOLTAGE      = 1e5                   # Initial cavity voltage [V]
QS_TARGET       = 0.010                 # Positive synchrotron tune [turn^-1]
QS_TOL          = 1.0e-8

########################################
# Voltage control
########################################
VOLTAGE_STEP    = 100.0                 # Finite-difference voltage step [V]
VOLTAGE_LIMITS  = [1.0e4, 3.0e5]         # Numerical search bounds [V]

################################################################################
# Match voltage at fixed RF phase and frequency
################################################################################

########################################
# Retain the committed RF phase and frequency conventions
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
# This is the 1500 m demo ring with a 1 GeV/c electron reference and h=200.
# The public Cavity kick uses abs(q0). With positive slip factor, phase pi
# focuses in this convention despite the electron's q0=-1. Only voltage varies.
env['voltage_rf'] = RF_VOLTAGE
env.set('rf.1', voltage = 'voltage_rf')
before = line.twiss6d(chrom = False)
print('Initial voltage [V] and positive qs:', env['voltage_rf'], before.qs)
print('Cavity phase [rad], frequency [Hz]:',
      line['rf.1'].phase, line['rf.1'].frequency)

########################################
# Match a positive synchrotron tune by changing voltage only
########################################
# Positive voltage bounds stay on this stable focusing branch. They are
# numerical search limits, rather than hardware ratings.
opt = line.match(
    method  = '6d',
    chrom   = False,
    vary    = xt.Vary(
        'voltage_rf',
        step    = VOLTAGE_STEP,
        limits  = VOLTAGE_LIMITS,
    ),
    targets = xt.Target('qs', QS_TARGET, tol = QS_TOL),
)

################################################################################
# Recompute the matched tune and verify the synchronous return
################################################################################

########################################
# Independently solve the final six-dimensional periodic optics
########################################
after = line.twiss6d(chrom = False)
opt.target_status()
print(f'Matched voltage: {env["voltage_rf"]:.6f} V')
print(f'Recomputed qs: {after.qs:.10f}; residual: {after.qs - QS_TARGET:+.3e}')

# qs is positive, while the signed longitudinal phase muzeta decreases around
# this above-transition ring. The two values have different sign conventions.
print('Signed longitudinal phase advance [turns]:',
      after.muzeta[-1] - after.muzeta[0])

########################################
# Track an independent copy of the synchronous particle for one turn
########################################
synchronous = after.particle_on_co.copy()
returned    = synchronous.copy()
line.track(returned, num_turns = 1)
print('Synchronous return: positions [m], normalized momenta dimensionless')
for coordinate in ['x', 'px', 'y', 'py', 'zeta', 'pzeta']:
    difference = getattr(returned, coordinate)[0] - getattr(synchronous,
                                                           coordinate)[0]
    print(f'{coordinate:6s} one-turn change: {difference:+.3e}')

# pzeta=(E-E0)/(beta0*p0*c); zeta=s-beta0*c*t at the reference clock origin.
# Twiss gives the local small-amplitude tune. Sinusoidal RF does not imply an
# amplitude-independent synchrotron tune or characterize bucket acceptance.
