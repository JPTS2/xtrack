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
LINE_NAME       = 'demo'
RF_VOLTAGE      = 1e5                    # Thin-cavity voltage [V]

################################################################################
# Load a ring with stable longitudinal focusing
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
env['voltage_rf'] = RF_VOLTAGE

################################################################################
# Solve periodic 6D optics and inspect the synchronous orbit
################################################################################

########################################
# Find the synchronous orbit and three linear modes
########################################
# Unlike the RF-free 4D examples, delta and zeta are solved as part of the
# periodic orbit. The model has one thin sinusoidal cavity and no radiation.
tw = line.twiss6d()
print('Tunes (mode 1, mode 2, synchrotron):', tw.qx, tw.qy, tw.qs)
print('Slip factor:', tw.slip_factor)
print('Longitudinal beta at the entrance [m]:', tw.bets0)

########################################
# Inspect longitudinal coordinates at the boundaries
########################################
# zeta is in metres; pzeta = (E-E0)/(beta0*p0*c) for the reference species.
# Positive zeta means earlier arrival at a fixed s and reference clock time.
# ptau = beta0*pzeta. muzeta is signed phase in turns: it decreases around this
# above-transition ring, while qs reports the positive oscillation tune.
tw.rows[['arc.s1.c1.qf.in', 'rf.1', '_end_point']].cols[
    'name s zeta delta ptau muzeta'].show(maxwidth = 'full')

################################################################################
# Track one turn from the synchronous particle
################################################################################

########################################
# Keep the initial fixed point and track an independent copy
########################################
initial = tw.particle_on_co.copy()
tracked = initial.copy()
line.track(tracked, num_turns = 1)

########################################
# Inspect return in all six coordinates
########################################
coordinates = ['x', 'px', 'y', 'py', 'zeta', 'pzeta']
before      = np.array([getattr(initial, name)[0] for name in coordinates])
after       = np.array([getattr(tracked, name)[0] for name in coordinates])
xt.Table(
    data    = {
        'coordinate': np.array(coordinates),
        'before':     before,
        'after':      after,
        'difference': after - before},
    index   = 'coordinate').show(maxwidth = 'full')

# Positions and zeta are in metres; the normalized momenta are dimensionless.
# Twiss is the linearization around this fixed point; the RF tracking remains
# sinusoidal, so finite-amplitude synchrotron motion can have a different tune.
