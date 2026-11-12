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
# Native ring and longitudinal RF
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
RF_VOLTAGE      = 1e5                    # Thin-cavity voltage [V]

########################################
# Illustrative transverse RF deflector
########################################
CRAB_VOLTAGE    = 1e3                    # [V]
CRAB_AT         = 'bpm.s1.f'
OBSERVATION     = 'ip.1'
ZETA_STEP       = 1e-2                   # Centred orbit probes [m]
OPEN_END        = 'bpm.s2.f'

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

################################################################################
# Measure the response to longitudinal displacement
################################################################################

########################################
# Add a weak crab cavity to an independent copy
########################################
# This single deflector illustrates synchrobetatron response, not a matched
# crossing scheme. Phase zero gives zero transverse kick at zeta = 0.
crab_line   = line.copy()
crab_line.insert(
    'crab.demo', xt.CrabCavity(
        crab_voltage    = CRAB_VOLTAGE,
        frequency       = crab_line['rf.1'].frequency,
        phase           = 0.0),
    at      = CRAB_AT)
crab_6d     = crab_line.twiss6d()
crab_4d     = crab_line.twiss4d()

########################################
# Compare with centred fixed-zeta closed-orbit probes
########################################
# In 4D, zeta0 is imposed while the transverse orbit closes. The 6D response
# instead comes from coupled normal modes with longitudinal motion active.
plus    = crab_line.twiss4d(zeta0 = ZETA_STEP)
minus   = crab_line.twiss4d(zeta0 = -ZETA_STEP)
dx_dzeta    = (
    plus['x', OBSERVATION] - minus['x', OBSERVATION]) / (2 * ZETA_STEP)
dpx_dzeta   = (
    plus['px', OBSERVATION] - minus['px', OBSERVATION]) / (2 * ZETA_STEP)
print('Crab optics at:', OBSERVATION)
print('6D dx_zeta [1], dpx_zeta [m^-1]:',
      crab_6d['dx_zeta', OBSERVATION], crab_6d['dpx_zeta', OBSERVATION])
print('4D dx_zeta [1], dpx_zeta [m^-1]:',
      crab_4d['dx_zeta', OBSERVATION], crab_4d['dpx_zeta', OBSERVATION])
print('Centred 4D orbit differences [1, m^-1]:', dx_dzeta, dpx_dzeta)

################################################################################
# Separate RF settings and open-boundary transport
################################################################################

########################################
# Switch off longitudinal RF, then the crab voltage
########################################
# RF-free optics use 4D; no stable longitudinal mode is requested.
crab_line['voltage_rf'] = 0.0
rf_off = crab_line.twiss4d()
print('RF-off 4D response [1, m^-1]:',
      rf_off['dx_zeta', OBSERVATION], rf_off['dpx_zeta', OBSERVATION])
crab_line['voltage_rf'] = RF_VOLTAGE
crab_line['crab.demo'].crab_voltage = 0.0
crab_off = crab_line.twiss6d()
print('Crab-off 6D response [1, m^-1]:',
      crab_off['dx_zeta', OBSERVATION], crab_off['dpx_zeta', OBSERVATION])

########################################
# Impose the same transverse entrance state at two arrival times
########################################
# Here only zeta changes at the cavity entrance; the downstream orbit need
# not close. This boundary condition differs from a periodic orbit probe.
crab_line['crab.demo'].crab_voltage = CRAB_VOLTAGE
open_orbits = []
for zeta in [-ZETA_STEP, ZETA_STEP]:
    init = crab_4d.get_twiss_init('crab.demo')
    init.particle_on_co.zeta = zeta
    open_orbits.append(crab_line.twiss(
        method  = '4d',
        start   = 'crab.demo',
        end     = OPEN_END,
        init    = init))
open_response   = (open_orbits[1]['x', OPEN_END]
                   - open_orbits[0]['x', OPEN_END]) / (2 * ZETA_STEP)
closed_response = (plus['x', OPEN_END] - minus['x', OPEN_END]) / (2 * ZETA_STEP)
print('Open and periodic dx/dzeta at', OPEN_END, '[1]:',
      open_response, closed_response)
