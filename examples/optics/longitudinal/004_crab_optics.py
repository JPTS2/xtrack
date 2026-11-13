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
repo_root           = Path(__file__).resolve().parents[3]
env                 = xt.load(repo_root / LATTICE_FILE)
line                = env[LINE_NAME]
env['voltage_rf']   = RF_VOLTAGE

################################################################################
# Measure the response to longitudinal displacement
################################################################################

########################################
# Add a weak crab cavity to the freshly loaded ring
########################################
# This single deflector illustrates synchrobetatron response, not a matched
# crossing scheme. Phase zero gives zero transverse kick at zeta = 0.
line.insert(
    'crab.demo', xt.CrabCavity(
        crab_voltage    = CRAB_VOLTAGE,
        frequency       = line['rf.1'].frequency,
        phase           = 0.0),
    at      = CRAB_AT)
crab_6d = line.twiss6d()
crab_4d = line.twiss4d()

########################################
# Compare with centred fixed-zeta closed-orbit probes
########################################
# In 4D, zeta0 is imposed while the transverse orbit closes. The 6D response
# instead comes from coupled normal modes with longitudinal motion active.
plus        = line.twiss4d(zeta0 = ZETA_STEP)
minus       = line.twiss4d(zeta0 = -ZETA_STEP)
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
line['voltage_rf']  = 0.0
rf_off              = line.twiss4d()
print('RF-off 4D response [1, m^-1]:',
      rf_off['dx_zeta', OBSERVATION], rf_off['dpx_zeta', OBSERVATION])
line['voltage_rf']              = RF_VOLTAGE
line['crab.demo'].crab_voltage  = 0.0
crab_off                        = line.twiss6d()
print('Crab-off 6D response [1, m^-1]:',
      crab_off['dx_zeta', OBSERVATION], crab_off['dpx_zeta', OBSERVATION])

########################################
# Impose the same transverse entrance state at two arrival times
########################################
# Here only zeta changes at the cavity entrance; the downstream orbit need
# not close. This boundary condition differs from a periodic orbit probe.
line['crab.demo'].crab_voltage  = CRAB_VOLTAGE
open_orbits                     = []
for zeta in [-ZETA_STEP, ZETA_STEP]:
    init                        = crab_4d.get_twiss_init('crab.demo')
    init.particle_on_co.zeta    = zeta
    open_orbits.append(line.twiss(
        method  = '4d',
        start   = 'crab.demo',
        end     = OPEN_END,
        init    = init))
open_response   = (open_orbits[1]['x', OPEN_END]
                   - open_orbits[0]['x', OPEN_END]) / (2 * ZETA_STEP)
closed_response = (plus['x', OPEN_END] - minus['x', OPEN_END]) / (2 * ZETA_STEP)
print('Open and periodic dx/dzeta at', OPEN_END, '[1]:',
      open_response, closed_response)
