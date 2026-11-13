# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################
P0C             = 1e9                  # Proton reference p0*c [eV]
LENGTH          = 100.0                # [m]
RF_FREQUENCY    = 100e6                # [Hz]
RF_TIMES        = [0.0, 1e-3, 2e-3]    # [s]
TOTAL_VOLTAGES  = [20e3, 21e3, 20e3]   # [V]
N_CAVITIES      = 3

################################################################################
# Bind a clock function to a native RF array entry
################################################################################

########################################
# Construct the map and its shared voltage programme
########################################
env = xt.Environment()
env.functions['voltage_programme'] = xt.FunctionPieceWiseLinear(
    x = RF_TIMES, y = TOTAL_VOLTAGES)
env.elements['segment'] = xt.LineSegmentMap(
    length                      = LENGTH,
    qx                          = 0.31,
    qy                          = 0.32,
    betx                        = 1.0,
    bety                        = 1.0,
    longitudinal_mode           = 'nonlinear',
    momentum_compaction_factor  = 0.01,
    voltage_rf                  = [0.0],
    frequency_rf                = [RF_FREQUENCY],
    phase_rf                    = [np.pi],
)
line = env.new_line(components = ['segment'])
line.set_particle_ref('proton', p0c = P0C)
env.ref['segment'].voltage_rf[0] = (
    env.functions['voltage_programme'](env.ref['t_turn_s']) / N_CAVITIES)

########################################
# Inspect expression and interpolated values
########################################
print('Array-entry expression:')
env.ref['segment'].voltage_rf[0].xdeps.info()
for time_s in [0.0, 0.5e-3, 1e-3, 1.5e-3, 2e-3]:
    env['t_turn_s'] = time_s
    print('Time [s], one RF-entry voltage [V]:',
          time_s, line['segment'].voltage_rf[0])
# One RF entry receives one third of the programmed total. No three-cavity
# layout is represented. This inspects a live dependency only; it does not
# track an accelerating beam,
# establish an RF bucket, or replace enabling time-dependent tracking updates.
