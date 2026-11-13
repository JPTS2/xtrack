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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
RF_VOLTAGE      = 1e5                   # Cavity voltage [V]
BPM_OBSERVATION = 'bpm.s1.d'            # Nonzero horizontal dispersion
KNOWN_TRIM      = 1e-7                  # Synthetic relative RF-frequency error
FREQUENCY_STEP  = 1e-8                  # Relative-frequency response step
ORBIT_TOL       = 1e-9                  # Position target tolerance [m]
SEARCH_TURNS    = 4

################################################################################
# Give the static RF system one laboratory-time origin
################################################################################

########################################
# Load the fixed-reference ring and define two independent controls
########################################
repo_root                       = Path(__file__).resolve().parents[3]
env                             = xt.load(repo_root / LATTICE_FILE)
line                            = env['demo']
p0c                             = float(line.particle_ref.p0c[0])
beta0                           = float(line.particle_ref.beta0[0])
period0                         = line.get_length() / (beta0 * c)
frequency0                      = float(line['rf.1'].frequency)
harmonic                        = round(frequency0 * period0)
cavity_s                        = float(line.get_table()['s', 'rf.1'])
cavity_time                     = cavity_s / (beta0 * c)
env['voltage_rf']               = RF_VOLTAGE
env['rf_relative_frequency']    = 0.0
env['rf_phase_trim']            = 0.0
# The native 1500 m ring has a 1 GeV/c electron reference and no radiation.
# Relative-time RF alone would not select an off-reference revolution period.
# absolute_time uses phase at lab t=0. Subtract omega*s_rf/(beta0*c)
# at the actual cavity location s_rf=250 m to retain reference phase pi.
env.set(
    'rf.1',
    absolute_time   = True,
    frequency       = f'{frequency0}*(1+rf_relative_frequency)',
    phase           = (
        f'{np.pi}-2*{np.pi}*{frequency0}*(1+rf_relative_frequency)'
        f'*{cavity_time}+rf_phase_trim'
    ),
)

########################################
# Compute the initial periodic state with a variable revolution period
########################################
reference = line.twiss(
    method                      = '6d',
    search_for_t_rev            = True,
    num_turns_search_t_rev      = SEARCH_TURNS,
)
# Positive entrance zeta means earlier arrival: Delta t=-zeta/(beta0*c).
# This is a static equilibrium: p0 stays fixed and the synchronous particle has
# the same momentum after each revolution. It is not an acceleration ramp.

################################################################################
# Generate a synthetic dispersive orbit measurement
################################################################################
env['rf_relative_frequency']    = KNOWN_TRIM
signal                          = line.twiss(
    method                  = '6d',
    search_for_t_rev        = True,
    num_turns_search_t_rev   = SEARCH_TURNS,
)

################################################################################
# Infer a frequency trim from a horizontal orbit measurement
################################################################################

########################################
# Generate a known signal and forget its frequency setting
########################################
# This synthetic measurement has no noise or alignment errors. Keep the RF
# phase fixed, so one dispersive position determines the one unknown trim.
# A zero-dispersion BPM would not identify its sign near the reference orbit.
known_trim                      = KNOWN_TRIM
measured_x                      = signal['x', BPM_OBSERVATION]
env['rf_relative_frequency']    = 0.0
print('Synthetic BPM x [m]:', BPM_OBSERVATION, measured_x)
print('Reference horizontal dispersion at the BPM [m]:',
      reference['dx', BPM_OBSERVATION])

########################################
# Fit the actual periodic orbit using the frequency control
########################################
inverse     = line.match(
    method                      = '6d',
    search_for_t_rev            = True,
    num_turns_search_t_rev       = SEARCH_TURNS,
    vary                        = [
        xt.Vary('rf_relative_frequency', step = FREQUENCY_STEP),
    ],
    targets                     = [
        xt.Target(
            'x',
            at      = BPM_OBSERVATION,
            value   = measured_x,
            tol     = ORBIT_TOL,
        ),
    ],
)
recovered   = line.twiss(
    method                      = '6d',
    search_for_t_rev            = True,
    num_turns_search_t_rev       = SEARCH_TURNS,
)
print('Known / inferred relative frequency trims:',
      KNOWN_TRIM, env['rf_relative_frequency'])
print('Recovered BPM residual [m]:',
      recovered['x', BPM_OBSERVATION] - measured_x)
print('Known / recovered synchronous momentum offsets:',
      signal.delta[0], recovered.delta[0])
