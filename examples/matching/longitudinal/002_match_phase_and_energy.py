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
# Static synchronous state at the ring entrance
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
RF_VOLTAGE      = 1e5                   # Initial cavity voltage [V]
DELTA_TARGET    = 1e-4                  # (p-p0)/p0, not relative total energy
ZETA_TARGET     = 3e-3                  # zeta=s-beta0*c*t at entrance [m]
BPM_OBSERVATION = 'bpm.s1.d'            # Nonzero horizontal dispersion
ORBIT_TOL       = 1e-9                  # Horizontal position tolerance [m]

########################################
# Matching and revolution-period search precision
########################################
FREQUENCY_STEP  = 1e-7                  # Relative-frequency step
PHASE_STEP      = 1e-5                  # Phase finite-difference step [rad]
DELTA_TOL       = 1e-8
ZETA_TOL        = 1e-7                  # [m]
SEARCH_TURNS    = 4

################################################################################
# Give the static RF system one laboratory-time origin
################################################################################

########################################
# Load the fixed-reference ring and define two independent controls
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
p0c         = float(line.particle_ref.p0c[0])
beta0       = float(line.particle_ref.beta0[0])
period0     = line.get_length() / (beta0 * c)
frequency0  = float(line['rf.1'].frequency)
harmonic    = round(frequency0 * period0)
cavity_s    = float(line.get_table()['s', 'rf.1'])
cavity_time = cavity_s / (beta0 * c)
env['voltage_rf'] = RF_VOLTAGE
env['rf_relative_frequency'] = 0.0
env['rf_phase_trim'] = 0.0
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
# Match synchronous momentum and arrival coordinate using frequency and phase
################################################################################

########################################
# Start from the linear slip-factor estimate
########################################
# Delta f/f approximately equals -eta*delta for small offsets.
# A uniform-slip estimate seeds the phase; the match supplies the final value.
env['rf_relative_frequency'] = -reference.slip_factor * DELTA_TARGET
frequency_guess         = frequency0 * (1 + env['rf_relative_frequency'])
period_guess            = harmonic / frequency_guess
zeta_at_cavity_guess    = (
    ZETA_TARGET + beta0 * c * (period0 - period_guess)
    * cavity_s / line.get_length()
)
env['rf_phase_trim'] = 2 * np.pi * frequency_guess * zeta_at_cavity_guess / (
    beta0 * c)

########################################
# Solve for the actual six-dimensional synchronous state
########################################
opt = line.match(
    method                      = '6d',
    search_for_t_rev            = True,
    num_turns_search_t_rev      = SEARCH_TURNS,
    vary                        = [
        xt.Vary('rf_relative_frequency', step = FREQUENCY_STEP),
        xt.Vary('rf_phase_trim', step = PHASE_STEP),
    ],
    targets                     = [
        xt.Target(
            'delta',
            at      = xt.START,
            value   = DELTA_TARGET,
            tol     = DELTA_TOL,
        ),
        xt.Target(
            'zeta',
            at      = xt.START,
            value   = ZETA_TARGET,
            tol     = ZETA_TOL,
        ),
    ],
)
matched = line.twiss(
    method                      = '6d',
    search_for_t_rev            = True,
    num_turns_search_t_rev      = SEARCH_TURNS,
)
print('Requested/matched entrance delta:', DELTA_TARGET, matched.delta[0])
print('Requested/matched entrance zeta [m]:', ZETA_TARGET, matched.zeta[0])
print('Reference p0*c remains [eV]:', line.particle_ref.p0c[0])
print('RF frequency [Hz], phase trim [rad]:',
      line['rf.1'].frequency, env['rf_phase_trim'])
print('RF cycles per synchronous turn; harmonic:',
      line['rf.1'].frequency * matched.t_rev, harmonic)
print('Matched/linearized relative frequency offsets:',
      env['rf_relative_frequency'], -reference.slip_factor * DELTA_TARGET)

################################################################################
# Track one revolution and check energy and all six coordinate changes
################################################################################

########################################
# Check closure in the RF-synchronous frame, not the reference clock
########################################
particle = line.build_particles(
    x       = matched.x[0],
    px      = matched.px[0],
    y       = matched.y[0],
    py      = matched.py[0],
    zeta    = matched.zeta[0],
    delta   = matched.delta[0],
)
initial = particle.copy()
line.track(particle)
expected_zeta_slip = beta0 * c * (period0 - harmonic / line['rf.1'].frequency)
print('One-turn coordinate changes; zeta includes reference-clock slip:')
for coordinate in ['x', 'px', 'y', 'py', 'zeta', 'delta']:
    actual      = float(
        getattr(particle, coordinate)[0] - getattr(initial, coordinate)[0])
    expected    = expected_zeta_slip if coordinate == 'zeta' else 0.0
    print(coordinate, 'actual / expected:', actual, expected)

########################################
# Compare revolution period and total energy independently
########################################
tracked_period  = period0 - float(particle.zeta[0] - initial.zeta[0]) / (
    beta0 * c)
energy_before   = float(initial.energy0[0] + initial.ptau[0] * p0c)
energy_after    = float(particle.energy0[0] + particle.ptau[0] * p0c)
energy_target   = np.sqrt(
    (p0c * (1 + DELTA_TARGET))**2 + float(initial.mass0)**2)
print('Tracked/Twiss/RF revolution periods [s]:',
      tracked_period, matched.t_rev, harmonic / line['rf.1'].frequency)
print('Target/matched total energy [eV]:', energy_target, energy_before)
print('One-turn energy change [eV]:', energy_after - energy_before)
print('Final particle state:', particle.state)

# No energy loss is included; the stable synchronous RF phase has zero net
# energy kick. Numerical matching/closed-orbit tolerances limit these checks.

################################################################################
# Infer a frequency trim from a horizontal orbit measurement
################################################################################

########################################
# Generate a known signal and forget its frequency setting
########################################
# This synthetic measurement has no noise or alignment errors. Keep the RF
# phase fixed, so one dispersive position determines the one unknown trim.
# A zero-dispersion BPM would not identify its sign near the reference orbit.
known_trim = env['rf_relative_frequency']
measured_x = matched['x', BPM_OBSERVATION]
env['rf_relative_frequency'] = 0.0
print('Synthetic BPM x [m]:', BPM_OBSERVATION, measured_x)
print('Reference horizontal dispersion at the BPM [m]:',
      reference['dx', BPM_OBSERVATION])

########################################
# Fit the actual periodic orbit using the frequency control
########################################
inverse = line.match(
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
recovered = line.twiss(
    method                      = '6d',
    search_for_t_rev            = True,
    num_turns_search_t_rev       = SEARCH_TURNS,
)
print('Known / inferred relative frequency trims:',
      known_trim, env['rf_relative_frequency'])
print('Recovered BPM residual [m]:',
      recovered['x', BPM_OBSERVATION] - measured_x)
print('Known / recovered synchronous momentum offsets:',
      matched.delta[0], recovered.delta[0])
