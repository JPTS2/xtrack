# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

from scipy.constants import c
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Native ring and bounded focusing-family ramp
########################################
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
KQF_FACTOR      = 1.005                 # Final strength / initial strength
RAMP_TURNS      = 6                     # Used to define the duration in seconds
N_TURNS         = 10

########################################
# Independent clock origin, turn label and particle launch
########################################
START_TIME      = 2.0e-6                # Initial reference clock time [s]
START_TURN      = 20                    # Independent integer turn label
X_START         = 1.0e-3                # [m]
DELTA_START     = 1.0e-3                # (p-p0)/p0

################################################################################
# Define a machine function of time in seconds
################################################################################

########################################
# Load the ring and calculate its reference revolution period
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['ring']
# The model supplies a 1 GeV/c electron reference, without RF or radiation.
# There is no EnergyProgram: p0 and the nominal clock increment stay fixed.
beta0       = float(line.particle_ref.beta0[0])
period      = line.get_length() / (beta0 * c)
kqf_start   = float(env['kqf'])
kqf_end     = KQF_FACTOR * kqf_start

########################################
# Connect a piecewise-linear function to the live family strength
########################################
# Function arguments are seconds, not at_turn. The repeated endpoint values
# hold the initial and final strengths before and after the linear ramp.
ramp_times  = [0.0, START_TIME, START_TIME + RAMP_TURNS * period,
               START_TIME + N_TURNS * period]
ramp_values = [kqf_start, kqf_start, kqf_end, kqf_end]
env.functions['qf_ramp'] = xt.FunctionPieceWiseLinear(
    x   = ramp_times,
    y   = ramp_values,
)
env['kqf'] = env.functions['qf_ramp'](env.ref['t_turn_s'])

# The existing quadrupole expressions already depend on kqf.

################################################################################
# Track with updates at each turn entrance
################################################################################

########################################
# Set the particle clock and enable automatic machine updates
########################################
particles = line.build_particles(x = X_START, delta = DELTA_START)
particles.time_s = START_TIME
particles.at_turn = START_TURN
line.enable_time_dependent_vars = True
line.dt_update_time_dependent_vars = None
# None refreshes at every turn entrance. A finite update interval is specified
# in seconds and can hold settings between updates. This is not continuous
# variation of the magnet strength within a tracked turn.

########################################
# Sample the settings actually applied during each tracked turn
########################################
samples = []
for _ in range(N_TURNS):
    turn            = int(particles.at_turn[0])
    entrance_clock  = float(particles.time_s)
    line.track(particles, num_turns = 1)
    samples.append((turn, entrance_clock, env['t_turn_s'], env['kqf'],
                    line['qf.1'].k1))

################################################################################
# Distinguish turn labels, the reference clock and the last applied setting
################################################################################

########################################
# Inspect the clock-to-field expression chain
########################################
print(f'Reference revolution period: {period:.12e} s')
print('Entrance clocks [s] and strengths [m^-2]:')
print(f'{"turn":>5} {"particle time_s":>18} {"applied t_turn_s":>18}'
      f' {"kqf":>14} {"qf.1 k1":>14}')
for turn, clock, applied, knob, strength in samples:
    print(f'{turn:5d} {clock:18.10e} {applied:18.10e}'
          f' {knob:14.10f} {strength:14.10f}')

########################################
# Compare the next entrance clock with the last machine update
########################################
print('After tracking, next-entrance particle time_s [s]:', particles.time_s)
print('Last applied machine t_turn_s [s]:', env['t_turn_s'])
print('Their difference [s]:', particles.time_s - env['t_turn_s'])
print('Final turn label:', particles.at_turn, 'state:', particles.state)
print('Final x [m] and delta:', particles.x, particles.delta)

# time_s is a reference clock, not this off-momentum particle's arrival time.
# At the entrance s=0, arrival time is time_s-zeta/(beta0*c). Its integer turn
# label can have a different origin.
