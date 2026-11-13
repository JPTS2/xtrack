# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path
from time import perf_counter

import xtrack as xt


################################################################################
# User parameters
################################################################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
K1_STEP         = 1e-5                  # Gradient response step [m^-2]

########################################
# Tracking-derived detuning targets
########################################
QX_TRACK        = 8.28
QY_TRACK        = 10.19
NEMITT          = 1.0e-6                # Normalized emittance [m]
NUM_TURNS       = 512
CHECK_TURNS     = 1024
CHECK_AMPLITUDE = 0.8                   # Scale relative to matching launches
K3L_INITIAL     = 500.0                 # Integrated octupole strength [m^-3]
K3L_STEP        = 10.0                  # Finite-difference step [m^-3]
K3L_LIMITS      = [0.0, 2000.0]         # Numerical bounds [m^-3]
DETUNING_TARGET = 80000.0               # dQ/dJ [m^-1], geometric J in metres
DETUNING_TOL    = 100.0                 # Slope tolerance [m^-1]

################################################################################
# Define an action that measures magnetic amplitude detuning by tracking
################################################################################
# The native measurement uses NAFF (optional package nafflib) on four probes.
# Geometric action Jx = (x_norm**2 + px_norm**2)*geometric_emittance/2 [m].
# The analogous expression defines Jy.
class ActionTrackedDetuning(xt.Action):

    def __init__(self, line, num_turns, amplitude = 1.0):
        self.line       = line
        self.num_turns  = num_turns
        self.amplitude  = amplitude

    def run(self):
        return self.line.get_amplitude_detuning_coefficients(
            nemitt_x    = NEMITT,
            nemitt_y    = NEMITT,
            num_turns   = self.num_turns,
            a0_sigmas   = 0.1,
            a1_sigmas   = self.amplitude,
            a2_sigmas   = 2.0 * self.amplitude,
        )

################################################################################
# Match two actual octupoles using tracking-derived observables
################################################################################

########################################
# Select a working point with unambiguous real-signal tunes
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
# The native real-coordinate tune estimator folds fractional tunes above 0.5.
# Keep both fractions below 0.5 so its slopes represent dQ/dJ directly.
line.match(
    method  = '4d',
    vary    = xt.VaryList(['kqf', 'kqd'], step = K1_STEP),
    targets = [
        xt.Target('qx', QX_TRACK, tol = 1.0e-8),
        xt.Target('qy', QY_TRACK, tol = 1.0e-8),
    ],
)
env['k3lf'] = K3L_INITIAL
env['k3ld'] = K3L_INITIAL
action      = ActionTrackedDetuning(line, NUM_TURNS)

########################################
# Fit self-detuning slopes with the two octupole controls
########################################
# Each action evaluation tracks four particles. These are finite-amplitude
# slopes between 1 and 2 sigma, not an exact zero-action normal form.
start_time  = perf_counter()
opt         = line.match(
    vary    = xt.VaryList(
        ['k3lf', 'k3ld'], step = K3L_STEP, limits = K3L_LIMITS),
    targets = [
        action.target('det_xx', DETUNING_TARGET, tol = DETUNING_TOL),
        action.target('det_yy', DETUNING_TARGET, tol = DETUNING_TOL),
    ],
)
opt.target_status()
print('Tracking-derived match time [s]:', perf_counter() - start_time)
for name in ['k3lf', 'k3ld']:
    print(name, '[m^-3]:', env[name])

################################################################################
# Independently remeasure with more turns and smaller launches
################################################################################
# Double the record and reduce the scanned amplitudes to 0.8 and 1.6 sigma.
# Agreement within the stated tolerance checks record length and amplitude bias.
check_action    = ActionTrackedDetuning(
    line        = line,
    num_turns   = CHECK_TURNS,
    amplitude   = CHECK_AMPLITUDE,
)
check_detuning  = check_action.run()
for name in ['det_xx', 'det_yy']:
    print(f'{name}: {check_detuning[name]:.3f} m^-1; '
          f'independent target residual = '
          f'{check_detuning[name] - DETUNING_TARGET:+.3f} m^-1')
