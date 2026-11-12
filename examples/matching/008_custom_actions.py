# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path
from time import perf_counter

import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Model and family perturbation
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
KQF_FACTOR      = 1.0002
KQD_FACTOR      = 0.9998

########################################
# Numerical matching settings
########################################
K1_STEP         = 1.0e-5                # Finite-difference step [m^-2]
KQF_LIMITS      = [0.06, 0.08]             # Numerical bounds [m^-2]
KQD_LIMITS      = [-0.08, -0.06]           # Numerical bounds [m^-2]
BETA_TOL        = 1.0e-8                # Average-beta target tolerance [m]

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
# Define an action returning approximate circumference-averaged beta functions
################################################################################
# This left-endpoint quadrature weights entrance beta by each element length.
# It approximates (1/C)*integral(beta ds), in metres. Zero-length markers have
# zero weight; this is not an unweighted average of the Twiss rows.

# Quadrature bias depends on longitudinal sampling. Reference and trial values
# use the same sampling; neither is claimed to be the exact integral.
class ActionAverageBeta(xt.Action):

    def __init__(self, line):
        self.line = line

    def run(self):
        tw              = self.line.twiss4d()
        lengths         = np.diff(tw.s)
        circumference   = tw.s[-1] - tw.s[0]
        return {
            'average_betx': np.sum(tw.betx[:-1] * lengths) / circumference,
            'average_bety': np.sum(tw.bety[:-1] * lengths) / circumference,
        }

################################################################################
# Measure the clean reference before perturbing the strengths
################################################################################

########################################
# Construct the action on a fresh native ring
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
# The ring has a 1 GeV/c electron reference, RF voltage zero and radiation off.
# The action requests periodic 4D optics at delta=0 on every evaluation.
action      = ActionAverageBeta(line)
reference   = action.run()

########################################
# Apply a stated perturbation and reevaluate
########################################
env['kqf'] *= KQF_FACTOR
env['kqd'] *= KQD_FACTOR
before = action.run()
print('Reference approximate average beta functions [m]:', reference)
print('Perturbed approximate average beta functions [m]:', before)

################################################################################
# Match named outputs of the custom action
################################################################################
# run() is reevaluated as the optimizer changes the family strengths. Its return
# dictionary supplies the observables named in action.target(...).
# These are numerical search bounds, rather than physical hardware limits.
opt = line.match(
    vary    = [
        xt.Vary('kqf', step = K1_STEP, limits = KQF_LIMITS),
        xt.Vary('kqd', step = K1_STEP, limits = KQD_LIMITS),
    ],
    targets = [
        action.target(
            'average_betx', reference['average_betx'],
            tol = BETA_TOL,
        ),
        action.target(
            'average_bety', reference['average_bety'],
            tol = BETA_TOL,
        ),
    ],
)

################################################################################
# Verify the derived observables and inspect the resulting optics
################################################################################

########################################
# Check a fresh action evaluation against the reference
########################################
opt.target_status()
after = action.run()
for name in reference:
    print(f'{name}: {after[name]:.10f} m; '
          f'reference residual = {after[name] - reference[name]:+.3e} m')

########################################
# Inspect quantities that were not directly targeted
########################################
tw_after = line.twiss4d()
print('Final tunes:', tw_after.qx, tw_after.qy)
for name in ['kqf', 'kqd']:
    print(name, '[m^-2]:', env[name])

# Matching these two averages is not a general constraint on beta beating,
# peak beam size, dispersion or stability away from this operating point.

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
tracking_env    = xt.load(repo_root / LATTICE_FILE)
tracking_line   = tracking_env['demo']
# The native real-coordinate tune estimator folds fractional tunes above 0.5.
# Keep both fractions below 0.5 so its slopes represent dQ/dJ directly.
tracking_line.match(
    method  = '4d',
    vary    = xt.VaryList(['kqf', 'kqd'], step = K1_STEP),
    targets = [
        xt.Target('qx', QX_TRACK, tol = 1.0e-8),
        xt.Target('qy', QY_TRACK, tol = 1.0e-8),
    ],
)
tracking_env['k3lf'] = K3L_INITIAL
tracking_env['k3ld'] = K3L_INITIAL
tracked_action      = ActionTrackedDetuning(tracking_line, NUM_TURNS)

########################################
# Fit self-detuning slopes with the two octupole controls
########################################
# Each action evaluation tracks four particles. These are finite-amplitude
# slopes between 1 and 2 sigma, not an exact zero-action normal form.
start_time = perf_counter()
tracked_opt = tracking_line.match(
    vary    = xt.VaryList(
        ['k3lf', 'k3ld'], step = K3L_STEP, limits = K3L_LIMITS),
    targets = [
        tracked_action.target('det_xx', DETUNING_TARGET, tol = DETUNING_TOL),
        tracked_action.target('det_yy', DETUNING_TARGET, tol = DETUNING_TOL),
    ],
)
tracked_opt.target_status()
print('Tracking-derived match time [s]:', perf_counter() - start_time)
for name in ['k3lf', 'k3ld']:
    print(name, '[m^-3]:', tracking_env[name])

################################################################################
# Independently remeasure with more turns and smaller launches
################################################################################
# Double the record and reduce the scanned amplitudes to 0.8 and 1.6 sigma.
# Agreement within the stated tolerance checks record length and amplitude bias.
check_action = ActionTrackedDetuning(
    line        = tracking_line,
    num_turns   = CHECK_TURNS,
    amplitude   = CHECK_AMPLITUDE,
)
check_detuning = check_action.run()
for name in ['det_xx', 'det_yy']:
    print(f'{name}: {check_detuning[name]:.3f} m^-1; '
          f'independent target residual = '
          f'{check_detuning[name] - DETUNING_TARGET:+.3f} m^-1')
