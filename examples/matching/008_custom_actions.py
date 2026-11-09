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
