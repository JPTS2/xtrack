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
# Ring and its installed thick-skew controls
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
SKEW_CONTROLS   = ['ks_s2', 'ks_s5']     # k1s on existing 1 m quads [m^-2]
K1S_STEP        = 1e-6                  # Finite-difference step [m^-2]
K1S_LIMITS      = (-0.005, 0.005)        # Numerical bounds, not hardware limits

########################################
# Two native coupling calibrations and small combinations
########################################
CALIBRATION     = 1e-3                  # Native coupling coefficient scale
COUPLING_TOL    = 1e-9
CASE_FACTORS    = [(0, 0), (1, 0), (0, 1), (0.5, 0.5), (-0.5, 0.5)]

########################################
# One fixed error on a controlled magnet
########################################
ERROR_ELEMENT   = 'arc.s2.c2.qd'
ERROR_K1S       = 1e-5                  # Additive skew-gradient error [m^-2]

################################################################################
# Calibrate two knobs using native coupled Twiss results
################################################################################

########################################
# Load a fresh ring and retain its nominal coupled-optics reference
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
reference   = line.twiss4d(chrom = False)

# The 1 GeV/c electron ring has RF voltage zero and radiation off.
# ks_s2 and ks_s5 already drive arc.s2.c2.qd and arc.s5.c2.qd, respectively.

########################################
# Match real and imaginary components in the fixed entrance convention
########################################
# These are native Twiss closest-tune-approach coefficients, derived from the
# normal modes. Their phase refers to the fixed line entrance and mode branch;
# they are not an exact thin-source resonance-driving-term sum.

# Each endpoint is calibrated with the other knob zero. These are controls
# of matched endpoints, not globally independent real/imaginary coordinates.

# Both generated knobs add expressions to the same installed skew controls.
for knob, real_target, imag_target in [
    ('coupling_real', CALIBRATION, 0.0),
    ('coupling_imag', 0.0, CALIBRATION),
]:
    opt = line.match_knob(
        knob_name           = knob,
        knob_value_start    = 0.0,
        knob_value_end      = CALIBRATION,
        run                 = False,
        method              = '4d',
        chrom               = False,
        vary                = xt.VaryList(
            SKEW_CONTROLS,
            step    = K1S_STEP,
            limits  = K1S_LIMITS),
        targets             = xt.TargetSet(
            c_minus_re_0    = real_target,
            c_minus_im_0    = imag_target,
            tol             = COUPLING_TOL),
    )
    opt.solve()
    opt.generate_knob()

# generate_knob binds the matched changes and returns its knob to zero.

################################################################################
# Recompute actual coupling at the calibrated endpoints and combinations
################################################################################
print('Requested real/imag; native real/imag; native c_minus; maximum error:')
results = []
for real_factor, imag_factor in CASE_FACTORS:
    requested_re    = CALIBRATION * real_factor
    requested_im    = CALIBRATION * imag_factor

    env['coupling_real']    = requested_re
    env['coupling_imag']    = requested_im

    tw          = line.twiss4d(chrom = False)
    measured    = np.array([tw.c_minus_re_0, tw.c_minus_im_0])
    error       = np.max(np.abs(measured - [requested_re, requested_im]))
    print(f'{requested_re:+.6e} {requested_im:+.6e}  '
          f'{measured[0]:+.6e} {measured[1]:+.6e}  '
          f'{tw.c_minus:.6e} {error:.3e}')
    results.append((requested_re, requested_im, measured, tw.c_minus, error))

# The gradients depend linearly on the knobs; coupled optics need not. The
# endpoint tolerance does not guarantee the same error at other combinations.
# Here the mixed cases visibly depart from their requested real/imag values.

################################################################################
# Use the calibrated knobs to cancel a fixed skew error
################################################################################

########################################
# Add an independent error term while retaining the original control
########################################
env['coupling_real']    = 0.0
env['coupling_imag']    = 0.0
env['skew_error']       = ERROR_K1S
env.set(ERROR_ELEMENT, k1s = 'ks_s2 + skew_error')
perturbed   = line.twiss4d(chrom = False)

########################################
# Correct the measured native coupling without varying the error
########################################
correction  = line.match(
    method  = '4d',
    chrom   = False,
    vary    = xt.VaryList(
        ['coupling_real', 'coupling_imag'],
        step    = CALIBRATION / 100),
    targets = xt.TargetSet(
        c_minus_re_0    = reference.c_minus_re_0,
        c_minus_im_0    = reference.c_minus_im_0,
        tol             = COUPLING_TOL),
)
corrected   = line.twiss4d(chrom = False)
print('Native c_minus before error / with error / after correction:',
      reference.c_minus, perturbed.c_minus, corrected.c_minus)
print('Correction knob values:', env['coupling_real'], env['coupling_imag'])
print('Fixed error / corrected net skew gradient [m^-2]:',
      env['skew_error'], line[ERROR_ELEMENT].k1s)

# The fixed error is at one of the actuated magnets: this is an ideal
# noiseless cancellation, not identification of an unknown error pattern.

# Larger settings, changed optics or a different phase origin require fresh
# calibration; these knobs do not guarantee global restoration of every optic.
