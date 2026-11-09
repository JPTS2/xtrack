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
# Ring and integrated skew-quadrupole controls
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
SKEW_SITES      = ['cor.s1', 'cor.ip.2']  # Existing thin multipoles
SKEW_STEP       = 1e-6                  # Integrated skew strength step [m^-1]

########################################
# Reference-phasor knob calibration and combinations
########################################
CALIBRATION     = 1e-4                  # Reference-phasor scale
COUPLING_TOL    = 1e-10
CASE_FACTORS    = [(0, 0), (1, 0), (0, 1), (0.5, 0.5), (-0.5, 0.5)]

################################################################################
# Define a coupling response in a fixed uncoupled reference phase convention
################################################################################

########################################
# Bind skew-quadrupole components on two existing thin multipoles
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
skew_knobs = []
for index, name in enumerate(SKEW_SITES):
    knob    = f'skew_strength_{index}'
    env[knob] = 0.0
    env.set(name, ksl = [0, knob])
    skew_knobs.append(knob)

# The 1500 m ring has a 1 GeV/c electron reference, with RF voltage zero and
# radiation off.
reference = line.twiss4d()

########################################
# Measure a first-order phasor using the saved uncoupled optics
########################################
# C_ref = sum(k1sl*sqrt(betx*bety)*exp(i*2*pi*(mux-muy)))/(2*pi).
# Only these two thin elements have nonzero k1sl; no thick-skew quadrature
# is used.

# Its phase is referenced to the fixed ring entrance. The weights remain
# frozen while strengths vary. This weak-coupling proxy neglects changes in
# optics and other resonance terms; it is not the exact Twiss complex c_minus.
class ReferenceCoupling(xt.Action):
    def __init__(self, line, reference):
        self.line = line
        self.weights = (
            np.sqrt(reference.betx * reference.bety)
            * np.exp(2j * np.pi * (reference.mux - reference.muy)) / (2 * np.pi)
        )

    def run(self):
        strengths   = self.line.get_table(attr = True).k1sl
        coefficient = np.sum(strengths * self.weights)
        return {'real': coefficient.real, 'imag': coefficient.imag}

coupling = ReferenceCoupling(line, reference)

################################################################################
# Generate two independently calibrated reference-phasor knobs
################################################################################
# Both generated knobs add expression terms to the same physical skew controls.
# They superpose in the fixed reference phasor, not necessarily in exact optics.
for knob, re_target, im_target in [
    ('coupling_real', CALIBRATION, 0.0),
    ('coupling_imag', 0.0, CALIBRATION),
]:
    opt = line.match_knob(
        knob_name           = knob,
        knob_value_start    = 0.0,
        knob_value_end      = CALIBRATION,
        run                 = False,
        vary                = xt.VaryList(skew_knobs, step = SKEW_STEP),
        targets             = [
            coupling.target('real', value = re_target, tol = COUPLING_TOL),
            coupling.target('imag', value = im_target, tol = COUPLING_TOL),
        ],
    )
    opt.solve()
    opt.generate_knob()

################################################################################
# Test combinations and compare with actual coupled periodic optics
################################################################################
# Twiss c_minus is its closest-tune-approach estimate. Neither its magnitude nor
# its entrance-phase real/imaginary components are identical to C_ref. In this
# example the discrepancy is visible even when the reference response is exact.
print('Requested real/imag; reference real/imag; |C_ref|; Twiss c_minus:')
results = []
for real_factor, imag_factor in CASE_FACTORS:
    requested_re    = CALIBRATION * real_factor
    requested_im    = CALIBRATION * imag_factor
    env['coupling_real'] = requested_re
    env['coupling_imag'] = requested_im
    measured        = coupling.run()
    tw              = line.twiss4d()
    magnitude       = np.hypot(measured['real'], measured['imag'])
    print(f'{requested_re:+.6e} {requested_im:+.6e}  '
          f'{measured["real"]:+.6e} {measured["imag"]:+.6e}  '
          f'{magnitude:.6e} {tw.c_minus:.6e}')
    results.append((requested_re, requested_im, measured, tw.c_minus))

# Exact calibration of this proxy is not an exact decoupling or stability
# guarantee. Larger settings need a fresh coupled-optics response calibration.
