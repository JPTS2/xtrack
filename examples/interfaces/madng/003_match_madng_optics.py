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
# Shared uncoupled electron ring, RF and radiation off
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
OBSERVATION     = 'bpm.s1.f'

########################################
# Initial offsets of the existing arc-family trim controls
########################################
DKQF_START      = 2e-5                  # Focusing-family k1 trim [m^-2]
DKQD_START      = -2e-5                 # Defocusing-family k1 trim [m^-2]
K1_STEP         = 1e-7                  # Finite-difference strength step [m^-2]
K1_LIMITS       = (-2e-4, 2e-4)         # Trim bounds [m^-2]
BETA_TOL        = 1e-7                  # Target tolerance [m]

################################################################################
# Prepare a reference from the attached MAD-NG model
################################################################################

########################################
# Load the same reference and expressions used in the optics comparison
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
line.twiss_default['method']    = '4d'
# The model supplies a 1 GeV/c electron reference. Only dkqf and dkqd will
# vary; final-focus settings and other circuits remain fixed.

########################################
# Record nominal mode beta functions at one named BPM
########################################
model       = line.build_madng_model()
reference   = line.madng_twiss(normal_form = False)
# beta11_ng and beta22_ng are MAD-NG mode beta functions in metres.
# In this uncoupled setting they correspond to native betx and bety.

################################################################################
# Match MAD-NG observables through the native Xtrack optimizer
################################################################################

########################################
# Change shared knobs and inspect the perturbed MAD-NG optics
########################################
env['dkqf'] = DKQF_START
env['dkqd'] = DKQD_START
before      = line.madng_twiss(normal_form = False)

########################################
# Restore the named observables using targets owned by the MAD-NG table
########################################
# reference.target retains the MAD-NG computation that produced the table.
# Every optimizer evaluation reruns that backend at the current knob settings.
optimizer   = line.match(
    vary    = xt.VaryList(['dkqf', 'dkqd'], step = K1_STEP, limits = K1_LIMITS),
    targets = reference.target(
        ['beta11_ng', 'beta22_ng'],
        at      = OBSERVATION,
        tol     = BETA_TOL,
    ),
)
optimizer.target_status()

################################################################################
# Recompute final optics independently of the optimizer's last evaluation
################################################################################

########################################
# Compare a fresh MAD-NG calculation with native Xtrack optics
########################################
checked     = line.madng_twiss(normal_form = False)
native      = line.twiss4d(chrom = False)
requested   = np.array([
    reference['beta11_ng', OBSERVATION], reference['beta22_ng', OBSERVATION]])
ng_after    = np.array([
    checked['beta11_ng', OBSERVATION], checked['beta22_ng', OBSERVATION]])
xt.Table(
    data    = {
        'quantity':      np.array(['beta_x [m]', 'beta_y [m]']),
        'requested':     requested,
        'MADNG_before':  np.array([
            before['beta11_ng', OBSERVATION],
            before['beta22_ng', OBSERVATION]]),
        'MADNG_after':   ng_after,
        'Xtrack_after':  np.array([
            native['betx', OBSERVATION], native['bety', OBSERVATION]]),
        'NG_residual':   ng_after - requested},
    index   = 'quantity').show(maxwidth = 'full', digits = 10)
# These targets restore two local beta functions. They do not specify tunes,
# global beta beating or the exact original strengths. The native comparison
# measures agreement of the exported optics for this particular setting.

########################################
# Inspect the resulting shared controls and release the external backend
########################################
print('Matched dkqf / dkqd [m^-2]:', env['dkqf'], env['dkqd'])
print('Xtrack minus MAD-NG beta functions at the BPM [m]:',
      native['betx', OBSERVATION] - ng_after[0],
      native['bety', OBSERVATION] - ng_after[1])
print('Final native tunes Qx / Qy:', native.qx, native.qy)
model.close()
