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
# Ring and desired tune derivatives
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
DQX_TARGET      = 1.0                   # dQx/d(delta), dimensionless
DQY_TARGET      = 2.0                   # dQy/d(delta), dimensionless

########################################
# Sextupole controls and numerical precision
########################################
K2L_STEP    = 1.0e-4                # Integrated k2 variation step [m^-2]
K2L_LIMITS  = [-1.0, 1.0]            # Numerical search bounds [m^-2]
CHROM_TOL   = 1.0e-6
DELTA_STEP  = 2.0e-6                # Chromaticity computation step
DELTA_CHECK = 1.0e-6                # Independent centered difference step

########################################
# Local IP chromatic response
########################################
IP_NAME             = 'ip.1'
IP_B_FRACTION       = 0.9                   # Retain 90% of each IP b_chrom
LOCAL_DELTA         = 2.5e-6                # Chromatic differentiation step
LOCAL_K2L_STEP      = 1.0e-3                # Individual sextupole step [m^-2]
LOCAL_K2L_LIMITS    = [-0.5, 0.5]          # Trim bounds [m^-2]
LOCAL_A_TOL         = 0.005
LOCAL_BX_TOL        = 0.002
LOCAL_BY_TOL        = 0.05
LOCAL_DQ_TOL        = 1.0e-4
LOCAL_SEXTUPOLES    = [
    'arc.s4.c3.sd', 'arc.s4.c3.sf', 'arc.s1.c3.sf',
    'arc.s3.c3.sf', 'arc.s1.c3.sd', 'arc.s2.c3.sd',
]

################################################################################
# Use the two existing normal-sextupole families
################################################################################

########################################
# Load the existing dispersed arc sextupoles
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
# The 1500 m ring supplies a 1 GeV/c electron reference, with RF voltage
# zero and radiation off.
# knl[2] is integrated normal k2 [m^-2], with Delta px=-knl[2]*(x^2-y^2)/2.
# Existing zero-length kicks preserve the reference geometry.
env['k2lf'] = 0.0
env['k2ld'] = 0.0

########################################
# Record the zero-sextupole reference
########################################
before = line.twiss4d(chrom = True, delta_chrom = DELTA_STEP)
print('Before dQ/d(delta):', before.dqx, before.dqy)

################################################################################
# Match chromaticity with the normal quadrupoles held fixed
################################################################################
# Off-momentum particles sample sextupoles through dispersion. The reference p0
# and all quadrupole strengths remain fixed. Bounds are not hardware limits.
opt = line.match(
    method      = '4d',
    chrom       = True,
    delta_chrom = DELTA_STEP,
    vary        = xt.VaryList(
        ['k2lf', 'k2ld'],
        step    = K2L_STEP,
        limits  = K2L_LIMITS,
    ),
    targets     = xt.TargetSet(
        dqx = DQX_TARGET,
        dqy = DQY_TARGET,
        tol = CHROM_TOL,
    ),
)

################################################################################
# Independently recompute the matched momentum dependence
################################################################################

########################################
# Check the API derivatives and the integrated family strengths
########################################
opt.target_status()
after = line.twiss4d(chrom = True, delta_chrom = DELTA_STEP)
print('Matched dQ/d(delta):', after.dqx, after.dqy)
for name in ['k2lf', 'k2ld']:
    print(name, 'integrated k2 [m^-2]:', env[name])

########################################
# Compare with centered differences of separate off-momentum tunes
########################################
plus        = line.twiss4d(delta0 = DELTA_CHECK, chrom = False)
minus       = line.twiss4d(delta0 = -DELTA_CHECK, chrom = False)
dqx_check   = (plus.qx - minus.qx) / (2 * DELTA_CHECK)
dqy_check   = (plus.qy - minus.qy) / (2 * DELTA_CHECK)
print('Independent centered tune derivatives:', dqx_check, dqy_check)
print('Centered/API derivative differences:',
      dqx_check - after.dqx, dqy_check - after.dqy)
print('On-momentum tunes before/after:',
      (before.qx, before.qy), (after.qx, after.qy))

# delta=(p-p0)/p0. Different finite-difference steps need not give identical
# derivatives. This match constrains first-order chromaticity, not higher-order
# momentum dependence, amplitude detuning or dynamic aperture.

################################################################################
# Reduce the local IP beta response while preserving global chromaticity
################################################################################

########################################
# Give six existing sextupoles independent trims
########################################
# These distributed trims continue from the globally corrected ring. Their
# different phases control four local IP quantities and two global derivatives.
local_knobs = []
for name in LOCAL_SEXTUPOLES:
    knob                    = f'local_trim_{name}'
    family                  = 'k2lf' if name.endswith('.sf') else 'k2ld'
    env[knob]               = 0.0
    env.ref[name].knl[2]    = env.ref[family] + env.ref[knob]
    local_knobs.append(knob)

########################################
# Set modest local targets and retain both global tune derivatives
########################################
# b_chrom = (d beta/d delta)/beta; a_chrom = d alpha/d delta - alpha*b_chrom.
# Hold both a values and reduce both b magnitudes by 10%, rather than imposing
# zero chromatic response. This constrains the IP, not the entire ring.
local_before    = line.twiss4d(chrom = True, delta_chrom = LOCAL_DELTA)
half_before     = line.twiss4d(chrom = True, delta_chrom = LOCAL_DELTA / 2)
local_opt       = line.match(
    method      = '4d',
    chrom       = True,
    delta_chrom = LOCAL_DELTA,
    vary        = xt.VaryList(
        local_knobs,
        step    = LOCAL_K2L_STEP,
        limits  = LOCAL_K2L_LIMITS,
    ),
    targets     = [
        xt.Target(
            'ax_chrom',
            at      = IP_NAME,
            value   = local_before['ax_chrom', IP_NAME],
            tol     = LOCAL_A_TOL),
        xt.Target(
            'bx_chrom',
            at      = IP_NAME,
            value   = IP_B_FRACTION * local_before['bx_chrom', IP_NAME],
            tol     = LOCAL_BX_TOL),
        xt.Target(
            'ay_chrom',
            at      = IP_NAME,
            value   = local_before['ay_chrom', IP_NAME],
            tol     = LOCAL_A_TOL),
        xt.Target(
            'by_chrom',
            at      = IP_NAME,
            value   = IP_B_FRACTION * local_before['by_chrom', IP_NAME],
            tol     = LOCAL_BY_TOL),
        xt.Target('dqx', value = local_before.dqx, tol = LOCAL_DQ_TOL),
        xt.Target('dqy', value = local_before.dqy, tol = LOCAL_DQ_TOL),
    ],
)

################################################################################
# Check the local correction with a smaller momentum step
################################################################################

########################################
# Compare local targets and the changes at half the step
########################################
local_opt.target_status()
local_after = line.twiss4d(chrom = True, delta_chrom = LOCAL_DELTA)
half_after  = line.twiss4d(chrom = True, delta_chrom = LOCAL_DELTA / 2)
fields      = ['ax_chrom', 'bx_chrom', 'ay_chrom', 'by_chrom']
fractions   = [1.0, IP_B_FRACTION, 1.0, IP_B_FRACTION]
local_table = xt.Table(
    data    = {
        'quantity':           np.array(fields),
        'before':             np.array([
            local_before[field, IP_NAME] for field in fields]),
        'target':             np.array([
            fraction * local_before[field, IP_NAME]
            for field, fraction in zip(fields, fractions)]),
        'matched':            np.array([
            local_after[field, IP_NAME] for field in fields]),
        'half_step_residual': np.array([
            half_after[field, IP_NAME] - fraction * half_before[field, IP_NAME]
            for field, fraction in zip(fields, fractions)]),
    },
    index   = 'quantity',
)
local_table.show()

########################################
# Check that the global tune derivatives and nominal tunes are retained
########################################
print('Half-step global chromaticity changes:',
      half_after.dqx - half_before.dqx, half_after.dqy - half_before.dqy)
print('On-momentum tune changes:',
      local_after.qx - local_before.qx, local_after.qy - local_before.qy)
for knob in local_knobs:
    print(knob, 'integrated k2 trim [m^-2]:', env[knob])

# Half-step changes agree on global chromaticity preservation within 5e-4.
# Local a residuals are below 0.005; b residuals are below 0.05. Smaller
# residuals at the matching step alone would overstate the derivative accuracy.
