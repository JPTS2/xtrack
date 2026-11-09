# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Ring and desired tune derivatives
########################################
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
N_CELLS         = 8
DQX_TARGET      = 1.0                   # dQx/d(delta), dimensionless
DQY_TARGET      = 2.0                   # dQy/d(delta), dimensionless

########################################
# Sextupole controls and numerical precision
########################################
K2L_STEP        = 1.0e-4                # Integrated k2 variation step [m^-2]
K2L_LIMITS      = [-20.0, 20.0]          # Numerical search bounds [m^-2]
CHROM_TOL       = 1.0e-6
DELTA_STEP      = 1.0e-4                # Chromaticity computation step
DELTA_CHECK     = 5.0e-5                # Independent centered difference step

################################################################################
# Add two normal-sextupole families to a fresh native ring
################################################################################

########################################
# Install integrated kicks at focusing and defocusing quad entrances
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['ring']
# The 40 m ring supplies a 1 GeV/c electron reference, without RF or radiation.
# knl[2] is integrated normal k2 [m^-2], with Delta px=-knl[2]*(x^2-y^2)/2.
# These zero-length kicks preserve the reference geometry.
env['k2lf'] = 0.0
env['k2ld'] = 0.0
placements = []
for cell in range(1, N_CELLS + 1):
    for prefix, quad, knob in [('sf', 'qf', 'k2lf'), ('sd', 'qd', 'k2ld')]:
        placements.append(env.new(
            f'{prefix}.{cell}', xt.Multipole,
            knl = [0.0, 0.0, knob],
            at  = f'{quad}.{cell}@start',
        ))
line.insert(placements)

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
