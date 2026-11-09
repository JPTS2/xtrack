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
# User parameters: geometry, optical targets and documented starting guesses
################################################################################

########################################
# Reference and arc-cell phase advance
########################################
GEOMETRY_FILE       = 'test_data/example_lattices/demo_ring/geometry.py'
OUTPUT_FILE         = 'outputs/matched_strengths.json'
P0C                 = 1e9               # Electron reference p0*c [eV]
HARMONIC            = 200
CELL_QX             = 0.25
CELL_QY             = 0.25
ARC_SEED            = np.sqrt(2) / 20    # Opposite initial k1 values [m^-2]
ARC_STEP            = 1e-6              # Finite-difference strength step [m^-2]
ARC_LIMITS          = (-0.2, 0.2)        # Illustrative numerical bounds [m^-2]
ARC_TOL             = 1e-11

########################################
# Outward half-straights and a bounded vertical squeeze
########################################
FF_SEED             = [-4, 2, -0.8, 0.5, -0.15, 0.1, -0.1, 0.1]  # [m^-2]
UTILITY_SEED        = [-0.03, 0.02, -0.02, 0.03, -0.02, 0.02, -0.01, 0.01]
IP_BETX             = 1.0               # [m]
IP_BETY_STAGES      = [0.5, 0.1, 0.03, 0.01]  # Last value is beta_y*=1 cm
INTERMEDIATE_BETA   = 60.0              # Both intermediate outer betas [m]
UTILITY_BETA        = 40.0              # Both utility-midpoint betas [m]
HALF_STEP           = 1e-5              # [m^-2]
MAX_STEPS           = 20

################################################################################
# Load geometry with zero quadrupole strengths, without the nominal match
################################################################################
repo_root       = Path(__file__).resolve().parents[2]
geometry_path   = repo_root / GEOMETRY_FILE
env             = xt.load(geometry_path)
arc             = env['arc.cell']
ff              = env['ff.half']
utility         = env['utility.half']
for line in [arc, ff, utility]:
    line.set_particle_ref('electron', p0c = P0C)

# Radiation, RF voltage and operating trims/errors/nonlinear fields stay off.
# Seeds are starting guesses, not the stored result in strengths.py.

################################################################################
# Match the periodic arc cell before using its beta/alpha as interface targets
################################################################################

########################################
# Solve the two-family arc problem
########################################
env['kqf'] = ARC_SEED
env['kqd'] = -ARC_SEED
arc.match(
    method  = '4d',
    chrom   = False,
    vary    = xt.VaryList(['kqf', 'kqd'], step = ARC_STEP, limits = ARC_LIMITS),
    targets = xt.TargetSet(qx = CELL_QX, qy = CELL_QY, tol = ARC_TOL),
)
port = arc.twiss4d(chrom = False)

########################################
# Distinguish the periodic cell dispersion from the achromat port
########################################
print('Arc cell Qx/Qy:', port.qx, port.qy)
print('Arc interface beta_x/beta_y [m]:', port.betx[0], port.bety[0])
print('Periodic-cell Dx [m] is NOT the four-cell D0 boundary:', port.dx[0])

# Four cells give transverse identity and return zero input dispersion to zero.
# The half-straights have no bends, so their zero input dispersion stays zero.

################################################################################
# Squeeze the final-focus waist while retaining an intermediate outer match
################################################################################

########################################
# Initialize eight independent thick-quadrupole controls
########################################
ff_knobs    = [f'ff_k{i}' for i in range(8)]
for knob, seed in zip(ff_knobs, FF_SEED):
    env[knob] = seed
ff_limits   = [(-15, 0), (0, 15)] + [(-5, 5)] * 6

########################################
# Continue through four specified vertical waist sizes
########################################
# Log beta ratios are dimensionless; alpha/beta residuals are in m^-1.
# Their separate tolerances set the matching scales. Each stage retains the
# previous strengths.
for beta_y in IP_BETY_STAGES:
    init    = xt.TwissInit(betx = IP_BETX, bety = beta_y, alfx = 0, alfy = 0)
    ff.match(
        method      = '4d',
        chrom       = False,
        init        = init,
        n_steps_max = MAX_STEPS,
        vary        = [
            xt.Vary(knob, step = HALF_STEP, limits = limits)
            for knob, limits in zip(ff_knobs, ff_limits)
        ],
        targets     = [
            xt.Target(
                lambda t: np.log(t.betx[-1] / INTERMEDIATE_BETA),
                value   = 0,
                tol     = 1e-6,
            ),
            xt.Target(
                lambda t: np.log(t.bety[-1] / INTERMEDIATE_BETA),
                value   = 0,
                tol     = 1e-6,
            ),
            xt.Target(
                lambda t: t.alfx[-1] / t.betx[-1],
                value   = 0,
                tol     = 1e-7,
            ),
            xt.Target(
                lambda t: t.alfy[-1] / t.bety[-1],
                value   = 0,
                tol     = 1e-7,
            ),
        ],
    )
    tw      = ff.twiss4d(init = init, chrom = False)
    print('IP beta_y / outer beta_x/beta_y [m]:',
          beta_y, tw.betx[-1], tw.bety[-1])

################################################################################
# Match both half-straights to the independently matched arc interface
################################################################################

########################################
# Set the utility starting guess and the two entrance waists
########################################
utility_knobs = [f'ut_k{i}' for i in range(8)]
for knob, seed in zip(utility_knobs, UTILITY_SEED):
    env[knob] = seed
half_lines = [
    (ff, ff_knobs, IP_BETX, IP_BETY_STAGES[-1]),
    (utility, utility_knobs, UTILITY_BETA, UTILITY_BETA),
]

########################################
# Solve the four endpoint conditions without assuming a unique strength vector
########################################
# Eight controls for four conditions leave freedom. The seeds, bounds and
# continuation path select one illustrative solution, not an optimum design.
for line, knobs, beta_x, beta_y in half_lines:
    init    = xt.TwissInit(betx = beta_x, bety = beta_y, alfx = 0, alfy = 0)
    line.match(
        method      = '4d',
        chrom       = False,
        init        = init,
        n_steps_max = MAX_STEPS,
        vary        = [
            xt.Vary(
                knob,
                step    = HALF_STEP,
                limits  = (-15, 15) if i < 2 else (-5, 5),
            )
            for i, knob in enumerate(knobs)
        ],
        targets     = [
            xt.Target(
                lambda t: np.log(t.betx[-1] / port.betx[0]),
                value   = 0,
                tol     = 1e-7,
            ),
            xt.Target(
                lambda t: np.log(t.bety[-1] / port.bety[0]),
                value   = 0,
                tol     = 1e-7,
            ),
            xt.Target(
                lambda t: t.alfx[-1] / t.betx[-1],
                value   = port.alfx[0] / port.betx[0],
                tol     = 1e-8,
            ),
            xt.Target(
                lambda t: t.alfy[-1] / t.bety[-1],
                value   = port.alfy[0] / port.bety[0],
                tol     = 1e-8,
            ),
        ],
    )
    tw      = line.twiss4d(init = init, chrom = False)
    print('Matched half-line outer beta/alpha:',
          tw.betx[-1], tw.bety[-1], tw.alfx[-1], tw.alfy[-1])

################################################################################
# Export strengths without overwriting the canonical operating point
################################################################################
strength_names  = ['kqf', 'kqd'] + ff_knobs + utility_knobs
strengths       = {name: float(env[name]) for name in strength_names}
output_path     = Path(__file__).resolve().parent / OUTPUT_FILE
output_path.parent.mkdir(parents = True, exist_ok = True)
xt.json.dump(strengths, output_path)
print('Exported matched strengths:', output_path)
print(strengths)

################################################################################
# Independently reload fresh geometry and the exported strength mapping
################################################################################

########################################
# Recreate independent geometry and ring
########################################
replayed = xt.load(geometry_path)
replayed.vars.load(output_path)
ring = replayed.new_line(
    name        = 'demo.replayed',
    components  = [
        'arc.s1', 'insertion', 'arc.s2', 'utility.s2',
        'arc.s3', 'utility.s3', 'arc.s4', 'utility.s4',
        'arc.s5', 'utility.s5', 'arc.s6', 'utility.s6',
    ],
)
ring.set_particle_ref('electron', p0c = P0C)
replayed['phase_rf'] = np.pi
replayed['frequency_rf'] = (
    HARMONIC * ring.particle_ref.beta0[0] * c / ring.get_length()
)

########################################
# Inspect the resulting periodic optics
########################################
full = ring.twiss4d(chrom = False)
print('Reloaded circumference [m], full-ring tunes:',
      ring.get_length(), full.qx, full.qy)
ip_table = full.rows[['ip.1']].cols['name betx bety alfx alfy dx dpx']
ip_table.show(maxwidth = 'full')

# This establishes the matched linear operating point. It does not correct
# chromaticity, qualify an action/momentum range or establish hardware limits.
