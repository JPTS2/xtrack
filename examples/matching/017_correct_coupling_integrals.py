# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt


################################################################################
# User parameters
################################################################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
START           = 'bump.start'
END             = 'bump.end'
START_S         = 160.0                 # [m]
END_S           = 250.0                 # [m]
SKEW_SITES      = [f'straight.s1.{side}.q{i}'
                   for side in ['l', 'r'] for i in [1, 3, 5, 7]]
N_SLICES        = 16                    # Thin slices per skew quadrupole
SOURCE_INDEX    = 3                     # Fixed source among SKEW_SITES
SOURCE_K1S      = 1e-4                  # Fixed skew gradient [m^-2]
INTEGRAL_STEP   = 1e-9                  # Corrector response step [m^-2]

################################################################################
# Select the fixed source and independent correction controls
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
knobs       = [f'correction_k1s_{i}' for i in range(len(SKEW_SITES))]

################################################################################
# Correct a separate fixed skew source with first-order integrals
################################################################################

########################################
# Resolve the skew sources along the thick quadrupoles
########################################
env     = xt.load(repo_root / LATTICE_FILE)
line    = env['demo']
for index, name in enumerate(SKEW_SITES):
    env[knobs[index]] = 0.0
    env.set(name, k1s = knobs[index])
for marker, position in [(START, START_S), (END, END_S)]:
    line.insert(marker, xt.Marker(), at = position)
# IntegralOptimization samples strengths at element entrances. Resolve the
# changing beta and phase inside these strong quadrupoles with thin slices.
line.slice_thick_elements(slicing_strategies = [
    xt.Strategy(slicing = None),
    *[xt.Strategy(slicing = xt.Teapot(N_SLICES), name = name, exact = True)
      for name in SKEW_SITES],
])
reference                   = line.twiss4d()
env[knobs[SOURCE_INDEX]]    = SOURCE_K1S
before                      = line.twiss4d()

########################################
# Supply named RDTs and a callable complex skew-source integral
########################################
def skew_integrand(twiss, strengths, phase_sign):
    # Direct skew strengths suffice here: no rotations, offsets or sextupoles.
    # An offset nonlinear source would require feed-down, as in optics/.
    return (strengths.k1sl * np.sqrt(twiss.betx * twiss.bety)
            * np.exp(-2j * np.pi * (twiss.mux + phase_sign * twiss.muy)))

# Each callable returns one real integrand component. Exposing all four signed
# components avoids a rank-deficient response from only two complex magnitudes.
correction = xt.IntegralOptimization(
    line                = line,
    twiss               = reference,
    start               = START,
    end                 = END,
    vary                = xt.VaryList(
        [knob for index, knob in enumerate(knobs) if index != SOURCE_INDEX],
        step = INTEGRAL_STEP),
    target_quantities   = {
        'difference': 'f1001',
        'sum':        'f1010',
        'diff_real':  lambda tw, tt: skew_integrand(tw, tt, -1).real,
        'diff_imag':  lambda tw, tt: skew_integrand(tw, tt, -1).imag,
        'sum_real':   lambda tw, tt: skew_integrand(tw, tt, +1).real,
        'sum_imag':   lambda tw, tt: skew_integrand(tw, tt, +1).imag},
    generated_knob_name = 'on_integral_correction',
    feed_down           = False,
)
# This fresh source has no requested interior bump target. run() returns
# absolute sums, also for the signed real/imaginary callable integrands.
# They are dimensionless and omit a periodic RDT denominator. The supplied
# optics stay fixed while live strengths vary.
print('Initial difference/sum integrals [1]:',
      correction.run()['difference'], correction.run()['sum'])
integral_opt = correction.get_optimizer()
integral_opt.target_status()
correction.correct(n_steps = 3)
after = line.twiss4d()

########################################
# Switch the generated knob off and on and check the actual transport
########################################
env['on_integral_correction']   = 0.0
reset                           = line.twiss4d()
print('Source-only difference/sum integrals after reset [1]:',
      correction.run()['difference'], correction.run()['sum'])
env['on_integral_correction'] = 1.0
print('Corrected difference/sum integrals [1]:',
      correction.run()['difference'], correction.run()['sum'])
for label, twiss in [('Source only', before),
                     ('Correction on', after),
                     ('Correction off', reset)]:
    print(label, 'exit upper-right map block:')
    print(twiss.get_R_matrix(START, END)[:2, 2:4])
    outside = ((twiss.s <= START_S) | (twiss.s >= END_S))
    print(label, 'maximum outside betx2/bety1 [m]:',
          np.max(twiss.betx2[outside]),
          np.max(twiss.bety1[outside]))
# Finite slicing and first-order optics leave a small map residual. Unlike the
# direct transfer-map matching, cancelling these integrals is approximate; it
# does not restore diagonal optics or demonstrate nonlinear decoupling.

################################################################################
# Compare cross-plane mode projections before and after correction
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (9, 7),
                         layout = 'constrained')
for label, tw in [('Source only', before), ('Correction on', after)]:
    axes[0].plot(tw.s, tw.betx2, label = label)
    axes[1].plot(tw.s, tw.bety1, label = label)
axes[0].set_ylabel('Mode 2 projection into x [m]')
axes[1].set_ylabel('Mode 1 projection into y [m]')
axes[1].set_xlabel('s [m]')
for axis in axes:
    axis.set_yscale('symlog', linthresh = 1e-9)
    axis.axvspan(START_S, END_S, alpha = 0.1, label = 'Correction section')
    axis.grid()
    axis.legend()
plt.show()
