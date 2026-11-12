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

########################################
# Local betatron-coupling bump boundaries
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
START           = 'bump.start'          # Zero-length marker at s=160 m
MIDDLE          = 'bump.middle'         # Zero-length marker at s=205 m
END             = 'bump.end'            # Zero-length marker at s=250 m
START_S         = 160.0                 # [m]
MIDDLE_S        = 205.0                 # [m]
END_S           = 250.0                 # [m]
SKEW_SITES      = [f'straight.s1.{side}.q{i}'
                   for side in ['l', 'r'] for i in [1, 3, 5, 7]]
R13_MIDDLE      = 0.001                 # dx_at_middle / dy_at_start [1]

########################################
# Skew-gradient controls and map-term tolerances
########################################
K1S_STEP        = 1.0e-5                # Skew k1 finite-difference step [m^-2]
K1S_LIMITS      = [-0.05, 0.05]         # Numerical search bounds [m^-2]
R13_TOL         = 1.0e-9                # x/y [1]
R14_TOL         = 1.0e-9                # x/py [m]
R23_TOL         = 1.0e-9                # px/y [m^-1]
R24_TOL         = 1.0e-9                # px/py [1]

########################################
# First-order integral comparison
########################################
N_SLICES        = 16                    # Thin slices per skew quadrupole
SOURCE_INDEX    = 3                     # Fixed source among SKEW_SITES
SOURCE_K1S      = 1.0e-4                # Fixed skew gradient [m^-2]
INTEGRAL_STEP   = 1.0e-9                # Corrector response step [m^-2]

################################################################################
# Create independent skew components inside the selected section
################################################################################

########################################
# Add live skew-gradient controls to eight existing quadrupoles
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
# The 1500 m ring supplies a 1 GeV/c electron reference, with RF voltage
# zero and radiation off.
# k1s is a skew gradient [m^-2]; the actual thick-magnet lengths are retained.
knobs = []
for index, name in enumerate(SKEW_SITES):
    knob    = f'bump_k1s_{index}'
    env[knob] = 0.0
    env.set(name, k1s = knob)
    knobs.append(knob)

########################################
# Fix the clean upstream normal-mode state
########################################
for marker, position in [(START, START_S), (MIDDLE, MIDDLE_S), (END, END_S)]:
    line.insert(marker, xt.Marker(), at = position)
reference   = line.twiss4d()
init        = reference.get_twiss_init(START)

################################################################################
# Open the coupling inside and close the cross-plane map at the exit
################################################################################

########################################
# Define interior and exit map constraints
########################################
# R maps (x,px,y,py), with momenta normalized to reference p0. All map locations
# below are zero-length marker boundaries, ahead of the quadrupole bodies.
# Four independent terms close the upper-right transverse block; symplecticity
# also closes the lower-left block. Both blocks are independently checked below.
targets = []
for term, tolerance in [
    ('r13', R13_TOL), ('r14', R14_TOL),
    ('r23', R23_TOL), ('r24', R24_TOL),
]:
    targets.append(xt.TargetRmatrixTerm(
        term, 0.0,
        start   = START,
        end     = END,
        tol     = tolerance,
    ))
targets.append(xt.TargetRmatrixTerm(
    'r13', R13_MIDDLE,
    start   = START,
    end     = MIDDLE,
    tol     = R13_TOL,
))

########################################
# Solve the skew-gradient controls
########################################
opt = line.match(
    method  = '4d',
    start   = START,
    end     = END,
    init    = init,
    vary    = xt.VaryList(knobs, step = K1S_STEP, limits = K1S_LIMITS),
    targets = targets,
)

# Eight controls and five conditions leave a nonunique solution. This closes
# transverse betatron coupling at delta=0. Diagonal optics and dispersion are
# not matched.

################################################################################
# Verify the full periodic solution and the external decoupling
################################################################################

########################################
# Inspect the physical map entries and the unused closure block
########################################
opt.target_status()
after       = line.twiss4d()
matrix_mid  = after.get_R_matrix(START, MIDDLE)
matrix_end  = after.get_R_matrix(START, END)
print('Interior R13, dimensionless:', matrix_mid[0, 2])
print('Exit upper-right block [[R13,R14],[R23,R24]]:')
print(matrix_end[:2, 2:4])
print('Exit lower-left block [[R31,R32],[R41,R42]]:')
print(matrix_end[2:4, :2])
print('Block units by row: [[1,m],[m^-1,1]]')

########################################
# Check normal-mode projections outside the selected section
########################################
outside = ((after.s <= after['s', START]) | (after.s >= after['s', END]))
print('Maximum outside betx2 and bety1 [m]:',
      np.max(np.abs(after.betx2[outside])),
      np.max(np.abs(after.bety1[outside])))
print('Final normal-mode tunes:', after.qx, after.qy)
print('Maximum outside vertical dispersion [m]:',
      np.max(np.abs(after.dy[outside])))
for knob in knobs:
    print(knob, 'skew gradient [m^-2]:', env[knob])

# Mode labels continue from the initially uncoupled x/y branches. Vanishing
# cross-plane mode projections outside demonstrate betatron decoupling there;
# neither zero coupling inside nor restoration of all optics is requested.

################################################################################
# Correct a separate fixed skew source with first-order integrals
################################################################################

########################################
# Resolve the skew sources along the thick quadrupoles
########################################
integral_env    = xt.load(repo_root / LATTICE_FILE)
integral_line   = integral_env['demo']
for index, name in enumerate(SKEW_SITES):
    integral_env[knobs[index]] = 0.0
    integral_env.set(name, k1s = knobs[index])
for marker, position in [(START, START_S), (END, END_S)]:
    integral_line.insert(marker, xt.Marker(), at = position)
# IntegralOptimization samples strengths at element entrances. Resolve the
# changing beta and phase inside these strong quadrupoles with thin slices.
integral_line.slice_thick_elements(slicing_strategies = [
    xt.Strategy(slicing = None),
    *[xt.Strategy(slicing = xt.Teapot(N_SLICES), name = name, exact = True)
      for name in SKEW_SITES],
])
integral_reference = integral_line.twiss4d()
integral_env[knobs[SOURCE_INDEX]] = SOURCE_K1S
integral_before = integral_line.twiss4d()

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
    line                = integral_line,
    twiss               = integral_reference,
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
integral_after = integral_line.twiss4d()

########################################
# Switch the generated knob off and on and check the actual transport
########################################
integral_env['on_integral_correction'] = 0.0
integral_reset = integral_line.twiss4d()
print('Source-only difference/sum integrals after reset [1]:',
      correction.run()['difference'], correction.run()['sum'])
integral_env['on_integral_correction'] = 1.0
print('Corrected difference/sum integrals [1]:',
      correction.run()['difference'], correction.run()['sum'])
for label, twiss in [('Source only', integral_before),
                     ('Correction on', integral_after),
                     ('Correction off', integral_reset)]:
    print(label, 'exit upper-right map block:')
    print(twiss.get_R_matrix(START, END)[:2, 2:4])
    outside_integral = ((twiss.s <= START_S) | (twiss.s >= END_S))
    print(label, 'maximum outside betx2/bety1 [m]:',
          np.max(twiss.betx2[outside_integral]),
          np.max(twiss.bety1[outside_integral]))
# Finite slicing and first-order optics leave a small map residual. Unlike the
# exact map match above, cancelling these integrals is an approximation; it
# does not restore diagonal optics or demonstrate nonlinear decoupling.

################################################################################
# Plot where each mode projects into the other physical plane
################################################################################

########################################
# Resolve the mode projections inside the insertion
########################################
fig, axes = plt.subplots(2, 1, figsize = (9, 7))
axes[0].plot(after.s, after.betx2, '.-', label = 'betx2: mode 2 in x')
axes[0].plot(after.s, after.bety1, '.-', label = 'bety1: mode 1 in y')
axes[0].axvline(MIDDLE_S, color = '0.5', linestyle = ':')
axes[0].set_xlim(START_S - 5, END_S + 5)
axes[0].set_xticks([START_S, MIDDLE_S, END_S],
                   labels   = ['bump.start', 'ip.1 / target', 'bump.end'])
axes[0].set_ylabel('Cross-plane projected beta [m]')
axes[0].set_title('Insertion detail')
axes[0].legend()

########################################
# Keep the full ring visible to check external decoupling
########################################
axes[1].plot(after.s, after.betx2, '.-', label = 'betx2: mode 2 in x')
axes[1].plot(after.s, after.bety1, '.-', label = 'bety1: mode 1 in y')
axes[1].axvspan(START_S, END_S, alpha = 0.1, label = 'Bump region')
axes[1].set_xlabel('s [m]')
axes[1].set_ylabel('Cross-plane projected beta [m]')
axes[1].set_title('Full ring: decoupled outside the insertion')
axes[1].legend()
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
