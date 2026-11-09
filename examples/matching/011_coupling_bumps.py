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
START_S         = 160.0                   # [m]
MIDDLE_S        = 205.0                  # [m]
END_S           = 250.0                  # [m]
SKEW_SITES      = [f'straight.s1.{side}.q{i}'
                   for side in ['l', 'r'] for i in [1, 3, 5, 7]]
R13_MIDDLE      = 0.001                  # dx_at_middle / dy_at_start [1]

########################################
# Skew-gradient controls and map-term tolerances
########################################
K1S_STEP        = 1.0e-5                # Skew k1 finite-difference step [m^-2]
K1S_LIMITS      = [-0.05, 0.05]            # Numerical search bounds [m^-2]
R13_TOL         = 1.0e-9                # x/y [1]
R14_TOL         = 1.0e-9                # x/py [m]
R23_TOL         = 1.0e-9                # px/y [m^-1]
R24_TOL         = 1.0e-9                # px/py [1]

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
