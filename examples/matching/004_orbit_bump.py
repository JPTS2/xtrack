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
# Native ring and local bump geometry
########################################
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
START_S         = 0.35                  # Upstream boundary [m]
CORRECTOR_S     = [0.4, 5.4, 10.4]       # Thin dipole positions [m]
MID_S           = 7.4                   # Observation between correctors 2/3 [m]
END_S           = 10.5                  # Boundary after the last corrector [m]
BUMP_HEIGHT     = 1e-3                  # Displacement at MID_S [m]

########################################
# Solver precision
########################################
KICK_STEP       = 1e-6                  # Normalized momentum-kick step
X_TOL           = 1e-10                 # Position tolerance [m]
PX_TOL          = 1e-10                 # Normalized canonical px tolerance

################################################################################
# Add independent steering knobs and zero-length boundary markers
################################################################################

########################################
# Insert three correctors
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['ring']
# The 40 m electron ring has p0*c=1 GeV, without RF or radiation; delta=0.
knobs = []
for index, position in enumerate(CORRECTOR_S):
    knob    = f'bump_kick_{index}'
    name    = f'bump.corr.{index}'
    env[knob] = 0.0
    # Multipole Delta px=-knl[0]: the knob is the positive horizontal kick.
    env.new(name, xt.Multipole, knl = ['-' + knob])
    line.insert(name, at = position)
    knobs.append(knob)

########################################
# Record the clean periodic orbit at named marker boundaries
########################################
for name, position in [('bump.start', START_S), ('bump.mid', MID_S),
                       ('bump.end', END_S)]:
    line.insert(name, xt.Marker(), at = position)
tw_ref      = line.twiss4d()
init        = tw_ref.get_twiss_init(at_element = 'bump.start')

################################################################################
# Match the local trajectory with its entrance state fixed
################################################################################
# Three variables control three conditions: height at the middle observation
# and restoration of x AND px downstream. Closure of position alone is not a
# closed bump. The fixed entrance state supplies the upstream boundary.
opt = line.match(
    method  = '4d',
    start   = 'bump.start',
    end     = 'bump.end',
    init    = init,
    vary    = xt.VaryList(knobs, step = KICK_STEP),
    targets = [
        xt.Target(
            'x',
            at      = 'bump.mid',
            value   = tw_ref['x', 'bump.mid'] + BUMP_HEIGHT,
            tol     = X_TOL,
        ),
        xt.Target(
            'x',
            at      = 'bump.end',
            value   = tw_ref['x', 'bump.end'],
            tol     = X_TOL,
        ),
        xt.Target(
            'px',
            at      = 'bump.end',
            value   = tw_ref['px', 'bump.end'],
            tol     = PX_TOL,
        ),
    ],
)

################################################################################
# Verify the resulting full-ring periodic orbit
################################################################################

########################################
# Inspect the target, boundary and external orbit deviations
########################################
tw_after    = line.twiss4d()
for name in ['bump.start', 'bump.mid', 'bump.end']:
    print(name, 'x difference [m], px difference [normalized to p0]:',
          tw_after['x', name] - tw_ref['x', name],
          tw_after['px', name] - tw_ref['px', name])
outside = (tw_after.s < START_S) | (tw_after.s > END_S)
for coordinate, unit in [('x', 'm'), ('px', 'dimensionless')]:
    print('Maximum outside-bump', coordinate, 'difference', f'[{unit}]:',
          np.max(np.abs((tw_after[coordinate] - tw_ref[coordinate])[outside])))
for knob in knobs:
    print(knob, 'normalized horizontal kick:', env[knob])

# At delta=0 these normalized kicks are paraxial angles [rad]. This confines
# the orbit deviation; it does not constrain beta functions or other optics.

########################################
# Plot the full-ring bump relative to the clean periodic orbit
########################################
fig, ax = plt.subplots(figsize = (9, 4))
ax.plot(tw_after.s, 1e3 * (tw_after.x - tw_ref.x), '.-', label = 'Matched bump')
ax.axhline(0, color = 'k', linestyle = '--', label = 'Clean orbit deviation')
ax.plot(MID_S, 1e3 * BUMP_HEIGHT, 'rx', label = 'Displacement target')
ax.axvspan(START_S, END_S, alpha = 0.1, color = 'C0')
ax.set_xlabel('s [m]')
ax.set_ylabel('x - clean orbit [mm]')
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()
