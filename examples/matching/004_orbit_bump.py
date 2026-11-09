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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
START_S         = 160.0                 # Upstream boundary [m]
CORRECTORS      = ['cor.ip.1', 'cor.ip.2', 'cor.ip.3']
MID_S           = 205.0                 # IP between correctors 2/3 [m]
END_S           = 250.0                 # Boundary after the last corrector [m]
BUMP_HEIGHT     = 1e-4                  # Displacement at MID_S [m]

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
# Bind the three existing insertion correctors
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
# The 1500 m electron ring has p0*c=1 GeV, with RF voltage zero and
# radiation off; delta=0.
knobs = []
for index, name in enumerate(CORRECTORS):
    knob    = f'bump_kick_{index}'
    env[knob] = 0.0
    # Multipole Delta px=-knl[0]: the knob is the positive horizontal kick.
    env.set(name, knl = ['-' + knob])
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
# Resolve the insertion shape and the selected IP displacement
########################################
fig, axes = plt.subplots(2, 1, figsize = (9, 7))
axes[0].plot(tw_after.s, 1e3 * (tw_after.x - tw_ref.x), '.-')
axes[0].axhline(0, color = 'k', linestyle = '--')
axes[0].plot(MID_S, 1e3 * BUMP_HEIGHT, 'rx', label = 'IP displacement target')
axes[0].axvline(MID_S, color = '0.5', linestyle = ':')
axes[0].set_xlim(START_S - 5, END_S + 5)
axes[0].set_xticks([START_S, MID_S, END_S],
                   labels   = ['bump.start', 'ip.1', 'bump.end'])
axes[0].set_ylabel('x - clean orbit [mm]')
axes[0].set_title('Insertion detail: target displacement is not the peak')
axes[0].legend()

########################################
# Retain the full-ring view to expose any external orbit leakage
########################################
axes[1].plot(tw_after.s, 1e3 * (tw_after.x - tw_ref.x), '.-',
             label  = 'Matched bump')
axes[1].axhline(0, color = 'k', linestyle = '--',
                label   = 'Clean orbit deviation')
axes[1].axvspan(START_S, END_S, alpha = 0.1, color = 'C0')
axes[1].set_xlabel('s [m]')
axes[1].set_ylabel('x - clean orbit [mm]')
axes[1].set_title('Full ring: closure outside the insertion')
axes[1].legend()
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
