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
# Vertical bump through a fixed skew quadrupole
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
START_S         = 160.0                 # Open-section entrance [m]
END_S           = 250.0                 # Downstream closure plane [m]
SKEW_ELEMENT    = 'straight.s1.l.q3'
K1S             = 1e-3                  # Fixed skew gradient [m^-2]
BUMP_Y          = 1e-4                  # IP vertical displacement [m]
BUMP_PY         = 0.0                   # Canonical momentum at the IP

########################################
# Four vertical and two horizontal correction controls
########################################
VERTICAL_S      = [185.5, 203.7, 211.5, 230.0]  # [m]
HORIZONTAL_S    = [195.0, 235.0]                # [m]
KICK_STEP       = 1e-7                  # Canonical kick response step
POSITION_TOL    = 1e-10                 # [m]
MOMENTUM_TOL    = 1e-10

################################################################################
# Construct a coupled insertion with six independent correctors
################################################################################

########################################
# Keep the skew source fixed during orbit matching
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
env.set(SKEW_ELEMENT, k1s = K1S)
for name, position in [('bump.start', START_S), ('bump.end', END_S)]:
    line.insert(name, xt.Marker(), at = position)

########################################
# Add both kick planes using the native multipole convention
########################################
vertical_knobs      = []
horizontal_knobs    = []
for plane, positions, knobs in [
    ('v', VERTICAL_S, vertical_knobs), ('h', HORIZONTAL_S, horizontal_knobs),
]:
    for index, position in enumerate(positions):
        name        = f'bump.{plane}.{index}'
        knob        = f'kick_{plane}_{index}'
        env[knob]   = 0.0
        line.insert(name, xt.Multipole(), at = position)
        if plane == 'v':
            env.set(name, ksl = [knob])
        else:
            env.set(name, knl = ['-' + knob])
        knobs.append(knob)
# Positive control values mean positive canonical py or px kicks.
reference   = line.twiss4d()
init        = reference.get_twiss_init('bump.start')

################################################################################
# First close only the vertical coordinates
################################################################################
vertical_targets = [
    xt.TargetSet(y = BUMP_Y, py = BUMP_PY, at = 'ip.1',
                 tol = POSITION_TOL),
    xt.Target('y', at = xt.END, value = reference, tol = POSITION_TOL),
    xt.Target('py', at = xt.END, value = reference, tol = MOMENTUM_TOL),
]
line.match(
    method  = '4d',
    start   = 'bump.start',
    end     = 'bump.end',
    init    = init,
    vary    = xt.VaryList(vertical_knobs, step = KICK_STEP),
    targets = vertical_targets,
)
vertical_only = line.twiss4d()

################################################################################
# Use two more controls to close the horizontal leakage
################################################################################

########################################
# Start again from zero kicks
########################################
for knob in vertical_knobs + horizontal_knobs:
    env[knob] = 0.0
line.match(
    method  = '4d',
    start   = 'bump.start',
    end     = 'bump.end',
    init    = init,
    vary    = xt.VaryList(vertical_knobs + horizontal_knobs, step = KICK_STEP),
    targets = vertical_targets + [
        xt.Target('x', at = xt.END, value = reference, tol = POSITION_TOL),
        xt.Target('px', at = xt.END, value = reference, tol = MOMENTUM_TOL),
    ],
)
closed = line.twiss4d()

########################################
# Recompute the periodic orbit independently
########################################
outside = (reference.s < START_S) | (reference.s > END_S)
for label, tw in [('Vertical closure only', vertical_only),
                  ('Both-plane closure', closed)]:
    print(label)
    print('IP y/py:', tw['y', 'ip.1'], tw['py', 'ip.1'])
    print('External maximum |x|/|y| [m]:',
          np.max(np.abs(tw.x[outside])), np.max(np.abs(tw.y[outside])))
# A vertical trajectory in a skew field drives x. Closing y/py alone is not a
# closed physical bump in both planes. Optics functions are not match targets.

################################################################################
# Compare local excitation with the external orbit leakage
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (10, 6),
                         layout = 'constrained')
for label, tw in [('Vertical closure only', vertical_only),
                  ('Both-plane closure', closed)]:
    axes[0].plot(tw.s, tw.x * 1e6, label = label)
    axes[1].plot(tw.s, tw.y * 1e6, label = label)
axes[0].set_ylabel('Horizontal orbit [µm]')
axes[1].set_ylabel('Vertical orbit [µm]')
axes[1].set_xlabel('s [m]')
for axis in axes:
    axis.axvspan(START_S, END_S, color = 'grey', alpha = 0.15)
    axis.grid()
    axis.legend()
plt.show()
