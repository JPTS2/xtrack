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
# Ring, bump and deliberately offset aperture
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
START_S         = 160.0                 # Upstream boundary [m]
END_S           = 250.0                 # Downstream boundary [m]
EXTRA_COR_S     = 230.0                 # Fourth corrector in this example [m]
BUMP_HEIGHT     = 1e-4                  # Horizontal displacement at ip.1 [m]
BUMP_SLOPE      = 2e-6                  # Canonical px at ip.1, delta=0
APERTURE_MIN_X  = 0.5e-4                # Offset horizontal opening [m]
APERTURE_MAX_X  = 1.5e-4                # [m]

########################################
# Matching precision
########################################
KICK_STEP       = 1e-6                  # Normalized momentum-kick step
X_TOL           = 1e-10                 # Position tolerance [m]
PX_TOL          = 1e-10                 # Canonical momentum tolerance

################################################################################
# Prepare a physical line and a shared optics twin
################################################################################

########################################
# Bind four independent horizontal correctors
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
# Add one zero-length corrector to this copy of the DEMO insertion.
line.insert('bump.cor.4', xt.Multipole(knl = [0.0]), at = EXTRA_COR_S)
correctors  = ['cor.ip.1', 'cor.ip.2', 'cor.ip.3', 'bump.cor.4']
knobs       = [f'bump_kick_{index}' for index in range(4)]
for name, knob in zip(correctors, knobs):
    env[knob] = 0.0
    # Multipole Delta px=-knl[0]; positive knob means positive kick.
    env.set(name, knl = ['-' + knob])
for name, position in [('bump.start', START_S), ('bump.end', END_S)]:
    line.insert(name, xt.Marker(), at = position)

########################################
# Keep the offset opening in tracking, replace it only for optics
########################################
line.insert(
    'bump.aperture',
    xt.LimitRect(min_x = APERTURE_MIN_X, max_x = APERTURE_MAX_X),
    at      = 'ip.1')
optics = line.copy(shallow = True)
env.new('bump.aperture.marker', xt.Marker)
optics.replace('bump.aperture', 'bump.aperture.marker')
reference   = optics.twiss4d()
initial     = line.build_particles()
line.track(initial)
print('Unsteered particle state at the offset aperture:', initial.state)

################################################################################
# Match the same bump with four choices of boundary data
################################################################################

########################################
# Fix the entrance and constrain IP position, slope and downstream closure
########################################
ip_targets = [
    xt.Target('x', at = 'ip.1', value = BUMP_HEIGHT, tol = X_TOL),
    xt.Target('px', at = 'ip.1', value = BUMP_SLOPE, tol = PX_TOL),
]
start_targets = [
    xt.Target('x', at = xt.START, value = reference, tol = X_TOL),
    xt.Target('px', at = xt.START, value = reference, tol = PX_TOL),
]
end_targets = [
    xt.Target('x', at = xt.END, value = reference, tol = X_TOL),
    xt.Target('px', at = xt.END, value = reference, tol = PX_TOL),
]
common = dict(
    method  = '4d',
    start   = 'bump.start',
    end     = 'bump.end',
    vary    = xt.VaryList(knobs, step = KICK_STEP))
solutions = {}
opt = optics.match(
    **common,
    init    = reference.get_twiss_init(at_element = 'bump.start'),
    targets = ip_targets + end_targets)
solutions['Entrance'] = [env[knob] for knob in knobs]

########################################
# Fix downstream conditions and close the orbit upstream
########################################
for knob in knobs:
    env[knob] = 0.0
opt = optics.match(
    **common,
    init    = reference.get_twiss_init(at_element = 'bump.end'),
    init_at = xt.END,
    targets = ip_targets + start_targets)
solutions['Exit'] = [env[knob] for knob in knobs]

########################################
# Fix the IP trajectory and close both external boundaries
########################################
for knob in knobs:
    env[knob] = 0.0
ip_init     = reference.get_twiss_init(at_element = 'ip.1')
ip_init.x   = BUMP_HEIGHT
ip_init.px  = BUMP_SLOPE
opt = optics.match(
    **common,
    init    = ip_init,
    init_at = 'ip.1',
    targets = start_targets + end_targets)
solutions['Interior'] = [env[knob] for knob in knobs]

########################################
# Extract downstream conditions and upstream targets from a Twiss table
########################################
for knob in knobs:
    env[knob] = 0.0
opt = optics.match(
    **common,
    init    = reference,
    init_at = xt.END,
    targets = ip_targets + start_targets)
solutions['Table'] = [env[knob] for knob in knobs]

################################################################################
# Independently recompute the full periodic orbit and physical transmission
################################################################################

########################################
# Compare fresh-start solutions and expose orbit leakage outside the bump
########################################
fig, axes = plt.subplots(2, 1, figsize = (9, 7))
for label, values in solutions.items():
    for knob, value in zip(knobs, values):
        env[knob] = value
    tw = optics.twiss4d()
    outside = (tw.s < START_S) | (tw.s > END_S)
    print(label, 'IP x [m], px:', tw['x', 'ip.1'], tw['px', 'ip.1'])
    print('  Maximum external |x| [m], |px|:',
          np.max(np.abs(tw.x[outside])), np.max(np.abs(tw.px[outside])))
    print('  Corrector kicks:', values)
    axes[0].plot(tw.s, 1e3 * tw.x, '.-', label = label)
    axes[1].plot(tw.s, 1e6 * tw.px, '.-', label = label)
# All cases start from zero kicks; none inherits a previously solved bump.
# Closure is orbital: beta functions and nonlinear response are unconstrained.

########################################
# Track through the unchanged physical aperture using the shared kicks
########################################
particle = line.build_particles()
line.track(particle)
print('Steered particle state after one physical turn:', particle.state)
print('Physical aperture:',
      line.get_table()['element_type', 'bump.aperture'])
print('Optics replacement:',
      optics.get_table()['element_type', 'bump.aperture.marker'])

########################################
# Resolve both coordinates throughout the insertion
########################################
axes[0].plot(
    [reference['s', 'ip.1']] * 2,
    1e3 * np.array([APERTURE_MIN_X, APERTURE_MAX_X]),
    color = '0.7', linewidth = 8, label = 'Aperture opening', zorder = 0)
axes[0].plot(reference['s', 'ip.1'], 1e3 * BUMP_HEIGHT, 'kx')
axes[1].plot(reference['s', 'ip.1'], 1e6 * BUMP_SLOPE, 'kx')
axes[0].set_ylabel('x [mm]')
axes[1].set_ylabel('px [1e-6]')
axes[1].set_xlabel('s [m]')
for axis in axes:
    axis.set_xlim(START_S - 5, END_S + 5)
    axis.axhline(0, color = 'k', linestyle = '--')
    axis.axvline(reference['s', 'ip.1'], color = '0.5', linestyle = ':')
    axis.grid()
    axis.legend()
fig.suptitle('Four-corrector bump: independent height and slope')
fig.tight_layout()
plt.show()
