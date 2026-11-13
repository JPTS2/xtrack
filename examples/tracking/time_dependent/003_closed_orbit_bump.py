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
from scipy.constants import c
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Shared model
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'

########################################
# Matched bump and time waveform
########################################
BUMP_START_S    = 160.0                 # [m]
BUMP_END_S      = 250.0                 # [m]
FOURTH_COR_S    = 230.0                 # [m]
BUMP_HEIGHT     = 1e-4                  # Peak horizontal IP position [m]
BUMP_TURNS      = 8                     # One finite pulse
CARRIER_TURNS   = 4                     # Sinusoidal period [reference turns]
PROBE_X         = [-1e-5, 0.0, 1e-5]   # Initial x outside the bump [m]
ORBIT_TOL       = 1e-10                 # Match x [m] and canonical px

################################################################################
# Load the nominal reference clock
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
################################################################################
# Drive a spatially closed bump with one common envelope
################################################################################

########################################
# Match four independent correctors on a fresh nominal ring
########################################
bump_env    = xt.load(repo_root / LATTICE_FILE)
bump_line   = bump_env['demo']
period      = bump_line.get_length() / (bump_line.particle_ref.beta0[0] * c)
bump_line.insert('bump.cor.4', xt.Multipole(knl = [0.0]), at = FOURTH_COR_S)
for name, position in [('bump.start', BUMP_START_S),
                       ('bump.end', BUMP_END_S)]:
    bump_line.insert(name, xt.Marker(), at = position)
correctors  = ['cor.ip.1', 'cor.ip.2', 'cor.ip.3', 'bump.cor.4']
bump_knobs  = [f'bump_kick_{index}' for index in range(4)]
for name, knob in zip(correctors, bump_knobs):
    bump_env[knob] = 0.0
    # Multipole Delta px=-knl[0]; positive knob means positive kick.
    bump_env.set(name, knl = ['-' + knob])
bump_reference          = bump_line.twiss4d()
bump_opt                = bump_line.match(
    method  = '4d',
    start   = 'bump.start',
    end     = 'bump.end',
    init    = bump_reference.get_twiss_init(at_element = 'bump.start'),
    vary    = xt.VaryList(bump_knobs, step = 1e-6),
    targets = [
        xt.TargetSet(x = BUMP_HEIGHT, px = 0.0, at = 'ip.1', tol = ORBIT_TOL),
        xt.TargetSet(x = 0.0, px = 0.0, at = xt.END, tol = ORBIT_TOL),
    ],
)
peak_kicks              = [bump_env[knob] for knob in bump_knobs]
bump_env['bump_scale']  = 0.0
for knob, strength in zip(bump_knobs, peak_kicks):
    bump_env[knob] = strength * bump_env.ref['bump_scale']

########################################
# Compare a finite pulse with a pulse-modulated sinusoid
########################################
# The same envelope multiplies every matched kick. With fixed linear optics,
# this scales the closed bump without changing its spatial cancellation.
# The carrier changes sign inside the pulse; it is not a second corrector.
bump_env.functions['bump_pulse']    = xt.FunctionPieceWiseLinear(
    x   = np.array([0, 2, 6, BUMP_TURNS]) * period,
    y   = [0.0, 1.0, 1.0, 0.0],
)
waveforms                           = {
    'Piecewise pulse': bump_env.functions['bump_pulse'](
        bump_env.ref['t_turn_s']),
    'Pulse times sine': bump_env.functions['bump_pulse'](
        bump_env.ref['t_turn_s']) * bump_env.functions.sin(
            2 * np.pi * bump_env.ref['t_turn_s'] / (CARRIER_TURNS * period)),
}
sample_times                        = np.linspace(0.0, BUMP_TURNS * period, 33)
profiles                            = {}
closure                             = {}
for label, expression in waveforms.items():
    bump_env['bump_scale']  = expression
    profiles[label]         = []
    closure[label]          = []
    for time in sample_times:
        bump_env['t_turn_s']    = time
        tw                      = bump_line.twiss4d()
        outside                 = (tw.s <= BUMP_START_S) | (tw.s >= BUMP_END_S)
        profiles[label].append(tw['x', 'ip.1'])
        closure[label].append((np.max(np.abs(tw.x[outside])),
                               np.max(np.abs(tw.px[outside]))))
    print(label, 'max outside |x| [m], |px|:',
          np.max(closure[label], axis = 0))

########################################
# Track the same three probes through each pulse
########################################
# Fields update once per turn. A closed bump can move rapidly without driving
# the on-axis probe outside this linear insertion; off-axis probes retain
# their ordinary betatron motion. This is not an adiabatic-following argument.
monitor = xt.ParticlesMonitor(
    num_particles   = len(PROBE_X),
    start_at_turn   = 0,
    stop_at_turn    = BUMP_TURNS,
)
bump_line.insert('bump.monitor', monitor, at = 'ip.1')
bump_line.enable_time_dependent_vars    = True
bump_line.dt_update_time_dependent_vars = None
tracked_ip                              = {}
probe_ends                              = {}
for label, expression in waveforms.items():
    bump_env['bump_scale']  = expression
    probes                  = bump_line.build_particles(x = PROBE_X)
    probes.time_s           = 0.0
    bump_line.track(probes, num_turns = BUMP_TURNS)
    tracked_ip[label]   = monitor.x.copy()
    probe_ends[label]   = probes.copy()
    print(label, 'probe states:', probes.state,
          '; on-axis probe final x [m], px:', probes.x[1], probes.px[1])
bump_line.enable_time_dependent_vars    = False
bump_env['bump_scale']                  = 0.0
control_probes                          = bump_line.build_particles(x = PROBE_X)
bump_line.track(control_probes, num_turns = BUMP_TURNS)
for label, probes in probe_ends.items():
    print(label, 'all three probes minus unpowered endpoint:')
    print('Delta x [m]:', probes.x - control_probes.x)
    print('Delta px:', probes.px - control_probes.px)

########################################
# Compare instantaneous optics with turn-sampled tracking at the IP
########################################
fig, axis = plt.subplots(figsize = (8, 4))
for label in waveforms:
    axis.plot(sample_times / period, np.array(profiles[label]) * 1e3,
              label = label + ': static optics')
    axis.plot(np.arange(BUMP_TURNS), tracked_ip[label][1] * 1e3,
              'o', label = label + ': on-axis probe')
axis.set_xlabel('Reference clock / revolution period')
axis.set_ylabel('IP x [mm]')
axis.grid()
axis.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.3),
            ncol = 2, fontsize = 9)
fig.tight_layout()

plt.show()
