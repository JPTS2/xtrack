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
# Shared geometry, chromaticity targets and momentum sample
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
DQX_TARGET      = 1.0                   # dQx/d(delta)
DQY_TARGET      = 2.0
K2L_STEP        = 1e-4                  # Integrated sextupole step [m^-2]
K2L_LIMITS      = [-1.0, 1.0]            # Numerical bounds [m^-2]
CHROM_TOL       = 1e-6
DELTA_STEP      = 2e-6                  # Chromatic derivative step
DELTAS          = np.array([-1e-4, 0.0, 1e-4])  # (p-p0)/p0
IP              = 'ip.1'

########################################
# Common finite launch sample and tracking horizon
########################################
NEMITT_X        = 1e-6                  # Normalization emittance [m]
NEMITT_Y        = 1e-6                  # [m]
AMPLITUDES      = np.array([0.5, 1.0, 2.0])  # x normalized amplitude
N_TURNS         = 64
TRACE_ID        = 8                    # Particle shown in the tracking panels
# px_norm=py_norm=0, y_norm=x_norm/2 fixes this launch phase slice.

################################################################################
# Load one operating point and retain an independent bare control
################################################################################

########################################
# Measure the bare reference before changing any controls
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
nominal     = line.twiss4d(chrom = True, delta_chrom = DELTA_STEP)
bare        = line.copy()
# The 1500 m ring supplies a 1 GeV/c electron reference (q0=-1).
# RF voltage, radiation, skew fields and octupole strengths remain zero.

########################################
# Use native normalization to prepare common physical entrance coordinates
########################################
amplitude, delta = np.meshgrid(AMPLITUDES, DELTAS)
initial = bare.build_particles(
    method      = '4d',
    x_norm      = amplitude.ravel(),
    px_norm     = 0.0,
    y_norm      = amplitude.ravel() / 2,
    py_norm     = 0.0,
    nemitt_x    = NEMITT_X,
    nemitt_y    = NEMITT_Y,
    delta       = delta.ravel(),
)
# Both tracking cases receive copies of these same coordinates, prepared using
# the bare linear optics. This is a fixed phase slice, not a matched bunch.

################################################################################
# Correct global chromaticities using the existing sextupole circuits
################################################################################
# This changes the SAME loaded environment used for the subsequent optics and
# tracking. Geometry, reference p0 and all quadrupole strengths stay fixed.
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
corrected = line.twiss4d(chrom = True, delta_chrom = DELTA_STEP)
print('Bare/corrected dQx/d(delta), dQy/d(delta):',
      (nominal.dqx, nominal.dqy), (corrected.dqx, corrected.dqy))
print('Corrected integrated sextupole strengths [m^-2]:',
      env['k2lf'], env['k2ld'])

################################################################################
# Compare off-momentum tunes and insertion optics at the same settings
################################################################################
optics = {}
rows   = []
for label, model in [('Bare', bare), ('Corrected', line)]:
    optics[label] = []
    for momentum in DELTAS:
        tw = model.twiss4d(delta0 = momentum, chrom = False)
        optics[label].append(tw)
        rows.append((label, momentum, tw.qx, tw.qy, tw['betx', IP],
                     tw['bety', IP], tw['x', IP], tw['dx', IP]))
columns = ['setting', 'delta', 'qx', 'qy',
           'betx_ip', 'bety_ip', 'x_ip', 'dx_ip']
print('IP beta, orbit and dispersion in metres; full tunes dimensionless:')
xt.Table(
    data    = {name: np.array(values)
               for name, values in zip(columns, zip(*rows))},
    index   = 'setting',
).show(maxwidth = 'full')

# Global dQ targets do not constrain insertion chromatic beta or dispersion.
# The finite-delta tunes include higher orders: matching the derivative at zero
# does not make these sampled tunes a straight line, especially vertically.

################################################################################
# Track the common launch sample without further operating-point changes
################################################################################
tracked = {}
records = {}
for label, model in [('Bare', bare), ('Corrected', line)]:
    particles       = initial.copy()
    model.track(particles, num_turns = N_TURNS, turn_by_turn_monitor = True)
    tracked[label]  = particles
    records[label]  = model.record_last_track
    print(label, 'final states/counts:',
          np.unique(particles.state, return_counts = True))
    particles.get_table().cols[
        'particle_id x px y py delta at_turn state'].show(maxwidth = 'full')

# State>0 denotes survival over this finite horizon. These nine launches do not
# establish momentum acceptance, dynamic aperture or long-term stability.

################################################################################
# Relate the momentum-dependent optics to actual tracked coordinates
################################################################################

########################################
# Compare momentum-dependent optics at the two settings
########################################
fig, axes = plt.subplots(2, 2, figsize = (11, 7))
for label in ['Bare', 'Corrected']:
    tables  = optics[label]
    axes[0, 0].plot(DELTAS * 1e4,
                    [1e3 * (tw.qy - nominal.qy) for tw in tables],
                    'o-', label = label)
    axes[0, 1].plot(DELTAS * 1e4, [1e2 * tw['bety', IP] for tw in tables],
                    'o-', label = label)
axes[0, 0].set_xlabel('delta [1e-4]')
axes[0, 0].set_ylabel('Qy - on-momentum Qy [1e-3]')
axes[0, 1].set_xlabel('delta [1e-4]')
axes[0, 1].set_ylabel('IP beta_y [cm]')

########################################
# Compare the same identified particle's physical tracking traces
########################################
launch_index    = np.flatnonzero(initial.particle_id == TRACE_ID)[0]
trace_label     = (
    f'id {TRACE_ID}, delta={initial.delta[launch_index]:+.0e}; '
    f'x_n={amplitude.ravel()[launch_index]:g}, '
    f'y_n={amplitude.ravel()[launch_index] / 2:g}')
for label in ['Bare', 'Corrected']:
    record  = records[label]
    index   = np.flatnonzero(record.particle_id[:, 0] == TRACE_ID)[0]
    turns   = record.at_turn[index]
    axes[1, 0].plot(turns, 1e3 * record.x[index], label = label)
    axes[1, 1].plot(turns, 1e3 * record.y[index], label = label)
axes[1, 0].set_ylabel('Entrance x [mm]')
axes[1, 1].set_ylabel('Entrance y [mm]')
for axis in axes[1]:
    axis.set_xlabel('Turn, before propagation')
    axis.set_title(trace_label)
for axis in axes.flat:
    axis.grid()
    axis.legend()
fig.tight_layout()
plt.show()
