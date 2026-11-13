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
# Corrected model and stored incoming coordinates
########################################
MODEL_FILE      = 'outputs/psb_corrected_collapse.json'
DISTRIBUTION    = 'test_data/psb_chicane/inj_distrib.dat'
INJECTION_SITE  = 'bi1.tstr1l1'

########################################
# Bounded multiturn injection on CPU
########################################
N_INJECT_TURNS  = 16
N_PER_TURN      = 4
N_TURNS         = 6000
BEND_SLICES     = 2
QUAD_SLICES     = 8
PARTICLE_WEIGHT = 10.0

################################################################################
# Load the corrected injection model and incoming distribution
################################################################################

########################################
# Load and slice the model while retaining the clock expressions
########################################
script_dir  = Path(__file__).resolve().parent
repo_root   = script_dir.parents[2]
line        = xt.load(script_dir / MODEL_FILE)
# Thin tracking retains the live parent circuits and their clock expressions.
line_thick  = line.copy()
line.slice_thick_elements(slicing_strategies = [
    xt.Strategy(slicing = None),
    xt.Strategy(slicing = xt.Teapot(BEND_SLICES), element_type = xt.Bend),
    xt.Strategy(slicing = xt.Teapot(QUAD_SLICES),
                element_type = xt.Quadrupole),
])

########################################
# Compare thick and thin optics at three collapse times
########################################
optics_times        = np.array([0.0, 2.75e-3, 5.5e-3])
foil_closed_orbit   = []
for time_s in optics_times:
    line['t_turn_s']        = time_s
    line_thick['t_turn_s']  = time_s
    tw                      = line.twiss()
    tw_thick                = line_thick.twiss()
    foil_closed_orbit.append(tw['x', INJECTION_SITE])
    print(f't = {time_s * 1e3:.2f} ms: thin - thick tunes:',
          tw.qx - tw_thick.qx, tw.qy - tw_thick.qy)
line['t_turn_s'] = 0.0
line.discard_tracker()

########################################
# Convert incoming energy and coordinates to the native reference
########################################
source      = np.loadtxt(repo_root / DISTRIBUTION, skiprows = 3)
reference   = line.particle_ref
energy      = source[:, 8] * 1e6 + reference.mass0
ptau        = (energy - reference.energy0[0]) / reference.p0c[0]
incoming    = xt.Particles(
    'proton',
    p0c     = reference.p0c[0],
    ptau    = ptau,
    x       = source[:, 0] * 1e-3,
    y       = source[:, 2] * 1e-3,
    zeta    = source[:, 4] * 1e-3,
    weight  = PARTICLE_WEIGHT,
)
# The input angles are slopes in mrad; native momenta are normalized to p0.
incoming.px = source[:, 1] * 1e-3 * (1 + incoming.delta)
incoming.py = source[:, 3] * 1e-3 * (1 + incoming.delta)
# The source is placed directly in the stored coordinates, without rematching
# to the displaced circulating orbit. The resulting mismatch is visible.
# Its H-minus labels are not a stripping/foil interaction model.

########################################
# Install the native injection sampler
########################################
injection = xt.ParticlesInjectionSample(
    particles_to_inject     = incoming,
    line                    = line,
    element_name            = 'injection',
    num_particles_to_inject = N_PER_TURN,
)
line.insert('injection', injection, at = INJECTION_SITE)
monitor = xt.ParticlesMonitor(
    start_at_turn = 0, stop_at_turn = N_TURNS,
    num_particles = N_INJECT_TURNS * N_PER_TURN + 1)
line.insert('foil.monitor', monitor, at = INJECTION_SITE)
# The public sampler draws randomly from the source each turn, without
# depleting it. Its random realization is not controlled by a seed here.
particles   = line.build_particles(_capacity = N_INJECT_TURNS * N_PER_TURN + 1)
particles.state[0] = -500
# The inactive starter leaves capacity for particles added by the sampler.

################################################################################
# Track the accumulating population with the collapse clock enabled
################################################################################
line['t_turn_s']                = 0.0
line.enable_time_dependent_vars = True
intensity                       = []
for turn in range(N_INJECT_TURNS):
    line.track(particles, num_turns = 1)
    alive = particles.state > 0
    intensity.append(np.sum(particles.weight[alive]))
injection.num_particles_to_inject = 0
line.track(particles, num_turns = N_TURNS - N_INJECT_TURNS)
# The source is now off; follow the stored population through the collapse.
# There is no foil scattering, space charge or validated aperture acceptance.
print('Final live macroparticle count:', np.count_nonzero(particles.state > 0))
print('Tracked laboratory interval [us]:', line['t_turn_s'] * 1e6)
print('Final represented population:',
      np.sum(particles.weight[particles.state > 0]))

################################################################################
# Inspect accumulation and the surviving transverse distribution
################################################################################
fig, axes   = plt.subplots(1, 2, figsize = (11, 4))
turns       = np.arange(1, N_INJECT_TURNS + 1)
axes[0].plot(turns, intensity, '.-', label = 'Circulating')
axes[0].plot(turns, turns * N_PER_TURN * PARTICLE_WEIGHT, '--',
             label = 'Injected')
axes[0].set(xlabel = 'Turns', ylabel = 'Represented population')
axes[0].legend()
alive = particles.state > 0
axes[1].plot(particles.x[alive] * 1e3, particles.px[alive] * 1e3, '.',
             markersize = 2)
axes[1].set(xlabel = 'x at line entrance [mm]', ylabel = '10^3 px')
fig.tight_layout()

########################################
# Compare the circulating centroid with the programmed closed orbit
########################################
alive_samples   = monitor.state > 0
counts          = np.sum(alive_samples, axis = 0)
centroid        = np.full(N_TURNS, np.nan)
valid           = counts > 0
centroid[valid] = np.sum(
    np.where(alive_samples, monitor.x, 0), axis = 0)[valid] / counts[valid]
time_ms         = np.arange(N_TURNS) * line.get_length() / (
    reference.beta0[0] * c) * 1e3
fig, axes       = plt.subplots(2, 1, figsize = (10, 7), sharex = True)
axes[0].plot(time_ms, centroid * 1e3, label = 'Circulating centroid')
axes[0].plot(optics_times * 1e3, np.array(foil_closed_orbit) * 1e3,
             'ko', label = 'Instantaneous closed-orbit samples')
axes[0].legend()
axes[0].set_ylabel('Centroid x at foil [mm]')
axes[1].plot(time_ms, counts)
axes[1].set(xlabel = 'Time from injection [ms]',
            ylabel = 'Live macroparticles at foil')
for axis in axes:
    axis.grid(alpha = 0.3)
fig.tight_layout()
plt.show()
