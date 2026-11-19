# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Sample one prescribed physical girder from two beams with different turn times.
# Separate the shared world motion from opposite local coordinates and clocks.

################################################################################
# Required packages
################################################################################
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Geometry, source and clocks
########################################
FORWARD_RETURN_M  = 100.
BACKWARD_RETURN_M = 180.
AMPLITUDE_M       = 50e-9
FREQUENCY_HZ      = 1e5                 # High on purpose: turn numbers differ visibly
TIME_ORIGIN_S     = 0.                  # Shared physical origin, not a shared turn counter
N_TURNS           = 64
SAMPLE_STEP_S     = 1e-8
N_SAMPLES         = 8192
COMPARE_TIME_S    = 1e-5

################################################################################
# Place two distinct line occurrences in one mechanical world
################################################################################
started = perf_counter()
env = xt.Environment()
env.new('q.forward', xt.Quadrupole, length = 1., k1 = 0.)
env.new('q.backward', xt.Quadrupole, length = 1., k1 = 0.)
env.new('return.forward', xt.Drift, length = FORWARD_RETURN_M)
env.new('return.backward', xt.Drift, length = BACKWARD_RETURN_M)
forward = env.new_line(name = 'forward', components = ['q.forward', 'return.forward'])
backward = env.new_line(name = 'backward', components = ['q.backward', 'return.backward'])
for line in (forward, backward):
    line.set_particle_ref('proton', p0c = 7e12)
backward.mechanics.set_reference_frame(xt.Frame.from_survey_angles(Z = 1., theta = np.pi))
mech = env.mechanics
girder = forward.mechanics.new_girder('shared', elements = ['q.forward'])
backward.mechanics.attach(girder, elements = ['q.backward'])
source = xt.mechanics.PlaneWave(
    amplitude = AMPLITUDE_M, frequency_hz = FREQUENCY_HZ, speed_m_s = 2e4,
    direction = (0., 0., 1.), polarization = (1., 0., 0.), phase_rad = np.pi / 2,
)
mech.add_source('surface_wave', source, targets = ['shared'])
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
motion = mech.compute_time_series(time_s = times)
timings = {'setup_and_record_s': perf_counter() - started}

################################################################################
# Advance each beam with its own clock and line-local effective fields
################################################################################
histories = {}
for label, line, element in [('Forward', forward, 'q.forward'), ('Backward', backward, 'q.backward')]:
    started = perf_counter()
    line.build_tracker()
    timings[label.lower() + '_tracker_build_s'] = perf_counter() - started
    particles = line.build_particles()
    sampled_times, sampled_x = [], []
    line.mechanics.enable_vibrations(time_series = motion, time_origin_s = TIME_ORIGIN_S)
    started = perf_counter()
    line.track(particles, num_turns = N_TURNS, log = xt.Log(
        time_s = lambda ll, pp: sampled_times.append(pp.time_s) or pp.time_s,
        shift_x = lambda ll, pp: sampled_x.append(float(ll[element].shift_x)) or float(ll[element].shift_x),
    ))
    timings[label.lower() + '_tracking_s'] = perf_counter() - started
    line.mechanics.disable_vibrations()
    histories[label] = (np.array(sampled_times) + TIME_ORIGIN_S, np.array(sampled_x))

########################################
# Compare the same physical time explicitly
########################################
forward.mechanics.apply(time_s = COMPARE_TIME_S, time_series = motion)
backward.mechanics.apply(time_s = COMPARE_TIME_S, time_series = motion)
np.testing.assert_allclose(env['q.forward'].shift_x, -env['q.backward'].shift_x, atol = 1e-15, rtol = 0)
print('Shared body and separate beam clocks\n------------------------------------')
print('Beam       Turn spacing [µs]    Last sampled time [µs]')
for label, (sampled_times, _) in histories.items():
    print(f'{label:<10} {np.diff(sampled_times).mean() * 1e6:16.6f} {sampled_times[-1] * 1e6:25.6f}')
print(f'Same-time local x forward / backward [nm]: {env["q.forward"].shift_x * 1e9:+.6f} / '
      f'{env["q.backward"].shift_x * 1e9:+.6f}')
print('Both lines query one immutable field; running one does not advance the other.')
print('This does not schedule causal shared filters, beam encounters or controllers.')

################################################################################
# Compare world coordinates against physical time and local coordinates by turn
################################################################################
fig, axes = plt.subplots(2, 1, figsize = (8, 6.5))
curve_time = np.linspace(0, max(t[-1] for t, _ in histories.values()), 1000)
curve_x = source.evaluate(curve_time, np.array([girder.nominal_frame.XYZ]))[:, 0, 0]
axes[0].plot(curve_time * 1e6, curve_x * 1e9, color = '0.6', label = 'Shared world motion')
for label, sign in [('Forward', 1.), ('Backward', -1.)]:
    sampled_times, local_x = histories[label]
    axes[0].plot(sampled_times * 1e6, sign * local_x * 1e9, '.', label = label)
    axes[1].plot(np.arange(N_TURNS), local_x * 1e9, '.-', label = label)
axes[0].set(xlabel = 'Physical time [µs]', ylabel = 'World X [nm]')
axes[1].set(xlabel = 'Turn index in each beam', ylabel = 'Line-local x [nm]')
axes[0].legend()
axes[1].legend()
fig.tight_layout()
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<30} {seconds:9.4f} s')
plt.show()
