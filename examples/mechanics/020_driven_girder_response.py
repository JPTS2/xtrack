# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# How do floor displacement and a girder-mounted pump reach a rigid magnet
# and its BPM? Use supplied response curves and one common bending coordinate.

################################################################################
# Required packages
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Girder, two rigid quads and the pickup next to q2
########################################
GIRDER_LENGTH_M = 6.
STATIONS_M      = np.array([1.2, 4.2])  # Stations at quad entrances in this model
HEIGHT_M        = .6                   # Neutral axis to magnetic axis
CONTACTS_M      = np.array([1.5, 4.5])  # Illustrative floor contacts
PUMP_STATION_M  = 4.8                   # Position of the vertical force on this girder

########################################
# Floor wave and one force input on this girder
########################################
FLOOR_PEAK_M = .35e-6
FLOOR_HZ     = 6.
WAVE_SPEED_M_S = 85.
PUMP_PEAK_N  = .2
PUMP_HZ      = 38.

########################################
# Illustrative conditional response parameters
########################################
RIGID_HZ        = 12.
BEND_HZ         = 37.
MAGNET_HZ       = 49.
PICKUP_HZ       = 72.
DAMPING         = .15
FLOOR_TO_BEND   = .18         # Low-frequency modal displacement / mean floor displacement
PUMP_COMPLIANCE_M_N = .4e-6   # Low-frequency modal compliance for forcing at the antinode

########################################
# Periodic input record and inspection
########################################
SAMPLE_STEP_S  = 1 / 1024
N_SAMPLES      = 2048         # Both input tones are exact Fourier bins
INSPECT_TIME_S = .125
PLOT_STOP_S    = .5

################################################################################
# Put the devices on a shared girder
################################################################################

########################################
# Keep the BPM on the q2 body at the same nominal longitudinal station
########################################
env = xt.Environment()
env.new('q1', xt.Quadrupole, length = .4, k1 = .12)
env.new('q2', xt.Quadrupole, length = .4, k1 = -.15)
env.new('bpm', xt.Device, length = 0.)
env.new('entry', xt.Drift, length = STATIONS_M[0])
env.new('gap', xt.Drift, length = STATIONS_M[1] - STATIONS_M[0] - .4)
env.new('exit', xt.Drift, length = GIRDER_LENGTH_M - STATIONS_M[1] - .4)
line = env.new_line(name = 'line', components = ['entry', 'q1', 'gap', 'bpm', 'q2', 'exit'])
line.mechanics.adopt_reference()

########################################
# Put each deformation port on the neutral axis below its magnet
########################################
mech = env.mechanics
girder = mech.new_girder('girder', anchor = xt.Frame.from_survey_angles(Y = -HEIGHT_M))
ports = ('mount.q1', 'mount.q2')
for name, port, station_m in zip(('q1', 'q2'), ports, STATIONS_M):
    mech.new_body(port, kind = 'mount', parent = girder,
                  mount = xt.Frame.from_survey_angles(Z = station_m))
    mech.new_body(name, kind = 'magnet', parent = port,
                  mount = xt.Frame.from_survey_angles(Y = HEIGHT_M),
                  elements = [name], line = line)
mech.new_body('pickup', kind = 'pickup', parent = 'q2', mount = xt.Frame(),
              elements = ['bpm'], line = line)

################################################################################
# Turn floor displacement and pump force into girder motion
################################################################################

########################################
# Sample the same wave at both declared contacts
########################################
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
wave = xt.mechanics.PlaneWave(FLOOR_PEAK_M, FLOOR_HZ, direction = (0, 0, 1),
                            speed_m_s = WAVE_SPEED_M_S)
receivers = np.column_stack((np.zeros_like(CONTACTS_M),
                             np.full(len(CONTACTS_M), -HEIGHT_M), CONTACTS_M))
floor = xt.mechanics.sample_field(wave, time_s = times, positions_m = receivers)
contact_y = floor.values[:, :, 1]
base_y = contact_y.mean(axis = 1)

########################################
# Drive rigid support translation and bending from the same floor input
########################################
# These supplied paths use the mean input. A single assumed support receiver
# can replace base_y if only a location is known. A measured multi-contact
# transfer can use the two columns directly and include differential input.
rigid_y = xt.mechanics.BaseExcitedOscillator(RIGID_HZ, DAMPING).apply(base_y, times)
floor_bend = FLOOR_TO_BEND * xt.mechanics.BaseExcitedOscillator(
    BEND_HZ, DAMPING).apply(base_y, times)

########################################
# Drive the same bending mode with a force on this support
########################################
# Project the force onto the assumed mode. Moving its injection point towards
# a node reduces excitation of the whole girder. An experimentally identified
# point-force receptance can already include this projection.
if not 0 <= PUMP_STATION_M <= GIRDER_LENGTH_M:
    raise ValueError('Place the pump force on this girder')
participation = np.sin(np.pi * PUMP_STATION_M / GIRDER_LENGTH_M)
frequency = np.fft.rfftfreq(len(times), SAMPLE_STEP_S)
ratio = frequency / BEND_HZ
pump_to_bend = xt.mechanics.TabulatedResponse(
    frequency, participation * PUMP_COMPLIANCE_M_N / (1 - ratio**2 + 2j * DAMPING * ratio))
pump_force = PUMP_PEAK_N * np.sin(2 * np.pi * PUMP_HZ * times)
pump_bend = pump_to_bend.apply(pump_force, times)
bend = floor_bend + pump_bend

########################################
# Share one bending history across both mount shapes
########################################
phase = np.pi * STATIONS_M / GIRDER_LENGTH_M
shape = np.zeros((1, len(ports), 6))
shape[0, :, 1] = np.sin(phase)                              # m of local y per m
shape[0, :, 3] = np.pi / GIRDER_LENGTH_M * np.cos(phase)    # rad of pitch per m
modal_motion = xt.mechanics.ModalMotion(
    times, bend[:, None], shape, target_names = ports, mode_names = ('vertical bending',))
rigid_values = np.zeros((len(times), 6))
rigid_values[:, 1] = rigid_y
mech.add_source('floor_rigid', xt.mechanics.RigidMotion(times, rigid_values),
                targets = 'girder')
mech.add_source('floor_and_pump_bending', modal_motion, targets = ports)

################################################################################
# Apply conditional magnet and pickup responses
################################################################################

########################################
# Compute the vertical inputs at the coincident q2 and BPM station
########################################
# This is a one-channel small-signal response law. The rigid pose still
# carries pitch and the height lever arm. The vertical effect of pitch at
# this station's height is second order and does not enter these FRFs.
q2_base_y = rigid_y + shape[0, 1, 1] * bend
q2_y = xt.mechanics.BaseExcitedOscillator(MAGNET_HZ, DAMPING).apply(q2_base_y, times)
bpm_y = xt.mechanics.BaseExcitedOscillator(PICKUP_HZ, DAMPING).apply(q2_y, times)

########################################
# Add only the relative responses to the inherited motion
########################################
# As displacement gains, the conditional FRFs supply absolute outputs at
# their ports. Subtract the input before binding on a child. The upstream
# responses are assumed to include loading from the installed assembly.
mech.add_source('magnet_relative_y', xt.mechanics.MeasuredMotion(times, q2_y - q2_base_y),
                targets = 'q2', component = 'y')
mech.add_source('pickup_relative_y', xt.mechanics.MeasuredMotion(times, bpm_y - q2_y),
                targets = 'pickup', component = 'y')

################################################################################
# Inspect the transfer paths and the moving pickup
################################################################################

########################################
# Inspect the sources and independently resolve physical output poses
########################################
print('A floor wave and a pump force drive the same girder\n'
      '--------------------------------------------------')
print(f'Floor contacts at s = {CONTACTS_M} m sample one {FLOOR_HZ:g} Hz traveling wave.')
print(f'A {PUMP_HZ:g} Hz force at s = {PUMP_STATION_M:g} m excites the shared bending coordinate.')
print('Floor input is displacement [m]; pump input is force [N]. Supplied responses set their gains and phase.')
print('The q2 magnet and its pickup inherit the support motion and add their relative responses.')
mech.get_source_table().show(cols = ['name', 'model', 'reference', 'n_targets'], maxwidth = 'full')
selected = np.flatnonzero(times <= PLOT_STOP_S)
frozen_index = int(np.argmin(abs(times - INSPECT_TIME_S)))
frozen_time = float(times[frozen_index])
t_plot = times[selected]
q2_poses = line.mechanics.get_table().rows['q2'].get_time_series(time_s = t_plot)
bpm_reading = np.array([line.mechanics.read_bpm('bpm', time_s = time_s, x = 0., y = 0.).y
                        for time_s in t_plot])
# Check the finite-height terms outside the scalar FRFs. The vertical
# projection of each rigid offset changes by (cos(pitch) - 1) times its size.
pitch = shape[0, 1, 3] * bend[selected]
cos_minus_one = -2 * np.sin(pitch / 2)**2
q2_expected = q2_y[selected] + cos_minus_one * (HEIGHT_M + (q2_y - q2_base_y)[selected])
bpm_expected = bpm_y[selected] + cos_minus_one * (HEIGHT_M + (bpm_y - q2_base_y)[selected])
assert np.allclose(q2_poses.dynamic_Y, q2_expected, rtol = 0, atol = 1e-16)
assert np.allclose(bpm_reading, -bpm_expected / np.cos(pitch), rtol = 0, atol = 1e-16)

########################################
# Report one frozen time with the driver and response units visible
########################################
at = frozen_index
reading = line.mechanics.read_bpm('bpm', time_s = frozen_time, x = 0., y = 0.).y
print(f'\nDrivers and responses at t = {frozen_time * 1e3:g} ms')
print('----------------------------------------')
print(f'Mean vertical floor motion [µm]       {base_y[at] * 1e6:+8.4f}')
print(f'Pump force at s = {PUMP_STATION_M:g} m [N]           {pump_force[at]:+8.4f}')
print(f'Pump modal participation             {participation:8.4f}')
print(f'Rigid girder translation [µm]        {rigid_y[at] * 1e6:+8.4f}')
print(f'Bending from floor / pump [µm]       {floor_bend[at] * 1e6:+8.4f} / {pump_bend[at] * 1e6:+.4f}')
print(f'q2 relative to its mount [µm]         {(q2_y[at] - q2_base_y[at]) * 1e6:+8.4f}')
print(f'BPM relative to q2 [µm]              {(bpm_y[at] - q2_y[at]) * 1e6:+8.4f}')
print(f'Reading for a design-axis beam [µm]   {reading * 1e6:+8.4f}')
print('Periodic response replay. A stationary on-axis ray isolates the mechanical BPM signal.')

################################################################################
# Plot how the inputs reach the magnet and pickup
################################################################################

########################################
# Show the two drives and their bending response
########################################
fig, axes = plt.subplots(2, 2, figsize = (11., 7.1), sharex = True, layout = 'constrained')
axes[0, 0].plot(t_plot, contact_y[selected] * 1e6, alpha = .65)
axes[0, 0].plot(t_plot, base_y[selected] * 1e6, color = '0.2', ls = '--')
axes[0, 0].set(title = 'Floor input at the declared contacts', ylabel = 'Vertical displacement [µm]')
axes[0, 0].legend([f's = {s:g} m' for s in CONTACTS_M] + ['Mean input'],
                  ncol = 3, loc = 'lower center', bbox_to_anchor = (.5, 1.), frameon = False)

axes[0, 1].plot(t_plot, floor_bend[selected] * 1e6, label = 'Floor contribution')
axes[0, 1].plot(t_plot, pump_bend[selected] * 1e6, label = 'Pump contribution')
axes[0, 1].set(title = 'Two inputs drive the same bending mode', ylabel = 'Bend amplitude [µm]')
axes[0, 1].legend(ncol = 2, loc = 'lower center', bbox_to_anchor = (.5, 1.), frameon = False)

########################################
# Compare physical magnet motion and the pickup signal
########################################
axes[1, 0].plot(t_plot, rigid_y[selected] * 1e6, ls = '--', label = 'Rigid girder')
axes[1, 0].plot(t_plot, q2_base_y[selected] * 1e6, label = 'q2 mount')
axes[1, 0].plot(t_plot, q2_poses.dynamic_Y * 1e6, label = 'q2 magnetic axis')
axes[1, 0].set(title = 'From common support input to magnet motion', xlabel = 'Time [s]', ylabel = 'Vertical motion [µm]')
axes[1, 0].legend(ncol = 3, loc = 'lower center', bbox_to_anchor = (.5, 1.),
                  frameon = False, fontsize = 9)

# The matched rigid-pickup case has the same pitch as q2. Project the fixed
# on-axis ray into that tilted pickup plane, just as for the responded pickup.
rigid_bpm_reading = -q2_poses.dynamic_Y / np.cos(pitch)
axes[1, 1].plot(t_plot, rigid_bpm_reading * 1e6, ls = '--', label = 'If BPM followed q2 exactly')
axes[1, 1].plot(t_plot, bpm_reading * 1e6, label = 'With BPM response')
axes[1, 1].set(title = 'Electrical reading for a fixed on-axis ray', xlabel = 'Time [s]', ylabel = 'BPM reading [µm]')
axes[1, 1].legend(ncol = 2, loc = 'lower center', bbox_to_anchor = (.5, 1.),
                  frameon = False, fontsize = 9)
for ax in axes.flat:
    ax.set_title(ax.get_title(), pad = 27)
    ax.axvline(frozen_time, color = '0.5', ls = ':', lw = 1.)
    ax.grid(alpha = .2)
plt.show()
