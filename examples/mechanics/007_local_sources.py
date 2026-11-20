# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Mount a pump in girder-local coordinates and prescribe its two coupling paths.
# Changing the mount changes the floor footprint, gain and travel delays.

################################################################################
# Required packages
################################################################################

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Line and support selection
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
QUADRUPOLES  = ['arc.s1.c1.qf.in', 'arc.s1.c1.qd', 'arc.s1.c1.qf.out']
OBSERVATION_BPM = 'bpm.s1.f'

########################################
# Current and alternative pump mounts
########################################
PUMP_LOCAL_M        = (.8, -.5, 0.)   # Local x,y,s on the first girder
ALTERNATIVE_LOCAL_M = (-.8, -.5, 2.)

########################################
# Pump excitation and direct support coupling
########################################
FREQUENCY_HZ        = 25.
EMITTED_PEAK_M      = 50e-9           # Prescribed displacement, not force
DIRECT_MOUNT_PEAK_M = 25e-9           # Same tone, direct support path [m]

########################################
# Supplied floor propagation law
########################################
DECAY_LENGTH_M        = 20.           # Amplitude gain exp(-distance/length)
PROPAGATION_SPEED_M_S = 50.
POLARIZATION          = (0., 1., 0.)  # World XYZ

########################################
# Record sampling and inspection
########################################
N_SAMPLES      = 4096
SAMPLE_STEP_S  = 1 / 512
INSPECT_TIME_S = .01

################################################################################
# Set up the ordinary Xtrack line
################################################################################

timings = {}
PRINT_TIMINGS = False                 # Optional diagnostics after the physical comparison

started = perf_counter()

########################################
# Locate and load the demonstration ring
########################################
repo_root = Path(__file__).resolve().parents[2]
nominal_env = xt.load(repo_root / LATTICE_FILE)
line = nominal_env['demo'].copy()
env = line.env
env['demo'] = line

timings['line_load_and_copy_s'] = perf_counter() - started

########################################
# Prepare and time the tracker before mechanics queries
########################################
started = perf_counter()
line.build_tracker()
timings['tracker_build_s'] = perf_counter() - started

################################################################################
# Place the supports and mounted equipment
################################################################################

########################################
# Access the environment mechanics model
########################################
mech = env.mechanics

started = perf_counter()

########################################
# Adopt existing offsets as the deliberate reference
########################################
line.mechanics.adopt_reference()

timings['reference_adoption_s'] = perf_counter() - started

started = perf_counter()

########################################
# Assign the quadrupoles to their supports
########################################
girders = [line.mechanics.new_girder(f'g.{name}', elements = [name]) for name in QUADRUPOLES]

########################################
# Mount the pump in the first support's local frame
########################################
pump = mech.new_body('pump', kind = 'equipment', parent = girders[0], mount = xt.Frame.from_survey_angles(
    X = PUMP_LOCAL_M[0], Y = PUMP_LOCAL_M[1], Z = PUMP_LOCAL_M[2],
))
timings['mechanics_definition_s'] = perf_counter() - started

started = perf_counter()

########################################
# Define the alternative local mount for comparison
########################################
alternative = girders[0].nominal_frame @ xt.Frame.from_survey_angles(
    X = ALTERNATIVE_LOCAL_M[0], Y = ALTERNATIVE_LOCAL_M[1], Z = ALTERNATIVE_LOCAL_M[2],
)

########################################
# Collect the support receiver positions
########################################
positions = np.array([g.nominal_frame.XYZ for g in girders])

timings['source_geometry_s'] = perf_counter() - started

################################################################################
# Define the two pump coupling paths
################################################################################

started = perf_counter()

########################################
# Define one common pump waveform
########################################
waveform = xt.mechanics.Harmonic(amplitude = EMITTED_PEAK_M, frequency_hz = FREQUENCY_HZ)

########################################
# Attach a propagation law to the current pump position
########################################
source = xt.mechanics.LocatedSource(
    waveform = waveform, position_m = tuple(pump.nominal_frame.XYZ),
    decay_length_m = DECAY_LENGTH_M, speed_m_s = PROPAGATION_SPEED_M_S,
    polarization = POLARIZATION,
)

########################################
# Use the same waveform at the alternative mount
########################################
other_source = xt.mechanics.LocatedSource(
    waveform = waveform, position_m = tuple(alternative.XYZ),
    decay_length_m = DECAY_LENGTH_M, speed_m_s = PROPAGATION_SPEED_M_S,
    polarization = POLARIZATION,
)

########################################
# Sample both floor fields at the same receivers and times
########################################
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
field = xt.mechanics.sample_field(source, time_s = times, positions_m = positions)
other_field = xt.mechanics.sample_field(other_source, time_s = times, positions_m = positions)

########################################
# Bind the floor path and direct mount path separately
########################################
mech.add_source('pump_floor_path', source, targets = [g.name for g in girders])
mech.add_source('pump_mount_path', xt.mechanics.Harmonic(
    amplitude = DIRECT_MOUNT_PEAK_M, frequency_hz = FREQUENCY_HZ,
), targets = [girders[0].name], component = 'y')

timings['sample_field_s'] = perf_counter() - started

################################################################################
# Inspect the supplied propagation model
################################################################################

########################################
# Calculate distances to the mounted source
########################################
distance = np.linalg.norm(positions - pump.nominal_frame.XYZ, axis = 1)

########################################
# Report amplitude gains, delays and direct coupling
########################################
print('Mounted pump and supplied propagation\n--------------------------------------')
print('Receiver       Distance [m]    Amplitude gain    Delay [ms]')
for index, radius in enumerate(distance):
    print(f'{"Support " + str(index + 1):<14} {radius:10.4f} {np.exp(-radius / DECAY_LENGTH_M):17.6f}'
          f' {radius / PROPAGATION_SPEED_M_S * 1e3:13.5f}')
print(f'Pump mount local x/y/s [m]: {PUMP_LOCAL_M}')
print(f'Direct mount tone [nm peak]: {DIRECT_MOUNT_PEAK_M * 1e9:.3f}')
print('The floor law and direct coupling are supplied models, not inferred elasticity.')
print('The pump body fixes nominal mounting; propagation uses that reference geometry.')

################################################################################
# Separate direct coupling, propagation and their combined beam effect
################################################################################

########################################
# Compare matched frozen closed orbits with each path enabled in turn
########################################
# Every case uses the same line and pump phase. The floor path includes supplied
# geometric attenuation/delay; direct coupling acts only on the mounted support.
mech.apply(time_s = INSPECT_TIME_S, disabled_sources = tuple(mech.sources))
baseline_twiss = line.twiss4d()
base_y = baseline_twiss['y', OBSERVATION_BPM]
path_cases = {
    'Direct mount path only': ('pump_floor_path',),
    'Propagation path only': ('pump_mount_path',),
    'Both paths': (),
}
closed_orbit_changes = {}
first_support_displacements = {}
for label, disabled in path_cases.items():
    mech.apply(time_s = INSPECT_TIME_S, disabled_sources = disabled)
    state = mech.survey(time_s = INSPECT_TIME_S) if not disabled else mech.get_body_poses(
        time_s = INSPECT_TIME_S, disabled_sources = disabled,
    )
    first_support_displacements[label] = float(env[QUADRUPOLES[0]].shift_y)
    closed_orbit_changes[label] = line.twiss4d()['y', OBSERVATION_BPM] - base_y
print(f'\nMatched physical states at t = {INSPECT_TIME_S:g} s\n'
      '------------------------------------------------')
print(f'{"Enabled coupling":<27} {"First quad y [nm]":>18} {"BPM orbit change [nm]":>24}')
for label in path_cases:
    print(f'{label:<27} {first_support_displacements[label] * 1e9:+18.3f}'
          f' {closed_orbit_changes[label] * 1e9:+24.3f}')
print(f'The BPM is {OBSERVATION_BPM}; these are fixed-map closed orbits, not a dynamic feedback run.')

########################################
# Draw where the pump is mounted and where the propagated disturbance arrives
########################################
offset = np.linspace(-50, 50, 101)
xx, zz = np.meshgrid(offset, offset)
grid = pump.nominal_frame.XYZ + np.column_stack((xx.ravel(), np.zeros(xx.size), zz.ravel()))
gain = np.exp(-np.linalg.norm(grid - pump.nominal_frame.XYZ, axis = 1) / DECAY_LENGTH_M)
fig, axes = plt.subplots(2, 2, figsize = (11, 7))
image = axes[0, 0].pcolormesh(xx, zz, gain.reshape(xx.shape), shading = 'auto', vmin = 0, vmax = 1)
axes[0, 0].plot(0, 0, 'r*', ms = 12, label = 'Current pump mount')
alt = alternative.XYZ - pump.nominal_frame.XYZ
axes[0, 0].plot(alt[0], alt[2], '*', color = 'lime', mec = 'k', ms = 10, label = 'Alternative mount')
relative = positions - pump.nominal_frame.XYZ
for index, point in enumerate(relative):
    axes[0, 0].plot(point[0], point[2], 'wo', mec = 'k')
    axes[0, 0].annotate(f'Support {index + 1}', (point[0], point[2]), xytext = (5, 4),
                       textcoords = 'offset points', fontsize = 8)
axes[0, 0].set(xlabel = 'World X relative to pump [m]', ylabel = 'World Z relative to pump [m]',
               aspect = 'equal', title = 'Pump, supports and propagation gain')
axes[0, 0].legend(fontsize = 8, loc = 'lower left')
fig.colorbar(image, ax = axes[0, 0], label = 'Amplitude gain [m/m]')

########################################
# Resolve the actual mounts and first receiver inside the larger footprint
########################################
zoom = axes[0, 0].inset_axes([.54, .53, .43, .42])
mount_points = np.array([[0., 0.], alt[[0, 2]], relative[0, [0, 2]]])
zoom.plot(mount_points[[0, 2], 0], mount_points[[0, 2], 1], ':', color = '0.6', lw = .8)
zoom.plot(mount_points[[1, 2], 0], mount_points[[1, 2], 1], ':', color = '0.6', lw = .8)
for point, marker, colour, text, offset_text in [
    (mount_points[0], '*', 'tab:red', 'Pump', (4, -10)),
    (mount_points[1], '*', 'limegreen', 'Alt', (4, 3)),
    (mount_points[2], 'o', 'white', '1', (-9, 4)),
]:
    zoom.plot(*point, marker, color = colour, mec = 'k', ms = 7 if marker == '*' else 5)
    zoom.annotate(text, point, xytext = offset_text, textcoords = 'offset points', fontsize = 7)
zoom.set(xlim = (mount_points[:, 0].min() - .7, mount_points[:, 0].max() + .7),
          ylim = (mount_points[:, 1].min() - .7, mount_points[:, 1].max() + .7), aspect = 'equal')
zoom.set_title('Mount detail: top view [m]', fontsize = 7, pad = 3)
zoom.tick_params(labelsize = 7)

########################################
# Show constructive or destructive combination at the mounted support
########################################
direct_y = DIRECT_MOUNT_PEAK_M * np.sin(2 * np.pi * FREQUENCY_HZ * times)
floor_y = field.values[:, 0, 1]
for values, label in [(direct_y, 'Direct mount'), (floor_y, 'Propagation'),
                       (direct_y + floor_y, 'Combined')]:
    axes[0, 1].plot(times, values * 1e9, label = label)
axes[0, 1].set(xlabel = 'Physical time [s]', ylabel = 'First support y [nm]', xlim = (0, .12),
               title = 'Two paths at the mounted support')
axes[0, 1].legend(fontsize = 7.5, loc = 'upper center', ncol = 3)
axes[0, 1].margins(y = .3)

########################################
# Compare receiver delay and the effect of moving the pump
########################################
for index in range(len(girders)):
    axes[1, 0].plot(times, field.values[:, index, 1] * 1e9, label = f'Support {index + 1}')
axes[1, 0].plot(times, other_field.values[:, 0, 1] * 1e9, '--', color = '0.35',
                label = 'Support 1: alternative mount')
axes[1, 0].set(xlabel = 'Physical time [s]', ylabel = 'Propagated y [nm]', xlim = (0, .12),
               title = 'Distance changes both gain and arrival phase')
axes[1, 0].legend(fontsize = 7.5, loc = 'upper center', ncol = 2)
axes[1, 0].margins(y = .4)

########################################
# Connect the support disturbance to a measurable beam quantity
########################################
labels = list(path_cases)
bars = axes[1, 1].barh(np.arange(len(labels)),
                       [closed_orbit_changes[label] * 1e9 for label in labels], color = ['tab:blue', 'tab:orange', 'tab:green'])
axes[1, 1].bar_label(bars, fmt = '%+.2f', padding = 4, fontsize = 9)
axes[1, 1].set(yticks = np.arange(len(labels)), yticklabels = ['Direct mount', 'Propagation', 'Both paths'],
               xlabel = 'Frozen BPM orbit change [nm]', title = 'Matched beam response at the selected phase')
axes[1, 1].margins(x = .28)
axes[1, 1].invert_yaxis()
for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
    ax.grid(alpha = .2)
    ax.set_axisbelow(True)
fig.tight_layout()

################################################################################
# Inspect the combined mechanical state
################################################################################

started = perf_counter()

########################################
# Apply both paths at the selected physical time
########################################
mech.apply(time_s = INSPECT_TIME_S)
timings['apply_s'] = perf_counter() - started

started = perf_counter()
survey = mech.survey(time_s = INSPECT_TIME_S)

timings['pose_inspection_s'] = perf_counter() - started

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the measured costs
########################################
if PRINT_TIMINGS:
    print('\nCPU operation costs\n-------------------')
    for name, seconds in timings.items():
        print(f'{name:<32} {seconds * 1e3:10.3f} ms')

########################################
# Display the figures
########################################
plt.show()
