# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Assign supports on the same demonstration ring as the feedback examples.
# Inspect wave, PSD or stated motion with native mechanics survey and plots.

################################################################################
# Required packages
################################################################################

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Ring and source selection
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
SOURCE       = 'wave'                                             # 'wave', 'psd' or 'measured'

########################################
# First-support installation errors
########################################
INSTALLATION_SHIFT_M    = (5e-9, 10e-9, 3e-9)
INSTALLATION_ANGLES_RAD = (1e-6, 2e-6, 3e-6)   # Survey pitch, yaw, roll

########################################
# Traveling-wave parameters
########################################
WAVE_AMPLITUDE_M  = 50e-9             # Peak vector displacement [m]
WAVE_FREQUENCY_HZ = 5.
WAVE_DIRECTION    = (1., 0., 1.)      # World XYZ
WAVE_POLARIZATION = (.3, 1., .2)
WAVE_SPEED_M_S    = 200.

########################################
# Correlated PSD alternative
########################################
PSD_FREQUENCY_HZ         = [0., 1., 2., 30., 40.]
PSD_DENSITY_M2_PER_HZ    = [0., 0., 1e-17, 1e-17, 0.]
PSD_CORRELATION_LENGTH_M = 100.
PSD_SEED                 = 956

########################################
# Frozen inspection time
########################################
TIME_S = .125
PRINT_TIMINGS = False

########################################
# Motion sampling
########################################
SAMPLE_STEP_S = 1 / 256
N_SAMPLES     = 2048

################################################################################
# Set up the ordinary Xtrack line
################################################################################

########################################
# Start performance monitoring
########################################
timings = {}

########################################
# Time line load and copy
########################################
started = perf_counter()

########################################
# Locate and load the demonstration ring
########################################
repo_root = Path(__file__).resolve().parents[2]
nominal_env = xt.load(repo_root / LATTICE_FILE)
line = nominal_env['demo'].copy()        # Reference backend rejects shared construction sublines
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
# Assign mechanical supports to the existing ring
################################################################################

########################################
# Access the environment mechanics model
########################################
mech = env.mechanics

########################################
# Time reference adoption
########################################
started = perf_counter()

########################################
# Adopt existing offsets as the deliberate reference
########################################
line.mechanics.adopt_reference()

timings['reference_adoption_s'] = perf_counter() - started

########################################
# Time geometry selection
########################################
started = perf_counter()

########################################
# Survey the nominal frames and select quadrupoles
########################################
nominal = line.survey(include_element_frames = True)
quads = list(line.get_table().rows.match(element_type = 'Quadrupole').name)

timings['geometry_selection_s'] = perf_counter() - started

########################################
# Create the illustrative support assignments
########################################
# Illustrative supports, not an FCC-ee mechanical layout. Replace this loop
# with a machine-specific env.call(...) script for actual assignments.
########################################
# Time mechanics definition
########################################
started = perf_counter()

assignments = {f'g.{name}': [name] for name in quads}
girders = list(line.mechanics.new_girders(assignments).values())

########################################
# Set the first support's installation placement
########################################
first = girders[0]

timings['mechanics_definition_s'] = perf_counter() - started
########################################
# Time installation definition
########################################
started = perf_counter()

first.installation.shift_x, first.installation.shift_y, first.installation.shift_s = INSTALLATION_SHIFT_M
first.installation.rot_x_rad, first.installation.rot_y_rad, first.installation.rot_s_rad = INSTALLATION_ANGLES_RAD

timings['installation_definition_s'] = perf_counter() - started

################################################################################
# Define and sample the floor input
################################################################################

########################################
# Start timing motion preparation
########################################
started = perf_counter()

########################################
# Choose physical sample times
########################################
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S

########################################
# Choose a wave, PSD or stated displacement trace
########################################
if SOURCE == 'wave':
    source = xt.mechanics.PlaneWave(
        amplitude = WAVE_AMPLITUDE_M, frequency_hz = WAVE_FREQUENCY_HZ,
        direction = WAVE_DIRECTION, polarization = WAVE_POLARIZATION,
        speed_m_s = WAVE_SPEED_M_S,
    )
elif SOURCE == 'psd':
    source = xt.mechanics.DisplacementPSD(
        frequency_hz = PSD_FREQUENCY_HZ, density = PSD_DENSITY_M2_PER_HZ,
        correlation_length_m = PSD_CORRELATION_LENGTH_M, seed = PSD_SEED,
    )
elif SOURCE == 'measured':
    source = xt.mechanics.MeasuredMotion(
        time_s = [0., 1., 3., times[-1]], displacement = [0., 30e-9, -20e-9, 0.],
    )
else:
    raise ValueError('Choose wave, psd or measured')

########################################
# Bind one jointly sampled source to every support
########################################
mech.add_source('floor_input', source, targets = [g.name for g in girders], component = 'y')

########################################
# Compute a reusable motion record
########################################
motion = mech.compute_time_series(time_s = times)

########################################
# Record the motion preparation cost
########################################
timings['sample_motion_s'] = perf_counter() - started

################################################################################
# Verify the applied element survey
################################################################################

########################################
# Start timing application and inspection
########################################
started = perf_counter()

########################################
# Predict the element poses without modifying the line
########################################
targets = line.mechanics.get_element_poses(time_s = TIME_S, time_series = motion)
timings['pose_prediction_s'] = perf_counter() - started

########################################
# Time alignment application separately from survey verification
########################################
started = perf_counter()

########################################
# Apply the selected mechanical state
########################################
mech.apply(time_s = TIME_S, time_series = motion)
timings['apply_s'] = perf_counter() - started

########################################
# Time the independent native-survey verification
########################################
started = perf_counter()
moved = line.survey(include_element_frames = True)

########################################
# Compare requested and actual positions and orientations
########################################
survey_error_m = max(np.max(np.abs(
    moved.get_frame(name, which = 'elem_start').XYZ - targets[name].XYZ,
)) for name in quads)
survey_error_rad = max(Rotation.from_matrix(
    moved.get_frame(name, which = 'elem_start').matrix[:3, :3] @ targets[name].matrix[:3, :3].T,
).magnitude() for name in quads)
assert survey_error_m < 1e-10 and survey_error_rad < 1e-12

########################################
# Read the mechanics state and record the inspection cost
########################################
survey = mech.survey(time_s = TIME_S, time_series = motion)
timings['survey_verification_s'] = perf_counter() - started

################################################################################
# Measure the frozen orbit change caused by the same support displacements
################################################################################

########################################
# Hold the installation errors fixed and remove only the dynamic input
########################################
started = perf_counter()
mech.apply(time_s = TIME_S, time_series = motion, disabled_sources = tuple(mech.sources))
installed_twiss = line.twiss4d()

########################################
# Apply the chosen wave phase and compute the changed closed orbit
########################################
mech.apply(time_s = TIME_S, time_series = motion)
moving_twiss = line.twiss4d()
# This compares two fixed maps. It does not assert that a driven beam follows
# their closed orbits; dynamic tracking appears in the later combined lesson.
dx = moving_twiss.x - installed_twiss.x
dy = moving_twiss.y - installed_twiss.y
timings['Matched frozen optics'] = perf_counter() - started

########################################
# State the physical change and the measured beam response
########################################
print('A spatial motion field across a surveyed ring\n--------------------------------------------')
print(f'Ring length: {line.get_length():.1f} m; quadrupoles on individual supports: {len(girders)}.')
print(f'Source model: {SOURCE}; displayed physical time: {TIME_S:g} s.')
if SOURCE == 'wave':
    print(f'Wave: {WAVE_FREQUENCY_HZ:g} Hz, speed {WAVE_SPEED_M_S:g} m/s, wavelength {WAVE_SPEED_M_S / WAVE_FREQUENCY_HZ:g} m.')
    print(f'Propagation direction XYZ: {WAVE_DIRECTION}; polarization XYZ: {WAVE_POLARIZATION} (normalized independently).')
print('Both optics cases retain the same first-support installation error.')
print('Only the prescribed dynamic support motion changes between the cases.')
print('\nFrozen mechanical and beam changes\n----------------------------------')
print(f'Largest |world-Y support displacement| [nm]: {np.max(np.abs(survey.dynamic_Y)) * 1e9:.3f}')
print(f'Largest |horizontal closed-orbit change| [µm]: {np.max(np.abs(dx)) * 1e6:.6f}')
print(f'Largest |vertical closed-orbit change| [µm]: {np.max(np.abs(dy)) * 1e6:.6f}')
print(f'Applied-frame check: largest position error {survey_error_m:.3g} m, orientation error {survey_error_rad:.3g} rad.')
print('These support assignments illustrate the API; they are not an engineering girder layout.')

################################################################################
# Connect spatial support motion to the resulting frozen orbit change
################################################################################

########################################
# Locate the moving quadrupoles on the actual surveyed ring
########################################
fig, axes = plt.subplots(2, 2, figsize = (10, 7.5))
axes[0, 0].plot(nominal.X, nominal.Z, color = '0.75', lw = 1)
centres = np.array([g.nominal_frame.XYZ for g in girders])
vertical_nm = survey.dynamic_Y * 1e9
limit = max(float(np.max(np.abs(vertical_nm))), 1e-12)
points = axes[0, 0].scatter(centres[:, 0], centres[:, 2], c = vertical_nm,
                           vmin = -limit, vmax = limit, cmap = 'coolwarm', s = 14)
fig.colorbar(points, ax = axes[0, 0], label = 'Support world-Y motion [nm]', shrink = .8)
if SOURCE == 'wave':
    direction = np.array(WAVE_DIRECTION, dtype = float)
    direction /= np.linalg.norm(direction)
    origin = centres[:, [0, 2]].mean(axis = 0)
    length = .3 * np.ptp(centres[:, [0, 2]], axis = 0).max()
    axes[0, 0].annotate('Wave travels', xy = origin + length * direction[[0, 2]], xytext = origin,
                       arrowprops = {'arrowstyle': '->', 'color': '0.2'}, fontsize = 9)
axes[0, 0].set(title = 'Support locations and wave phase around the ring', xlabel = 'World X [m]', ylabel = 'World Z [m]')
axes[0, 0].set_aspect('equal', adjustable = 'box')

########################################
# Show the actual prescribed displacements at the attached supports
########################################
for axis in 'XYZ':
    axes[0, 1].plot(nominal['s', quads], survey['dynamic_' + axis] * 1e9, '.-', ms = 2, label = 'World ' + axis)
axes[0, 1].set(title = 'Different supports sample different spatial phases', xlabel = 'Nominal s [m]', ylabel = 'Dynamic support displacement [nm]')
axes[0, 1].legend(fontsize = 8)

########################################
# Show both measured orbit-plane changes on equal footing
########################################
axes[1, 0].plot(moving_twiss.s, dx * 1e6, color = 'tab:blue')
axes[1, 0].set(title = 'Horizontal closed-orbit change', xlabel = 'Nominal s [m]', ylabel = 'With motion − installed baseline [µm]')
axes[1, 1].plot(moving_twiss.s, dy * 1e6, color = 'tab:orange')
axes[1, 1].set(title = 'Vertical closed-orbit change', xlabel = 'Nominal s [m]', ylabel = 'With motion − installed baseline [µm]')
for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
    ax.grid(alpha = .2)
fig.suptitle(f'Same ring and installation errors; dynamic field evaluated at t = {TIME_S:g} s', fontsize = 12)
fig.tight_layout(h_pad = 2.)

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the measured costs
########################################
if PRINT_TIMINGS:
    print('\nCPU operation costs\n-------------------')
    for name, seconds in timings.items():
        print(f'{name:<24} {seconds:9.4f} s')

########################################
# Display the figures
########################################
plt.show()
