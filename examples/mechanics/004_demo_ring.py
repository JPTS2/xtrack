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

########################################
# Motion sampling
########################################
SAMPLE_STEP_S = 1 / 256
N_SAMPLES     = 2048

########################################
# Display magnification
########################################
DISPLACEMENT_SCALE = 1e8
ROTATION_SCALE     = 1e5

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

girders = [line.mechanics.new_girder(f'g.{name}', elements = [name]) for name in quads]

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

########################################
# Report numerical agreement and input memory
########################################
print('Surveyed ring and prescribed input\n----------------------------------')
print(f'Ring length [m]: {line.get_length():.1f}; supports: {len(girders)}; source: {SOURCE}')
print(f'Largest entrance position / orientation discrepancy: {survey_error_m:.3g} m / {survey_error_rad:.3g} rad')
print(f'Input payload [MiB]: {sum(v.nbytes for v in motion.values.values()) / 1024**2:.3f}')
print('Only quadrupoles move here. Feet and mechanical responses follow in 005 and 006.')

################################################################################
# Inspect the ring motion and input histories
################################################################################

########################################
# Draw magnified body translations and orientations
########################################
layout_fig, layout_axes = mech.plot(
    time_s = TIME_S, time_series = motion, line = line,
    displacement_scale = DISPLACEMENT_SCALE, rotation_scale = ROTATION_SCALE,
)

########################################
# Prepare the spatial and temporal diagnostic panels
########################################
fig, axes = plt.subplots(1, 2, figsize = (10, 4))

########################################
# Plot dynamic world displacement along the ring
########################################
for axis in 'XYZ':
    axes[0].plot(nominal['s', quads], survey['dynamic_' + axis] * 1e9, '.-', ms = 2, label = axis)
axes[0].set(xlabel = 'Nominal s [m]', ylabel = 'Dynamic world displacement [nm]')

########################################
# Plot the first support's local input history
########################################
table = motion.get_table(first.name)
for name, label in [('shift_x', 'x'), ('shift_y', 'y'), ('shift_s', 's')]:
    axes[1].plot(table.time_s, table[name] * 1e9, label = label)
axes[1].set(xlabel = 'Physical time [s]', ylabel = 'First-girder local input [nm]', xlim = (0, 1))
axes[1].axvline(TIME_S, color = '0.4', ls = ':')

########################################
# Finish the diagnostic figure
########################################
axes[0].legend()
axes[1].legend()
fig.tight_layout()

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the measured costs
########################################
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<24} {seconds:9.4f} s')

########################################
# Display the figures
########################################
plt.show()
