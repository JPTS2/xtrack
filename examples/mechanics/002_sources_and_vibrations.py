# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Translate and rotate a rigid girder carrying two quadrupoles.
# See which magnet centres and axes move, then measure the tracked beam response.
# Declare base mounts and heights, then see how a girder sways the whole magnets.
# Compare roll about the girder base with roll about the magnetic axis.

################################################################################
# Required packages
################################################################################
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.transform import Rotation
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Two quadrupoles on a girder, followed by an observation point
########################################
QUAD_LENGTH_M       = .5
QUAD_K1_M2          = .2
GAP_LENGTH_M        = .5
OBSERVATION_DRIFT_M = 10.
INSTALLATION_Y_M    = .2e-6             # Static girder installation error [m]
LAUNCH_X_M          = .5e-3            # Common off-axis launch reveals quadrupole roll coupling [m]
LAUNCH_Y_M          = .25e-3

########################################
# Harmonic motion and independent source-model choices
########################################
SOURCE              = 'harmonic'       # 'harmonic', 'measured' or 'psd' translation
ROTATION_SOURCE     = 'harmonic'       # 'harmonic', 'measured' or 'psd' angles
VERTICAL_PEAK_M     = 50e-9
PITCH_PEAK_RAD      = 200e-9
YAW_PEAK_RAD        = 300e-9
ROLL_PEAK_RAD       = 400e-9
FREQUENCY_HZ        = 500.

########################################
# PSD and supplied-sample alternatives
########################################
PSD_FREQUENCY_HZ    = [0., 100., 200., 800., 900.]
PSD_DENSITY_M2_HZ   = [0., 0., 1e-18, 1e-18, 0.]
PSD_SEED            = 956
# The 700 Hz integrated trapezoidal band gives an angular expected variance
# of peak²/2 when its plateau is peak²/1400 rad²/Hz. Finite realizations vary.

########################################
# Physical times, record sampling and secondary timing diagnostics
########################################
INSPECT_TIME_S      = .0005            # Geometry and terminal comparison: first harmonic maximum
PROBE_DURATION_S    = .004             # Two harmonic cycles with these inputs
N_PROBES            = 81               # Identical fresh launches at these physical arrival times
N_SAMPLES           = 4096
SAMPLE_STEP_S       = 1e-5
PRINT_TIMINGS       = False

########################################
# Mechanical dimensions used for mounts and rigid drawing envelopes
########################################
MAGNET_HEIGHT_M     = .40              # Base to top; magnetic axis is at half-height here [m]
MAGNET_WIDTH_M      = .28
GIRDER_HEIGHT_M     = .25              # Bottom-to-top depth of the girder [m]
GIRDER_WIDTH_M      = .36
MOUNT_HEIGHT_M      = .08              # Stand between the girder top and magnet base [m]
DISPLAY_GAIN       = 1e6               # Reference-point translation and angles; drawing only

################################################################################
# Build the transport line and define the rigid girder reference
################################################################################

########################################
# Set up performance monitoring
########################################
timings = {}

########################################
# Construct the same magnetic line for every comparison
########################################
started = perf_counter()
env = xt.Environment()
env.new('q1', xt.Quadrupole, length = QUAD_LENGTH_M, k1 = QUAD_K1_M2)
env.new('gap', xt.Drift, length = GAP_LENGTH_M)
env.new('q2', xt.Quadrupole, length = QUAD_LENGTH_M, k1 = -QUAD_K1_M2)
env.new('to_observation', xt.Drift, length = OBSERVATION_DRIFT_M)
env.new('observation', xt.Marker)
line = env.new_line(name = 'line', components = ['q1', 'gap', 'q2', 'to_observation', 'observation'])
line.set_particle_ref('proton', p0c = 7e12)
line.build_tracker()
launch = line.build_particles(x = LAUNCH_X_M, y = LAUNCH_Y_M)
# Make the comparison environment before attaching mechanics. Both scenarios
# contain independent elements with identical design magnetic geometry.
axis_env = env.copy()
axis_line = axis_env['line']
axis_line.build_tracker()
axis_launch = axis_line.build_particles(x = LAUNCH_X_M, y = LAUNCH_Y_M)
timings['Lines, trackers and common launches'] = perf_counter() - started

########################################
# Put the girder reference at its base and mount each magnet by its bottom face
########################################
# The surveyed magnetic axes stay at design Y = 0. The magnet-base frame is
# half a magnet height below its axis. Each stand lifts that base above the
# girder top. These offsets enter the mechanics tree, not only the drawing.
started = perf_counter()
mech = env.mechanics
line.mechanics.adopt_reference()
girder_length_m = 2 * QUAD_LENGTH_M + GAP_LENGTH_M
axis_height_above_base_m = GIRDER_HEIGHT_M + MOUNT_HEIGHT_M + MAGNET_HEIGHT_M / 2
girder_base_y_m = -axis_height_above_base_m
girder = mech.new_girder('girder', anchor = xt.Frame.from_survey_angles(
    Y = girder_base_y_m, Z = girder_length_m / 2,
))
magnet_bodies = {}
for name, centre_s in [('q1', QUAD_LENGTH_M / 2), ('q2', girder_length_m - QUAD_LENGTH_M / 2)]:
    magnet_bodies[name] = mech.new_body(
        'magnet.' + name, kind = 'magnet', parent = girder,
        mount = xt.Frame.from_survey_angles(
            Y = GIRDER_HEIGHT_M + MOUNT_HEIGHT_M, Z = centre_s - girder_length_m / 2,
        ), elements = [name], line = line,
    )
girder.installation.shift_y = INSTALLATION_Y_M

########################################
# Build the same assembly with a reference point on the magnetic axis
########################################
# Holding this point fixed during roll is a different physical motion from
# holding the girder base fixed. Both start with exactly the same magnet poses.
axis_mech = axis_env.mechanics
axis_line.mechanics.adopt_reference()
axis_girder = axis_mech.new_girder('girder', anchor = xt.Frame.from_survey_angles(
    Z = girder_length_m / 2,
))
for name, centre_s in [('q1', QUAD_LENGTH_M / 2), ('q2', girder_length_m - QUAD_LENGTH_M / 2)]:
    axis_mech.new_body(
        'magnet.' + name, kind = 'magnet', parent = axis_girder,
        mount = xt.Frame.from_survey_angles(
            Y = -MAGNET_HEIGHT_M / 2, Z = centre_s - girder_length_m / 2,
        ), elements = [name], line = axis_line,
    )
axis_girder.installation.shift_y = INSTALLATION_Y_M
timings['Height-aware bodies and base mounts'] = perf_counter() - started

################################################################################
# Prescribe translation and three rigid rotations through the source interface
################################################################################

########################################
# Choose the vertical displacement law
########################################
started = perf_counter()
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
probe_times = np.linspace(0., PROBE_DURATION_S, N_PROBES)
if max(INSPECT_TIME_S, PROBE_DURATION_S) > times[-1]:
    raise ValueError('Extend the source record to cover inspection and all probe times')
if SOURCE == 'harmonic':
    vertical = xt.mechanics.Harmonic(VERTICAL_PEAK_M, FREQUENCY_HZ)
elif SOURCE == 'measured':
    vertical = xt.mechanics.MeasuredMotion(
        [0., .002, .004, times[-1]], [0., VERTICAL_PEAK_M, 0., 0.],
    )
elif SOURCE == 'psd':
    vertical = xt.mechanics.DisplacementPSD(PSD_FREQUENCY_HZ, PSD_DENSITY_M2_HZ, seed = PSD_SEED)
else:
    raise ValueError('Choose harmonic, measured or psd')
mech.add_source('vertical', vertical, targets = ['girder'], component = 'y')

########################################
# Bind pitch, yaw and roll as separate named motion sources
########################################
# These are survey angles: R = Ry(yaw) Rx(-pitch) Rz(roll) for one jointly
# specified orientation. Separate source rotations compose in insertion order.
# Positive pitch raises the downstream end; positive yaw moves it towards +X.
for index, (component, amplitude) in enumerate([
    ('pitch', PITCH_PEAK_RAD), ('yaw', YAW_PEAK_RAD), ('roll', ROLL_PEAK_RAD),
]):
    if ROTATION_SOURCE == 'harmonic':
        angular = xt.mechanics.Harmonic(amplitude, FREQUENCY_HZ, unit = 'rad')
    elif ROTATION_SOURCE == 'measured':
        angular = xt.mechanics.MeasuredMotion(
            [0., .002, .004, times[-1]], [0., amplitude, 0., 0.], unit = 'rad',
        )
    elif ROTATION_SOURCE == 'psd':
        angular = xt.mechanics.AngularPSD(
            PSD_FREQUENCY_HZ,
            np.array([0., 0., 1., 1., 0.]) * amplitude**2 / 1400., seed = PSD_SEED + 1 + index,
        )
    else:
        raise ValueError('Choose harmonic, measured or psd angular motion')
    mech.add_source(component, angular, targets = ['girder'], component = component)
    if component == 'roll':
        roll_source = angular

########################################
# Compute one record shared by every controlled comparison
########################################
motion = mech.compute_time_series(time_s = times)
axis_mech.add_source('roll', roll_source, targets = ['girder'], component = 'roll')
axis_motion = axis_mech.compute_time_series(time_s = times)
np.testing.assert_array_equal(motion.values['roll'], axis_motion.values['roll'])
timings['Source definitions and common record'] = perf_counter() - started
source_table = mech.get_source_table()

################################################################################
# Track the baseline with static installation error and no vibration
################################################################################

########################################
# Capture the installed magnet frames and girder reference
########################################
all_sources = tuple(mech.sources)
mech.apply(disabled_sources = all_sources)
installed_survey = line.survey(include_element_frames = True)
installed_frames = {
    name: installed_survey.get_frame(name, which = 'elem_start').matrix.copy()
    for name in ('q1', 'q2')
}
installed_body = mech.get_body_poses(disabled_sources = all_sources)['girder'].matrix.copy()
installed_magnet_bases = {
    name: mech.get_body_poses(disabled_sources = all_sources)['magnet.' + name].XYZ.copy()
    for name in ('q1', 'q2')
}
installed_centres = {
    name: matrix[:3, 3] + matrix[:3, :3] @ [0., 0., QUAD_LENGTH_M / 2]
    for name, matrix in installed_frames.items()
}

########################################
# Track the common off-axis launch to the observation point
########################################
particle = launch.copy()
line.track(particle)
baseline_exit = np.array([float(particle.x[0]), float(particle.y[0])])

########################################
# Verify that the beam-axis-reference assembly has the same installed magnets
########################################
axis_mech.apply(disabled_sources = ('roll',))
axis_installed_survey = axis_line.survey(include_element_frames = True)
axis_installed_body = axis_mech.get_body_poses(disabled_sources = ('roll',))['girder'].matrix.copy()
for name in ('q1', 'q2'):
    np.testing.assert_allclose(
        axis_installed_survey.get_frame(name, which = 'elem_start').matrix,
        installed_frames[name], rtol = 0, atol = 1e-15,
    )
particle = axis_launch.copy()
axis_line.track(particle)
axis_baseline_exit = np.array([float(particle.x[0]), float(particle.y[0])])
np.testing.assert_array_equal(axis_baseline_exit, baseline_exit)

################################################################################
# Isolate each rigid motion and measure its effect on magnets and beam
################################################################################

########################################
# Select which sources act in each controlled comparison
########################################
cases = {
    'Vertical translation only': ('vertical',),
    'Pitch only': ('pitch',),
    'Yaw only': ('yaw',),
    'Roll about girder base': ('roll',),
    'All four motions': all_sources,
}
frames = {}
body_poses = {}
centre_changes = {}
snapshot_exits = {}
screen_changes = {}

########################################
# Apply the selected pose and inspect actual Xtrack element frames
########################################
started = perf_counter()
for label, active in cases.items():
    disabled = tuple(name for name in all_sources if name not in active)
    mech.apply(time_s = INSPECT_TIME_S, time_series = motion, disabled_sources = disabled)
    actual_survey = line.survey(include_element_frames = True)
    frames[label] = {
        name: actual_survey.get_frame(name, which = 'elem_start').matrix.copy()
        for name in ('q1', 'q2')
    }
    body_poses[label] = mech.get_body_poses(
        time_s = INSPECT_TIME_S, time_series = motion, disabled_sources = disabled,
    )['girder'].matrix.copy()
    centre_changes[label] = {}
    rigid_map = body_poses[label] @ np.linalg.inv(installed_body)
    for name, matrix in frames[label].items():
        # The actual lowered fields/survey must agree with the requested rigid map.
        np.testing.assert_allclose(matrix, rigid_map @ installed_frames[name], rtol = 0, atol = 3e-14)
        centre = matrix[:3, 3] + matrix[:3, :3] @ [0., 0., QUAD_LENGTH_M / 2]
        centre_changes[label][name] = centre - installed_centres[name]
    particle = launch.copy()
    line.track(particle)
    snapshot_exits[label] = np.array([float(particle.x[0]), float(particle.y[0])]) - baseline_exit

    ########################################
    # Track identical fresh launches at different vibration phases
    ########################################
    # Each is one passage through a held mechanical pose, never a closed orbit.
    response = []
    for time_s in probe_times:
        mech.apply(time_s = float(time_s), time_series = motion, disabled_sources = disabled)
        particle = launch.copy()
        line.track(particle)
        assert particle.state[0] == 1
        response.append([float(particle.x[0]), float(particle.y[0])])
    screen_changes[label] = np.array(response) - baseline_exit

timings['Controlled poses and tracked probe scans'] = perf_counter() - started

########################################
# Check the height lever arm in the base-referenced roll
########################################
roll_rotation = body_poses['Roll about girder base'][:3, :3] @ installed_body[:3, :3].T
for name in ('q1', 'q2'):
    lever = installed_centres[name] - installed_body[:3, 3]
    np.testing.assert_allclose(
        centre_changes['Roll about girder base'][name],
        (roll_rotation - np.eye(3)) @ lever, rtol = 0, atol = 2e-15,
    )
    np.testing.assert_allclose(
        installed_centres[name] - installed_magnet_bases[name],
        [0., MAGNET_HEIGHT_M / 2, 0.], rtol = 0, atol = 1e-15,
    )

########################################
# Compare equal roll angles with the magnetic axis held fixed instead
########################################
axis_mech.apply(time_s = INSPECT_TIME_S, time_series = axis_motion)
axis_roll_body = axis_mech.get_body_poses(time_s = INSPECT_TIME_S, time_series = axis_motion)['girder'].matrix.copy()
axis_roll_survey = axis_line.survey(include_element_frames = True)
axis_roll_centres = {}
for name in ('q1', 'q2'):
    matrix = axis_roll_survey.get_frame(name, which = 'elem_start').matrix
    axis_roll_centres[name] = matrix[:3, 3] + matrix[:3, :3] @ [0., 0., QUAD_LENGTH_M / 2]
    np.testing.assert_allclose(axis_roll_centres[name], installed_centres[name], rtol = 0, atol = 1e-15)
particle = axis_launch.copy()
axis_line.track(particle)
axis_roll_snapshot_exit = np.array([float(particle.x[0]), float(particle.y[0])]) - axis_baseline_exit
axis_roll_response = []
for time_s in probe_times:
    axis_mech.apply(time_s = float(time_s), time_series = axis_motion)
    particle = axis_launch.copy()
    axis_line.track(particle)
    axis_roll_response.append([float(particle.x[0]), float(particle.y[0])])
axis_roll_changes = np.array(axis_roll_response) - axis_baseline_exit
axis_mech.restore()

################################################################################
# Enable all motions and track probes arriving at different physical times
################################################################################

########################################
# Let line.mechanics apply the combined pose at each probe's arrival time
########################################
# This is a train of independent, identically launched probes. Their arrival
# times span the vibration cycle; no periodic transport model is needed.
started = perf_counter()
line.mechanics.enable_vibrations(time_series = motion)
automatic_exits = []
for time_s in probe_times:
    particle = launch.copy()
    particle.time_s = float(time_s)
    line.track(particle)
    automatic_exits.append([float(particle.x[0]), float(particle.y[0])])
line.mechanics.disable_vibrations()
for name in ('q1', 'q2'):
    np.testing.assert_equal(env[name].shift_y, INSTALLATION_Y_M)
automatic_changes = np.array(automatic_exits) - baseline_exit
np.testing.assert_allclose(automatic_changes, screen_changes['All four motions'], rtol = 0, atol = 2e-15)
automatic_difference_m = np.max(np.abs(automatic_changes - screen_changes['All four motions']))
timings['Combined motion with automatic updates'] = perf_counter() - started

########################################
# Read the physical translation and angles used in the geometry comparison
########################################
vertical_at_inspection_m = body_poses['Vertical translation only'][1, 3] - installed_body[1, 3]
isolated_rotations = {
    label: Rotation.from_matrix(body_poses[label][:3, :3] @ installed_body[:3, :3].T).as_rotvec()
    for label in ('Pitch only', 'Yaw only', 'Roll about girder base')
}
pitch_at_inspection_rad = -isolated_rotations['Pitch only'][0]
yaw_at_inspection_rad = isolated_rotations['Yaw only'][1]
roll_at_inspection_rad = isolated_rotations['Roll about girder base'][2]

################################################################################
# Explain the applied transforms and measured beam response
################################################################################

########################################
# State the physical setup and the selected source models
########################################
print('Rigid girder motion and the tracked beam\n'
      '---------------------------------------')
print(f'Line: q1 ({QUAD_LENGTH_M:g} m), gap ({GAP_LENGTH_M:g} m), '
      f'q2 ({QUAD_LENGTH_M:g} m), drift ({OBSERVATION_DRIFT_M:g} m), observation point.')
print(f'Magnet height: {MAGNET_HEIGHT_M:g} m; base mount is {MAGNET_HEIGHT_M / 2:g} m below the magnetic axis.')
print(f'Girder height: {GIRDER_HEIGHT_M:g} m; stand height above the girder: {MOUNT_HEIGHT_M:g} m.')
print(f'Magnetic axis above girder-base reference: {axis_height_above_base_m:g} m.')
print('Main case: girder-base reference held fixed during pure rotation.')
print(f'Every case retains the same static installation error: y = {INSTALLATION_Y_M * 1e6:g} µm.')
print(f'Every probe starts at x = {LAUNCH_X_M * 1e3:g} mm, y = {LAUNCH_Y_M * 1e3:g} mm, px = py = 0.')
print(f'Sources: translation {SOURCE}, angles {ROTATION_SOURCE}.')
source_table.cols['name model component unit'].show()
print(f'At the displayed phase: vertical {vertical_at_inspection_m * 1e9:+.3f} nm; '
      f'pitch {pitch_at_inspection_rad * 1e6:+.3f} µrad; '
      f'yaw {yaw_at_inspection_rad * 1e6:+.3f} µrad; roll {roll_at_inspection_rad * 1e6:+.3f} µrad.')

########################################
# Connect magnet-centre changes to the tracked observation-point changes
########################################
print(f'\nMechanical and beam changes at t = {INSPECT_TIME_S * 1e3:g} ms\n'
      '------------------------------------------------------')
print('All changes below are relative to the same static installation-error case, in nm.')
print(f'{"Applied motion":<26} {"q1 dX":>9} {"q1 dY":>9} {"q2 dX":>9} {"q2 dY":>9}'
      f' {"Beam dx":>12} {"Beam dy":>12}')
for label in cases:
    q1, q2 = centre_changes[label]['q1'], centre_changes[label]['q2']
    dx, dy = snapshot_exits[label] * 1e9
    print(f'{label:<26} {q1[0] * 1e9:9.3f} {q1[1] * 1e9:9.3f}'
          f' {q2[0] * 1e9:9.3f} {q2[1] * 1e9:9.3f} {dx:+12.6f} {dy:+12.6f}')
    if label == 'Roll about girder base':
        q1 = axis_roll_centres['q1'] - installed_centres['q1']
        q2 = axis_roll_centres['q2'] - installed_centres['q2']
        dx, dy = axis_roll_snapshot_exit * 1e9
        print(f'{"Roll about magnetic axis":<26} {q1[0] * 1e9:9.3f} {q1[1] * 1e9:9.3f}'
              f' {q2[0] * 1e9:9.3f} {q2[1] * 1e9:9.3f} {dx:+12.6f} {dy:+12.6f}')
print('Translation moves both magnet centres together.')
print('Pitch gives opposite vertical centre displacements; yaw gives opposite horizontal displacements.')
print('Rolling the girder about its base sways both magnet centres and rotates their field axes.')
print(f'Height × roll predicts the first-order horizontal sway: '
      f'{-axis_height_above_base_m * roll_at_inspection_rad * 1e9:+.3f} nm.')
print('At the same roll angle, holding the magnetic axis fixed removes centre sway.')
print('Its smaller beam response remains because the quadrupole field axes still rotate.')
print('These are different imposed motions: zero translation is specified at different reference points.')
print(f'Baseline beam at the observation point: x = {baseline_exit[0] * 1e6:+.6f} µm, '
      f'y = {baseline_exit[1] * 1e6:+.6f} µm.')

########################################
# Explain the combined dynamic scan without making clocks the lesson
########################################
print('\nCombined vibration during tracking\n----------------------------------')
print(f'{N_PROBES} independent probes arrive between 0 and {PROBE_DURATION_S * 1e3:g} ms.')
print('enable_vibrations() applies the rigid pose at each probe arrival, held for its passage.')
print(f'Largest difference from explicitly applying the same poses [nm]: {automatic_difference_m * 1e9:.3e}')
print('Vibration is now removed; the static installation error remains.')
print(f'Geometry-only motion magnification: ×{DISPLAY_GAIN:,.0f}. Beam graphs and terminal values are unscaled.')

################################################################################
# Draw actual rigid geometry at the selected vibration phase
################################################################################

########################################
# Prepare installed magnet outlines and the common rotation reference
########################################
# Envelopes are illustrative rigid drawings. Use the same magnified rigid map
# for the whole assembly; do not independently move or deform magnet vertices.
half_width, half_height = MAGNET_WIDTH_M / 2, MAGNET_HEIGHT_M / 2
side = np.array([[0., -half_height, 0.], [0., -half_height, QUAD_LENGTH_M],
                 [0., half_height, QUAD_LENGTH_M], [0., half_height, 0.], [0., -half_height, 0.]])
plan = np.array([[-half_width, 0., 0.], [-half_width, 0., QUAD_LENGTH_M],
                 [half_width, 0., QUAD_LENGTH_M], [half_width, 0., 0.], [-half_width, 0., 0.]])
end = np.array([[-half_width, -half_height, QUAD_LENGTH_M / 2],
                [half_width, -half_height, QUAD_LENGTH_M / 2],
                [half_width, half_height, QUAD_LENGTH_M / 2],
                [-half_width, half_height, QUAD_LENGTH_M / 2],
                [-half_width, -half_height, QUAD_LENGTH_M / 2]])
# The girder outline uses its declared height, from the base reference to its
# top. Each stand runs from that top to the actual magnet-base mounting frame.
deck_s = girder_length_m / 2 + .05
deck_side = np.array([[0., 0., -deck_s], [0., 0., deck_s],
                      [0., GIRDER_HEIGHT_M, deck_s], [0., GIRDER_HEIGHT_M, -deck_s], [0., 0., -deck_s]])
deck_plan = np.array([[-GIRDER_WIDTH_M / 2, 0., -deck_s], [-GIRDER_WIDTH_M / 2, 0., deck_s],
                      [GIRDER_WIDTH_M / 2, 0., deck_s], [GIRDER_WIDTH_M / 2, 0., -deck_s],
                      [-GIRDER_WIDTH_M / 2, 0., -deck_s]])
deck_end = np.array([[-GIRDER_WIDTH_M / 2, 0., 0.], [GIRDER_WIDTH_M / 2, 0., 0.],
                     [GIRDER_WIDTH_M / 2, GIRDER_HEIGHT_M, 0.], [-GIRDER_WIDTH_M / 2, GIRDER_HEIGHT_M, 0.],
                     [-GIRDER_WIDTH_M / 2, 0., 0.]])
colours = {'q1': 'tab:blue', 'q2': 'tab:orange'}
pivot = installed_body[:3, 3]
geometry_fig = plt.figure(figsize = (11, 7.5))
grid = geometry_fig.add_gridspec(2, 2)
geometry_axes = [geometry_fig.add_subplot(grid[0, 0]), geometry_fig.add_subplot(grid[0, 1]),
                 geometry_fig.add_subplot(grid[1, 0])]
longitudinal_view_points = []
roll_grid = grid[1, 1].subgridspec(1, 2)
roll_axes = [geometry_fig.add_subplot(roll_grid[0, 0]), geometry_fig.add_subplot(roll_grid[0, 1])]

########################################
# Show translation, pitch and yaw in the plane where each acts
########################################
for ax, label, local_outline, coordinates, ordinate in [
    (geometry_axes[0], 'Vertical translation only', side, [2, 1], 'World Y [m]'),
    (geometry_axes[1], 'Pitch only', side, [2, 1], 'World Y [m]'),
    (geometry_axes[2], 'Yaw only', plan, [2, 0], 'World X [m]'),
]:
    moved = installed_body if label == 'Vertical translation only' else body_poses[label]
    enlarged_rotation = Rotation.from_matrix(moved[:3, :3] @ installed_body[:3, :3].T)
    enlarged_rotation = Rotation.from_rotvec(DISPLAY_GAIN * enlarged_rotation.as_rotvec()).as_matrix()
    enlarged_translation = DISPLAY_GAIN * (moved[:3, 3] - pivot)
    support = deck_plan if label == 'Yaw only' else deck_side
    original_deck = support @ installed_body[:3, :3].T + pivot
    shown_deck = (original_deck - pivot) @ enlarged_rotation.T + pivot + enlarged_translation
    ax.plot(*original_deck[:, coordinates].T, '--', color = '0.6', lw = .8)
    ax.plot(*shown_deck[:, coordinates].T, color = '0.25', lw = 1.2)
    longitudinal_view_points.extend([original_deck[:, coordinates], shown_deck[:, coordinates]])
    if label != 'Yaw only':
        for centre in installed_centres.values():
            mount_base = centre - [0., MAGNET_HEIGHT_M / 2, 0.]
            mount = np.array([mount_base - [0., MOUNT_HEIGHT_M, 0.], mount_base])
            moved_mount = (mount - pivot) @ enlarged_rotation.T + pivot + enlarged_translation
            ax.plot(*moved_mount[:, coordinates].T, color = '0.25', lw = 1.2)
            ax.plot(*mount_base[coordinates], 'o', ms = 4, mfc = 'white', mec = 'k')
    for name, matrix in installed_frames.items():
        original = local_outline @ matrix[:3, :3].T + matrix[:3, 3]
        shown = (original - pivot) @ enlarged_rotation.T + pivot + enlarged_translation
        ax.plot(*original[:, coordinates].T, '--', color = '0.6')
        ax.fill(*shown[:, coordinates].T, color = colours[name], alpha = .15)
        ax.plot(*shown[:, coordinates].T, color = colours[name])
        longitudinal_view_points.extend([original[:, coordinates], shown[:, coordinates]])
        centre = (installed_centres[name] - pivot) @ enlarged_rotation.T + pivot + enlarged_translation
        ax.text(centre[coordinates[0]], centre[coordinates[1]], name, ha = 'center', va = 'center',
                bbox = {'facecolor': 'white', 'edgecolor': 'none', 'alpha': .7, 'pad': 1})
    ax.plot(pivot[coordinates[0]], pivot[coordinates[1]], 'kD', ms = 4)
    ax.axhline(INSTALLATION_Y_M if ordinate == 'World Y [m]' else 0., color = '0.85', lw = .8)
    titles = {
        'Vertical translation only': 'Installed assembly: declared heights and mounts',
        'Pitch only': f'Pitch: {pitch_at_inspection_rad * 1e6:+.3g} µrad',
        'Yaw only': f'Yaw: {yaw_at_inspection_rad * 1e6:+.3g} µrad',
    }
    ax.set(title = titles[label], xlabel = 'World Z [m]', ylabel = ordinate)
    ax.title.set_fontsize(11)
    ax.set_aspect('equal', adjustable = 'box')

# Keep these three views on common metric limits, including the full swept
# outlines when the source parameters change.
points = np.concatenate(longitudinal_view_points)
lower, upper = points.min(axis = 0), points.max(axis = 0)
padding = .08 * (upper - lower)
for ax in geometry_axes:
    ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])

########################################
# Mark the declared dimensions on the installed elevation
########################################
height_ax = geometry_axes[0]
for z, bottom, top, text in [
    (.65, INSTALLATION_Y_M - MAGNET_HEIGHT_M / 2, INSTALLATION_Y_M + MAGNET_HEIGHT_M / 2,
     f'Magnet\n{MAGNET_HEIGHT_M:g} m'),
    (.80, pivot[1], pivot[1] + GIRDER_HEIGHT_M, f'Girder\n{GIRDER_HEIGHT_M:g} m'),
]:
    height_ax.annotate('', xy = (z, top), xytext = (z, bottom),
                       arrowprops = {'arrowstyle': '<->', 'color': '0.25', 'lw': .8})
    height_ax.text(z + .025, (bottom + top) / 2, text, fontsize = 8, va = 'center')
height_ax.text(.55, INSTALLATION_Y_M - MAGNET_HEIGHT_M / 2 - MOUNT_HEIGHT_M / 2,
               f'{MOUNT_HEIGHT_M:g} m stand', fontsize = 8, va = 'center')

########################################
# Compare equal roll angles about the beam axis and about the girder base
########################################
# Both magnets have the same end projection, so one section represents q1/q2.
# The physical girder/magnet envelopes are identical in both comparisons.
roll_view_points = []
for ax, title, reference, moved in [
    (roll_axes[0], 'Magnetic axis held fixed', axis_installed_body, axis_roll_body),
    (roll_axes[1], 'Girder base held fixed', installed_body, body_poses['Roll about girder base']),
]:
    reference_point = reference[:3, 3]
    roll = moved[:3, :3] @ reference[:3, :3].T
    shown_roll = Rotation.from_rotvec(DISPLAY_GAIN * Rotation.from_matrix(roll).as_rotvec()).as_matrix()
    original = end @ installed_frames['q1'][:3, :3].T + installed_frames['q1'][:3, 3]
    shown = (original - reference_point) @ shown_roll.T + reference_point
    original_deck = deck_end @ installed_body[:3, :3].T + pivot
    shown_deck = (original_deck - reference_point) @ shown_roll.T + reference_point
    ax.plot(original_deck[:, 0], original_deck[:, 1], '--', color = '0.6', lw = .8)
    ax.fill(shown_deck[:, 0], shown_deck[:, 1], color = '0.5', alpha = .12)
    ax.plot(shown_deck[:, 0], shown_deck[:, 1], color = '0.25', lw = 1.2)
    original_mount = np.array([installed_magnet_bases['q1'] - [0., MOUNT_HEIGHT_M, 0.],
                               installed_magnet_bases['q1']])
    mount = (original_mount - reference_point) @ shown_roll.T + reference_point
    ax.plot(mount[:, 0], mount[:, 1], color = '0.25', lw = 1.2)
    ax.plot(mount[-1, 0], mount[-1, 1], 'o', ms = 4, mfc = 'white', mec = 'k')
    ax.plot(original[:, 0], original[:, 1], '--', color = '0.6')
    ax.fill(shown[:, 0], shown[:, 1], color = 'tab:blue', alpha = .15)
    ax.plot(shown[:, 0], shown[:, 1], color = 'tab:blue')
    shown_centre = (installed_centres['q1'] - reference_point) @ shown_roll.T + reference_point
    ax.plot(reference_point[0], reference_point[1], 'kD', ms = 4)
    ax.plot(shown_centre[0], shown_centre[1], 'o', color = 'tab:blue', ms = 3)
    ax.axhline(INSTALLATION_Y_M, color = '0.85', lw = .8)
    ax.set(title = title + f'\nRoll {roll_at_inspection_rad * 1e6:+.3g} µrad; both magnets', xlabel = 'World X [m]')
    ax.title.set_fontsize(10)
    ax.set_aspect('equal', adjustable = 'box')
    roll_view_points.extend([original[:, :2], shown[:, :2], original_deck[:, :2], shown_deck[:, :2]])
points = np.concatenate(roll_view_points)
lower, upper = points.min(axis = 0), points.max(axis = 0)
padding = .12 * (upper - lower)
for ax in roll_axes:
    ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
roll_axes[0].set_ylabel('World Y [m]')
roll_axes[1].tick_params(labelleft = False)
geometry_fig.suptitle('Mounted magnets and rigid girder motion\n'
                      f'Installed geometry at upper left; angles ×{DISPLAY_GAIN:,.0f} elsewhere '
                      f'(t = {INSPECT_TIME_S * 1e3:g} ms)', fontsize = 11)
geometry_fig.legend(handles = [
    Line2D([], [], ls = '--', color = '0.6', label = 'Static installation-error placement'),
    Line2D([], [], color = 'tab:blue', label = 'q1 transformed'),
    Line2D([], [], color = 'tab:orange', label = 'q2 transformed'),
    Line2D([], [], color = '0.25', label = 'Rigid girder and mounts'),
    Line2D([], [], marker = 'D', ls = '', color = 'k', label = 'Installed rotation reference'),
    Line2D([], [], marker = 'o', mfc = 'white', ls = '', color = 'k', label = 'Magnet base mounts'),
    Line2D([], [], marker = 'o', ls = '', color = 'tab:blue', label = 'Magnetic axis (end views)'),
], loc = 'lower center', ncol = 4, fontsize = 8.5)
geometry_fig.tight_layout(rect = (0., .08, 1., 1.), h_pad = 2.4, w_pad = 2.)

################################################################################
# Plot the actual tracked beam response to each rigid motion
################################################################################

########################################
# Isolate four motions, then show their combined automatic tracking response
########################################
beam_fig = plt.figure(figsize = (10, 8))
beam_grid = beam_fig.add_gridspec(3, 2)
beam_axes = [beam_fig.add_subplot(beam_grid[i // 2, i % 2]) for i in range(4)]
beam_axes.append(beam_fig.add_subplot(beam_grid[2, :]))
for ax, label in zip(beam_axes, cases):
    values = automatic_changes if label == 'All four motions' else screen_changes[label]
    ax.plot(probe_times * 1e3, values[:, 0] * 1e9, label = 'Horizontal beam change')
    ax.plot(probe_times * 1e3, values[:, 1] * 1e9, label = 'Vertical beam change')
    if label == 'Roll about girder base':
        ax.plot(probe_times * 1e3, axis_roll_changes[:, 0] * 1e9, '--', color = 'tab:blue')
        ax.plot(probe_times * 1e3, axis_roll_changes[:, 1] * 1e9, '--', color = 'tab:orange')
    ax.axvline(INSPECT_TIME_S * 1e3, ls = ':', color = '0.5')
    ax.set(title = label + (' — automatic updates' if label == 'All four motions' else ''),
           xlabel = 'Probe arrival time [ms]', ylabel = 'Beam change [nm]')
    ax.grid(alpha = .2)
beam_fig.suptitle('Tracked beam at the downstream observation point\n'
                  'Changes from the static installation-error case; panels have individual vertical scales', fontsize = 11)
handles, labels = beam_axes[0].get_legend_handles_labels()
handles.extend([Line2D([], [], ls = ':', color = '0.5'), Line2D([], [], ls = '--', color = '0.4')])
labels.extend([f'Geometry shown at t = {INSPECT_TIME_S * 1e3:g} ms', 'Roll control: magnetic axis fixed'])
beam_fig.legend(handles, labels, loc = 'lower center', ncol = 2, fontsize = 9)
beam_fig.tight_layout(rect = (0., .065, 1., 1.))

################################################################################
# Keep performance details available after the physical comparison
################################################################################

########################################
# Print optional operation costs without making them the teaching result
########################################
if PRINT_TIMINGS:
    print('\nCPU operation costs\n-------------------')
    for name, seconds in timings.items():
        print(f'{name:<43} {seconds * 1e3:10.3f} ms')

########################################
# Display the transformed magnets and tracked beam response
########################################
plt.show()
