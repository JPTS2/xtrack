# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Translate and rotate a rigid girder carrying two quadrupoles.
# See which magnet centres and axes move, then measure the tracked beam response.
# Isolate translation, pitch, yaw and roll before enabling their combined motion.

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
# Illustrative rigid envelopes and motion magnification
########################################
MAGNET_WIDTH_M      = .28              # Drawing only; not an aperture model
MAGNET_HEIGHT_M     = .28
DISPLAY_GAIN        = 1e6              # Applied equally to translation and rotation in geometry only

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
timings['Line, tracker and common launch'] = perf_counter() - started

########################################
# Place the rotation reference on the beam axis, midway between magnet centres
########################################
# This is the point about which the prescribed angular motion acts. Choosing
# another anchor changes the required translation for the same physical motion.
# With this choice, pure roll rotates the magnet axes without moving their centres.
started = perf_counter()
mech = env.mechanics
line.mechanics.adopt_reference()
girder_length_m = 2 * QUAD_LENGTH_M + GAP_LENGTH_M
girder = line.mechanics.new_girder(
    'girder', elements = ['q1', 'q2'],
    anchor = xt.Frame.from_survey_angles(Z = girder_length_m / 2),
)
girder.installation.shift_y = INSTALLATION_Y_M
timings['Girder and static installation error'] = perf_counter() - started

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

########################################
# Compute one record shared by every controlled comparison
########################################
motion = mech.compute_time_series(time_s = times)
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
# Prepare a second launch exactly on the installed magnetic axis
########################################
# Its roll response is a useful control: a rolled centred quadrupole cannot
# kick a particle travelling on its axis. The off-axis launch exposes coupling.
centred_launch = line.build_particles(y = INSTALLATION_Y_M)
centred = centred_launch.copy()
line.track(centred)
centred_baseline_exit = np.array([float(centred.x[0]), float(centred.y[0])])

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
    'Roll only': ('roll',),
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
# Check that roll rotates the magnet axes while keeping both centres fixed
########################################
for name in ('q1', 'q2'):
    np.testing.assert_allclose(centre_changes['Roll only'][name], 0., rtol = 0, atol = 1e-15)
mech.apply(time_s = INSPECT_TIME_S, time_series = motion,
           disabled_sources = tuple(name for name in all_sources if name != 'roll'))
centred = centred_launch.copy()
line.track(centred)
centred_roll_change = np.array([float(centred.x[0]), float(centred.y[0])]) - centred_baseline_exit
np.testing.assert_allclose(centred_roll_change, 0., rtol = 0, atol = 1e-15)

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
    for label in ('Pitch only', 'Yaw only', 'Roll only')
}
pitch_at_inspection_rad = -isolated_rotations['Pitch only'][0]
yaw_at_inspection_rad = isolated_rotations['Yaw only'][1]
roll_at_inspection_rad = isolated_rotations['Roll only'][2]

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
print(f'Rotation reference: on the installed beam axis, halfway along the {girder_length_m:g} m girder.')
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
print('Translation moves both magnet centres together.')
print('Pitch gives opposite vertical centre displacements; yaw gives opposite horizontal displacements.')
print('Roll leaves both centres fixed here, but rotates both quadrupole field axes and couples x/y.')
print(f'Centred-launch roll control: largest beam change = {np.max(np.abs(centred_roll_change)) * 1e9:.3e} nm.')
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
# A thin deck and two short mounts make the common rigid support visible.
# These are drawing envelopes, transformed by the same map as both magnets.
deck_y = -half_height - .04
deck_s = girder_length_m / 2 + .05
deck_side = np.array([[0., deck_y - .02, -deck_s], [0., deck_y - .02, deck_s],
                      [0., deck_y, deck_s], [0., deck_y, -deck_s], [0., deck_y - .02, -deck_s]])
deck_plan = np.array([[-half_width - .02, deck_y, -deck_s], [-half_width - .02, deck_y, deck_s],
                      [half_width + .02, deck_y, deck_s], [half_width + .02, deck_y, -deck_s],
                      [-half_width - .02, deck_y, -deck_s]])
deck_end = np.array([[-half_width - .02, deck_y - .02, 0.], [half_width + .02, deck_y - .02, 0.],
                     [half_width + .02, deck_y, 0.], [-half_width - .02, deck_y, 0.],
                     [-half_width - .02, deck_y - .02, 0.]])
colours = {'q1': 'tab:blue', 'q2': 'tab:orange'}
pivot = installed_body[:3, 3]
geometry_fig = plt.figure(figsize = (10, 6.5))
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
    moved = body_poses[label]
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
            mount = np.array([centre + [0., deck_y, 0.], centre + [0., -half_height, 0.]])
            moved_mount = (mount - pivot) @ enlarged_rotation.T + pivot + enlarged_translation
            ax.plot(*moved_mount[:, coordinates].T, color = '0.25', lw = 1.2)
    for name, matrix in installed_frames.items():
        original = local_outline @ matrix[:3, :3].T + matrix[:3, 3]
        shown = (original - pivot) @ enlarged_rotation.T + pivot + enlarged_translation
        ax.plot(*original[:, coordinates].T, '--', color = '0.6')
        ax.fill(*shown[:, coordinates].T, color = colours[name], alpha = .15)
        ax.plot(*shown[:, coordinates].T, color = colours[name])
        longitudinal_view_points.extend([original[:, coordinates], shown[:, coordinates]])
        centre = (installed_centres[name] - pivot) @ enlarged_rotation.T + pivot + enlarged_translation
        ax.text(centre[coordinates[0]], centre[coordinates[1]], name, ha = 'center', va = 'center')
    ax.plot(pivot[coordinates[0]], pivot[coordinates[1]], 'kD', ms = 4)
    ax.axhline(INSTALLATION_Y_M if ordinate == 'World Y [m]' else 0., color = '0.85', lw = .8)
    titles = {
        'Vertical translation only': f'Vertical translation: {vertical_at_inspection_m * 1e9:+.3g} nm',
        'Pitch only': f'Pitch: {pitch_at_inspection_rad * 1e6:+.3g} µrad',
        'Yaw only': f'Yaw: {yaw_at_inspection_rad * 1e6:+.3g} µrad',
    }
    ax.set(title = titles[label], xlabel = 'World Z [m]', ylabel = ordinate)
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
# Show both rolled cross-sections equally, with fixed centres and rotated axes
########################################
roll = body_poses['Roll only'][:3, :3] @ installed_body[:3, :3].T
shown_roll = Rotation.from_rotvec(DISPLAY_GAIN * Rotation.from_matrix(roll).as_rotvec()).as_matrix()
for ax, name in zip(roll_axes, ('q1', 'q2')):
    matrix = installed_frames[name]
    original = end @ matrix[:3, :3].T + matrix[:3, 3]
    shown = (original - pivot) @ shown_roll.T + pivot
    original_deck = deck_end @ installed_body[:3, :3].T + pivot
    shown_deck = (original_deck - pivot) @ shown_roll.T + pivot
    ax.plot(original_deck[:, 0], original_deck[:, 1], '--', color = '0.6', lw = .8)
    ax.plot(shown_deck[:, 0], shown_deck[:, 1], color = '0.25', lw = 1.2)
    mount = np.array([[0., deck_y, 0.], [0., -half_height, 0.]]) @ shown_roll.T + pivot
    ax.plot(mount[:, 0], mount[:, 1], color = '0.25', lw = 1.2)
    ax.plot(original[:, 0], original[:, 1], '--', color = '0.6')
    ax.fill(shown[:, 0], shown[:, 1], color = colours[name], alpha = .15)
    ax.plot(shown[:, 0], shown[:, 1], color = colours[name])
    for direction in ([half_width, 0., 0.], [0., half_height, 0.]):
        axis = np.array([np.zeros(3), direction]) @ shown_roll.T + installed_centres[name]
        ax.plot(axis[:, 0], axis[:, 1], color = colours[name], lw = 1)
    ax.plot(pivot[0], pivot[1], 'kD', ms = 4)
    ax.set(title = f'{name}: roll {roll_at_inspection_rad * 1e6:+.3g} µrad', xlabel = 'World X [m]')
    ax.set_aspect('equal', adjustable = 'box')
    ax.margins(x = .15, y = .15)
roll_axes[0].set_ylabel('World Y [m]')
roll_axes[1].tick_params(labelleft = False)
geometry_fig.suptitle(f'Rigid magnet geometry at t = {INSPECT_TIME_S * 1e3:g} ms\n'
                      f'Translation and rotation changes ×{DISPLAY_GAIN:,.0f} for display', fontsize = 12)
geometry_fig.legend(handles = [
    Line2D([], [], ls = '--', color = '0.6', label = 'Static installation-error placement'),
    Line2D([], [], color = 'tab:blue', label = 'q1 transformed'),
    Line2D([], [], color = 'tab:orange', label = 'q2 transformed'),
    Line2D([], [], color = '0.25', label = 'Rigid girder and mounts'),
    Line2D([], [], marker = 'D', ls = '', color = 'k', label = 'Installed rotation reference'),
], loc = 'lower center', ncol = 3, fontsize = 9)
geometry_fig.tight_layout(rect = (0., .08, 1., 1.))

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
    ax.axvline(INSPECT_TIME_S * 1e3, ls = ':', color = '0.5')
    ax.set(title = label + (' — automatic updates' if label == 'All four motions' else ''),
           xlabel = 'Probe arrival time [ms]', ylabel = 'Beam change [nm]')
    ax.grid(alpha = .2)
beam_fig.suptitle('Tracked beam at the downstream observation point\n'
                  'Changes from the static installation-error case; panels have individual vertical scales', fontsize = 11)
handles, labels = beam_axes[0].get_legend_handles_labels()
handles.append(Line2D([], [], ls = ':', color = '0.5'))
labels.append(f'Geometry shown at t = {INSPECT_TIME_S * 1e3:g} ms')
beam_fig.legend(handles, labels, loc = 'lower center', ncol = 3, fontsize = 9)
beam_fig.tight_layout(rect = (0., .04, 1., 1.))

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
