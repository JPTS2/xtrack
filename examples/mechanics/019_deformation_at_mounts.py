# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# How can a deforming support carry several magnets that remain rigid?
# Give Xtrack two mode shapes at three stations and two amplitude histories.
# Compare a rigid parent, bending alone, and bending plus torsion.

################################################################################
# Required packages
################################################################################
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.constants import c
from scipy.spatial.transform import Rotation
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Girder stations and rigid magnet geometry
########################################
GIRDER_LENGTH_M      = 6.
MAGNET_STATIONS_M    = np.array([1., 3., 5.])
GIRDER_HALF_HEIGHT_M = .2
STAND_HEIGHT_M       = .2
MAGNET_HEIGHT_M      = .4
MAGNET_LENGTH_M      = .4
MAGNET_WIDTH_M       = .3

########################################
# Two common prescribed modal amplitudes
########################################
BENDING_PEAK_M       = 2e-6
BENDING_HZ           = 5.
TORSION_PEAK_RAD     = 8e-6
TORSION_HZ           = 7.
# w(s,t) = q_b(t) sin(pi s/L); twist(s,t) = q_t(t) cos(pi s/L).
# These are stated kinematic shapes, not calculated elastic eigenmodes.

########################################
# Common parent motion and static placement
########################################
PARENT_SHIFT_M       = (20e-6, -30e-6, 10e-6)
PARENT_ANGLES_RAD    = (.04, -.03, .05)   # Finite survey pitch, yaw, roll
INSTALLATION_Y_M     = .7e-6
INSTALLATION_ROLL_RAD = .002
DELIBERATE_Q1_X_M    = .3e-3
DELIBERATE_Q1_ROLL_RAD = .04

########################################
# Physical record, toy tracking clock and display
########################################
N_TURNS             = 64
TURN_PERIOD_S       = .001
SAMPLE_STEP_S       = .001
INSPECT_TIME_S      = .025
DISPLAY_GAIN        = 100000.           # Magnifies both modal amplitudes in geometry only
PRINT_TIMINGS       = False

################################################################################
# Resolve the prescribed field into rigid motion at each mount
################################################################################

########################################
# Evaluate one cross-section from its displacement, tangent and twist
########################################
# Each transverse section is rigid. Different stations have different poses.
# A magnet uses one effective mount and remains rigid over its full length.
# Contact compliance, elastic loading and bent magnetic fields are not solved.
def section_pose(station_m, bending_m, torsion_rad):
    phase = np.pi * station_m / GIRDER_LENGTH_M
    vertical_m = bending_m * np.sin(phase)
    slope = bending_m * np.pi / GIRDER_LENGTH_M * np.cos(phase)
    pitch_rad = slope
    roll_rad = torsion_rad * np.cos(phase)
    # Small-deformation pitch is the slope. The display amplifies this angle.
    # The subsequent local roll twists about that tangent.
    rotation = Rotation.from_euler('XZ', [-pitch_rad, roll_rad]).as_matrix()
    centre = np.array([0., vertical_m, station_m])
    return centre, rotation, pitch_rad, roll_rad

########################################
# Keep optional timing diagnostics separate from physical inputs
########################################
timings = {}
started = perf_counter()
n_samples = int(np.ceil(N_TURNS * TURN_PERIOD_S / SAMPLE_STEP_S)) + 2
times = np.arange(n_samples) * SAMPLE_STEP_S
bending = BENDING_PEAK_M * np.sin(2 * np.pi * BENDING_HZ * times)
torsion = TORSION_PEAK_RAD * np.cos(2 * np.pi * TORSION_HZ * times)
port_names = tuple(f'port.q{i + 1}' for i in range(len(MAGNET_STATIONS_M)))

########################################
# Specify bending and torsion at the neutral-axis stations
########################################
phase = np.pi * MAGNET_STATIONS_M / GIRDER_LENGTH_M
shapes = np.zeros((2, len(port_names), 6))    # [mode, mount, x/y/s/pitch/yaw/roll]
shapes[0, :, 1] = np.sin(phase)              # dy per metre of bending
shapes[0, :, 3] = np.pi / GIRDER_LENGTH_M * np.cos(phase)  # pitch [rad/m]
shapes[1, :, 5] = np.cos(phase)              # roll per radian of torsion
amplitudes = np.column_stack((bending, torsion))
flex_motion = xt.mechanics.ModalMotion(
    times, amplitudes, shapes,
    target_names = port_names, mode_names = ('vertical bending', 'torsion'),
)
timings['Prescribed mount-record construction'] = perf_counter() - started

################################################################################
# Build the ordinary magnetic line and its mounting tree
################################################################################

########################################
# Preserve an intentionally offset and rolled quadrupole
########################################
started = perf_counter()
env = xt.Environment()
for name, strength in [('q1', .12), ('q2', -.15), ('q3', .1)]:
    env.new(name, xt.Quadrupole, length = MAGNET_LENGTH_M, k1 = strength)
env['q1'].shift_x = DELIBERATE_Q1_X_M
env['q1'].rot_s_rad = DELIBERATE_Q1_ROLL_RAD

########################################
# Put the three magnet centres at the declared stations
########################################
components = []
previous_end_m = 0.
for index, station_m in enumerate(MAGNET_STATIONS_M):
    drift_name = f'before.q{index + 1}'
    drift_length_m = station_m - MAGNET_LENGTH_M / 2 - previous_end_m
    env.new(drift_name, xt.Drift, length = drift_length_m)
    components.extend([drift_name, f'q{index + 1}'])
    previous_end_m = station_m + MAGNET_LENGTH_M / 2
env.new('end_of_girder', xt.Drift, length = GIRDER_LENGTH_M - previous_end_m)
components.append('end_of_girder')

########################################
# Use a stable toy return map to establish a visible physical tracking clock
########################################
# The map is a transport/clock model; it is not a surveyed ring layout.
reference = xt.Particles(p0c = 7e12)
env.elements['return'] = xt.LineSegmentMap(
    length = float(reference.beta0[0]) * c * TURN_PERIOD_S - GIRDER_LENGTH_M,
    qx = .23, qy = .31, betx = 8., bety = 9.,
    damping_rate_x = .02, damping_rate_px = .02,
    damping_rate_y = .02, damping_rate_py = .02,
)
line = env.new_line(name = 'ring', components = components + ['return'])
line.particle_ref = reference
line.build_tracker()
timings['Line and tracker construction'] = perf_counter() - started

########################################
# Retain the surveyed design field frames for the rigid drawing envelopes
########################################
started = perf_counter()
design_survey = line.survey(include_element_frames = True)
timings['Original surveyed field frames'] = perf_counter() - started

########################################
# Create neutral-axis ports and put each rigid magnet above its port
########################################
started = perf_counter()
axis_height_m = GIRDER_HALF_HEIGHT_M + STAND_HEIGHT_M + MAGNET_HEIGHT_M / 2
mech = env.mechanics
line.mechanics.adopt_reference()
girder = mech.new_girder('girder_reference',
    anchor = xt.Frame.from_survey_angles(Y = -axis_height_m))
girder.installation.shift_y = INSTALLATION_Y_M
girder.installation.rot_s_rad = INSTALLATION_ROLL_RAD
for index, station_m in enumerate(MAGNET_STATIONS_M):
    port = mech.new_body(port_names[index], kind = 'mount', parent = girder,
        mount = xt.Frame.from_survey_angles(Z = station_m))
    mech.new_body(f'magnet.q{index + 1}', kind = 'magnet', parent = port,
        mount = xt.Frame.from_survey_angles(Y = GIRDER_HALF_HEIGHT_M + STAND_HEIGHT_M),
        elements = [f'q{index + 1}'], line = line)

########################################
# Compose common rigid motion with relative deformation at the ports
########################################
parent_values = np.tile([*PARENT_SHIFT_M, *PARENT_ANGLES_RAD], (len(times), 1))
parent_motion = xt.mechanics.RigidMotion(times, parent_values, ('girder_reference',))
mech.add_source('common_parent', parent_motion, targets = 'girder_reference')
mech.add_source('flexural_mounts', flex_motion, targets = port_names)
timings['Mounting tree and source definitions'] = perf_counter() - started

################################################################################
# Inspect the physical ownership before applying any motion
################################################################################

########################################
# Select the mount rows from the ordinary mechanics table
########################################
mounts = mech.get_table().rows.match(kind = 'mount')
print('Three deformation ports carrying rigid magnets\n'
      '---------------------------------------------')
mounts.show(cols = ['name', 'kind', 'parent'])
print(f'Girder length [m]: {GIRDER_LENGTH_M:g}; mount stations [m]: {MAGNET_STATIONS_M}.')
print(f'Neutral-axis to magnetic-axis height [m]: {axis_height_m:g}.')
print(f'Common parent pitch / yaw / roll [rad]: {PARENT_ANGLES_RAD}.')
print(f'Static parent installation: y = {INSTALLATION_Y_M * 1e6:g} µm; roll = {INSTALLATION_ROLL_RAD * 1e3:g} mrad.')
print('A port carries one rigid magnet; the girder reference supplies common finite motion.')
print('The bending slope sets pitch. Torsion and mount height jointly set horizontal sway.')
print('This is prescribed kinematics, without an elastic/contact solver or bent magnet fields.')

########################################
# Inspect the common source and the single multi-port deformation record
########################################
source_table = mech.get_source_table()
source_table.show(cols = ['name', 'model', 'reference', 'n_targets'], maxwidth = 'full')
print(f'Modal input record: {flex_motion.nbytes:,} bytes for shape and amplitude data.')

################################################################################
# Resolve the magnet motion relative to the same rigid-parent baseline
################################################################################

########################################
# Query both cases at identical times without changing the tracked line
########################################
started = perf_counter()
parent_rotation = Rotation.from_euler('YXZ',
    [PARENT_ANGLES_RAD[1], -PARENT_ANGLES_RAD[0], PARENT_ANGLES_RAD[2]]).as_matrix()
parent_rotation = parent_rotation @ Rotation.from_euler('Z', INSTALLATION_ROLL_RAD).as_matrix()
magnet_delta_m = np.zeros((len(times), len(port_names), 3))
for row, time_s in enumerate(times):
    moving = line.mechanics.get_element_poses(time_s = float(time_s))
    rigid = line.mechanics.get_element_poses(time_s = float(time_s),
                                            disabled_sources = ['flexural_mounts'])
    for column in range(len(port_names)):
        name = f'q{column + 1}'
        # Track the same magnetic-axis centre in each rigid element frame.
        local_centre = np.array([0., 0., MAGNET_LENGTH_M / 2])
        moved_centre = moving[name].XYZ + moving[name].matrix[:3, :3] @ local_centre
        rigid_centre = rigid[name].XYZ + rigid[name].matrix[:3, :3] @ local_centre
        magnet_delta_m[row, column] = parent_rotation.T @ (moved_centre - rigid_centre)
timings['Read-only matched pose inspection'] = perf_counter() - started

########################################
# Report an actual stored sample with unscaled physical units
########################################
selected = int(np.argmin(np.abs(times - INSPECT_TIME_S)))
shown_time_s = float(times[selected])
print(f'\nDeformation relative to the same rigid-parent pose at t = {shown_time_s * 1e3:g} ms\n'
      '----------------------------------------------------------------------------')
print('Magnet    Centre dx [µm]    Centre dy [µm]    Centre ds [µm]    Pitch [µrad]    Roll [µrad]')
mount_values = flex_motion.evaluate(times[selected:selected + 1], MAGNET_STATIONS_M)[0]
for column, name in enumerate(('q1', 'q2', 'q3')):
    delta_um = magnet_delta_m[selected, column] * 1e6
    pitch_urad, roll_urad = mount_values[column, [3, 5]] * 1e6
    print(f'{name:<8} {delta_um[0]:+14.6f} {delta_um[1]:+17.6f} {delta_um[2]:+17.6f}'
          f' {pitch_urad:+15.6f} {roll_urad:+14.6f}')
print('Translations above use the co-moving parent axes; source pitch/roll use nominal port axes.')
print('The opposite end slopes produce opposite pitch; opposite twist produces opposite horizontal sway.')
print('The deliberate q1 offset also contributes its small, preserved rotational lever arm.')

########################################
# Inspect the same state with its installation and common parent motion
########################################
print('\nMagnetic-axis entrance poses in the line survey frame [m]')
line.mechanics.get_table(time_s = shown_time_s).show(
    cols = ['name', 'reference_Y', 'static_Y', 'dynamic_Y', 'total_Y'],
)

########################################
# Compare the same mounts with zero, one and two excited modes
########################################
print(f'\nMount inputs at t = {shown_time_s * 1e3:g} ms (neutral-axis stations)')
print('Case                    q2 dy [µm]    q1 pitch [µrad]    q1 roll [µrad]')
for label, mask in [('Rigid parent', [0, 0]), ('Bending', [1, 0]), ('Bending and torsion', [1, 1])]:
    mount_sample = np.einsum('m,mbc->bc', amplitudes[selected] * mask, shapes)
    print(f'{label:<21} {mount_sample[1, 1] * 1e6:+12.5f}'
          f' {mount_sample[0, 3] * 1e6:+17.5f} {mount_sample[0, 5] * 1e6:+16.5f}')

################################################################################
# Use the same records in native turn-held tracking
################################################################################

########################################
# Freeze each pose once per turn with the existing compiled backend
########################################
started = perf_counter()
particles = line.build_particles(x = 1e-4, y = -2e-4)
line.mechanics.enable_vibrations(backend = 'compiled')
line.track(particles, num_turns = N_TURNS)
timings['Native compiled tracking'] = perf_counter() - started
assert line.mechanics._runtime is not None
assert np.all(particles.state > 0)
assert int(particles.at_turn[0]) == N_TURNS
assert line['q1'].rot_s_rad == DELIBERATE_Q1_ROLL_RAD
assert girder.installation.shift_y == INSTALLATION_Y_M
line.mechanics.freeze_updates()
print('\nNative execution and retained static inputs\n'
      '------------------------------------------')
print(f'Completed turns: {int(particles.at_turn[0])}; physical clock [ms]: {particles.time_s * 1e3:.3f}.')
print('Records are interpolated on the host; finite hierarchy composition and element lowering use native kernels.')
print(f'Deliberate q1 roll is unchanged [rad]: {line["q1"].rot_s_rad:g}.')
print('Static installation inputs are unchanged. Automatic updates are now frozen for inspection.')
print('env.mechanics.clear() restores adopted element alignment and removes this mechanical model.')

################################################################################
# Draw the prescribed deformation and the rigid carried magnets
################################################################################

########################################
# Compare the undeformed and deformed assembly in the parent frame
########################################
colours = ['tab:blue', 'tab:orange', 'tab:green']
fig = plt.figure(figsize = (11, 7.5))
grid = fig.add_gridspec(2, 3, height_ratios = (1.15, 1.))
side = fig.add_subplot(grid[0, :])
ends = [fig.add_subplot(grid[1, column]) for column in range(3)]
display_bend = DISPLAY_GAIN * bending[selected]
display_twist = DISPLAY_GAIN * torsion[selected]
curve_s = np.linspace(0., GIRDER_LENGTH_M, 300)
upper, lower, centreline = [], [], []
for station_m in curve_s:
    centre, rotation, _, _ = section_pose(station_m, display_bend, display_twist)
    centreline.append(centre)
    upper.append(centre + rotation @ [0., GIRDER_HALF_HEIGHT_M, 0.])
    lower.append(centre + rotation @ [0., -GIRDER_HALF_HEIGHT_M, 0.])
upper, lower, centreline = np.array(upper), np.array(lower), np.array(centreline)
side.fill(np.r_[upper[:, 2], lower[::-1, 2]], np.r_[upper[:, 1], lower[::-1, 1]],
          color = '0.85', alpha = .6)
for edge in (upper, lower):
    side.plot(edge[:, 2], edge[:, 1], color = '0.35', lw = 1.)
side.plot(centreline[:, 2], centreline[:, 1], ':', color = '0.35', lw = 1.)
for height in (-GIRDER_HALF_HEIGHT_M, GIRDER_HALF_HEIGHT_M):
    side.plot([0., GIRDER_LENGTH_M], [height, height], '--', color = '0.65', lw = .9)

########################################
# Carry each complete rigid magnet using only its own mount-section frame
########################################
magnet_bottom_m = GIRDER_HALF_HEIGHT_M + STAND_HEIGHT_M
magnet_top_m = magnet_bottom_m + MAGNET_HEIGHT_M
side_outline = np.array([
    [0., magnet_bottom_m, -MAGNET_LENGTH_M / 2],
    [0., magnet_top_m, -MAGNET_LENGTH_M / 2],
    [0., magnet_top_m, MAGNET_LENGTH_M / 2],
    [0., magnet_bottom_m, MAGNET_LENGTH_M / 2],
    [0., magnet_bottom_m, -MAGNET_LENGTH_M / 2],
])
end_outline = np.array([
    [-MAGNET_WIDTH_M / 2, magnet_bottom_m, 0.],
    [MAGNET_WIDTH_M / 2, magnet_bottom_m, 0.],
    [MAGNET_WIDTH_M / 2, magnet_top_m, 0.],
    [-MAGNET_WIDTH_M / 2, magnet_top_m, 0.],
    [-MAGNET_WIDTH_M / 2, magnet_bottom_m, 0.],
])
for column, (station_m, colour) in enumerate(zip(MAGNET_STATIONS_M, colours)):
    centre, rotation, _, _ = section_pose(station_m, display_bend, display_twist)
    # Use the surveyed original field frame, preserving deliberate offsets and
    # roll exactly. Only the deformation amplitudes receive DISPLAY_GAIN.
    design_frame = design_survey.get_frame(f'q{column + 1}', which = 'elem_start')
    design_rotation = design_frame.matrix[:3, :3]
    nominal_section = np.array([0., -axis_height_m, station_m])
    nominal_axis = np.array([0., axis_height_m, 0.])
    element_centre = np.array([0., 0., MAGNET_LENGTH_M / 2])
    design_origin = design_frame.XYZ - nominal_section
    design_side = (side_outline - nominal_axis + element_centre) @ design_rotation.T + design_origin
    design_end = (end_outline - nominal_axis + element_centre) @ design_rotation.T + design_origin
    design_axis = design_origin + design_rotation @ element_centre
    shown_side = design_side @ rotation.T + centre
    side.plot(design_side[:, 2] + station_m, design_side[:, 1], '--', color = '0.65', lw = .9)
    side.fill(shown_side[:, 2], shown_side[:, 1], color = colour, alpha = .18)
    side.plot(shown_side[:, 2], shown_side[:, 1], color = colour, lw = 1.5)
    stand = np.array([[0., GIRDER_HALF_HEIGHT_M, 0.], [0., magnet_bottom_m, 0.]])
    shown_stand = stand @ rotation.T + centre
    side.plot(shown_stand[:, 2], shown_stand[:, 1], color = colour, lw = 1.3)
    side.text(station_m, magnet_top_m + .25, f'q{column + 1}', ha = 'center', color = colour)
    ax = ends[column]
    shown_end = design_end @ rotation.T + centre
    ax.plot(design_end[:, 0], design_end[:, 1], '--', color = '0.65', lw = .9)
    ax.fill(shown_end[:, 0], shown_end[:, 1], color = colour, alpha = .18)
    ax.plot(shown_end[:, 0], shown_end[:, 1], color = colour, lw = 1.5)
    axis_point = centre + rotation @ design_axis
    ax.plot(axis_point[0], axis_point[1], 'o', color = colour, ms = 4)
    ax.plot(shown_stand[:, 0], shown_stand[:, 1], color = colour, lw = 1.3)
    ax.set(title = f'q{column + 1}: end view at s = {station_m:g} m',
           xlabel = 'Parent-frame x [m]', ylabel = 'Parent-frame y [m]',
           xlim = (-.45, .45), ylim = (.12, 1.02))
    ax.set_aspect('equal', adjustable = 'box')
side.set(title = 'Side view: one prescribed girder shape, three rigid carried magnets',
         xlabel = 'Parent-frame s [m]', ylabel = 'Parent-frame y [m]',
         xlim = (-.15, GIRDER_LENGTH_M + .15), ylim = (-.3, 1.12))
fig.suptitle(f't = {shown_time_s * 1e3:g} ms; modal amplitudes ×{DISPLAY_GAIN:,.0f}; common parent pose removed', fontsize = 12)
fig.legend(handles = [Line2D([], [], color = '0.65', ls = '--', label = 'Zero deformation'),
                      Line2D([], [], color = '0.35', label = 'Prescribed support shape'),
                      Line2D([], [], marker = 'o', color = '0.25', ls = '', label = 'Magnetic-axis centre')],
           loc = 'lower center', ncol = 3, fontsize = 9)
fig.tight_layout(rect = (0., .045, 1., 1.))

################################################################################
# Compare unscaled common inputs and relative magnet motion
################################################################################

########################################
# Put each modal input on its own physical scale
########################################
history_fig, axes = plt.subplots(2, 2, figsize = (10, 6.5), sharex = True)
axes[0, 0].plot(times * 1e3, bending * 1e6, color = 'tab:purple')
axes[0, 0].set(title = 'One bending amplitude drives every station', ylabel = 'q_b [µm]')
axes[0, 1].plot(times * 1e3, torsion * 1e6, color = 'tab:red')
axes[0, 1].set(title = 'One torsion amplitude drives every station', ylabel = 'q_t [µrad]')

########################################
# Show differential motion after removing the common rigid-parent case
########################################
for column, colour in enumerate(colours):
    style = '--' if column == 2 else '-'
    axes[1, 0].plot(times * 1e3, magnet_delta_m[:, column, 1] * 1e6,
                    color = colour, ls = style, label = f'q{column + 1}')
    axes[1, 1].plot(times * 1e3, magnet_delta_m[:, column, 0] * 1e6,
                    color = colour, ls = style, label = f'q{column + 1}')
axes[1, 0].set(title = 'Symmetric end stations share the bending motion',
               xlabel = 'Physical time [ms]', ylabel = 'Centre Δy [µm]')
axes[1, 1].set(title = 'Twist sways elevated magnetic axes',
               xlabel = 'Physical time [ms]', ylabel = 'Centre Δx [µm]')
for ax in axes.ravel():
    ax.axvline(shown_time_s * 1e3, color = '0.5', ls = ':', lw = 1.)
    ax.grid(alpha = .2)
    ax.margins(y = .15)
for ax in axes[1]:
    ax.legend(fontsize = 8)
history_fig.suptitle('Unscaled inputs and relative magnet motion; '
                    f'dotted line = displayed pose at {shown_time_s * 1e3:g} ms', fontsize = 12)
history_fig.tight_layout()

################################################################################
# Display figures and optional execution costs
################################################################################

########################################
# Print diagnostic costs only when requested
########################################
if PRINT_TIMINGS:
    print('\nSeparate operation costs [s]\n----------------------------')
    for name, seconds in timings.items():
        print(f'{name:<42} {seconds:.6f}')

########################################
# Keep the frozen model available for interactive inspection
########################################
plt.show()
