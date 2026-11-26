# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# The same magnet motion can coexist with a fixed, girder-owned or magnet-owned
# chamber. Keep intentional magnetic-axis offsets distinct from pipe placement.
# Each beam comparison is frozen at one physical time; moving-pipe automatic
# tracking is not implemented. Extract fresh fixed-survey-plane aperture cuts.

################################################################################
# Required packages
################################################################################
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Deliberate magnetic geometry and installed mechanical positions
########################################
QUAD_LENGTH_M       = .5
QUAD_K1_M2          = 1.2
DELIBERATE_X_M      = .6e-3
DELIBERATE_Y_M      = 50e-6
GIRDER_INSTALL_Y_M = .1e-3
MAGNET_INSTALL_Y_M = 20e-6

########################################
# Stated displacement inputs, enlarged to reveal aperture consequences
########################################
# These are deliberately large prescribed motions, not a ground-noise budget.
GIRDER_PEAK_M = .4e-3
MAGNET_PEAK_M = .15e-3
INSPECT_TIME_S = .25

########################################
# Chamber geometry, independent of the intentional magnetic-axis offsets
########################################
PIPE_HALF_X_M = 2e-3
PIPE_HALF_Y_M = 1e-3
PIPE_X_M      = .2e-3

########################################
# Frozen beam probe and secondary timing output
########################################
PROBE_X_M     = np.linspace(-4e-3, 4e-3, 41)
PROBE_Y_M     = np.linspace(-1.5e-3, 1.5e-3, 41)
P0C_EV        = 45.6e9
PRINT_TIMINGS = False

################################################################################
# Compare three mechanical ownership choices
################################################################################

########################################
# Keep identical magnet physics and input particles in every case
########################################
cases = {}
timings = {}
labels = {'fixed': 'Fixed installed pipe',
          'girder': 'Pipe follows girder',
          'magnet': 'Pipe follows q1 body'}
colours = {'fixed': '.4', 'girder': 'tab:blue', 'magnet': 'tab:orange'}
line_length_m = 1.5 + 2 * QUAD_LENGTH_M
check_s_m = 1. + QUAD_LENGTH_M
window_s = np.array([.15, check_s_m, line_length_m - .15])
probe_x, probe_y = np.meshgrid(PROBE_X_M, PROBE_Y_M)
probe_x, probe_y = probe_x.ravel(), probe_y.ravel()

for policy in labels:

    ########################################
    # Construct the same two deliberately offset magnets
    ########################################
    started = perf_counter()
    env = xt.Environment()
    env.new('up', xt.Drift, length = .5)
    env.new('gap', xt.Drift, length = .5)
    env.new('down', xt.Drift, length = .5)
    env.new('q1', xt.Quadrupole, length = QUAD_LENGTH_M, k1 = QUAD_K1_M2,
            shift_x = DELIBERATE_X_M, shift_y = DELIBERATE_Y_M)
    env.new('q2', xt.Quadrupole, length = QUAD_LENGTH_M, k1 = -QUAD_K1_M2)
    line = env.new_line(name = 'line', components = ['up', 'q1', 'gap', 'q2', 'down'])
    line.particle_ref = xt.Particles(p0c = P0C_EV, mass0 = xt.ELECTRON_MASS_EV, q0 = 1)

    ########################################
    # Put the magnets above a shared girder and give q1 relative motion
    ########################################
    mech = env.mechanics
    line.mechanics.adopt_reference()
    girder = mech.new_girder('girder', anchor = xt.Frame.from_survey_angles(
        Y = -.6, Z = .75 + QUAD_LENGTH_M))
    half_spacing_m = (QUAD_LENGTH_M + .5) / 2
    magnet = mech.new_body('magnet1', parent = girder,
        mount = xt.Frame.from_survey_angles(Y = .6, Z = -half_spacing_m),
        elements = ['q1'], line = line, kind = 'magnet')
    mech.new_body('magnet2', parent = girder,
        mount = xt.Frame.from_survey_angles(Y = .6, Z = half_spacing_m),
        elements = ['q2'], line = line, kind = 'magnet')
    girder.installation.shift_y = GIRDER_INSTALL_Y_M
    magnet.installation.shift_y = MAGNET_INSTALL_Y_M
    mech.add_source('girder_y', xt.mechanics.Harmonic(GIRDER_PEAK_M, 1.),
                    targets = 'girder', component = 'y')
    mech.add_source('magnet_y', xt.mechanics.Harmonic(MAGNET_PEAK_M, 2., np.pi / 2),
                    targets = 'magnet1', component = 'y')

    ########################################
    # Define one physical chamber independently of the magnetic-axis reference
    ########################################
    profile = env.aperture.new_profile('section', 'Rectangle',
        max_x = PIPE_HALF_X_M, max_y = PIPE_HALF_Y_M)
    pipe = env.aperture.new_pipe('chamber_shape', positions = [
        profile.place(at = .1), profile.place(at = line_length_m - .1)])
    line.aperture.place_pipe('chamber', pipe, at = 'up', shift_x = PIPE_X_M)
    line.aperture._with_progress = False
    owner = {'fixed': None, 'girder': girder, 'magnet': magnet}[policy]
    line.mechanics.attach_pipe('chamber', body = owner)

    ########################################
    # Inspect the declared ownership before calculating a pose
    ########################################
    print(f'\n{labels[policy]}\n' + '-' * len(labels[policy]))
    line.mechanics.get_pipe_table().show()
    timings[policy + ': construction and reference aperture'] = perf_counter() - started

    ########################################
    # Freeze magnet fields and pipe cuts from one mechanical state
    ########################################
    started = perf_counter()
    frozen = line.mechanics.freeze_at(time_s = INSPECT_TIME_S, aperture_at_s = [check_s_m])
    pipe_table = frozen.pipe_table
    aperture = frozen.aperture
    windows = aperture.get_cross_sections_at_s(window_s).cross_section
    timings[policy + ': matched freeze and aperture cuts'] = perf_counter() - started

    ########################################
    # Inspect the same magnet poses and sampled aperture before q2
    ########################################
    limit = frozen.limits[check_s_m]
    magnet_poses = frozen.element_poses
    magnetic_y = np.array([magnet_poses[name].Y for name in ('q1', 'q2')])
    magnetic_x = np.array([magnet_poses[name].X for name in ('q1', 'q2')])
    check_section = np.column_stack((limit.x_vertices, limit.y_vertices))
    assert line['q1'].shift_x == DELIBERATE_X_M
    assert line.aperture.pipe_positions['chamber'].shift_y == 0.

    ########################################
    # Track a frozen copy of the upstream transport and this sampled limit
    ########################################
    # Copy elements, rather than attaching a second physical line to the bodies.
    # The copied numeric magnet fields and aperture both belong to this time.
    started = perf_counter()
    probe = xt.Line(elements = [line.element_dict[name].copy()
                               for name in ('up', 'q1', 'gap')] + [limit],
                    element_names = ['up', 'q1', 'gap', 'pipe_check'])
    probe.particle_ref = line.particle_ref.copy()
    particles = probe.build_particles(x = probe_x, y = probe_y)
    probe.track(particles)
    order = np.argsort(particles.particle_id)
    alive = np.array(particles.state)[order] > 0
    beam_xy = np.column_stack((particles.x, particles.y))[order]
    timings[policy + ': frozen beam probe including first use'] = perf_counter() - started
    cases[policy] = dict(env = env, line = line, aperture = aperture,
        pipe_table = pipe_table, windows = windows,
        magnetic_x = magnetic_x, magnetic_y = magnetic_y, alive = alive,
        check_section = check_section, beam_xy = beam_xy)

################################################################################
# Read the mechanical reference and beam acceptance together
################################################################################

########################################
# Compare millimetre chamber positions with the common magnetic-axis position
########################################
print(f'\nFrozen physical time: {INSPECT_TIME_S:g} s')
print('The intentional q1 magnetic-axis offset does not move the chamber.')
print('Every case has identical magnet motion and identical incident particles.')
print(f'Incident probe grid: {len(PROBE_X_M)} x {len(PROBE_Y_M)} points; a stated scan, not an emittance distribution.')
print(f'{"Ownership":<28} {"Pipe Y [mm]":>13} {"q1 axis Y [mm]":>16}'
      f' {"q2 axis Y [mm]":>16}'
      f' {"Accepted incident y [mm]":>27}')
for policy, case in cases.items():
    accepted_y = probe_y[case['alive']] * 1e3
    interval = (f'[{accepted_y.min():+.4f}, {accepted_y.max():+.4f}]'
                if len(accepted_y) else 'No surviving probes')
    print(f'{labels[policy]:<28} {case["pipe_table"].total_Y[0] * 1e3:13.6f}'
          f' {case["magnetic_y"][0] * 1e3:16.6f}'
          f' {case["magnetic_y"][1] * 1e3:16.6f}'
          f' {interval:>27}')
np.testing.assert_allclose(cases['fixed']['magnetic_y'], cases['girder']['magnetic_y'])
np.testing.assert_allclose(cases['fixed']['magnetic_y'], cases['magnet']['magnetic_y'])
np.testing.assert_allclose(cases['fixed']['beam_xy'], cases['girder']['beam_xy'])
np.testing.assert_allclose(cases['fixed']['beam_xy'], cases['magnet']['beam_xy'])
print(f'This is discrete frozen aperture checking at s={check_s_m:g} m, not continuous wall interception.')
print('The transverse windows below mark q2 at that checking plane; q1 owns the relative magnet/pipe motion.')
print('The original line.aperture retains the adopted reference geometry.')
print('Automatic moving-pipe updates are guarded; these limits are frozen samples.')

################################################################################
# Compare actual transverse aperture windows
################################################################################

########################################
# Keep the same tracked beam and magnetic axis in every transverse view
########################################
beam_mm = cases['fixed']['beam_xy'] * 1e3
reference = cases['fixed']['check_section'] * 1e3
span_mm = max(2.4 * PIPE_HALF_X_M * 1e3, np.ptp(beam_mm[:, 1]) + .4)
centre_y_mm = (beam_mm[:, 1].max() + beam_mm[:, 1].min()) / 2
fig, axes = plt.subplots(1, 3, figsize = (11.5, 5.3), sharex = True, sharey = True)
fig.suptitle(f'Same tracked beam; chamber opening at q2 entrance (s={check_s_m:g} m, t={INSPECT_TIME_S:g} s)',
             fontsize = 13, y = .98)
for ax, (policy, case) in zip(axes, cases.items()):
    colour = colours[policy]
    opening = case['check_section'] * 1e3
    ax.add_patch(Polygon(reference, closed = True, fill = False,
                         edgecolor = '.6', ls = '--', lw = 1.2))
    ax.add_patch(Polygon(opening, closed = True, edgecolor = colour,
                         facecolor = to_rgba(colour, .09), lw = 2.))
    for accepted, colour_of_points in [(False, '#ba4e4e'), (True, '#257a45')]:
        selected = case['alive'] == accepted
        ax.scatter(beam_mm[selected, 0], beam_mm[selected, 1],
                    s = 5, color = colour_of_points, alpha = .65, zorder = 3)
    ax.plot(case['magnetic_x'][1] * 1e3, case['magnetic_y'][1] * 1e3,
             '+', color = 'black', ms = 12, mew = 1.7, zorder = 4)
    ax.plot(PIPE_X_M * 1e3, case['pipe_table'].total_Y[0] * 1e3,
             'o', mfc = 'white', mec = colour, ms = 5, zorder = 4)
    ax.set_title(f'{labels[policy]}\nChamber y = {case["pipe_table"].total_Y[0] * 1e3:+.3f} mm',
                  fontsize = 11, pad = 12)
    ax.set(xlabel = 'x [mm]',
            xlim = (PIPE_X_M * 1e3 - span_mm / 2, PIPE_X_M * 1e3 + span_mm / 2),
            ylim = (centre_y_mm - span_mm / 2, centre_y_mm + span_mm / 2))
    ax.set_aspect('equal', adjustable = 'box')
    ax.grid(alpha = .15)
axes[0].set_ylabel('y [mm]')
fig.legend(handles = [
    Line2D([], [], color = '.6', ls = '--', label = 'Adopted chamber position'),
    Line2D([], [], marker = '+', color = 'black', ls = '', ms = 10, label = 'q2 magnetic axis'),
    Line2D([], [], marker = 'o', mfc = 'white', mec = '.4', color = '.4', ls = '', label = 'Chamber centre'),
    Line2D([], [], marker = '.', color = '#257a45', ls = '', label = 'Survives aperture check'),
    Line2D([], [], marker = '.', color = '#ba4e4e', ls = '', label = 'Lost at aperture check'),
], loc = 'lower center', ncol = 3, frameon = False, fontsize = 9)
fig.subplots_adjust(left = .07, right = .98, bottom = .20, top = .79, wspace = .12)

################################################################################
# Locate the chamber windows and magnets in three dimensions
################################################################################

########################################
# Draw sampled sections, longitudinal rails and rigid magnetic axes
########################################
geometry_fig = plt.figure(figsize = (13, 5.4))
geometry_fig.suptitle('Chamber windows around the same two magnets', fontsize = 14, y = .98)
reference_closed = np.vstack((reference, reference[0]))
for column, (policy, case) in enumerate(cases.items(), start = 1):
    ax = geometry_fig.add_subplot(1, 3, column, projection = '3d')
    colour = colours[policy]
    windows = case['windows']
    for station, section in zip(window_s, windows):
        points = np.column_stack((np.full(len(section), station), section * 1e3))
        ax.add_collection3d(Poly3DCollection([points],
            facecolor = to_rgba(colour, .06), edgecolor = colour, linewidth = 1.2))
    first = windows[0]
    corner_indices = [np.argmax(first[:, 0] + first[:, 1]),
                      np.argmax(first[:, 0] - first[:, 1]),
                      np.argmin(first[:, 0] + first[:, 1]),
                      np.argmin(first[:, 0] - first[:, 1])]
    for index in corner_indices:
        rail = np.array([section[index] for section in windows]) * 1e3
        ax.plot(window_s, rail[:, 0], rail[:, 1], color = colour, lw = .8, alpha = .5)
    ax.plot(np.full(len(reference_closed), check_s_m),
             reference_closed[:, 0], reference_closed[:, 1],
             color = '.6', ls = '--', lw = 1.)
    ax.plot(window_s[[0, -1]], [0., 0.], [0., 0.], ':', color = '.5', lw = 1.)
    for index, start_s in enumerate((.5, check_s_m)):
        x_mm = case['magnetic_x'][index] * 1e3
        y_mm = case['magnetic_y'][index] * 1e3
        ax.plot([start_s, start_s + QUAD_LENGTH_M], [x_mm, x_mm], [y_mm, y_mm],
                 color = 'black', lw = 2.4)
        ax.text(start_s + QUAD_LENGTH_M / 2, x_mm, y_mm + .14,
                 f'q{index + 1}', ha = 'center', fontsize = 9)
    ax.set(title = labels[policy], xlabel = 's [m]', ylabel = 'x [mm]', zlabel = 'y [mm]',
            xlim = (0., line_length_m),
            ylim = ((PIPE_X_M - 1.2 * PIPE_HALF_X_M) * 1e3,
                    (PIPE_X_M + 1.2 * PIPE_HALF_X_M) * 1e3),
            zlim = (-1.3 * PIPE_HALF_Y_M * 1e3,
                    (1.3 * PIPE_HALF_Y_M + GIRDER_INSTALL_Y_M + GIRDER_PEAK_M) * 1e3))
    ax.set_box_aspect((2.6, 1.4, 1.6))
    ax.view_init(elev = 20, azim = -55)
    ax.tick_params(labelsize = 8, pad = 1)
    ax.set_zlabel('y [mm]', labelpad = 1)
    ax.grid(False)
geometry_fig.legend(handles = [
    Line2D([], [], color = 'black', lw = 2.4, label = 'Magnetic axes'),
    Line2D([], [], color = '.6', ls = '--', label = 'Adopted chamber at check plane'),
    Line2D([], [], color = '.5', ls = ':', label = 'Nominal beam reference'),
], loc = 'lower center', ncol = 3, frameon = False, fontsize = 9)
geometry_fig.text(.5, .91, 'Transverse scale enlarged relative to length; axes retain physical units.',
                   ha = 'center', fontsize = 10)
geometry_fig.subplots_adjust(left = .01, right = .92, bottom = .13, top = .87, wspace = .12)

########################################
# Keep preparation and tracking costs separate and optional
########################################
if PRINT_TIMINGS:
    print('\nSeparate operation costs [s]')
    for name, value in timings.items():
        print(f'{name:<55} {value:10.6f}')
plt.show()
