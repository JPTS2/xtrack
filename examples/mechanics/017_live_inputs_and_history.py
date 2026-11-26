# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Supply an applied mover position independently of ground vibration.
# Track three blocks, remove vibration, and reconstruct the motion actually used.
# This example has no feedback controller or actuator-response model.

################################################################################
# Required packages
################################################################################
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.constants import c
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Design, installation and provided mover positions
########################################
DELIBERATE_X_M       = .3e-3
DELIBERATE_ROLL_RAD  = .04
INSTALLATION_Y_M    = .2e-6
APPLIED_POSITIONS_M = (0., 2e-6, -1e-6)   # One supplied position per tracking block

########################################
# Independent ground disturbance
########################################
GROUND_PEAK_M = 50e-9
GROUND_HZ     = 20.

########################################
# Tracking and optional operation costs
########################################
P0C            = 45.6e9                 # Positron momentum times c [eV]
TURN_PERIOD_S  = .3e-3                  # Reference period of the toy one-turn map
BLOCK_TURNS    = 64
PRINT_TIMINGS  = False

if len(APPLIED_POSITIONS_M) != 3 or BLOCK_TURNS < 2:
    raise ValueError('Supply three applied positions and at least two turns per block')

################################################################################
# Define a pure reader of the currently supplied position
################################################################################

########################################
# Keep physical actuator evolution outside source evaluation
########################################
# A real external owner would set applied_y_m from its physical actuator state,
# including limits and settling. Here the three supplied values are already
# applied positions, not targets passed through an unmodelled response.
class AppliedPosition:
    frame = 'body_local'
    unit = 'm'

    def __init__(self, applied_y_m):
        self.applied_y_m = applied_y_m
        self.evaluations = 0

    def evaluate(self, time_s, positions_m):
        # Counting reads is diagnostic only. A query does not move an actuator.
        self.evaluations += 1
        return np.full((len(time_s), len(positions_m)), self.applied_y_m)

################################################################################
# Build the line and its mounting hierarchy
################################################################################

########################################
# Set up operation timings separately from the physical inputs
########################################
timings = {}
started = perf_counter()

########################################
# Preserve an intentionally offset and rolled quadrupole
########################################
env = xt.Environment()
env.new('q', xt.Quadrupole, length = .5, k1 = .12,
        shift_x = DELIBERATE_X_M, rot_s_rad = DELIBERATE_ROLL_RAD)
reference = xt.Particles(p0c = P0C, mass0 = xt.ELECTRON_MASS_EV, q0 = 1)

########################################
# Use a simple damped map for beam transport and the reference clock
########################################
# This is a toy tracking ring, not a geometrical ring-layout example. The map
# length sets its period; only the quadrupole's small assembly is drawn below.
env.elements['arc'] = xt.LineSegmentMap(
    length          = float(reference.beta0[0]) * c * TURN_PERIOD_S - .5,
    qx              = .28,
    qy              = .31,
    betx            = 10.,
    bety            = 10.,
    damping_rate_x  = .05,
    damping_rate_px = .05,
    damping_rate_y  = .05,
    damping_rate_py = .05,
)
line = env.new_line(name = 'ring', components = ['q', 'arc'])
line.particle_ref = reference
line.build_tracker()
particles = line.build_particles()
timings['Line and tracker preparation'] = perf_counter() - started

########################################
# Mount a carriage between the installed girder and the magnet
########################################
started = perf_counter()
mech = env.mechanics
line.mechanics.adopt_reference()
girder = mech.new_girder('girder', anchor = xt.Frame.from_survey_angles(Y = -.6, Z = .25))
girder.installation.shift_y = INSTALLATION_Y_M
carriage = mech.new_body('carriage', parent = girder,
                        mount = xt.Frame.from_survey_angles(Y = .4), kind = 'mover')
magnet = mech.new_body('magnet', parent = carriage,
                      mount = xt.Frame.from_survey_angles(Y = .2),
                      elements = ['q'], line = line, kind = 'magnet')

########################################
# Classify the disturbance and the live applied-control input explicitly
########################################
mech.add_source('ground_y', xt.mechanics.Harmonic(GROUND_PEAK_M, GROUND_HZ),
                targets = 'girder', component = 'y')
provided = AppliedPosition(APPLIED_POSITIONS_M[0])
mech.add_source('carriage_y', provided, targets = 'carriage', component = 'y',
                role = 'control', live = True)
source_table = mech.get_source_table()
line.mechanics.enable_vibrations(backend = 'compiled')
timings['Mechanical hierarchy and inputs'] = perf_counter() - started

################################################################################
# Track two blocks with vibration and capture the applied inputs
################################################################################

########################################
# Bound the recording to the two blocks that will use the same source mode
########################################
# This stores only the selected live six-coordinate inputs on the host.
# The prescribed harmonic is reproducible; it is not a GPU pose-history cache.
recorder_on = line.mechanics.record_applied_sources(
    sources = ['carriage_y'], max_samples = 2 * BLOCK_TURNS,
)

########################################
# Track the first supplied position in one complete call
########################################
started = perf_counter()
line.track(particles, num_turns = BLOCK_TURNS)
timings['First tracking block'] = perf_counter() - started

########################################
# Change the supplied position and track the next complete block
########################################
provided.applied_y_m = APPLIED_POSITIONS_M[1]
started = perf_counter()
line.track(particles, num_turns = BLOCK_TURNS)
timings['Second tracking block'] = perf_counter() - started
history_on = recorder_on.snapshot()
recorder_on.stop()

################################################################################
# Remove vibration while keeping control active
################################################################################

########################################
# Retain the last applied carriage position without reading its provider
########################################
reads_before = provided.evaluations
line.mechanics.disable_vibrations()
assert provided.evaluations == reads_before
assert not line.mechanics.vibrations_enabled
assert line.mechanics.controls_enabled
assert line.mechanics.automatic_updates_enabled
held_after_removal_m = float(line['q'].shift_y)
np.testing.assert_allclose(held_after_removal_m,
                           INSTALLATION_Y_M + APPLIED_POSITIONS_M[1], atol = 1e-20)
vibration_removed_s = float(particles.time_s)

########################################
# Start a new recording for the controls-only mode
########################################
# Each recording pins its source mask and prescribed record. Separate segments
# make the mode change explicit instead of relabelling past samples afterward.
recorder_off = line.mechanics.record_applied_sources(
    sources = ['carriage_y'], max_samples = BLOCK_TURNS,
)
provided.applied_y_m = APPLIED_POSITIONS_M[2]
started = perf_counter()
line.track(particles, num_turns = BLOCK_TURNS)
timings['Controls-only tracking block'] = perf_counter() - started
history_off = recorder_off.snapshot()
recorder_off.stop()

################################################################################
# Reconstruct the recorded magnetic motion through a selected table row
################################################################################

########################################
# Compose histories from the inputs actually captured at each freeze
########################################
quad = line.mechanics.get_table().rows['q']
reads_before = provided.evaluations
quad_on = quad.get_time_series(applied_history = history_on)
quad_off = quad.get_time_series(applied_history = history_off)
assert provided.evaluations == reads_before
times = np.concatenate((quad_on.time_s, quad_off.time_s))
control_y = np.concatenate((history_on.values['carriage_y'][:, 0, 1],
                            history_off.values['carriage_y'][:, 0, 1]))
dynamic_y = np.concatenate((quad_on.dynamic_Y, quad_off.dynamic_Y))
ground_y = dynamic_y - control_y
expected_ground = np.concatenate((
    GROUND_PEAK_M * np.sin(2 * np.pi * GROUND_HZ * quad_on.time_s),
    np.zeros(len(quad_off)),
))
np.testing.assert_allclose(ground_y, expected_ground, atol = 1e-20)
assert not history_on.overflowed and not history_off.overflowed
assert int(particles.at_turn[0]) == 3 * BLOCK_TURNS
assert np.all(particles.state > 0)

########################################
# Show that a later provider change cannot rewrite the saved history
########################################
provided.applied_y_m = 10e-6
reads_before = provided.evaluations
replayed = quad.get_time_series(applied_history = history_on)
np.testing.assert_array_equal(replayed.total_matrix, quad_on.total_matrix)
assert provided.evaluations == reads_before

########################################
# Freeze the current maps, then clear mechanics and recover the design fields
########################################
line.mechanics.freeze_updates()
frozen_shift_y_m = float(quad.get_element().shift_y)
assert not line.mechanics.automatic_updates_enabled
mech.clear()
assert line['q'].shift_x == DELIBERATE_X_M
assert line['q'].rot_s_rad == DELIBERATE_ROLL_RAD
assert line['q'].shift_y == 0.

################################################################################
# Read the source roles and the applied states
################################################################################

########################################
# Inspect the two distinct input contracts
########################################
print('Supplied control position and independent vibration\n'
      '---------------------------------------------------')
source_table.show(cols=['name', 'role', 'capability', 'component', 'unit',
                        'n_targets'], maxwidth='full')
print('The carriage receives an externally supplied applied position.')
print('No controller or actuator settling model is included in this example.')
print('Three tracking calls advance one particle clock; inputs freeze once per turn.')

########################################
# Compare the actual block endpoints in physical units
########################################
print('\nRecorded magnetic motion relative to static installation\n'
      '--------------------------------------------------------')
print('Block                 Last freeze [ms]    Control [µm]    Ground [nm]    Quad change [µm]')
for label, index in [('First: vibration on', BLOCK_TURNS - 1),
                     ('Second: vibration on', 2 * BLOCK_TURNS - 1),
                     ('Third: vibration off', 3 * BLOCK_TURNS - 1)]:
    print(f'{label:<22} {times[index] * 1e3:17.3f}'
          f' {control_y[index] * 1e6:15.3f} {ground_y[index] * 1e9:14.3f}'
          f' {dynamic_y[index] * 1e6:13.3f}')
print(f'Vibration removal retained the applied control: {APPLIED_POSITIONS_M[1] * 1e6:+.3f} µm.')
print('Control updates continued in the third block; vibration did not.')
print(f'Provider changed afterward to {provided.applied_y_m * 1e6:+.3f} µm; saved histories stayed unchanged.')
print(f'Frozen effective shift_y before clearing [µm]: {frozen_shift_y_m * 1e6:+.3f}')
print('Clearing restored the original deliberate offset and roll.')
print('Inspect source_table, history_on/history_off and quad_on/quad_off after %run.')

########################################
# Report detailed costs only when requested
########################################
if PRINT_TIMINGS:
    print('\nSeparate operation costs [s]\n----------------------------')
    for label, value in timings.items():
        print(f'{label:<38} {value:.6f}')

################################################################################
# Plot ownership, applied motion and vibration removal
################################################################################

########################################
# Draw the mounting tree and identify who supplies each input
########################################
fig = plt.figure(figsize = (11, 7.4))
grid = fig.add_gridspec(2, 2, height_ratios = (1., .85))
tree = fig.add_subplot(grid[0, 0])
motion = fig.add_subplot(grid[0, 1])
residual = fig.add_subplot(grid[1, :])
tree.set(xlim = (0, 1), ylim = (0, 1))
tree.axis('off')
for bottom, colour, title, text in [
    (.70, '0.92', 'Girder', 'Static installation + prescribed ground'),
    (.39, '#e2f1e6', 'Carriage', 'Live applied position; role="control"'),
    (.08, '#e3edfa', 'Quadrupole', 'Original design offset and roll retained'),
]:
    tree.add_patch(FancyBboxPatch((.08, bottom), .84, .20,
        boxstyle = 'round,pad=.015', facecolor = colour, edgecolor = '0.4'))
    tree.text(.50, bottom + .13, title, ha = 'center', va = 'center', weight = 'bold')
    tree.text(.50, bottom + .06, text, ha = 'center', va = 'center', fontsize = 9)
for top, bottom in [(.70, .59), (.39, .28)]:
    tree.annotate('', xy = (.5, bottom), xytext = (.5, top),
                  arrowprops = dict(arrowstyle = '->', color = '0.4'))
tree.set_title('Mounting and input ownership')

########################################
# Keep the micrometre control and nanometre disturbance on separate scales
########################################
motion.step(times * 1e3, control_y * 1e6, where = 'post', color = 'tab:green',
            label = 'Recorded applied carriage position')
motion.plot(times * 1e3, dynamic_y * 1e6, color = 'tab:blue', lw = 1.,
            label = 'Recorded quadrupole motion')
motion.set(xlabel = 'Physical freeze time [ms]', ylabel = 'World Y change [µm]',
           title = 'Control remains after vibration removal')
motion.legend(fontsize = 8, loc = 'lower left')
residual.plot(times * 1e3, ground_y * 1e9, color = 'tab:blue',
              label = 'Quadrupole motion minus recorded control')
residual.plot(quad_on.time_s * 1e3, expected_ground[:len(quad_on)] * 1e9,
              '--', color = '0.35', label = 'Prescribed ground while enabled')
for ax in (motion, residual):
    ax.axvline(vibration_removed_s * 1e3, color = '0.4', ls = ':')
    ax.axvspan(vibration_removed_s * 1e3, times[-1] * 1e3, color = '0.93', zorder = -1)
    ax.grid(alpha = .2)
residual.text(vibration_removed_s * 1e3 + .5, .85 * GROUND_PEAK_M * 1e9,
              'Vibration removed; control still active', fontsize = 9)
residual.set(xlabel = 'Physical freeze time [ms]', ylabel = 'Residual ground motion [nm]',
             title = 'Replay uses captured control samples, never the current provider value')
residual.legend(fontsize = 9, loc = 'lower left')
fig.tight_layout()
plt.show()
