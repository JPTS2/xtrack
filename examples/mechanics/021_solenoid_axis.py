# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# A longitudinal solenoid field and the multipoles inside it can have parallel
# but different mechanical axes. Bind the two physical owners once, rather
# than assigning an independent magnet to each integration segment.

################################################################################
# Required packages
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from time import perf_counter

################################################################################
# User parameters
################################################################################

########################################
# Three consecutive field segments
########################################
SEGMENT_LENGTH_M = .5
KS_PER_M         = .12
K1L_PER_M        = .08       # Normal quadrupole integral in the middle segment

########################################
# Deliberate, installed and dynamic positions
########################################
CHART_DESIGN_X_M     = -5e-6
SOLENOID_DESIGN_X0_M = 25e-6  # Measured relative to the multipole chart
CHART_INSTALL_X_M    = 10e-6
AXIS_INSTALL_X_M     = -8e-6
CHART_VIBRATION_M    = 20e-6
CHART_VIBRATION_HZ   = 6.
AXIS_VIBRATION_M     = 6e-6
AXIS_VIBRATION_HZ    = 27.
INSPECT_TIME_S       = .05

########################################
# Optional execution-cost output
########################################
PRINT_TIMINGS       = False

################################################################################
# Describe the existing field model
################################################################################

########################################
# Separate setup, pose application and transport timings
########################################
timings = {}

########################################
# One solenoid profile, with multipoles in its central interval
########################################
started = perf_counter()
env = xt.Environment()
names = ['sol.entrance', 'sol.overlap', 'sol.exit']
profiles = [(0., KS_PER_M), (KS_PER_M, KS_PER_M), (KS_PER_M, 0.)]
for name, profile in zip(names, profiles):
    env.new(name, xt.VariableSolenoid, length = SEGMENT_LENGTH_M,
            ks_profile = profile,
            knl = [0., K1L_PER_M if name == 'sol.overlap' else 0.],
            shift_x = CHART_DESIGN_X_M, x0 = SOLENOID_DESIGN_X0_M)
line = env.new_line(name = 'line', components = names)
line.set_particle_ref('positron', p0c = 45.6e9)
line.build_tracker()
timings['Field construction and tracker'] = perf_counter() - started

########################################
# Track through the design field
########################################
launch = line.build_particles(x = 2e-5, y = 1e-5)
particles = launch.copy()
started = perf_counter()
line.track(particles)
timings['Design one-pass tracking'] = perf_counter() - started
design_exit = np.array([particles.x[0], particles.px[0],
                        particles.y[0], particles.py[0]])

################################################################################
# Assign physical owners to the field
################################################################################

########################################
# Adopt the existing chart alignment and relative solenoid centres
########################################
mech = env.mechanics
started = perf_counter()
line.mechanics.adopt_reference()

# The chart carries the multipoles and the entire tracking coordinate system.
chart = line.mechanics.new_girder('multipole_chart', elements = names)

# One axis for all three segments, relative to the common chart.
# Existing x0/y0 are automatically adopted: no alignment is entered again.
axis = line.mechanics.new_solenoid_axis('solenoid_axis', elements = names)
chart.installation.shift_x = CHART_INSTALL_X_M
axis.installation.shift_x = AXIS_INSTALL_X_M
timings['Chart and axis binding'] = perf_counter() - started

########################################
# Add motion common to both fields, and motion relative to the chart
########################################
started = perf_counter()
mech.add_source('support_sway', xt.mechanics.Harmonic(
    amplitude = CHART_VIBRATION_M, frequency_hz = CHART_VIBRATION_HZ,
), targets = chart.name, component = 'x')
mech.add_source('axis_motion', xt.mechanics.Harmonic(
    amplitude = AXIS_VIBRATION_M, frequency_hz = AXIS_VIBRATION_HZ,
), targets = axis.name, component = 'x')
timings['Source definition'] = perf_counter() - started

# The relative axis accepts transverse x/y only. For a common finite rigid
# rotation or longitudinal shift, move the chart body. Independent tilt or
# longitudinal deformation of the field profile is not represented by x0/y0.

################################################################################
# Inspect the installed and moving field axes
################################################################################

########################################
# Apply installation only
########################################
started = perf_counter()
mech.apply(disabled_sources = tuple(mech.sources))
timings['Installed pose application'] = perf_counter() - started
installed_chart = line['sol.overlap'].shift_x
installed_x0 = line['sol.overlap'].x0
particles = launch.copy()
started = perf_counter()
line.track(particles)
timings['Installed one-pass tracking'] = perf_counter() - started
installed_exit = np.array([particles.x[0], particles.px[0],
                           particles.y[0], particles.py[0]])

########################################
# Freeze chart vibration with the relative axis held at its installed offset
########################################
started = perf_counter()
mech.apply(time_s = INSPECT_TIME_S, disabled_sources = 'axis_motion')
timings['Chart-only pose application'] = perf_counter() - started
chart_only_shift = line['sol.overlap'].shift_x
chart_only_x0 = line['sol.overlap'].x0
particles = launch.copy()
started = perf_counter()
line.track(particles)
timings['Chart-only one-pass tracking'] = perf_counter() - started
chart_only_exit = np.array([particles.x[0], particles.px[0],
                           particles.y[0], particles.py[0]])

########################################
# Freeze both fields at a chosen physical time
########################################
started = perf_counter()
mech.apply(time_s = INSPECT_TIME_S)
timings['Dynamic pose application'] = perf_counter() - started
moving_chart = line['sol.overlap'].shift_x
moving_x0 = line['sol.overlap'].x0
particles = launch.copy()
started = perf_counter()
line.track(particles)
timings['Frozen-vibration one-pass tracking'] = perf_counter() - started
moving_exit = np.array([particles.x[0], particles.px[0],
                        particles.y[0], particles.py[0]])

########################################
# The usual tables expose both physical owners
########################################
print('One shared magnetic chart and one relative solenoid axis\n'
      '-------------------------------------------------------')
print('The three tracking segments share the same two physical owners.')
print('The relative solenoid axis accepts transverse x/y translation only; common rigid rotations belong to the chart.')
print('\nPhysical body hierarchy\n-----------------------')
mech.get_table().cols['name kind parent'].show()

print('\nField owners for each tracking segment\n'
      '--------------------------------------')
line.mechanics.get_table().cols['name body solenoid_axis_body'].show()

print(f'\nAdopted and effective local axis offsets [m] at t = {INSPECT_TIME_S:.3f} s')
line.mechanics.get_solenoid_axis_table().show()
print('Multipole strengths and ks_profile are not changed by axis lowering.')

print('\nHorizontal field-centre positions in survey coordinates [µm]')
print(f'{"State":<24} {"Multipole chart":>16} {"Solenoid axis":>16}')
for label, chart_x, x0 in [
    ('Design', CHART_DESIGN_X_M, SOLENOID_DESIGN_X0_M),
    ('Installed', installed_chart, installed_x0),
    ('Chart vibration only', chart_only_shift, chart_only_x0),
    ('Chart and axis vibration', moving_chart, moving_x0),
]:
    print(f'{label:<24} {chart_x*1e6:16.3f} {(chart_x+x0)*1e6:16.3f}')

print('\nChanges in beam exit relative to design [µm, µrad]')
print(f'{"State":<24} {"x":>11} {"px":>11} {"y":>11} {"py":>11}')
for label, coordinates in [('Installed', installed_exit),
                           ('Chart vibration only', chart_only_exit),
                           ('Chart and axis vibration', moving_exit)]:
    dx, dpx, dy, dpy = (coordinates-design_exit)*1e6
    print(f'{label:<24} {dx:11.4f} {dpx:11.4f} {dy:11.4f} {dpy:11.4f}')
print('The last two cases share chart motion and installation; only the relative axis source changes.')
axis_only_effect = (moving_exit - chart_only_exit) * 1e6
print('Additional axis-motion effect [x µm, px µrad, y µm, py µrad]: '
      + ', '.join(f'{value:+.6f}' for value in axis_only_effect))

################################################################################
# Visualise the two fields and their shared mechanical motion
################################################################################

########################################
# Read complete histories without modifying the applied pose
########################################
times = np.linspace(0., .4, 241)
chart_history = line.mechanics.get_table().get_time_series('sol.overlap',
                                                          time_s = times)
axis_history = mech.get_table().get_time_series(axis.name, time_s = times)

########################################
# Field profile and frozen geometry
########################################
boundaries = np.arange(4)*SEGMENT_LENGTH_M
midpoint = .5*(boundaries[1]+boundaries[2])
figure = plt.figure(figsize = (11, 6.2), layout = 'constrained')
grid = figure.add_gridspec(2, 2, height_ratios = [1., 1.5])
ax_profile = figure.add_subplot(grid[0, :])
ax_positions = figure.add_subplot(grid[1, 0])
ax_history = figure.add_subplot(grid[1, 1])

chart_color = '#255c99'
axis_color = '#bd601b'
overlap_color = '#dbe8f4'

ax_profile.axvspan(boundaries[1], boundaries[2], color = overlap_color)
ax_profile.plot(boundaries, [0., KS_PER_M, KS_PER_M, 0.],
                color = axis_color, lw = 2.5, label = 'Longitudinal solenoid field')
ax_profile.text(midpoint, .3*KS_PER_M, 'Quadrupole field in this chart interval',
                ha = 'center', color = chart_color)
for boundary in boundaries[1:-1]:
    ax_profile.axvline(boundary, color = '.55', ls = ':')
ax_profile.set(xlabel = 'Reference longitudinal position s [m]',
               ylabel = r'$k_s$ [m$^{-1}$]', xlim = (0., boundaries[-1]),
               ylim = (0., KS_PER_M*1.24), title = 'One field profile, three tracking segments')
ax_profile.legend(loc = 'upper left', frameon = False)

ax_positions.axvspan(boundaries[1], boundaries[2], color = overlap_color)
ax_positions.axhline(CHART_DESIGN_X_M*1e6, color = chart_color, ls = ':', alpha = .8)
ax_positions.axhline((CHART_DESIGN_X_M+SOLENOID_DESIGN_X0_M)*1e6,
                     color = axis_color, ls = ':', alpha = .8)
ax_positions.plot(boundaries, np.full(4, moving_chart*1e6),
                  color = chart_color, lw = 2.5, label = 'Multipole chart')
ax_positions.plot(boundaries, np.full(4, (moving_chart+moving_x0)*1e6),
                  color = axis_color, lw = 2.5, label = 'Solenoid field axis')
ax_positions.annotate('', xy = (midpoint, (moving_chart+moving_x0)*1e6),
                      xytext = (midpoint, moving_chart*1e6),
                      arrowprops = dict(arrowstyle = '<->', color = '.25'))
ax_positions.text(midpoint+.08*SEGMENT_LENGTH_M, (moving_chart+.5*moving_x0)*1e6, r'$x_0$',
                  va = 'center', color = '.25')
ax_positions.set(xlim = (0., boundaries[-1]),
                 xlabel = 's [m]', ylabel = 'Horizontal position [µm]',
                 title = f'Frozen geometry at {INSPECT_TIME_S:.3f} s\nDotted lines: deliberate design')
ax_positions.legend(loc = 'center right', bbox_to_anchor = (1., .25),
                    frameon = False)

ax_history.plot(times, chart_history.total_X*1e6, color = chart_color,
                lw = 2., label = 'Multipole chart')
ax_history.plot(times, axis_history.total_X*1e6, color = axis_color,
                lw = 2., label = 'Solenoid field axis')
ax_history.axvline(INSPECT_TIME_S, ls = ':', color = '.3')
ax_history.set(xlabel = 'Physical time [s]', ylabel = 'Horizontal position [µm]',
               title = 'Common sway and additional relative axis motion')
ax_history.grid(alpha = .2)

################################################################################
# Display figures and optional execution costs
################################################################################

########################################
# Print diagnostic costs only when requested
########################################
if PRINT_TIMINGS:
    print('\nSeparate operation costs [ms]\n-----------------------------')
    for label, elapsed in timings.items():
        print(f'{label:<38} {elapsed*1e3:9.3f}')

########################################
# Display the physical field and axis comparisons
########################################
plt.show()
