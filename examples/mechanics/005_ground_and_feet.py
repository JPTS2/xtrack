# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Sample one floor field at four feet per girder and reconstruct rigid motion.
# Show the part a rigid body can follow and the remaining contact mismatch.

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
# Line selection and local support feet
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
QUADRUPOLES  = ['arc.s1.c1.qf.in', 'arc.s1.c1.qd', 'arc.s1.c1.qf.out']
FEET_LOCAL_M = [(-.3, -.5, -1.), (.3, -.5, -1.),
                (-.3, -.5, 1.), (.3, -.5, 1.)]

########################################
# Floor excitation model
########################################
SOURCE       = 'wave'                 # 'wave' or 'psd'
AMPLITUDE_M  = 50e-9                  # Peak vector displacement [m]
FREQUENCY_HZ = 5.
SPEED_M_S    = 100.
DIRECTION    = (1., 0., 1.)           # World XYZ
POLARIZATION = (.3, 1., .2)

########################################
# Correlated PSD alternative
########################################
PSD_SEED             = 956
CORRELATION_LENGTH_M = 30.

########################################
# Record sampling
########################################
N_SAMPLES     = 4096
SAMPLE_STEP_S = 1 / 256

########################################
# Stored inspection time and one rigid-geometry display gain
########################################
INSPECT_TIME_S     = .125
GEOMETRY_GAIN      = 1e7             # Same gain for local translation and rotation
PRINT_TIMINGS      = False

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
line = nominal_env['demo'].copy()        # Independent elements for reference lowering
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
# Define supports and their floor contacts
################################################################################

########################################
# Access the shared mechanics model
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
# Time mechanics definition
########################################
started = perf_counter()

########################################
# Store four local feet on each rigid support
########################################
girders = [line.mechanics.new_girder(f'g.{name}', elements = [name], feet_local_m = FEET_LOCAL_M)
           for name in QUADRUPOLES]
timings['mechanics_definition_s'] = perf_counter() - started

########################################
# Time the receiver geometry preparation separately
########################################
started = perf_counter()

########################################
# Collect the same contact geometry in world coordinates
########################################
frames = [g.nominal_frame for g in girders]
feet = np.array([g.feet_local_m for g in girders])
positions = np.concatenate([g.nominal_feet_m for g in girders])

timings['source_geometry_s'] = perf_counter() - started

################################################################################
# Sample the floor and reconstruct rigid motion
################################################################################

########################################
# Start timing sampling and fitting
########################################
started = perf_counter()

########################################
# Choose a prescribed wave or correlated floor PSD
########################################
if SOURCE == 'wave':
    source = xt.mechanics.PlaneWave(
        amplitude = AMPLITUDE_M, frequency_hz = FREQUENCY_HZ,
        direction = DIRECTION, polarization = POLARIZATION, speed_m_s = SPEED_M_S,
    )
elif SOURCE == 'psd':
    source = xt.mechanics.DisplacementPSD(
        frequency_hz = [0., 1., 2., 30., 40.],
        density = [0., 0., 1e-17, 1e-17, 0.], seed = PSD_SEED,
        correlation_length_m = CORRELATION_LENGTH_M,
    )
else:
    raise ValueError("SOURCE must be 'wave' or 'psd'")

########################################
# Sample one field jointly at every contact
########################################
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
floor = xt.mechanics.sample_field(source, time_s = times, positions_m = positions)

########################################
# Fit all six rigid coordinates and retain contact mismatch
########################################
motion, residual = floor.fit_rigid_bodies(
    frames = frames, feet_local_m = feet, target_names = [g.name for g in girders],
)

########################################
# Bind the fitted motion as an absolute support response
########################################
mech.add_source('floor_following', motion, targets = motion.target_names, reference = 'absolute')

########################################
# Record the floor sampling and fitting cost
########################################
timings['sample_and_fit_s'] = perf_counter() - started

################################################################################
# Inspect what the rigid fit can reproduce
################################################################################

########################################
# Read the first support's six-coordinate history
########################################
table = motion.get_table(girders[0].name)

################################################################################
# Compare individual floor contacts with the fitted rigid body
################################################################################

########################################
# Select one stored physical instant and reconstruct its fitted contact positions
########################################
index = int(np.argmin(np.abs(times - INSPECT_TIME_S)))
selected_time_s = float(times[index])
values = motion.values[index, 0]
local_rotation = Rotation.from_euler('YXZ', values[[4, 3, 5]] * [1., -1., 1.]).as_matrix()
local_translation = values[:3]
first_feet = feet[0]
floor_local = floor.values[index, :len(first_feet)] @ frames[0].matrix[:3, :3]
fitted_local = first_feet @ local_rotation.T + local_translation
fit_displacement = fitted_local - first_feet
contact_difference = floor_local - fit_displacement
np.testing.assert_allclose(np.sqrt(np.mean(np.sum(contact_difference**2, axis = 1))),
                           residual[index, 0], rtol = 0, atol = 3e-13)

########################################
# Apply the fitted motion and check the actual quadrupole frames
########################################
started = perf_counter()
mech.apply(time_s = selected_time_s)
actual = line.survey(include_element_frames = True)
requested = line.mechanics.get_element_poses(time_s = selected_time_s)
for name in QUADRUPOLES:
    np.testing.assert_allclose(actual.get_frame(name, which = 'elem_start').matrix,
                               requested[name].matrix, rtol = 0, atol = 1e-10)
timings['Apply and verify rigid element poses'] = perf_counter() - started

########################################
# Explain what the rigid fit follows and what remains incompatible
########################################
print('Four moving feet, one rigid girder\n---------------------------------')
print(f'Floor field: {SOURCE}; {len(girders)} supports with four contacts each.')
print('Contacts span 0.6 m across the support and 2 m along it; the body reference is 0.5 m above them.')
if SOURCE == 'wave':
    print(f'Wave wavelength: {SPEED_M_S / FREQUENCY_HZ:g} m; propagation XYZ {DIRECTION}; polarization XYZ {POLARIZATION}.')
print(f'First support, stored sample at t = {selected_time_s:g} s:')
print('Contact    Floor vertical [nm]    Rigid-fit vertical [nm]    Full mismatch norm [nm]')
for name, raw, fitted, difference in zip('ABCD', floor_local, fit_displacement, contact_difference):
    print(f'{name:<8} {raw[1] * 1e9:+20.4f} {fitted[1] * 1e9:+26.4f} {np.linalg.norm(difference) * 1e9:27.4f}')
print(f'Full three-dimensional contact mismatch RMS [nm]: {residual[index, 0] * 1e9:.6f}')
print('The fitted translation and rotation move the entire attached magnet rigidly.')
print('The residual is the contact motion that this rigid pose cannot reproduce.')
print('This is a kinematic fit to supplied displacements, without a structural/contact force law.')

################################################################################
# Draw the floor contacts and the rigid motion they produce
################################################################################

########################################
# Show where the four contacts sit relative to the magnetic-axis reference
########################################
fig, axes = plt.subplots(2, 2, figsize = (10, 7.5))
outline_order = [0, 1, 3, 2, 0]
axes[0, 0].plot(first_feet[outline_order, 2], first_feet[outline_order, 0], color = '0.5')
for name, point in zip('ABCD', first_feet):
    axes[0, 0].plot(point[2], point[0], 'o', color = 'tab:orange')
    axes[0, 0].annotate(name, (point[2], point[0]), xytext = (5, 5), textcoords = 'offset points')
axes[0, 0].plot(0., 0., 'kD', label = 'Body / magnetic-axis reference')
axes[0, 0].annotate('Beam direction', xy = (.7, 0.), xytext = (-.7, 0.),
                   arrowprops = {'arrowstyle': '->', 'color': 'tab:blue'}, fontsize = 9)
axes[0, 0].set(title = 'Contact footprint: local plan view', xlabel = 'Local s [m]', ylabel = 'Local x [m]',
               xlim = (-1.3, 1.3), ylim = (-.55, .55))
axes[0, 0].set_aspect('equal', adjustable = 'box')
axes[0, 0].legend(loc = 'upper center', fontsize = 8)

########################################
# Show the same whole rigid object carried by the fitted pose
########################################
enlarged_rotation = Rotation.from_rotvec(GEOMETRY_GAIN * Rotation.from_matrix(local_rotation).as_rotvec()).as_matrix()
enlarged_translation = GEOMETRY_GAIN * local_translation
shown_feet = first_feet @ enlarged_rotation.T + enlarged_translation
shown_floor = first_feet + GEOMETRY_GAIN * floor_local
length = float(line[QUADRUPOLES[0]].length)
magnet = np.array([[0., -.15, -length / 2], [0., -.15, length / 2],
                   [0., .15, length / 2], [0., .15, -length / 2], [0., -.15, -length / 2]])
shown_magnet = magnet @ enlarged_rotation.T + enlarged_translation
axes[0, 1].plot(magnet[:, 2], magnet[:, 1], '--', color = '0.65')
axes[0, 1].fill(shown_magnet[:, 2], shown_magnet[:, 1], alpha = .15, color = 'tab:blue')
axes[0, 1].plot(shown_magnet[:, 2], shown_magnet[:, 1], color = 'tab:blue', label = 'Carried rigid magnet')
axes[0, 1].plot(shown_feet[outline_order, 2], shown_feet[outline_order, 1], color = 'tab:blue', lw = 1)
# A rigid upper deck joins the post tops and carries the magnet bottom.
# This schematic uses the same full rigid map as the contacts and magnet.
upper_deck = first_feet.copy()
upper_deck[:, 1] = -.15
shown_deck = upper_deck @ enlarged_rotation.T + enlarged_translation
axes[0, 1].fill(shown_deck[outline_order, 2], shown_deck[outline_order, 1], color = 'tab:blue', alpha = .15)
axes[0, 1].plot(shown_deck[outline_order, 2], shown_deck[outline_order, 1], color = 'tab:blue', lw = 1.2)
for foot, fitted, raw in zip(first_feet, shown_feet, shown_floor):
    post = np.array([foot, [foot[0], -.15, foot[2]]]) @ enlarged_rotation.T + enlarged_translation
    axes[0, 1].plot(post[:, 2], post[:, 1], color = 'tab:blue', lw = .8)
    axes[0, 1].plot([fitted[2], raw[2]], [fitted[1], raw[1]], color = 'tab:red', lw = 1.5)
axes[0, 1].plot(shown_floor[:, 2], shown_floor[:, 1], 'o', color = 'tab:orange', label = 'Supplied floor contacts')
axes[0, 1].plot(shown_feet[:, 2], shown_feet[:, 1], 's', mfc = 'none', color = 'tab:blue', label = 'Contacts of fitted rigid body')
axes[0, 1].set(title = f'Local elevation; motion ×{GEOMETRY_GAIN:,.0f}', xlabel = 'Local s [m]', ylabel = 'Local y [m]')
axes[0, 1].set_aspect('equal', adjustable = 'box')
axes[0, 1].margins(.15)
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc = 'lower center', ncol = 3, fontsize = 8)

########################################
# Compare each floor height with the closest rigid-body contact height
########################################
xx = np.arange(4)
width = .36
axes[1, 0].bar(xx - width / 2, floor_local[:, 1] * 1e9, width, color = 'tab:orange', label = 'Supplied floor motion')
axes[1, 0].bar(xx + width / 2, fit_displacement[:, 1] * 1e9, width, color = 'tab:blue', label = 'Fitted rigid contact motion')
axes[1, 0].set(xticks = xx, xticklabels = list('ABCD'), title = 'Most of the floor motion is followed',
               xlabel = 'Contact', ylabel = 'Vertical displacement [nm]')
axes[1, 0].legend(fontsize = 8)

########################################
# Resolve the smaller residual on its own physical scale
########################################
for component, colour, offset in [(0, 'tab:blue', -.24), (1, 'tab:orange', 0.), (2, 'tab:green', .24)]:
    axes[1, 1].bar(xx + offset, contact_difference[:, component] * 1e9, .23,
                   color = colour, label = ['Local x', 'Local y', 'Local s'][component])
axes[1, 1].set(xticks = xx, xticklabels = list('ABCD'), title = 'Incompatible motion remains as a residual',
               xlabel = 'Contact', ylabel = 'Floor − rigid fit [nm]')
axes[1, 1].axhline(0., color = '0.5', lw = .7)
axes[1, 1].legend(fontsize = 8)
fig.suptitle(f'One sampled floor field → one rigid pose per support; first support at t = {selected_time_s:g} s', fontsize = 11)
fig.tight_layout(rect = (0., .055, 1., 1.), h_pad = 2.)

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
