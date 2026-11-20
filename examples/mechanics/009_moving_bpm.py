# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Does centring a moving BPM also centre the beam in the design coordinates?
# Correct frozen states using each readout; this is a quasi-static study, not feedback tracking.

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
# Line, pickup and corrector selection
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
QUADRUPOLE   = 'arc.s1.c1.qf.in'
BPM          = 'bpm.s1.f'
CORRECTOR    = 'v_cor_s1'

########################################
# Floor excitation
########################################
AMPLITUDE_M  = 50e-9
FREQUENCY_HZ = 5.

########################################
# Supplied conditional mechanical responses
########################################
GIRDER_HZ = 12.
MAGNET_HZ = 35.
BPM_HZ    = 8.
DAMPING   = .2

########################################
# Deliberate pickup orientation
########################################
PICKUP_ROLL_RAD = .03                 # Deliberate electrical-plane roll

########################################
# Signed corrector calibration probe
########################################
PROBE_RAD = 1e-7

########################################
# Motion record sampling
########################################
N_SAMPLES     = 4096
SAMPLE_STEP_S = 1 / 256

########################################
# Frozen observation times
########################################
N_STATES   = 24
DURATION_S = .4

########################################
# Optional performance diagnostics
########################################
PRINT_TIMINGS = False

################################################################################
# Set up the ordinary Xtrack line
################################################################################

########################################
# Start performance monitoring
########################################
timings = {}

started = perf_counter()

########################################
# Locate and load the demonstration ring
########################################
repo_root = Path(__file__).resolve().parents[2]
nominal_env = xt.load(repo_root / LATTICE_FILE)
line = nominal_env['demo'].copy()
env = line.env
env['demo'] = line

################################################################################
# Define the intended pickup and mounting geometry
################################################################################

########################################
# Set the deliberate electrical-plane roll
########################################
env[BPM].rot_s_rad_no_frame = PICKUP_ROLL_RAD

timings['line_load_and_copy_s'] = perf_counter() - started

########################################
# Prepare and time the tracker before mechanics queries
########################################
started = perf_counter()
line.build_tracker()
timings['tracker_build_s'] = perf_counter() - started

started = perf_counter()

########################################
# Adopt existing offsets as the deliberate reference
########################################
line.mechanics.adopt_reference()

timings['reference_adoption_s'] = perf_counter() - started

########################################
# Access the shared mechanics model
########################################
mech = env.mechanics

started = perf_counter()

########################################
# Survey the nominal body attachment frames
########################################
nominal = line.survey(include_element_frames = True)

timings['geometry_selection_s'] = perf_counter() - started

started = perf_counter()

########################################
# Place the girder
########################################
girder = mech.new_girder('girder', anchor = nominal.get_frame(QUADRUPOLE, which = 'elem_start'))

########################################
# Mount the quadrupole on its support
########################################
magnet = mech.new_body('magnet', kind = 'magnet', parent = girder, mount = xt.Frame(),
                       elements = [QUADRUPOLE], line = line)

########################################
# Mount the electrical pickup on the magnet
########################################
pickup = mech.new_body('pickup', kind = 'pickup', parent = magnet,
                       anchor = nominal.get_frame(BPM, which = 'ref_start'), elements = [BPM], line = line)

timings['mechanics_definition_s'] = perf_counter() - started

################################################################################
# Calibrate the actual electrical readout
################################################################################

started = perf_counter()

########################################
# Measure the reference orbit in the pickup frame
########################################
reference_twiss = line.twiss4d()
reference_y = float(line.mechanics.read_bpm(
    BPM, x = reference_twiss['x', BPM], y = reference_twiss['y', BPM],
    px = reference_twiss['px', BPM], py = reference_twiss['py', BPM],
).y)

########################################
# Probe the signed corrector response in both directions
########################################
readings = []
nominal_readings = []
for kick in (-PROBE_RAD, PROBE_RAD):
    env[CORRECTOR] = kick
    probe = line.twiss4d()
    nominal_readings.append(float(probe['y', BPM]))
    readings.append(float(line.mechanics.read_bpm(
        BPM, x = probe['x', BPM], y = probe['y', BPM],
        px = probe['px', BPM], py = probe['py', BPM],
    ).y))

########################################
# Calculate the response and restore the corrector
########################################
response = (readings[1] - readings[0]) / (2 * PROBE_RAD)
nominal_response = (nominal_readings[1] - nominal_readings[0]) / (2 * PROBE_RAD)
env[CORRECTOR] = 0.

timings['readout_calibration_s'] = perf_counter() - started

################################################################################
# Prepare phase-aware mechanical response records
################################################################################

started = perf_counter()

########################################
# Sample the prescribed six-coordinate floor input
########################################
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
values = np.zeros((N_SAMPLES, 1, 6))
values[:, 0, 1] = AMPLITUDE_M * np.sin(2 * np.pi * FREQUENCY_HZ * times)
floor = xt.mechanics.RigidMotion(times, values, ('girder',))

########################################
# State the response-loading assumption before cascading
########################################
# These are supplied conditional responses. The upstream girder response is
# assumed to include loading; this lesson does not derive a structural model.

########################################
# Apply the support response
########################################
girder_motion = floor.through(xt.mechanics.BaseExcitedOscillator(GIRDER_HZ, DAMPING))

########################################
# Apply the magnet's conditional response
########################################
magnet_motion = girder_motion.through(
    xt.mechanics.BaseExcitedOscillator(MAGNET_HZ, DAMPING), target_names = ('magnet',),
)

########################################
# Apply the pickup's conditional response
########################################
pickup_motion = magnet_motion.through(
    xt.mechanics.BaseExcitedOscillator(BPM_HZ, DAMPING), target_names = ('pickup',),
)

########################################
# Bind each body's absolute motion
########################################
for name, record in [('girder_motion', girder_motion), ('magnet_motion', magnet_motion), ('pickup_motion', pickup_motion)]:
    mech.add_source(name, record, targets = record.target_names, reference = 'absolute')

timings['response_records_s'] = perf_counter() - started

################################################################################
# Compare the two correction targets
################################################################################

started = perf_counter()

########################################
# Choose observation times and prepare result histories
########################################
state_times = np.linspace(0., DURATION_S, N_STATES, endpoint = False)
beam_y, pickup_y, actual_y, nominal_corrected, pickup_corrected = [], [], [], [], []
nominal_beam_after, pickup_beam_after = [], []

########################################
# Repeat the frozen correction comparison
########################################
for time_s in state_times:

    ########################################
    # Remove the previous correction before each state
    ########################################
    env[CORRECTOR] = 0.

    ########################################
    # Apply the mechanical state and compute its closed orbit
    ########################################
    mech.apply(time_s = time_s)
    twiss = line.twiss4d()

    ########################################
    # Project the beam into the moving electrical plane
    ########################################
    observed = float(line.mechanics.read_bpm(
        BPM, time_s = time_s, x = twiss['x', BPM], y = twiss['y', BPM],
        px = twiss['px', BPM], py = twiss['py', BPM],
    ).y) - reference_y

    ########################################
    # Retain nominal beam, pickup and measured positions
    ########################################
    beam_y.append(float(twiss['y', BPM] - reference_twiss['y', BPM]))
    pickup_y.append(float(mech.survey(time_s = time_s)['dynamic_Y', 'pickup']))
    actual_y.append(observed)

    ########################################
    # Correct using each readout and remeasure the actual pickup
    ########################################
    for error, calibration, destination, beam_destination in [
        (beam_y[-1], nominal_response, nominal_corrected, nominal_beam_after),
        (observed, response, pickup_corrected, pickup_beam_after),
    ]:
        env[CORRECTOR] = -error / calibration
        corrected = line.twiss4d()
        beam_destination.append(float(corrected['y', BPM] - reference_twiss['y', BPM]))
        destination.append(float(line.mechanics.read_bpm(
            BPM, time_s = time_s, x = corrected['x', BPM], y = corrected['y', BPM],
            px = corrected['px', BPM], py = corrected['py', BPM],
        ).y) - reference_y)

timings['frozen_states_and_correction_s'] = perf_counter() - started

################################################################################
# Compare the correction outcomes
################################################################################

########################################
# Report the signed response and residual RMS
########################################
print('Moving pickup and frozen orbit correction\n-----------------------------------------')
print(f'Signed corrector -> design-coordinate response [m/rad]: {nominal_response:.6f}')
print(f'Signed corrector -> electrical BPM response [m/rad]: {response:.6f}')
print('Frozen states share the same floor realization and corrector calibration.')
print('Question: centre the beam on its design orbit, or centre it in the moving pickup?')
print('Quantity                                             RMS [nm]')
for label, values in [('Beam in nominal coordinates', beam_y), ('Pickup motion', pickup_y),
                       ('Actual pickup before correction', actual_y),
                       ('After nominal-coordinate correction', nominal_corrected),
                       ('After moving-pickup correction', pickup_corrected),
                       ('True beam after nominal-coordinate correction', nominal_beam_after),
                       ('True beam after moving-pickup correction', pickup_beam_after)]:
    print(f'{label:<50} {np.sqrt(np.mean(np.asarray(values)**2)) * 1e9:12.6f}')
print('Readout uses ray/plane intersection, including pickup orientation and beam slope.')
print('Each state is frozen Twiss. No causal controller or turn-by-turn feedback is claimed.')

################################################################################
# Draw the moving electrical zero and compare the two correction objectives
################################################################################

########################################
# Inspect the state with the largest pickup displacement
########################################
state_index = int(np.argmax(np.abs(pickup_y)))
chosen_time = state_times[state_index]
setup_fig, setup_axes = plt.subplots(2, 1, figsize = (9, 5.7), gridspec_kw = {'height_ratios': [1, 2]})
chain_ax, position_ax = setup_axes
chain_ax.set_axis_off()
box_left = (.015, .270, .525, .780)
box_width = .205
for left, label in zip(box_left,
                       ('Prescribed\nfloor', f'Loaded girder\n{GIRDER_HZ:g} Hz',
                        f'Quad / girder\nresponse\n{MAGNET_HZ:g} Hz',
                        f'Pickup / quad\nresponse\n{BPM_HZ:g} Hz')):
    chain_ax.add_patch(plt.Rectangle((left, .25), box_width, .60, transform = chain_ax.transAxes,
                                     fc = '0.95', ec = '0.4', lw = .8))
    chain_ax.text(left + box_width / 2, .55, label, ha = 'center', va = 'center',
                  transform = chain_ax.transAxes, fontsize = 9)
for left, next_left in zip(box_left[:-1], box_left[1:]):
    chain_ax.annotate('', xy = (next_left, .55), xytext = (left + box_width, .55),
                      xycoords = 'axes fraction',
                      arrowprops = {'arrowstyle': '->', 'color': '0.4', 'shrinkA': 0, 'shrinkB': 0})
chain_ax.set_title('Supplied response chain (schematic); pickup and beam have different origins')
positions = np.array([beam_y[state_index], nominal_beam_after[state_index], pickup_beam_after[state_index]]) * 1e9
pickup_position = float(pickup_y[state_index]) * 1e9
for index, value in enumerate(positions):
    position_ax.plot(index, value, 'o', color = 'tab:blue', ms = 7)
    position_ax.plot(index, pickup_position, 's', mfc = 'none', mec = 'tab:orange', ms = 8)
    position_ax.plot([index, index], [pickup_position, value], ':', color = '0.4')
position_ax.axhline(0., color = '0.5', ls = '--', label = 'Design beam coordinate')
position_ax.plot([], [], 'o', color = 'tab:blue', label = 'True beam position')
position_ax.plot([], [], 's', mfc = 'none', mec = 'tab:orange', label = 'Moving pickup centre')
position_ax.set(xticks = range(3), xticklabels = ['No correction', 'Correct design-coordinate error', 'Correct electrical BPM error'],
                ylabel = 'Vertical displacement from design [nm]', title = f'One frozen state at t = {chosen_time:.4g} s')
position_ax.legend(fontsize = 8, loc = 'best')
position_ax.margins(y = .25)
setup_fig.suptitle(f'Electrical pickup roll is {PICKUP_ROLL_RAD * 1e3:g} mrad; points below show nominal vertical coordinates', fontsize = 10)
setup_fig.tight_layout()

########################################
# Compare actual electrical residuals and true beam motion over the same states
########################################
fig, axes = plt.subplots(2, 2, sharex = True, figsize = (10, 6.5))
for label, values in [('Beam in design coordinates', beam_y), ('Pickup-centre displacement', pickup_y)]:
    axes[0, 0].plot(state_times, np.asarray(values) * 1e9, '.-', label = label)
axes[0, 0].set(title = 'Before correction: two independently moving objects', ylabel = 'Vertical displacement [nm]')
axes[0, 1].plot(state_times, np.asarray(actual_y) * 1e9, '.-', label = 'Actual electrical readout')
axes[0, 1].set(title = 'What the moving BPM measures', ylabel = 'Electrical BPM error [nm]')
for label, electrical, beam in [
    ('Correct design coordinates', nominal_corrected, nominal_beam_after),
    ('Correct electrical readout', pickup_corrected, pickup_beam_after),
]:
    axes[1, 0].plot(state_times, np.asarray(electrical) * 1e9, '.-', label = label)
    axes[1, 1].plot(state_times, np.asarray(beam) * 1e9, '.-', label = label)
axes[1, 0].set(title = 'Electrical residual after correction', ylabel = 'Electrical BPM error [nm]')
axes[1, 1].set(title = 'True beam position after correction', ylabel = 'Beam error in design coordinates [nm]')
for ax in axes.flat:
    ax.set_xlabel('Frozen physical time [s]')
    ax.grid(alpha = .2)
    ax.legend(fontsize = 8)
fig.suptitle('Centring a moving electrical zero and restoring the design orbit are different objectives', fontsize = 11)
fig.tight_layout()
env[CORRECTOR] = 0.
mech.restore()

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the measured costs
########################################
if PRINT_TIMINGS:
    print('\nReference CPU costs\n-------------------')
    for name, seconds in timings.items():
        print(f'{name:<36} {seconds:9.4f} s')

########################################
# Display the figures
########################################
plt.show()
