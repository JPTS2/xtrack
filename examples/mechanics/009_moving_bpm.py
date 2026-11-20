# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Compare nominal beam coordinates with the electrical readout of a moving BPM.
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
# Access the shared mechanics model
########################################
mech = env.mechanics

########################################
# Time geometry selection
########################################
started = perf_counter()

########################################
# Survey the nominal body attachment frames
########################################
nominal = line.survey(include_element_frames = True)

timings['geometry_selection_s'] = perf_counter() - started

########################################
# Time mechanics definition
########################################
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

########################################
# Time readout calibration
########################################
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
for kick in (-PROBE_RAD, PROBE_RAD):
    env[CORRECTOR] = kick
    probe = line.twiss4d()
    readings.append(float(line.mechanics.read_bpm(
        BPM, x = probe['x', BPM], y = probe['y', BPM],
        px = probe['px', BPM], py = probe['py', BPM],
    ).y))

########################################
# Calculate the response and restore the corrector
########################################
response = (readings[1] - readings[0]) / (2 * PROBE_RAD)
env[CORRECTOR] = 0.

timings['readout_calibration_s'] = perf_counter() - started

################################################################################
# Prepare phase-aware mechanical response records
################################################################################

########################################
# Start timing response preparation
########################################
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

########################################
# Record response preparation cost
########################################
timings['response_records_s'] = perf_counter() - started

################################################################################
# Compare the two correction targets
################################################################################

########################################
# Start timing the frozen-state study
########################################
started = perf_counter()

########################################
# Choose observation times and prepare result histories
########################################
state_times = np.linspace(0., DURATION_S, N_STATES, endpoint = False)
beam_y, pickup_y, actual_y, nominal_corrected, pickup_corrected = [], [], [], [], []

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
    for error, destination in [(beam_y[-1], nominal_corrected), (observed, pickup_corrected)]:
        env[CORRECTOR] = -error / response
        corrected = line.twiss4d()
        destination.append(float(line.mechanics.read_bpm(
            BPM, time_s = time_s, x = corrected['x', BPM], y = corrected['y', BPM],
            px = corrected['px', BPM], py = corrected['py', BPM],
        ).y) - reference_y)

########################################
# Record the frozen-state and correction cost
########################################
timings['frozen_states_and_correction_s'] = perf_counter() - started

################################################################################
# Compare the correction outcomes
################################################################################

########################################
# Report the signed response and residual RMS
########################################
print('Moving pickup and frozen orbit correction\n-----------------------------------------')
print(f'Measured signed corrector response [m/rad]: {response:.6f}')
print('Quantity                         RMS [nm]')
for label, values in [('Beam in nominal coordinates', beam_y), ('Pickup motion', pickup_y),
                       ('Actual pickup before correction', actual_y),
                       ('After nominal-coordinate correction', nominal_corrected),
                       ('After moving-pickup correction', pickup_corrected)]:
    print(f'{label:<38} {np.sqrt(np.mean(np.asarray(values)**2)) * 1e9:12.6f}')
print('Readout uses ray/plane intersection, including pickup orientation and beam slope.')
print('Each state is frozen Twiss. No causal controller or turn-by-turn feedback is claimed.')

################################################################################
# Inspect why the observation frame matters
################################################################################

########################################
# Prepare motion and correction panels
########################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6.5))

########################################
# Separate nominal beam motion from pickup motion
########################################
for label, values in [('Beam, nominal frame', beam_y), ('Pickup motion', pickup_y), ('Actual BPM', actual_y)]:
    axes[0].plot(state_times, np.asarray(values) * 1e9, '.-', label = label)

########################################
# Compare actual pickup residuals after each correction
########################################
for label, values in [('Nominal-coordinate correction', nominal_corrected), ('Moving-BPM correction', pickup_corrected)]:
    axes[1].plot(state_times, np.asarray(values) * 1e9, '.-', label = label)

########################################
# Label and arrange both comparisons
########################################
axes[0].set(ylabel = 'Vertical coordinate [nm]')
axes[1].set(xlabel = 'Frozen physical time [s]', ylabel = 'Actual BPM residual [nm]')
axes[0].legend(fontsize = 8)
axes[1].legend(fontsize = 8)
fig.tight_layout()

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the measured costs
########################################
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<36} {seconds:9.4f} s')

########################################
# Display the figures
########################################
plt.show()
