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
# Magnet, pickup and mechanical response
########################################
LATTICE_FILE  = 'test_data/example_lattices/demo_ring/lattice.py'
QUADRUPOLE   = 'arc.s1.c1.qf.in'
BPM          = 'bpm.s1.f'
CORRECTOR    = 'v_cor_s1'
AMPLITUDE_M  = 50e-9
FREQUENCY_HZ = 5.
GIRDER_HZ    = 12.
MAGNET_HZ    = 35.
BPM_HZ       = 8.
DAMPING      = .2
PICKUP_ROLL_RAD = .03                  # Deliberate electrical-plane roll
PROBE_RAD    = 1e-7

########################################
# Offline motion and frozen observation times
########################################
N_SAMPLES     = 4096
SAMPLE_STEP_S = 1 / 256
N_STATES      = 24
DURATION_S    = .4

################################################################################
# Mount the pickup on the magnet and calibrate its signed response
################################################################################
started = perf_counter()
repo_root = Path(__file__).resolve().parents[2]
nominal_env = xt.load(repo_root / LATTICE_FILE)
line = nominal_env['demo'].copy()
env = line.env
env['demo'] = line
env[BPM].rot_s_rad_no_frame = PICKUP_ROLL_RAD
line.mechanics.adopt_reference()
mech = env.mechanics
nominal = line.survey(include_element_frames = True)
girder = mech.new_girder('girder', anchor = nominal.get_frame(QUADRUPOLE, which = 'elem_start'))
magnet = mech.new_body('magnet', parent = girder, mount = xt.Frame(),
                       elements = [QUADRUPOLE], line = line)
pickup = mech.new_body('pickup', parent = magnet,
                       anchor = nominal.get_frame(BPM, which = 'ref_start'), elements = [BPM], line = line)
line.build_tracker()

########################################
# Read the beam in the actual pickup plane
########################################
reference_twiss = line.twiss4d()
reference_y = float(line.mechanics.read_bpm(
    BPM, x = reference_twiss['x', BPM], y = reference_twiss['y', BPM],
    px = reference_twiss['px', BPM], py = reference_twiss['py', BPM],
).y)
readings = []
for kick in (-PROBE_RAD, PROBE_RAD):
    env[CORRECTOR] = kick
    probe = line.twiss4d()
    readings.append(float(line.mechanics.read_bpm(
        BPM, x = probe['x', BPM], y = probe['y', BPM],
        px = probe['px', BPM], py = probe['py', BPM],
    ).y))
response = (readings[1] - readings[0]) / (2 * PROBE_RAD)
env[CORRECTOR] = 0.
timings = {'setup_and_calibration_s': perf_counter() - started}

################################################################################
# Prescribe absolute body responses with complex phase
################################################################################
started = perf_counter()
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
values = np.zeros((N_SAMPLES, 1, 6))
values[:, 0, 1] = AMPLITUDE_M * np.sin(2 * np.pi * FREQUENCY_HZ * times)
floor = xt.mechanics.RigidMotion(times, values, ('girder',))
# These are supplied conditional responses. The upstream girder response is
# assumed to include loading; this lesson does not derive a structural model.
girder_motion = floor.through(xt.mechanics.BaseExcitedOscillator(GIRDER_HZ, DAMPING))
magnet_motion = girder_motion.through(
    xt.mechanics.BaseExcitedOscillator(MAGNET_HZ, DAMPING), target_names = ('magnet',),
)
pickup_motion = magnet_motion.through(
    xt.mechanics.BaseExcitedOscillator(BPM_HZ, DAMPING), target_names = ('pickup',),
)
for name, record in [('girder_motion', girder_motion), ('magnet_motion', magnet_motion), ('pickup_motion', pickup_motion)]:
    mech.add_source(name, record, targets = record.target_names, reference = 'absolute')
timings['response_records_s'] = perf_counter() - started

################################################################################
# Compare correction from nominal coordinates and actual BPM readout
################################################################################
started = perf_counter()
state_times = np.linspace(0., DURATION_S, N_STATES, endpoint = False)
beam_y, pickup_y, actual_y, nominal_corrected, pickup_corrected = [], [], [], [], []
for time_s in state_times:
    env[CORRECTOR] = 0.
    mech.apply(time_s = time_s)
    twiss = line.twiss4d()
    observed = float(line.mechanics.read_bpm(
        BPM, time_s = time_s, x = twiss['x', BPM], y = twiss['y', BPM],
        px = twiss['px', BPM], py = twiss['py', BPM],
    ).y) - reference_y
    beam_y.append(float(twiss['y', BPM] - reference_twiss['y', BPM]))
    pickup_y.append(float(mech.survey(time_s = time_s)['dynamic_Y', 'pickup']))
    actual_y.append(observed)
    for error, destination in [(beam_y[-1], nominal_corrected), (observed, pickup_corrected)]:
        env[CORRECTOR] = -error / response
        corrected = line.twiss4d()
        destination.append(float(line.mechanics.read_bpm(
            BPM, time_s = time_s, x = corrected['x', BPM], y = corrected['y', BPM],
            px = corrected['px', BPM], py = corrected['py', BPM],
        ).y) - reference_y)
timings['frozen_states_and_correction_s'] = perf_counter() - started
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
# Show why the measurement frame changes the correction target
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6.5))
for label, values in [('Beam, nominal frame', beam_y), ('Pickup motion', pickup_y), ('Actual BPM', actual_y)]:
    axes[0].plot(state_times, np.asarray(values) * 1e9, '.-', label = label)
for label, values in [('Nominal-coordinate correction', nominal_corrected), ('Moving-BPM correction', pickup_corrected)]:
    axes[1].plot(state_times, np.asarray(values) * 1e9, '.-', label = label)
axes[0].set(ylabel = 'Vertical coordinate [nm]')
axes[1].set(xlabel = 'Frozen physical time [s]', ylabel = 'Actual BPM residual [nm]')
axes[0].legend(fontsize = 8)
axes[1].legend(fontsize = 8)
fig.tight_layout()
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<36} {seconds:9.4f} s')
plt.show()
