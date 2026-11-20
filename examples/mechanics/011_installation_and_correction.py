# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Draw static installation cases independently of intentional geometry and motion.
# Correct their frozen BPM orbits with the same measured response matrix.

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
# Deliberate reference geometry
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
DELIBERATE_QUAD = 'arc.s1.c1.qf.in'
DELIBERATE_X_M  = 1e-3                                               # Intentional off-axis magnetic reference

########################################
# Independent installation cases
########################################
INSTALLATION_SEEDS = (956, 957, 958, 959)

########################################
# Quadrupole installation scales
########################################
QUAD_SIGMA_M   = (5e-6, 5e-6, 0.)
QUAD_SIGMA_RAD = (0., 0., 1e-4)

########################################
# Pickup installation scales
########################################
BPM_SIGMA_M = (2e-6, 2e-6, 0.)

########################################
# Distribution truncation and common fixtures
########################################
TRUNCATE_SIGMA = 3.                   # Parent Gaussian sigma; truncated RMS is smaller
COMMON_GROUPS  = {}                   # Optional {'fixture': ['g.name1', 'g.name2']}

########################################
# Static response and correction controls
########################################
PROBE_RAD      = 1e-7
SVD_REL_CUTOFF = 1e-3
N_CORRECTIONS  = 2

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
# Define the intended magnetic reference
################################################################################

########################################
# Set the intentional off-axis quadrupole
########################################
env[DELIBERATE_QUAD].shift_x = DELIBERATE_X_M

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

################################################################################
# Assign mechanical bodies to the reference lattice
################################################################################

########################################
# Access the environment mechanics model
########################################
mech = env.mechanics

########################################
# Time geometry selection
########################################
started = perf_counter()

########################################
# Select the magnetic, pickup and steering devices
########################################
table = line.get_table()
quads = list(table.rows.match(element_type = 'Quadrupole').name)
bpms = list(table.rows['bpm[.].*'].name)
correctors = list(table.rows['cor[.].*'].name)
knobs = ['v_' + name.replace('.', '_') for name in correctors]

timings['geometry_selection_s'] = perf_counter() - started

########################################
# Time mechanics definition
########################################
started = perf_counter()

########################################
# Assign the quadrupoles to supports
########################################
girders = [line.mechanics.new_girder(f'g.{name}', elements = [name]) for name in quads]

########################################
# Give each pickup its own mechanical body
########################################
pickups = [mech.new_body(f'pickup.{name}', kind = 'pickup', elements = [name], line = line) for name in bpms]

timings['mechanics_definition_s'] = perf_counter() - started

################################################################################
# Define the electrical pickup readout
################################################################################

########################################
# Project Twiss coordinates into the surveyed pickup planes
########################################
def read_pickups(twiss):
    return np.array([float(line.mechanics.read_bpm(
        name, x = twiss['x', name], y = twiss['y', name],
        px = twiss['px', name], py = twiss['py', name],
    ).y) for name in bpms])

################################################################################
# Measure the static correction response
################################################################################

########################################
# Start timing calibration
########################################
started = perf_counter()

########################################
# Measure the adopted reference orbit
########################################
reference_twiss = line.twiss4d()
reference = read_pickups(reference_twiss)

########################################
# Scan both signs of every corrector probe
########################################
response = np.empty((len(bpms), len(knobs)))
for index, knob in enumerate(knobs):
    probe = []
    for kick in (-PROBE_RAD, PROBE_RAD):
        env[knob] = kick
        probe.append(read_pickups(line.twiss4d()))
    response[:, index] = (probe[1] - probe[0]) / (2 * PROBE_RAD)
    env[knob] = 0.

########################################
# Retain the supported SVD response modes
########################################
uu, singular, vh = np.linalg.svd(response, full_matrices = False)
keep = singular > SVD_REL_CUTOFF * singular[0]
inverse = (vh[keep].T / singular[keep]) @ uu[:, keep].T

########################################
# Record response calibration cost
########################################
timings['response_calibration_s'] = perf_counter() - started

################################################################################
# Generate and correct the installation cases
################################################################################

########################################
# Start timing the seed study
########################################
started = perf_counter()

########################################
# Prepare records for comparing the seeds
########################################
cases = []
for seed in INSTALLATION_SEEDS:

    ########################################
    # Reset the correctors before each installation case
    ########################################
    for knob in knobs:
        env[knob] = 0.

    ########################################
    # Draw the quadrupole installation case
    ########################################
    installed = mech.generate_installation_errors(
        seed = seed, bodies = [g.name for g in girders],
        translation_sigma_m = QUAD_SIGMA_M, rotation_sigma_rad = QUAD_SIGMA_RAD,
        truncate_sigma = TRUNCATE_SIGMA, groups = COMMON_GROUPS,
    )

    ########################################
    # Draw pickup placement errors with the same installation seed
    ########################################
    installed_pickups = mech.generate_installation_errors(
        seed = seed, bodies = [p.name for p in pickups],
        translation_sigma_m = BPM_SIGMA_M, truncate_sigma = TRUNCATE_SIGMA,
    )

    ########################################
    # Apply the two static error layers
    ########################################
    installed.apply(mech)
    installed_pickups.apply(mech)
    mech.apply()

    ########################################
    # Measure the installed orbit before correction
    ########################################
    before = read_pickups(line.twiss4d()) - reference
    kicks = np.zeros(len(knobs))
    after = before.copy()

    ########################################
    # Correct using the retained response modes
    ########################################
    for iteration in range(N_CORRECTIONS):
        kicks -= inverse @ after
        for knob, kick in zip(knobs, kicks):
            env[knob] = kick
        after = read_pickups(line.twiss4d()) - reference

    ########################################
    # Retain the immutable installation case and correction result
    ########################################
    cases.append(dict(seed = seed, installed = installed, before = before, after = after, kicks = kicks.copy()))

########################################
# Record the complete seed-study cost
########################################
timings['installation_cases_and_correction_s'] = perf_counter() - started

################################################################################
# Compare the static correction results
################################################################################

########################################
# Report reference optics, installation scatter and residuals
########################################
print('Static installation and orbit correction\n----------------------------------------')
print(f'Intentional quad shift_x [mm]: {DELIBERATE_X_M * 1e3:.6f}')
print(f'BPMs / correctors / retained modes: {len(bpms)} / {len(knobs)} / {keep.sum()}')
print(f'Bare demonstration tunes: {reference_twiss.qx:.6f} / {reference_twiss.qy:.6f}')
print('The near-integer vertical tune and insertion beta make this demo sensitive to installation errors.')
print('Seed     Quad y RMS [µm]    BPM error before [µm]    After [µm]    Kick RMS [µrad]')
for case in cases:
    rms = lambda values: np.sqrt(np.mean(np.asarray(values)**2))
    print(f'{case["seed"]:4d} {rms(case["installed"].values[:, 1]) * 1e6:19.5f}'
          f' {rms(case["before"]) * 1e6:24.5f} {rms(case["after"]) * 1e6:13.5f}'
          f' {rms(case["kicks"]) * 1e6:19.5f}')
print('Installation records remain immutable. Corrector trims do not redefine the installation.')
print('No dynamic source seed enters these static installation draws.')

################################################################################
# Inspect seed variation and correction limits
################################################################################

########################################
# Prepare seed, spatial and response-mode panels
########################################
fig, axes = plt.subplots(1, 3, figsize = (12, 3.8))

########################################
# Compare orbit RMS across installation seeds
########################################
before_rms = [np.sqrt(np.mean(case['before']**2)) * 1e6 for case in cases]
after_rms = [np.sqrt(np.mean(case['after']**2)) * 1e6 for case in cases]
axes[0].plot(INSTALLATION_SEEDS, before_rms, 'o-', label = 'Before')
axes[0].plot(INSTALLATION_SEEDS, after_rms, 'o-', label = 'After')
axes[0].set(xlabel = 'Installation seed', ylabel = 'Actual BPM RMS [µm]')
axes[0].legend()

########################################
# Inspect the final seed around the ring
########################################
last = cases[-1]
axes[1].plot(table['s', bpms], last['before'] * 1e6, '.-', label = 'Before')
axes[1].plot(table['s', bpms], last['after'] * 1e6, '.-', label = 'After')
axes[1].set(xlabel = 'BPM s [m]', ylabel = 'BPM error [µm]', title = f'Installation seed {last["seed"]}')
axes[1].legend()

########################################
# Show which response modes the correction retained
########################################
axes[2].semilogy(np.arange(1, len(singular) + 1), singular / singular[0], 'o-')
axes[2].axhline(SVD_REL_CUTOFF, ls = '--', color = '0.4')
axes[2].set(xlabel = 'Response mode', ylabel = 'Relative singular value')
fig.tight_layout()

################################################################################
# Remove mechanics without redefining the magnetic reference
################################################################################

########################################
# Clear the installation and mechanical definitions
########################################
mech.clear()

########################################
# Verify that the intentional quadrupole offset remains
########################################
np.testing.assert_equal(env[DELIBERATE_QUAD].shift_x, DELIBERATE_X_M)
print(f'Deliberate quad shift after clearing mechanics [mm]: {env[DELIBERATE_QUAD].shift_x * 1e3:.6f}')

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the measured costs
########################################
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<40} {seconds:9.4f} s')

########################################
# Display the figures
########################################
plt.show()
