# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Restore the design orbit after installation errors without erasing intentional quad offsets.
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

timings['response_calibration_s'] = perf_counter() - started

################################################################################
# Generate and correct the installation cases
################################################################################

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
    before_twiss = line.twiss4d()
    before = read_pickups(before_twiss) - reference
    beam_before = np.asarray(before_twiss['y', bpms] - reference_twiss['y', bpms])
    kicks = np.zeros(len(knobs))
    after = before.copy()
    after_twiss = before_twiss

    ########################################
    # Correct using the retained response modes
    ########################################
    for iteration in range(N_CORRECTIONS):
        kicks -= inverse @ after
        for knob, kick in zip(knobs, kicks):
            env[knob] = kick
        after_twiss = line.twiss4d()
        after = read_pickups(after_twiss) - reference
    beam_after = np.asarray(after_twiss['y', bpms] - reference_twiss['y', bpms])

    ########################################
    # Retain the immutable installation case and correction result
    ########################################
    cases.append(dict(seed = seed, installed = installed, installed_pickups = installed_pickups,
                      before = before, after = after, beam_before = beam_before, beam_after = beam_after, kicks = kicks.copy()))

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
print('The correction targets electrical BPM errors, relative to the adopted design orbit.')
print('Seed     Quad y RMS [µm]    BPM before [µm]    BPM after [µm]    True beam after [µm]')
for case in cases:
    rms = lambda values: np.sqrt(np.mean(np.asarray(values)**2))
    print(f'{case["seed"]:4d} {rms(case["installed"].values[:, 1]) * 1e6:19.5f}'
          f' {rms(case["before"]) * 1e6:18.5f} {rms(case["after"]) * 1e6:17.5f}'
          f' {rms(case["beam_after"]) * 1e6:23.5f}')
print('Installation records remain immutable. Corrector trims do not redefine the installation.')
print('No dynamic source seed enters these static installation draws.')
print('Displaced BPM electrical zeros limit how closely electrical correction restores the true design orbit.')

################################################################################
# See the installation errors and what the correction actually restores
################################################################################

########################################
# Draw the three independent placement layers
########################################
setup_fig, setup_axes = plt.subplots(1, 2, figsize = (10, 3.6))
ax = setup_axes[0]
ax.set_axis_off()
for index, (title, detail) in enumerate([
    ('Design placement', f'Intentional quad x = {DELIBERATE_X_M * 1e3:g} mm\nAdopted before errors'),
    ('Static installation errors', 'Seeded magnet and pickup offsets\nHeld fixed during correction'),
    ('Corrector trims', 'Steering changes the beam\nInstallation geometry is retained'),
]):
    y = .82 - index * .31
    ax.text(.5, y, title + '\n' + detail, transform = ax.transAxes, ha = 'center', va = 'center', fontsize = 9,
            bbox = {'boxstyle': 'round,pad=.4', 'fc': '0.96', 'ec': '0.45'})
last = cases[-1]
quad_s = table['s', quads]
quad_y = last['installed'].values[:, 1] * 1e6
pickup_y = last['installed_pickups'].values[:, 1] * 1e6
setup_axes[1].plot(quad_s, quad_y, '.', label = 'Quad support installation y')
setup_axes[1].plot(table['s', bpms], pickup_y, 's', mfc = 'none', label = 'Pickup installation y')
setup_axes[1].set(xlabel = 'Survey position s [m]', ylabel = 'Static installation error [µm]',
                   title = f'One actual installation realization: seed {last["seed"]}')
setup_axes[1].legend(fontsize = 8)
setup_axes[1].grid(alpha = .2)
setup_fig.tight_layout()

########################################
# Compare electrical errors with true beam errors at the same BPM locations
########################################
fig, axes = plt.subplots(2, 2, figsize = (10, 6.5))
before_rms = [np.sqrt(np.mean(case['before']**2)) * 1e6 for case in cases]
after_rms = [np.sqrt(np.mean(case['after']**2)) * 1e6 for case in cases]
true_after_rms = [np.sqrt(np.mean(case['beam_after']**2)) * 1e6 for case in cases]
axes[0, 0].plot(INSTALLATION_SEEDS, before_rms, 'o-', label = 'Electrical error before correction')
axes[0, 0].plot(INSTALLATION_SEEDS, after_rms, 'o-', label = 'Electrical error after correction')
axes[0, 0].set(xlabel = 'Installation seed', ylabel = 'Electrical BPM RMS [µm]', title = 'Same correction procedure for every static seed')
axes[0, 1].plot(INSTALLATION_SEEDS, after_rms, 'o-', label = 'Electrical residual')
axes[0, 1].plot(INSTALLATION_SEEDS, true_after_rms, 'o-', label = 'True design-coordinate beam residual')
axes[0, 1].set(xlabel = 'Installation seed', ylabel = 'Residual RMS [µm]', title = 'What remains after correction?')
axes[1, 0].plot(table['s', bpms], last['beam_before'] * 1e6, '.-', label = 'True beam before correction')
axes[1, 0].plot(table['s', bpms], last['beam_after'] * 1e6, '.-', label = 'True beam after correction')
axes[1, 0].set(xlabel = 'BPM s [m]', ylabel = 'Beam error from design [µm]', title = f'Actual beam around the ring: seed {last["seed"]}')
axes[1, 1].semilogy(np.arange(1, len(singular) + 1), singular / singular[0], 'o-', label = 'Measured response modes')
axes[1, 1].axhline(SVD_REL_CUTOFF, ls = '--', color = '0.4', label = 'Retained-mode threshold')
axes[1, 1].set(xlabel = 'Response mode', ylabel = 'Relative singular value', title = 'Correction uses the observable response directions')
for ax in axes.flat:
    ax.legend(fontsize = 8)
    ax.grid(alpha = .2)
fig.tight_layout()

################################################################################
# Remove mechanics without redefining the magnetic reference
################################################################################

########################################
# Reset study correctors, then clear the installation and mechanical definitions
########################################
for knob in knobs:
    env[knob] = 0.
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
if PRINT_TIMINGS:
    print('\nReference CPU costs\n-------------------')
    for name, seconds in timings.items():
        print(f'{name:<40} {seconds:9.4f} s')

########################################
# Display the figures
########################################
plt.show()
