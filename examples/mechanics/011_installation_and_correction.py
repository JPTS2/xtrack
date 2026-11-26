# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Restore the design orbit after installation errors without erasing intentional quad offsets.
# Solve independent frozen closed orbits and apply offline corrector trims.
# Electrical pickup residuals and true design-orbit residuals are different quantities.

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
girders = list(line.mechanics.new_girders({f'g.{name}': [name] for name in quads}).values())

########################################
# Give each pickup its own mechanical body
########################################
pickups = list(line.mechanics.new_bodies(
    {f'pickup.{name}': [name] for name in bpms}, kind = 'pickup',
).values())

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
# Generate static installation cases and solve their offline correction
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
    # Compute the frozen installed closed orbit and project it into pickup frames
    ########################################
    before_twiss = line.twiss4d()
    before = read_pickups(before_twiss) - reference
    beam_before = np.asarray(before_twiss['y', bpms] - reference_twiss['y', bpms])
    kicks = np.zeros(len(knobs))
    after = before.copy()
    after_twiss = before_twiss

    ########################################
    # Solve static corrector trims using the retained response modes
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
print('Static installation errors and offline closed-orbit correction\n'
      '-------------------------------------------------------------')
print('Each seed defines one fixed geometry. Twiss supplies the orbit used by the electrical readout.')
print('Corrector probes and trims are static calculations; no monitor acquisition or online feedback is used.')
print(f'Intentional quad shift_x [mm]: {DELIBERATE_X_M * 1e3:.6f}')
print(f'BPMs / correctors / retained modes: {len(bpms)} / {len(knobs)} / {keep.sum()}')
print(f'Bare demonstration tunes: {reference_twiss.qx:.6f} / {reference_twiss.qy:.6f}')
print('The near-integer vertical tune and insertion beta make this demo sensitive to installation errors.')
print('One illustrative support per quad and one independent body per pickup are used.')
print('The correction targets projected electrical pickup errors relative to the adopted design readout.')
print('Corrector limits, alignment tolerances and beam survival are not established by these frozen solves.')
print('Seed     Quad y RMS [µm]    BPM before [µm]    BPM after [µm]    True beam after [µm]')
for case in cases:
    rms = lambda values: np.sqrt(np.mean(np.asarray(values)**2))
    print(f'{case["seed"]:4d} {rms(case["installed"].values[:, 1]) * 1e6:19.5f}'
          f' {rms(case["before"]) * 1e6:18.5f} {rms(case["after"]) * 1e6:17.5f}'
          f' {rms(case["beam_after"]) * 1e6:23.5f}')
print('Installation records remain immutable. Corrector trims do not redefine the installation.')
print('No dynamic source seed enters these static installation draws.')
print('Pickup offsets bias the electrical zero. Discarded response modes and the finite trim iterations also leave residuals.')
print('An electrical residual and a design-coordinate beam residual therefore need not coincide.')
print('Largest absolute corrector trim per seed [µrad]:',
      ', '.join(f'{case["seed"]}: {np.max(np.abs(case["kicks"])) * 1e6:.3f}' for case in cases))

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
    ('Offline corrector trims', 'Frozen Twiss -> electrical error -> SVD trim\nRecompute Twiss; retain installation'),
]):
    y = .82 - index * .31
    ax.text(.5, y, title + '\n' + detail, transform = ax.transAxes, ha = 'center', va = 'center', fontsize = 9,
            bbox = {'boxstyle': 'round,pad=.4', 'fc': '0.96', 'ec': '0.45'})
for upper_y, lower_y in ((.71, .63), (.40, .32)):
    ax.annotate('', xy = (.5, lower_y), xytext = (.5, upper_y), xycoords = 'axes fraction',
                arrowprops = {'arrowstyle': '->', 'color': '.4'})
last = cases[-1]
quad_s = table['s', quads]
quad_y = last['installed'].values[:, 1] * 1e6
pickup_y = last['installed_pickups'].values[:, 1] * 1e6
setup_axes[1].plot(quad_s, quad_y, '.', label = 'Quad support installation y')
setup_axes[1].plot(table['s', bpms], pickup_y, 's', mfc = 'none', label = 'Pickup installation y')
setup_axes[1].set(xlabel = 'Survey position s [m]', ylabel = 'Static installation error [µm]',
                   title = f'One actual installation realization: seed {last["seed"]}')
setup_axes[1].margins(y = .25)
setup_axes[1].legend(loc = 'upper center', ncol = 2, fontsize = 8)
setup_axes[1].grid(alpha = .2)
setup_fig.tight_layout()

########################################
# Compare electrical errors with true beam errors at the same BPM locations
########################################
fig, axes = plt.subplot_mosaic(
    [['seeds', 'residuals'], ['before', 'modes'], ['after', 'modes']],
    figsize = (11, 8), gridspec_kw = {'height_ratios': [1.1, .75, .75]}, layout = 'constrained',
)
before_rms = [np.sqrt(np.mean(case['before']**2)) * 1e6 for case in cases]
after_rms = [np.sqrt(np.mean(case['after']**2)) * 1e6 for case in cases]
true_after_rms = [np.sqrt(np.mean(case['beam_after']**2)) * 1e6 for case in cases]
axes['seeds'].semilogy(INSTALLATION_SEEDS, before_rms, 'o-', label = 'Electrical error before trim')
axes['seeds'].semilogy(INSTALLATION_SEEDS, after_rms, 'o-', label = 'Electrical residual after trim')
axes['seeds'].set(xticks = INSTALLATION_SEEDS, xlabel = 'Installation seed', ylabel = 'Electrical RMS [µm]',
                  title = 'Same offline solve; log scale resolves the correction')
axes['residuals'].plot(INSTALLATION_SEEDS, after_rms, 'o-', label = 'Electrical residual')
axes['residuals'].plot(INSTALLATION_SEEDS, true_after_rms, 'o-', label = 'True design-orbit residual')
axes['residuals'].set(xticks = INSTALLATION_SEEDS, xlabel = 'Installation seed', ylabel = 'Residual RMS [µm]',
                      title = 'Residuals after the same finite number of trims')
axes['before'].plot(table['s', bpms], last['beam_before'] * 1e6, '.-', color = 'tab:blue')
axes['before'].set(ylabel = 'True beam error [µm]', title = f'Seed {last["seed"]}: before offline correction')
axes['before'].tick_params(labelbottom = False)
axes['after'].sharex(axes['before'])
axes['after'].plot(table['s', bpms], last['beam_after'] * 1e6, '.-', label = 'True design-orbit error')
axes['after'].plot(table['s', bpms], last['after'] * 1e6, 's--', mfc = 'none', label = 'Electrical readout error')
axes['after'].set(xlabel = 'Pickup position s [m]', ylabel = 'Residual [µm]',
                  title = 'After correction, on a separate vertical scale')
axes['modes'].semilogy(np.arange(1, len(singular) + 1), singular / singular[0], 'o-', label = 'Calibrated response modes')
axes['modes'].axhline(SVD_REL_CUTOFF, ls = '--', color = '.4', label = 'Retained-mode threshold')
axes['modes'].set(xlabel = 'Response mode', ylabel = 'Relative singular value',
                  title = 'Weak response directions are omitted from the solve')
for name, ax in axes.items():
    if name != 'before':
        ax.legend(fontsize = 8)
    ax.grid(alpha = .2)
fig.suptitle('Static installation geometry is fixed while offline steering trims change the frozen orbit', fontsize = 11)

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
