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
# Reference, installation and observation
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
DELIBERATE_QUAD = 'arc.s1.c1.qf.in'
DELIBERATE_X_M  = 1e-3                  # Intentional off-axis magnetic reference
INSTALLATION_SEEDS = (956, 957, 958, 959)
QUAD_SIGMA_M  = (5e-6, 5e-6, 0.)
QUAD_SIGMA_RAD = (0., 0., 1e-4)
BPM_SIGMA_M    = (2e-6, 2e-6, 0.)
TRUNCATE_SIGMA = 3.                     # Parent Gaussian sigma; truncated RMS is smaller
COMMON_GROUPS  = {}                     # Optional {'fixture': ['g.name1', 'g.name2']}

########################################
# Static orbit correction
########################################
PROBE_RAD      = 1e-7
SVD_REL_CUTOFF = 1e-3
N_CORRECTIONS  = 2

################################################################################
# Adopt the intended magnetic geometry and attach supports and pickups
################################################################################
started = perf_counter()
repo_root = Path(__file__).resolve().parents[2]
nominal_env = xt.load(repo_root / LATTICE_FILE)
line = nominal_env['demo'].copy()
env = line.env
env['demo'] = line
env[DELIBERATE_QUAD].shift_x = DELIBERATE_X_M
line.mechanics.adopt_reference()
mech = env.mechanics
table = line.get_table()
quads = list(table.rows.match(element_type = 'Quadrupole').name)
bpms = list(table.rows['bpm[.].*'].name)
correctors = list(table.rows['cor[.].*'].name)
knobs = ['v_' + name.replace('.', '_') for name in correctors]
girders = [line.mechanics.new_girder(f'g.{name}', elements = [name]) for name in quads]
pickups = [mech.new_body(f'pickup.{name}', elements = [name], line = line) for name in bpms]
line.build_tracker()
timings = {'setup_and_tracker_s': perf_counter() - started}

########################################
# Read nominal Twiss coordinates in the surveyed electrical frames
########################################
def read_pickups(twiss):
    return np.array([float(line.mechanics.read_bpm(
        name, x = twiss['x', name], y = twiss['y', name],
        px = twiss['px', name], py = twiss['py', name],
    ).y) for name in bpms])

################################################################################
# Measure one signed response at the deliberate reference geometry
################################################################################
started = perf_counter()
reference_twiss = line.twiss4d()
reference = read_pickups(reference_twiss)
response = np.empty((len(bpms), len(knobs)))
for index, knob in enumerate(knobs):
    probe = []
    for kick in (-PROBE_RAD, PROBE_RAD):
        env[knob] = kick
        probe.append(read_pickups(line.twiss4d()))
    response[:, index] = (probe[1] - probe[0]) / (2 * PROBE_RAD)
    env[knob] = 0.
uu, singular, vh = np.linalg.svd(response, full_matrices = False)
keep = singular > SVD_REL_CUTOFF * singular[0]
inverse = (vh[keep].T / singular[keep]) @ uu[:, keep].T
timings['response_calibration_s'] = perf_counter() - started

################################################################################
# Draw installation cases and compare frozen correction results
################################################################################
started = perf_counter()
cases = []
for seed in INSTALLATION_SEEDS:
    for knob in knobs:
        env[knob] = 0.
    installed = mech.generate_installation_errors(
        seed = seed, bodies = [g.name for g in girders],
        translation_sigma_m = QUAD_SIGMA_M, rotation_sigma_rad = QUAD_SIGMA_RAD,
        truncate_sigma = TRUNCATE_SIGMA, groups = COMMON_GROUPS,
    )
    installed_pickups = mech.generate_installation_errors(
        seed = seed, bodies = [p.name for p in pickups],
        translation_sigma_m = BPM_SIGMA_M, truncate_sigma = TRUNCATE_SIGMA,
    )
    installed.apply(mech)
    installed_pickups.apply(mech)
    mech.apply()
    before = read_pickups(line.twiss4d()) - reference
    kicks = np.zeros(len(knobs))
    after = before.copy()
    for iteration in range(N_CORRECTIONS):
        kicks -= inverse @ after
        for knob, kick in zip(knobs, kicks):
            env[knob] = kick
        after = read_pickups(line.twiss4d()) - reference
    cases.append(dict(seed = seed, installed = installed, before = before, after = after, kicks = kicks.copy()))
timings['installation_cases_and_correction_s'] = perf_counter() - started
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
# Inspect seed-to-seed correction and retained response modes
################################################################################
fig, axes = plt.subplots(1, 3, figsize = (12, 3.8))
before_rms = [np.sqrt(np.mean(case['before']**2)) * 1e6 for case in cases]
after_rms = [np.sqrt(np.mean(case['after']**2)) * 1e6 for case in cases]
axes[0].plot(INSTALLATION_SEEDS, before_rms, 'o-', label = 'Before')
axes[0].plot(INSTALLATION_SEEDS, after_rms, 'o-', label = 'After')
axes[0].set(xlabel = 'Installation seed', ylabel = 'Actual BPM RMS [µm]')
axes[0].legend()
last = cases[-1]
axes[1].plot(table['s', bpms], last['before'] * 1e6, '.-', label = 'Before')
axes[1].plot(table['s', bpms], last['after'] * 1e6, '.-', label = 'After')
axes[1].set(xlabel = 'BPM s [m]', ylabel = 'BPM error [µm]', title = f'Installation seed {last["seed"]}')
axes[1].legend()
axes[2].semilogy(np.arange(1, len(singular) + 1), singular / singular[0], 'o-')
axes[2].axhline(SVD_REL_CUTOFF, ls = '--', color = '0.4')
axes[2].set(xlabel = 'Response mode', ylabel = 'Relative singular value')
fig.tight_layout()
mech.clear()
np.testing.assert_equal(env[DELIBERATE_QUAD].shift_x, DELIBERATE_X_M)
print(f'Deliberate quad shift after clearing mechanics [mm]: {env[DELIBERATE_QUAD].shift_x * 1e3:.6f}')
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<40} {seconds:9.4f} s')
plt.show()
