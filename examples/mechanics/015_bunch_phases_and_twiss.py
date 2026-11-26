# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Run each selected bunch phase independently through the same moving quadrupole.
# Compare actual frozen Twiss with the independently tracked driven response.
# Bunches run separately; this does not implement packed multibunch mechanics.

################################################################################
# Required packages
################################################################################

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Choose the bundled model or an existing machine
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME    = 'demo'
QUAD_NAME    = None                   # None selects the first quadrupole
CONTEXT      = xo.ContextCpu()

# For the pinned FCC input used in the review, replace the parameters above:
# LATTICE_FILE = '../inputs/acc-models-fcc-ee-108.1.0/lattices/z/fccee_z_beam_1_positrons.json'
# LINE_NAME = 'fccee_p_ring'
# QUAD_NAME = 'qf2r.0'
# Set FREQUENCIES_HZ in the next block to (10., 1000.) for that FCC input.

########################################
# Choose source frequencies and amplitude
########################################
FREQUENCIES_HZ       = None           # None uses the fractions below
FREQUENCY_OVER_FREV  = (.003, .30)    # Cycles/turn; high case resolves the finite-frequency beam response
AMPLITUDE_M         = 100e-9          # Peak body-local horizontal displacement
DRIVE_PHASE_RAD     = np.pi / 4

########################################
# Choose filled RF buckets
########################################
FILLED_BUCKETS = None                 # Or explicit integer bucket IDs
FILL_FRACTIONS = (0., .08, .16, .60, .68, .76)
# Fractions select six centroids in two illustrative trains, with empty gaps.
# They are not an engineering fill. Every bunch samples the same source.

########################################
# Enable weak sextupole feed-down in the bundled model
########################################
DEMO_K2LF_M2 = .1                     # Integrated normal sextupole [m^-2]
DEMO_K2LD_M2 = -.1
# These explicit teaching settings make orbit-dependent frozen optics visible.
# They are applied only when loading the bundled demo ring.

########################################
# Choose tracking and the initial beam state
########################################
N_TURNS           = 256
INITIAL_CONDITION = 'driven'          # 'driven' or 'zero_motion_closed_orbit'
CALIBRATION_M     = 100e-9            # Central difference of the one-turn map
BETA_SAMPLE_STEP  = 8                 # Plot sampling only; Twiss uses all elements
PRINT_TIMINGS     = False

################################################################################
# Set up the line, reference optics and fill clock
################################################################################

########################################
# Start performance monitoring
########################################
timings = {}
started = perf_counter()

########################################
# Load an independent line
########################################
repo_root = Path(__file__).resolve().parents[2]
loaded = xt.load(repo_root / LATTICE_FILE)
line = loaded[LINE_NAME].copy()
env = line.env
if LATTICE_FILE == 'test_data/example_lattices/demo_ring/lattice.py':
    env['k2lf'] = DEMO_K2LF_M2
    env['k2ld'] = DEMO_K2LD_M2
timings['line_load_and_copy_s'] = perf_counter() - started

########################################
# Prepare the tracker and freeze longitudinal motion
########################################
started = perf_counter()
line.build_tracker(_context = CONTEXT)
line.freeze_longitudinal()
timings['tracker_build_s'] = perf_counter() - started
# Radiation, feedback and collective effects are absent in these input models.

########################################
# Compute the zero-vibration reference
########################################
started = perf_counter()
baseline = line.twiss(method = '4d')
closed_orbit = baseline.particle_on_co.copy()
one_turn = np.asarray(baseline.R_matrix[:4, :4])
coordinates = ('x', 'px', 'y', 'py')
reference_state = np.array([
    float(CONTEXT.nparray_from_context_array(getattr(closed_orbit, name))[0])
    for name in coordinates
])
timings['reference_twiss_s'] = perf_counter() - started

########################################
# Define reference revolution and RF bucket times
########################################
period_s = line.get_length() / (float(line.particle_ref.beta0[0]) * c)
frev_hz = 1 / period_s
table = line.get_table()
cavity = table.rows.match(element_type = 'Cavity').name[0]
rf_frequency_hz = float(line[cavity].frequency)
harmonic = round(rf_frequency_hz * period_s)
if FILLED_BUCKETS is None:
    buckets = np.rint(harmonic * np.asarray(FILL_FRACTIONS)).astype(int)
else:
    buckets = np.asarray(FILLED_BUCKETS)
if (buckets.ndim != 1 or len(buckets) < 2
        or not np.issubdtype(buckets.dtype, np.integer)
        or len(np.unique(buckets)) != len(buckets)
        or np.any(buckets < 0) or np.any(buckets >= harmonic)):
    raise ValueError('Select at least two unique integer RF buckets in [0, harmonic)')
arrival_offsets_s = buckets / rf_frequency_hz
frequencies_hz = (np.asarray(FREQUENCY_OVER_FREV) * frev_hz
                  if FREQUENCIES_HZ is None else np.asarray(FREQUENCIES_HZ))
if (frequencies_hz.shape != (2,) or not np.all(np.isfinite(frequencies_hz))
        or frequencies_hz[0] < 0 or frequencies_hz[1] <= frequencies_hz[0]):
    raise ValueError('Choose two increasing nonnegative source frequencies')
if INITIAL_CONDITION not in ('driven', 'zero_motion_closed_orbit'):
    raise ValueError('Choose driven or zero_motion_closed_orbit initial condition')

########################################
# Select the quadrupole and adopt deliberate reference offsets
########################################
quad = (table.rows.match(element_type = 'Quadrupole').name[0]
        if QUAD_NAME is None else QUAD_NAME)
started = perf_counter()
line.mechanics.adopt_reference()
timings['reference_adoption_s'] = perf_counter() - started

################################################################################
# Calibrate a displacement against the native one-turn map
################################################################################

########################################
# Attach the quadrupole to a named support
########################################
mech = env.mechanics
body = line.mechanics.new_girder('quad_support', elements = [quad])
started = perf_counter()

########################################
# Track two fixed displacements about the zero-motion closed orbit
########################################
outputs = []
for displacement in (-CALIBRATION_M, CALIBRATION_M):
    body.shift_x = displacement
    line.mechanics.apply()
    particle = closed_orbit.copy()
    line.track(particle, num_turns = 1)
    outputs.append(np.array([
        float(CONTEXT.nparray_from_context_array(getattr(particle, name))[0])
        for name in coordinates
    ]))
drive = (outputs[1] - outputs[0]) / (2 * CALIBRATION_M)
static_response = np.linalg.solve(np.eye(4) - one_turn, drive)
timings['one_turn_drive_calibration_s'] = perf_counter() - started

########################################
# Restore the reference before comparing motion scenarios
########################################
mech.clear()
# Clear restores captured deliberate element fields. Each scenario below builds
# the same support and one source; no source is reseeded for individual bunches.

################################################################################
# Compare frozen optics at each bunch phase with independently driven tracking
################################################################################

results = []
frozen_rows = []
sampled = np.arange(0, len(baseline.s), BETA_SAMPLE_STEP)
turns = np.arange(N_TURNS)

for scenario_index, frequency_hz in enumerate(frequencies_hz):

    ########################################
    # Build one source shared by all bunches
    ########################################
    body = line.mechanics.new_girder('quad_support', elements = [quad])
    mech.add_source('quad_motion', xt.mechanics.Harmonic(
        amplitude = AMPLITUDE_M, frequency_hz = float(frequency_hz),
        phase_rad = DRIVE_PHASE_RAD,
    ), targets = 'quad_support', component = 'x')

    ########################################
    # Calculate the finite-frequency linear particular solution
    ########################################
    mu = 2 * np.pi * frequency_hz * period_s
    particular = np.linalg.solve(
        np.exp(1j * mu) * np.eye(4) - one_turn, drive * AMPLITUDE_M)
    gain = abs(particular[0]) / abs(AMPLITUDE_M * static_response[0])

    ########################################
    # Recompute actual frozen Twiss at each bunch arrival phase
    ########################################
    started = perf_counter()
    frozen_orbits, beta_beating = [], []
    for bucket, arrival_s in zip(buckets, arrival_offsets_s):
        line.mechanics.apply(time_s = float(arrival_s))
        optics = line.twiss(method = '4d')
        frozen_orbits.append(float(optics.x[0] - baseline.x[0]))
        full_beta_beating = optics.betx / baseline.betx - 1
        beta_beating.append(full_beta_beating[sampled])
        frozen_rows.append(dict(name = f'{frequency_hz:g}Hz.bucket{bucket}',
            bucket = int(bucket), frequency_hz = float(frequency_hz),
            arrival_offset_s = float(arrival_s),
            phase_rad = float(DRIVE_PHASE_RAD + 2 * np.pi * frequency_hz * arrival_s),
            x_co_m = float(optics.x[0]), y_co_m = float(optics.y[0]),
            betx_m = float(optics.betx[0]), bety_m = float(optics.bety[0]),
            peak_relative_betx_change = float(np.max(np.abs(full_beta_beating))),
            qx = float(optics.qx), qy = float(optics.qy)))
    timings[f'frozen_twiss_{frequency_hz:g}_hz_s'] = perf_counter() - started

    ########################################
    # Track each bunch from an explicit initial condition
    ########################################
    started = perf_counter()
    bunches = []
    for bucket, arrival_s in zip(buckets, arrival_offsets_s):
        phase_b = DRIVE_PHASE_RAD + 2 * np.pi * frequency_hz * arrival_s
        steady = np.imag(np.exp(1j * (mu * turns + phase_b))[:, None]
                         * particular[None, :])
        initial = steady[0] if INITIAL_CONDITION == 'driven' else np.zeros(4)
        particle = closed_orbit.copy()
        for index, name in enumerate(coordinates):
            getattr(particle, name)[:] = reference_state[index] + initial[index]

        # Positive bucket delay would be negative bunch-centre zeta in a packed
        # representation. Here each bunch is a separate run with that delay.
        line.mechanics.enable_vibrations(time_origin_s = float(arrival_s))
        line.track(particle, num_turns = N_TURNS, turn_by_turn_monitor = True)
        record = line.record_last_track
        actual = np.column_stack([
            CONTEXT.nparray_from_context_array(getattr(record, name))[0]
            - reference_state[index]
            for index, name in enumerate(coordinates)
        ])
        line.mechanics.disable_vibrations()

        ########################################
        # Retain the free transient required by the chosen initial state
        ########################################
        free = np.empty_like(steady)
        free[0] = initial - steady[0]
        for turn in range(1, N_TURNS):
            free[turn] = one_turn @ free[turn - 1]
        prediction = steady + free
        frozen_sequence = (np.sin(mu * turns + phase_b) * AMPLITUDE_M
                           * static_response[0])
        error = (np.sqrt(np.mean((actual[:, 0] - prediction[:, 0])**2))
                 / max(np.sqrt(np.mean(actual[:, 0]**2)), 1e-30))
        assert np.all(CONTEXT.nparray_from_context_array(particle.state) > 0)
        assert error < .03, (frequency_hz, int(bucket), error)
        bunches.append(dict(bucket = int(bucket), actual = actual[:, 0],
            prediction = prediction[:, 0], frozen_sequence = frozen_sequence,
            relative_rms_error = error))
    timings[f'bunch_tracking_{frequency_hz:g}_hz_s'] = perf_counter() - started

    ########################################
    # Retain results and clear only before the next scenario
    ########################################
    results.append(dict(frequency_hz = float(frequency_hz), gain = gain,
        frozen_orbits = np.array(frozen_orbits), beta_beating = np.array(beta_beating),
        bunches = bunches))
    if scenario_index + 1 < len(frequencies_hz):
        mech.clear()

########################################
# Leave the final source and one frozen bunch available for inspection
########################################
representative_bunch = len(buckets) // 2
started = perf_counter()
line.mechanics.apply(time_s = float(arrival_offsets_s[representative_bunch]))
timings['final_frozen_application_s'] = perf_counter() - started
# After %run, mech/body remain live and vibrations are disabled. Ordinary Twiss
# and line.mechanics.get_table(time_s=...) can inspect another bunch's phase.

################################################################################
# Inspect source phase, frozen optics and driven orbit
################################################################################

########################################
# Print the clock, assumptions and numerical comparison
########################################
bunch_table = xt.Table({key: np.array([row[key] for row in frozen_rows])
                        for key in frozen_rows[0]})
high_frequency_bunches = bunch_table.rows[
    bunch_table.frequency_hz == frequencies_hz[-1]]
print('One moving quadrupole seen by a sparse bunch train')
print('-------------------------------------------------')
print(f'C = {line.get_length():.3f} m; T_rev = {period_s:.9g} s; RF harmonic = {harmonic}')
print(f'Quad: {quad}; observation: {table.name[0]}; initial state: {INITIAL_CONDITION}')
print('Buckets:', buckets.tolist())
print('Each occupied RF bucket is an independent tracking run with its own arrival-time offset.')
print('The bunch cases sample one common prescribed source; this is not a packed multibunch simulation.')
print('Empty bucket gaps advance the source phase; they do not restart it.')
print('Whole-line turn-held maps omit s/(beta0*c) passage delays and within-bunch arrival spread.')
print('Frozen Twiss is an optics snapshot, not a time-dependent equilibrium beam.')
print(f'The final source remains defined; the line is frozen for bucket {buckets[representative_bunch]}.')
print(f'Quadrupole displacement amplitude: {AMPLITUDE_M * 1e9:g} nm peak.')
if INITIAL_CONDITION == 'driven':
    print('Launches use the calibrated linear driven particular solution; no free betatron transient is imposed.')
else:
    print('Launches use the zero-motion closed orbit; the prediction retains the resulting free betatron transient.')
print('The low-frequency case tests how closely the driven motion follows a sequence of frozen closed orbits.')
print('High frequency: the fill samples different positions, and beam dynamics differ from a sequence of static orbits.')
print('\nSource and beam comparison')
print('Frequency [Hz]  Phase span [rad]  Frozen orbit spread [nm]  Particular/static gain  Relative map error')
for result in results:
    frequency_hz = result['frequency_hz']
    spread = 2 * np.pi * frequency_hz * np.ptp(arrival_offsets_s)
    error = max(bunch['relative_rms_error'] for bunch in result['bunches'])
    print(f'{frequency_hz:14.3f} {spread:17.5f} {np.ptp(result["frozen_orbits"]) * 1e9:25.3f}'
          f' {result["gain"]:19.5f} {error:22.3e}')
print('\nHigh-frequency bunches: actual magnet positions and frozen Twiss at their arrival phases')
print('Bucket  Arrival [µs]  Magnet x [nm]  Frozen orbit at origin [nm]  Peak ring beta change [ppm]')
for index, bucket in enumerate(buckets):
    row = high_frequency_bunches.rows[high_frequency_bunches.bucket == bucket]
    phase = float(row.phase_rad[0])
    print(f'{bucket:6d} {arrival_offsets_s[index] * 1e6:13.6f} {AMPLITUDE_M * np.sin(phase) * 1e9:14.3f}'
          f' {(float(row.x_co_m[0]) - baseline.x[0]) * 1e9:20.3f}'
          f' {float(row.peak_relative_betx_change[0]) * 1e6:27.3f}')
if PRINT_TIMINGS:
    print('\nOperation costs [s], including cold preparation in the first track:')
    for name, value in timings.items():
        print(f'{name:42s} {value:.4f}')

########################################
# Show the fill, sampled magnet positions and the resulting beam response
########################################
figure = plt.figure(figsize = (11, 8.5), layout = 'constrained')
grid = figure.add_gridspec(3, 2, height_ratios = [.45, 1., 1.])
fill_axis = figure.add_subplot(grid[0, :])
axes = np.array([[figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
                 [figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1])]])
# This strip is a physical arrival-time map, including empty gaps.
fill_axis.hlines(0., 0., period_s * 1e6, color = '.75', lw = 2)
for index, (bucket, arrival) in enumerate(zip(buckets, arrival_offsets_s)):
    fill_axis.plot(arrival * 1e6, 0., 'o', color = 'tab:blue', ms = 6)
    fill_axis.text(arrival * 1e6, .12, f'Bucket {bucket}', ha = 'center', fontsize = 8)
fill_axis.annotate('Next turn', xy = (period_s * 1e6, 0.), xytext = (period_s * 1e6, -.28),
                   ha = 'right', fontsize = 8, arrowprops = {'arrowstyle': '->', 'color': '.4'})
fill_axis.set(xlim = (-.03 * period_s * 1e6, 1.03 * period_s * 1e6), ylim = (-.4, .45),
              xlabel = 'Arrival time at the line origin within one revolution [µs]', yticks = [],
              title = 'Occupied RF phases used in independent runs; gaps retain their physical arrival times')
for edge in ('left', 'right', 'top'):
    fill_axis.spines[edge].set_visible(False)
arrival_grid = np.linspace(0., period_s, 400)
for result in results:
    frequency_hz = result['frequency_hz']
    label = f'{frequency_hz:g} Hz ({frequency_hz / frev_hz:.3g} f_rev)'
    curve, = axes[0, 0].plot(arrival_grid * 1e6,
        AMPLITUDE_M * np.sin(DRIVE_PHASE_RAD + 2 * np.pi * frequency_hz * arrival_grid) * 1e9, label = label)
    axes[0, 0].plot(arrival_offsets_s * 1e6,
        AMPLITUDE_M * np.sin(DRIVE_PHASE_RAD + 2 * np.pi * frequency_hz * arrival_offsets_s) * 1e9,
        'o', color = curve.get_color())
    axes[0, 1].plot(arrival_offsets_s * 1e6, result['frozen_orbits'] * 1e9, 'o-', label = label)
axes[0, 0].set(xlabel = 'Bunch arrival offset [µs]', ylabel = 'Quadrupole displacement [nm]',
               title = 'Filled-bucket markers sample the same moving magnet')
axes[0, 1].set(xlabel = 'Bunch arrival offset [µs]', ylabel = 'Frozen closed-orbit change [nm]',
               title = 'Actual frozen Twiss at each bunch arrival')
for axis, result in zip(axes[1], results):
    bunch = result['bunches'][len(buckets) // 2]
    axis.plot(turns, bunch['actual'] * 1e9, lw = 2, label = 'Tracked particle')
    axis.plot(turns, bunch['prediction'] * 1e9, 'o', ms = 3, mfc = 'none',
              markevery = max(1, N_TURNS // 32), label = 'Calibrated driven-map prediction')
    axis.plot(turns, bunch['frozen_sequence'] * 1e9, ':', lw = 1.5, color = '.35', label = 'Frozen-orbit sequence')
    axis.set(xlabel = 'Turn', ylabel = 'Observation x [nm]',
        title = f'{result["frequency_hz"]:g} Hz; particular/static gain {result["gain"]:.3f}')
    if result['frequency_hz'] / frev_hz > .1:
        axis.set_xlim(0, 40)
for axis in axes.flat:
    axis.margins(y = .25)
    axis.legend(fontsize = 7.5, loc = 'upper center')
    axis.grid(alpha = .2)
figure.suptitle('Independent bunch phases: frozen lattice snapshots versus the dynamically driven beam', fontsize = 12)

########################################
# Plot actual frozen beta beating for the higher source frequency
########################################
beta_figure, beta_axes = plt.subplots(1, 2, figsize = (12, 4.5),
                                      gridspec_kw = {'width_ratios': [1.7, 1.]}, layout = 'constrained')
spread_by_position = np.ptp(results[-1]['beta_beating'], axis = 0)
location = int(np.argmax(spread_by_position))
location_s = baseline.s[sampled[location]]
for bucket, beating, arrival in zip(buckets, results[-1]['beta_beating'], arrival_offsets_s):
    curve, = beta_axes[0].plot(baseline.s[sampled], beating * 1e6, label = f'Bucket {bucket}')
    beta_axes[1].plot(arrival * 1e6, beating[location] * 1e6, 'o', color = curve.get_color())
beta_axes[0].set(xlabel = 'Ring position s [m]', ylabel = 'Frozen Δβx / βx [ppm]',
                 title = 'Frozen snapshot optics at each selected bunch phase')
beta_axes[0].axvline(float(table['s', quad]), color = '.3', ls = ':', label = 'Moving quadrupole')
beta_axes[0].axvline(float(location_s), color = '.5', ls = '--', label = 'Largest plotted bunch-phase spread')
beta_axes[0].margins(y = .25)
beta_axes[0].legend(ncol = 2, fontsize = 7.5, loc = 'upper center')
beta_axes[1].set(xlabel = 'Bunch arrival offset [µs]', ylabel = 'Frozen Δβx / βx [ppm]',
                 title = f'Bunch-phase dependence at s = {location_s:.1f} m')
for ax in beta_axes:
    ax.grid(alpha = .2)
beta_figure.suptitle('Recomputed Twiss for fixed mechanical snapshots; not dynamic-equilibrium optics', fontsize = 11)
plt.show()
