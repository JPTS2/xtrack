# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
import xtrack as xt
import xtrack.synctime as st

################################################################################
# User parameters
################################################################################

########################################
# Constant-speed transport around a loop
########################################
CIRCUMFERENCE   = 100.0                  # [m]
P0C             = 1e9                    # Proton reference momentum [eV/c]
DELTA           = np.array([-0.2, 0.0, 0.2])

########################################
# Frame clock
########################################
FRAME_FRACTION  = 0.8                    # Reference revolutions per frame
N_FRAMES        = 24

########################################
# Pickup passage acquisition
########################################
PICKUP_FRAMES   = 128                    # Additional complete frames
PICKUP_BINS     = 64                     # Time bins per reference revolution
SHOW_PERIODS    = 5                      # Event-raster display range

################################################################################
# Install synchronization boundaries before building the tracker
################################################################################

########################################
# Build an illustrative constant-path loop
########################################
# Zero transverse coordinates keep each drift path equal to its design length.
# This loop isolates timing; it is not a transversely confining storage ring.
line        = xt.Line(elements = {
    'first_half':   xt.Drift(length = CIRCUMFERENCE / 2),
    'pickup':       xt.Marker()})
line.set_particle_ref('proton', p0c = P0C)
particles   = xt.Particles('proton', p0c = P0C, delta = DELTA)
beta0       = particles.beta0[0]

reference_period    = CIRCUMFERENCE / (beta0 * c)

########################################
# Install separate native pickups for the three momentum classes
########################################
# Each monitor accepts one particle ID. Coasting acquisition bins LOCAL
# arrival time, independently of that particle's changing revolution counter.
pickups = []
for index in range(len(DELTA)):
    pickup = xt.BeamStatsMonitor(
        coasting            = True,
        num_slices          = PICKUP_BINS,
        start_at_turn       = 0,
        stop_at_turn        = int(np.ceil(
            (N_FRAMES + PICKUP_FRAMES) * FRAME_FRACTION)) + 2,
        particle_id_range   = (index, index + 1),
        stats               = ['num_particles'],
    )
    line.append(f'pickup.{index}', pickup)
    pickups.append(pickup)
line.append('second_half', xt.Drift(length = CIRCUMFERENCE / 2))

########################################
# Put the pickup and line ends on the same frame clock
########################################
# The frame window must move faster than every particle:
# beta0 / FRAME_FRACTION > max(beta0 * particles.rvv).
# Here its speed is about 0.91c, compared with at most 0.79c for the probes.
# Cavities and collective elements are selected automatically; the marker is
# included explicitly. Install after any construction or slicing changes.
st.install_sync_time_at_collective_elements(
    line,
    frame_relative_length   = FRAME_FRACTION,
    at_element_names        = ['pickup'],
    with_progress           = False)
line.build_tracker()
line.get_table().cols['name s element_type'].show()

################################################################################
# Advance complete frames and inspect the particle passage counters
################################################################################

########################################
# Track one frame at a time
########################################
clock       = np.zeros(N_FRAMES + 1)
passages    = np.zeros((N_FRAMES + 1, len(DELTA)), dtype = int)
waiting     = np.zeros(N_FRAMES + 1, dtype = int)
for frame in range(1, N_FRAMES + 1):
    # With SyncTime installed, num_turns counts frames, not revolutions.
    line.track(particles, num_turns = 1)
    order   = np.argsort(particles.particle_id)
    clock[frame]        = particles.time_s
    passages[frame]     = particles.at_turn[order]
    waiting[frame]      = np.count_nonzero(particles.state < 0)

########################################
# Separate the common clock from each particle's location and arrival time
########################################
# Negative SyncTime states mean waiting at a frame boundary, not particle loss.
# Waiting particles resume automatically in the appropriate later frame.
order   = np.argsort(particles.particle_id)
arrival = particles.time_s + (particles.s - particles.zeta) / (beta0 * c)
xt.Table(
    data    = {
        'particle_id':      particles.particle_id[order],
        'delta':            particles.delta[order],
        'completed_turns':  particles.at_turn[order],
        's_m':              particles.s[order],
        'arrival_time_us':  arrival[order] * 1e6,
        'state':            particles.state[order]},
    index   = 'particle_id').show(maxwidth = 'full')
print('Common frame clock [us]:', particles.time_s * 1e6)
print('Reference-turn counter:', particles.reference_turns)
print('Expected reference turns:', N_FRAMES * FRAME_FRACTION)
print('Frames with particles waiting:', np.count_nonzero(waiting))

# Arrival times refer to the stored particle positions. Frame windows schedule
# events: the stored particles need not represent one simultaneous lab snapshot.
# Each at_turn belongs to that particle's stored event, not precisely time_s.
# These boundaries schedule transport; they do not equalize particle speeds.
# This example uses the CPU controller and complete frames only.

################################################################################
# Compare the frame clock with stored particle revolution counters
################################################################################
fig, ax = plt.subplots(figsize = (8, 4))
for index, delta in enumerate(DELTA):
    ax.step(clock / reference_period, passages[:, index], where = 'post',
            label   = f'delta = {delta:+.1f}')
ax.plot(clock / reference_period, clock / reference_period, '--',
        label   = 'Reference turns')
ax.set_xlabel('Shared clock / reference revolution period')
ax.set_ylabel('Stored particle revolution counter')
ax.grid()
ax.legend()
fig.tight_layout()

################################################################################
# Measure revolution frequencies from actual pickup passages
################################################################################

########################################
# Continue the same particles across many relative turn slips
########################################
line.track(particles, num_turns = PICKUP_FRAMES)
# The pickup history includes the preceding frames. More than 100 passages
# per probe separate the period measurement from a few integer frame counts.

########################################
# Convert occupied pickup bins to passage times
########################################
event_times         = []
measured_frequency  = []
bin_bound           = []
passage_count       = []
for pickup in pickups:
    time_grid   = pickup.time_centers(
        line_length = CIRCUMFERENCE,
        beta0       = beta0,
    )
    times       = time_grid[pickup.num_particles > 0]
    span        = times[-1] - times[0]
    count       = len(times)
    width       = reference_period / PICKUP_BINS
    frequency   = (count - 1) / span
    event_times.append(times)
    measured_frequency.append(frequency)
    passage_count.append(count)
    # Each endpoint lies within half a bin of its assigned centre. Thus the
    # measured span differs by at most one bin; this is not a statistical error.
    bin_bound.append(frequency * width / (span - width))

# time_centers uses the local pickup clock, omitting the constant reference
# flight time from s=0 to this pickup. That offset cancels in passage intervals.
# One isolated probe occupies each populated bin: bin width is far below its
# revolution period. These are detected passages, not a sampled FFT spectrum.

########################################
# Compare measured frequencies with constant-path speeds
########################################
order               = np.argsort(particles.particle_id)
expected_frequency  = beta0 * particles.rvv[order] * c / CIRCUMFERENCE
measured_frequency  = np.array(measured_frequency)
bin_bound           = np.array(bin_bound)
xt.Table(
    data    = {
        'delta':            DELTA,
        'pickup_passages':  np.array(passage_count),
        'measured_Hz':      measured_frequency,
        'speed_Hz':         expected_frequency,
        'difference_Hz':    measured_frequency - expected_frequency,
        'bin_bound_Hz':     bin_bound},
    index   = 'delta').show(maxwidth = 'full')
print('Pickup time-bin width [ns]:', width * 1e9)
print('Acquisition spans [us]:',
      [(times[-1] - times[0]) * 1e6 for times in event_times])
# beta*c/C is exact for this fixed-path, zero-transverse-motion model.
# A magnetic ring also has momentum-dependent path length; use its orbit
# and revolution-time calculation rather than transplanting this speed formula.

################################################################################
# Inspect detected passages and their frequency quantization
################################################################################

########################################
# Show early pickup events on the local clock
########################################
fig, axes = plt.subplots(1, 2, figsize = (11, 4))
for index, times in enumerate(event_times):
    visible = times < SHOW_PERIODS * reference_period
    axes[0].plot(
        times[visible] / reference_period,
        np.full(np.count_nonzero(visible), index), '|', markersize = 15)
axes[0].set_yticks(range(len(DELTA)),
                   labels = [f'delta = {value:+.1f}' for value in DELTA])
axes[0].set_xlabel('Local pickup time / reference revolution period')
axes[0].set_ylim(-0.5, len(DELTA) - 0.5)
axes[0].grid(axis = 'x')

########################################
# Show the deterministic binning bound on the frequency measurement
########################################
axes[1].errorbar(DELTA, measured_frequency - expected_frequency,
                 yerr = bin_bound, fmt = 'o', label = 'Endpoint-bin bound')
axes[1].axhline(0.0, color = '0.5', linestyle = '--')
axes[1].set_xlabel('Relative momentum offset delta')
axes[1].set_ylabel('Measured minus speed prediction [Hz]')
axes[1].legend()
axes[1].grid()
fig.tight_layout()
plt.show()
