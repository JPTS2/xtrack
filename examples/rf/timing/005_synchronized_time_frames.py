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

################################################################################
# Install synchronization boundaries before building the tracker
################################################################################

########################################
# Build an illustrative constant-path loop
########################################
# Zero transverse coordinates keep each drift path equal to its design length.
# This loop isolates timing; it is not a transversely confining storage ring.
line    = xt.Line(elements = {
    'first_half':   xt.Drift(length = CIRCUMFERENCE / 2),
    'pickup':       xt.Marker(),
    'second_half':  xt.Drift(length = CIRCUMFERENCE / 2)})
line.particle_ref = xt.Particles(mass0 = xt.PROTON_MASS_EV, p0c = P0C)
particles   = xt.Particles(
    mass0   = xt.PROTON_MASS_EV,
    p0c     = P0C,
    delta   = DELTA)
beta0       = particles.beta0[0]

reference_period    = CIRCUMFERENCE / (beta0 * c)

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
plt.show()
