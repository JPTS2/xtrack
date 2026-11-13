# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Extraction sextupole settings
########################################
LATTICE_FILE    = 'test_data/example_lattices/pimms_extraction/lattice.py'
SETTINGS        = [('Initial', 1.0, -6.5), ('Rotated', 4.897, 0.2315)]
# Each setting gives (label, kse1, kse2), with k2 in m^-3.
SEPTUM_X        = 0.035              # Proposed intercepting boundary [m]

########################################
# Finite-turn boundary estimate
########################################
N_TURNS         = 1024
SEARCH_MAX_X    = 0.03               # Positive-x, px=0 search interval [m]
SEARCH_TOL      = 2e-6               # Launch bracket width [m]

################################################################################
# Locate and inspect the extraction boundary for each setting
################################################################################
repo_root   = Path(__file__).resolve().parents[4]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['pimms']
tw          = line.twiss4d()
fig, axes   = plt.subplots(1, 2, figsize = (10, 4))
for label, kse1, kse2 in SETTINGS:
    ########################################
    # Bracket the launch that first reaches the proposed septum
    ########################################
    env['kse1'] = kse1
    env['kse2'] = kse2
    lower       = 0.0
    upper       = SEARCH_MAX_X
    particle    = line.build_particles(x = upper)
    line.track(particle, num_turns = N_TURNS, turn_by_turn_monitor = True)
    record      = line.record_last_track
    if not np.any((record.x[0] >= SEPTUM_X) & (record.state[0] > 0)):
        raise ValueError('SEARCH_MAX_X does not bracket a septum crossing.')
    while upper - lower > SEARCH_TOL:
        midpoint    = (lower + upper) / 2
        particle    = line.build_particles(x = midpoint)
        line.track(particle, num_turns = N_TURNS,
                   turn_by_turn_monitor = True)
        record  = line.record_last_track
        reached = np.any((record.x[0] >= SEPTUM_X) & (record.state[0] > 0))
        if reached:
            upper = midpoint
        else:
            lower = midpoint

    ########################################
    # Estimate the bounded region from the inner tracked trajectory
    ########################################
    particles   = line.build_particles(x = [lower, upper])
    line.track(particles, num_turns = N_TURNS, turn_by_turn_monitor = True)
    record      = line.record_last_track
    normalized  = tw.get_normalized_coordinates(record)
    x_norm      = normalized.x_norm[0]
    px_norm     = normalized.px_norm[0]
    order       = np.argsort(np.arctan2(px_norm, x_norm))
    x_boundary  = x_norm[order]
    px_boundary = px_norm[order]
    # Shoelace area includes the factor 1/2. This is a sampled contour, not a
    # proof of infinite-time stability or a precision separatrix calculation.
    area = 0.5 * np.abs(np.sum(
        x_boundary * np.roll(px_boundary, -1)
        - px_boundary * np.roll(x_boundary, -1)))

    ########################################
    # Measure the outgoing branch slope near the proposed septum
    ########################################
    valid       = np.flatnonzero(record.state[1] > 0)
    nearest     = valid[np.argmin(np.abs(record.x[1, valid] - SEPTUM_X))]
    neighbours  = np.array([nearest - 3, nearest, nearest + 3])
    neighbours  = neighbours[(neighbours >= 0) & (neighbours < N_TURNS)]
    neighbours  = neighbours[record.state[1, neighbours] > 0]
    if len(neighbours) < 2:
        raise ValueError('Not enough same-branch samples near the septum.')
    # Every third turn returns to the same resonance branch.
    slope, intercept = np.polyfit(record.x[1, neighbours],
                                 record.px[1, neighbours], 1)
    print(f'{label}: launch bracket [{lower:.7f}, {upper:.7f}] m')
    print(f'  Sampled bounded area: {area:.6g} m')
    print(f'  Branch dpx/dx near x={SEPTUM_X:g} m: {slope:.6g} m^-1')

    ########################################
    # Compare the bounded contour and outgoing branch
    ########################################
    closed_order    = np.append(order, order[0])
    contour,        = axes[0].plot(record.x[0, closed_order] * 1e3,
                            record.px[0, closed_order] * 1e3, label = label)
    color           = contour.get_color()
    axes[1].plot(np.append(x_boundary, x_boundary[0]),
                 np.append(px_boundary, px_boundary[0]),
                 color = color, label = label)
    for branch in range(3):
        branch_turns = valid[valid % 3 == branch]
        axes[0].plot(record.x[1, branch_turns] * 1e3,
                     record.px[1, branch_turns] * 1e3,
                     color = color, alpha = 0.4)
    x_fit = np.array([SEPTUM_X - 0.005, SEPTUM_X + 0.005])
    axes[0].plot(x_fit * 1e3, (slope * x_fit + intercept) * 1e3,
                 '--', color = color)

axes[0].axvline(SEPTUM_X * 1e3, color = 'k', linestyle = ':')
axes[0].set(xlabel = 'x [mm]', ylabel = r'$p_x$ [$10^{-3}$]',
            xlim = (-40, 45), ylim = (-4, 4))
axes[1].set(xlabel = r'$x/\sqrt{\beta_x}$ [$\sqrt{m}$]',
            ylabel = r'$(\alpha_x x+\beta_x p_x)/\sqrt{\beta_x}$ [$\sqrt{m}$]')
axes[1].set_aspect('equal')
for axis in axes:
    axis.grid(alpha = 0.3)
    axis.legend()
fig.tight_layout()
plt.show()
