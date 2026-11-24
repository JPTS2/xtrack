# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Common straight reference and exterior planes
########################################
P0C             = 1e9                   # Proton reference momentum [eV/c]
K0              = 0.15                  # Vertical field / rigidity [m^-1]
MAGNET_LENGTH   = 1.0                   # Hard-edge magnetic length [m]
DRIFT_LENGTH    = 0.5                   # Each surrounding drift [m]
RAMP_LENGTHS    = [0.025, 0.05]         # Length of each quintic ramp [m]
Y_START         = np.linspace(-3e-3, 3e-3, 7)  # [m]
Y_LINEAR        = 1e-6                  # Small-amplitude slope probe [m]

########################################
# Fringe parameters for the Bend edge maps
########################################
HALF_GAP        = 0.02                  # Pole half-gap [m]

########################################
# Integration controls
########################################
RAMP_STEPS      = 64
BODY_STEPS      = 128

################################################################################
# Keep the field integral and exterior planes fixed
################################################################################

########################################
# Hard-edge field with full dipole edges
########################################
hard_edge = xt.Line(elements = [
    xt.Drift(length = DRIFT_LENGTH),
    xt.Bend(
        length = MAGNET_LENGTH, angle = 0.0, k0 = K0,
        edge_entry_model = 'full', edge_exit_model = 'full'),
    xt.Drift(length = DRIFT_LENGTH),
])
# Reference curvature is zero and the powered field bends the ray.
# The unsmoothed reference has full edge maps with hgap = fint = 0.

########################################
# Resolve two fringe lengths with the same BFieldExpansion model
########################################
# Within the smooth family, only the ramp length changes. Each quintic ramp
# contributes half its length at peak field, so shorten the plateau by one
# ramp length to preserve the integral.
initial        = xt.Particles('proton', p0c = P0C, y = np.r_[Y_LINEAR, Y_START])
hard_particles = initial.copy()
hard_edge.track(hard_particles)
matched_bends = {}
outputs       = {}
print('Match the Bend fringe integral to each resolved profile')
print('--------------------------------------------------------')
print(f'Common entrance-to-exit length [m]: {hard_edge.get_length():.3f}')
print(f'Common integrated B_y/(B rho) [1]: {K0 * MAGNET_LENGTH:.6f}')
print('Bend baseline: full edges, hgap = 0, fint = 0.')
print()

for ramp_length in RAMP_LENGTHS:
    # For f = B_y/B_peak, 2*hgap*fint = integral f*(1-f) ds at one face.
    # The quintic f(t) = 10*t**3 - 15*t**4 + 6*t**5 gives 25/231
    # on t in [0, 1]. The falling ramp gives the same integral.
    gap_fint = (25 / 462) * ramp_length
    fint     = gap_fint / HALF_GAP
    gap_edge = hard_edge.copy()
    gap_bend = gap_edge['e1']
    gap_bend.edge_entry_hgap = HALF_GAP
    gap_bend.edge_exit_hgap  = HALF_GAP
    gap_bend.edge_entry_fint = fint
    gap_bend.edge_exit_fint  = fint
    gap_particles = initial.copy()
    gap_edge.track(gap_particles)
    matched_bends[ramp_length] = gap_particles

    ramp_mm = ramp_length * 1e3
    print(f'{ramp_mm:g} mm ramp: hgap = {HALF_GAP * 1e3:g} mm, '
          f'fint = {fint:.6f}, hgap * fint [mm] = {gap_fint * 1e3:.6f}')
    rising = K0 * np.array([
        0.0, 0.0, 0.0, 10 / ramp_length**3,
        -15 / ramp_length**4, 6 / ramp_length**5])
    falling = -rising.copy()
    falling[0] = K0

    for refinement in [1, 2]:
        smooth = xt.Line(elements = [
            xt.Drift(length = DRIFT_LENGTH - ramp_length / 2),
            xt.BFieldExpansion(
                length = ramp_length, knc = [rising],
                num_integration_steps = refinement * RAMP_STEPS,
                pkin_const = True),
            xt.BFieldExpansion(
                length = MAGNET_LENGTH - ramp_length, knc = [[K0]],
                num_integration_steps = refinement * BODY_STEPS,
                pkin_const = True),
            xt.BFieldExpansion(
                length = ramp_length, knc = [falling],
                num_integration_steps = refinement * RAMP_STEPS,
                pkin_const = True),
            xt.Drift(length = DRIFT_LENGTH - ramp_length / 2),
        ])
        particles = initial.copy()
        smooth.track(particles)
        if refinement == 1:
            coarse_py = particles.py.copy()

    outputs[ramp_length] = particles
    step_error = np.max(np.abs(particles.py - coarse_py))
    print(f'{ramp_length * 1e3:g} mm ramp, maximum exit py change on '
          f'doubling RK4 steps [1]: {step_error:.2e}')

################################################################################
# Compare the vertical kick at the common exit plane
################################################################################
print('\nExit momentum at several launch heights')
print('---------------------------------------')
table_data = {'Entrance y [mm]': 1e3 * Y_START,
              'Bend zero py [1e-6]': 1e6 * hard_particles.py[1:]}
for ramp_length, particles in outputs.items():
    ramp_label = f'{ramp_length * 1e3:g} mm'
    table_data[f'{ramp_label} Bend py [1e-6]'] = (
        1e6 * matched_bends[ramp_length].py[1:])
    table_data[f'{ramp_label} field py [1e-6]'] = 1e6 * particles.py[1:]
xt.Table(table_data, index = 'Entrance y [mm]').show(maxwidth = 'full')

################################################################################
# Put the nonlinear residual on the scale of the total kick
################################################################################
fig, (field_ax, kick_ax, error_ax) = plt.subplots(
    1, 3, figsize = (13, 4), layout = 'constrained')
window = 0.7 * max(RAMP_LENGTHS)
field_ax.step(
    [-window * 1e3, 0.0, window * 1e3], [0.0, K0, K0],
    where = 'post', color = '0.35', linestyle = '--', label = 'Bend')
s_edge = np.linspace(-window, window, 200)
print('\nCompare the residual with the total kick')
print('----------------------------------------')
for index, (ramp_length, particles) in enumerate(outputs.items()):
    t = np.clip(s_edge / ramp_length + 0.5, 0.0, 1.0)
    field = K0 * t**3 * (10 - 15 * t + 6 * t**2)
    delta_py = particles.py - matched_bends[ramp_length].py
    # Use the extra 1 µm launch to estimate the slope of the residual at y=0.
    # The symmetric midplane has zero vertical kick at y=0 in both models.
    linear_py    = delta_py[0] / Y_LINEAR * Y_START
    nonlinear_py = delta_py[1:] - linear_py
    matched_py   = matched_bends[ramp_length].py[1:]
    # At y = 0 both kicks vanish by symmetry. The percentage is undefined.
    nonzero      = matched_py != 0
    error_pct    = np.full(Y_START.shape, np.nan)
    error_pct[nonzero] = (
        100 * np.abs(nonlinear_py[nonzero] / matched_py[nonzero]))
    label = f'{ramp_length * 1e3:g} mm ramp'
    color = f'C{index}'
    field_ax.plot(1e3 * s_edge, field, color = color, label = label)
    kick_ax.plot(1e3 * Y_START, 1e6 * matched_py, color = color,
                 label = 'Matched Bend' if index == 0 else None)
    kick_ax.plot(1e3 * Y_START, 1e6 * particles.py[1:], 'o', color = color,
                 fillstyle = 'none', label = 'Resolved' if index == 0 else None)
    error_ax.plot(1e3 * Y_START, error_pct, 'o-', color = color, label = label)
    print(f'{label}: max linear part = {np.max(np.abs(linear_py)):.3e}, '
          f'max nonlinear part = {np.max(np.abs(nonlinear_py)):.3e} [1]')
    print(f'  Largest nonlinear residual relative to matched kick '
          f'[%]: {np.nanmax(error_pct):.3f}')
field_ax.set(title = 'Entrance field profile',
             xlabel = 'Distance from Bend face [mm]',
             ylabel = r'$B_y/(B\rho)$ [m$^{-1}$]')
kick_ax.set(title = 'Total kick (both ramp lengths)',
            xlabel = 'Entrance y [mm]', ylabel = r'Exit $p_y$ [$10^{-6}$]')
error_ax.set(title = 'Nonlinear difference',
             xlabel = 'Entrance y [mm]',
             ylabel = 'Fraction of matched kick [%]',
             ylim = (0.0, None))
for ax in (field_ax, kick_ax, error_ax):
    ax.legend(frameon = False)
    ax.grid(alpha = 0.25)
plt.show()
