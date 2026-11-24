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
# Lens and reference particle
########################################
P0C             = 1e9                  # Proton reference momentum [eV/c]
LENS_STRENGTH   = 1e-6                 # Integrated knll [m]
LENS_SCALE      = 1e-2                 # Singularities at x = +/-cnll [m]

########################################
# Horizontal-axis probes and phase-space grid
########################################
LENS_AMPLITUDES = np.array([0.1, 0.3, 0.6])  # x/cnll; y = 0
GRID_X_MAX      = 0.7                   # Maximum |x/cnll|
GRID_PX_MAX     = 2e-4                  # Maximum |p_x/p0|
GRID_NX         = 21
GRID_NPX        = 7

################################################################################
# Track an elliptic nonlinear lens
################################################################################

########################################
# Launch inside the two singular points
########################################
# This potential is non-polynomial. Keep |x/cnll| < 1 at y = 0; its small-
# amplitude quadrupole term has integrated strength 2*knll/cnll**2 [m^-1].
lens = xt.NonLinearLens(knll = LENS_STRENGTH, cnll = LENS_SCALE)
lens_line = xt.Line(elements = [lens])
lens_line.set_particle_ref('proton', p0c = P0C)
lens_particles = lens_line.build_particles(x = LENS_SCALE * LENS_AMPLITUDES)
lens_line.track(lens_particles)

########################################
# Compare with the exact horizontal-axis derivative
########################################
# For u = x/cnll, the potential on this interval is
# U(u) = -u*asin(u)/sqrt(1-u**2); the kick is knll/cnll * dU/du.
u = LENS_AMPLITUDES
lens_expected_px = -LENS_STRENGTH / LENS_SCALE * (
    np.arcsin(u) / (1 - u**2)**1.5 + u / (1 - u**2))
linear_px = -2 * LENS_STRENGTH / LENS_SCALE * u
print('Track an elliptic nonlinear lens\n--------------------------------')
print('Elliptic lens x/cnll:', u)
print('Exact elliptic kick P_x/p0:', lens_expected_px)
print('Tracked elliptic kick P_x/p0:', lens_particles.px)
print('Small-amplitude quadrupole approximation P_x/p0:', linear_px)

################################################################################
# Track a horizontal phase-space grid through the lens
################################################################################
x_grid, px_grid = np.meshgrid(
    LENS_SCALE * np.linspace(-GRID_X_MAX, GRID_X_MAX, GRID_NX),
    np.linspace(-GRID_PX_MAX, GRID_PX_MAX, GRID_NPX))
grid_particles = lens_line.build_particles(
    x   = x_grid.ravel(),
    px  = px_grid.ravel())
lens_line.track(grid_particles)

# The thin lens leaves x fixed. Each row starts at a common px, so its
# curvature after tracking shows how the kick depends on x.
linear_k1l  = 2 * LENS_STRENGTH / LENS_SCALE**2
px_linear   = px_grid - linear_k1l * x_grid
x_tracked   = grid_particles.x.reshape(x_grid.shape)
px_tracked  = grid_particles.px.reshape(x_grid.shape)
print('\nCompare the tracked grid with the linear shear\n'
      '----------------------------------------------')
print(f'Surviving grid particles: {np.count_nonzero(grid_particles.state > 0)}'
      f' / {x_grid.size}')
print(f'Maximum nonlinear correction to px [1]: '
      f'{np.max(np.abs(px_tracked - px_linear)):.3e}')

################################################################################
# Compare the exit grids on common axes
################################################################################
fig, axes = plt.subplots(1, 2, sharex = True, sharey = True,
                         figsize = (10, 4), layout = 'constrained')
row_colors = plt.cm.viridis(np.linspace(0.12, 0.82, GRID_NPX))
for ax, x, px, title in [
    (axes[0], x_grid, px_linear, 'Linear quadrupole approximation'),
    (axes[1], x_tracked, px_tracked, 'Tracked nonlinear lens'),
]:
    ax.axhline(0, color = '0.85', linewidth = 0.8)
    ax.axvline(0, color = '0.85', linewidth = 0.8)
    ax.plot(1e3 * x, 1e3 * px, color = '0.8', linewidth = 0.7)
    # Match colours across panels for rows with the same entrance momentum.
    for x_row, px_row, color in zip(x, px, row_colors):
        ax.plot(1e3 * x_row, 1e3 * px_row, '.-',
                color = color, linewidth = 1.2, markersize = 4)
    ax.set(xlabel = 'Exit x [mm]', title = title)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel(r'Exit $p_x$ [$10^{-3}$]')
plt.show()
