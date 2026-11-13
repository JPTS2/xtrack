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
# Integrable detuning map and Gaussian emittances
########################################
P0C         = 7e12                  # Proton reference p0*c [eV]
QX          = 0.28
QY          = 0.31
DET_XX      = 1e6                   # dQx/dJx [m^-1]
DET_XY      = -5e5                  # dQx/dJy = dQy/dJx [m^-1]
DET_YY      = 1.2e6                 # dQy/dJy [m^-1]
NEMITT_X    = 2.5e-6                # Normalized rms emittance [m]
NEMITT_Y    = 4.5e-6                # [m]

########################################
# Finite sampling and dispersion-integral regularization
########################################
N_ACTION        = 24
SIGMA_RANGE     = (0.05, 6.0)           # Action bounds from amplitude in sigma
N_TURNS         = 512
N_FFT           = 2**16
N_INTERPOLATE   = 192
N_DIAGRAM       = 40
EPSILON         = 5e-5                  # Positive imaginary tune regulator

################################################################################
# Generate a tracked uniform-action footprint
################################################################################

########################################
# Build the integrable map
########################################
# This is an integrable normal-form map, not a lattice with an impedance model.
# Symmetric cross detuning is consistent with a Hamiltonian depending on Jx,Jy.
# A harmonic longitudinal map keeps the 6D normalization nonsingular. All
# launches have zeta=delta=0: no momentum spread or synchrotron averaging here.
line = xt.Line(elements = [
    xt.LineSegmentMap(
        qx      = QX,
        qy      = QY,
        det_xx  = DET_XX,
        det_xy  = DET_XY,
        det_yx  = DET_XY,
        det_yy  = DET_YY,
        qs      = 1e-3,
        bets    = 1.0,
    ),
])
line.set_particle_ref('proton', p0c = P0C)

########################################
# Track the uniform-action grid
########################################
footprint = line.get_footprint(
    nemitt_x        = NEMITT_X,
    nemitt_y        = NEMITT_Y,
    mode            = 'uniform_action_grid',
    n_x_norm        = N_ACTION,
    n_y_norm        = N_ACTION,
    x_norm_range    = SIGMA_RANGE,
    y_norm_range    = SIGMA_RANGE,
    n_turns         = N_TURNS,
    n_fft           = N_FFT,
    keep_fft        = False,
)

################################################################################
# Check tune extraction against the known action-dependent tunes
################################################################################
# Footprint action grids use normalized actions; the map uses geometric J [m].
betagamma   = float(line.particle_ref.beta0[0] * line.particle_ref.gamma0[0])
jx          = footprint.Jx_2d / betagamma
jy          = footprint.Jy_2d / betagamma
print('Maximum tracked-minus-analytic tune errors x/y:',
      np.max(np.abs(footprint.qx - (QX + DET_XX * jx + DET_XY * jy))),
      np.max(np.abs(footprint.qy - (QY + DET_XY * jx + DET_YY * jy))))

################################################################################
# Evaluate the Gaussian transverse dispersion integral
################################################################################

########################################
# Compute the regularized integral
########################################
# The API assumes an uncoupled Gaussian action density exp(-Jx/epsx-Jy/epsy).
# It returns complex coherent tune-shift boundaries with positive imaginary
# part by convention. EPSILON regularizes the pole.
diagram_x, diagram_y = footprint.get_stability_diagram(
    n_points_stability_diagram   = N_DIAGRAM,
    n_points_interpolate        = N_INTERPOLATE,
    epsilon0                    = EPSILON,
    epsilon_factor              = 0.0,
)

########################################
# Interpret the finite-regulator boundary
########################################
# This fixed epsilon, action cutoff and quadrature grid do not establish a
# converged threshold. No coherent mode or growth rate is supplied, so no
# stability verdict follows.
print('Fixed imaginary regulator:', EPSILON)
print('Peak imaginary boundary x/y:',
      np.max(diagram_x.imag), np.max(diagram_y.imag))

################################################################################
# Plot the finite-resolution boundaries in the API sign convention
################################################################################
fig, ax = plt.subplots(figsize = (8, 5))
for diagram, label in [(diagram_x, 'Horizontal'), (diagram_y, 'Vertical')]:
    ax.plot(diagram.real, diagram.imag, '.-', label = label)
ax.set_xlabel('Re coherent tune shift')
ax.set_ylabel('Im coherent tune shift (positive convention)')
ax.set_title('Gaussian dispersion integral: finite regulator and action cutoff')
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()
