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
# Reference line
########################################
P0C             = 1.0e9                 # Proton reference p0*c [eV]
DRIFT_LENGTH    = 0.5                   # [m]
K1L             = 0.2                   # Normal quadrupole [m^-1]
K2L             = 20.0                  # Normal sextupole [m^-2]
K3L             = 600.0                 # Normal octupole [m^-3]

########################################
# Entrance amplitude scan
########################################
AMPLITUDES      = np.geomspace(1e-4, 1e-2, 11)  # Entrance x [m]
Y_OVER_X        = 0.4
PX_PER_X        = 0.02                  # P_x/p0 per metre of entrance x
PY_PER_Y        = -0.01                 # P_y/p0 per metre of entrance y

########################################
# Longitudinal coordinate probe
########################################
ZETA_PROBE      = 0.01                  # [m]
DELTA_PROBE     = 1.0e-4                # (p - p0) / p0

################################################################################
# Build the known nonlinear reference map
################################################################################
line = xt.Line(
    elements        = [
        xt.Marker(),
        xt.Drift(length = DRIFT_LENGTH, model = 'exact'),
        xt.Multipole(knl = [0.0, K1L, K2L, K3L]),
        xt.Marker()],
    element_names   = ['start', 'drift', 'multipole', 'end'],
)
line.set_particle_ref('proton', p0c = P0C)

################################################################################
# Extract the quadratic map about the on-axis trajectory
################################################################################

########################################
# Supply open-line entrance optics
########################################
# The reference trajectory is zero. Entrance beta values supply the open
# Twiss calculation used to identify that trajectory; no periodicity is assumed.
tw = line.twiss(method = '4d', betx = 1.0, bety = 1.0)
second_order = xt.SecondOrderTaylorMap.from_line(
    line,
    start       = 'start',
    end         = 'end',
    twiss_table = tw,
)

########################################
# Inspect the tensor convention
########################################
# z_out = k + R @ z + sum_jk T[:, j, k]*z[j]*z[k]; no extra 1/2.
# Here z = (x, px, y, py, zeta, pzeta), with pzeta = ptau/beta0.
print('Extracted T[px, x, x] and -K2L/2 [m^-2]:',
      second_order.T[1, 0, 0], -K2L / 2)
print('Mixed py entries, whose sum multiplies x*y [m^-2]:',
      second_order.T[3, 0, 2], second_order.T[3, 2, 0])

################################################################################
# Construct the corresponding affine first-order map
################################################################################

########################################
# Convert the fixed-reference coordinate basis
########################################
# FirstOrderTaylorMap uses u = (x, px, y, py, tau, ptau), tau = zeta/beta0.
# With fixed reference momentum, z = basis @ u converts the two conventions.
beta0       = line.particle_ref.beta0[0]
basis       = np.diag([1.0, 1.0, 1.0, 1.0, beta0, 1 / beta0])
first_order = xt.FirstOrderTaylorMap(
    length  = second_order.length,
    m0      = np.linalg.solve(basis, second_order.k),
    m1      = np.linalg.solve(basis, second_order.R @ basis),
)
print('Longitudinal shear coefficients in the two coordinate conventions [m]:')
print('R[zeta, pzeta]:', second_order.R[4, 5])
print('m1[tau, ptau]:', first_order.m1[4, 5])

########################################
# Inspect a small longitudinal response
########################################
longitudinal_probe = xt.Particles(
    mass0   = xt.PROTON_MASS_EV,
    p0c     = P0C,
    zeta    = ZETA_PROBE,
    delta   = DELTA_PROBE,
)
# The linear drift shear is d(zeta)/d(pzeta) = L/gamma0^2 at delta = 0.
# The zero-length multipole adds no longitudinal advance.
linear_zeta = ZETA_PROBE + DRIFT_LENGTH / line.particle_ref.gamma0[0]**2 * (
    longitudinal_probe.pzeta[0])
ptau_before = longitudinal_probe.ptau[0]
first_order.track(longitudinal_probe)
print('Expected linear zeta and mapped zeta [m]:',
      linear_zeta, longitudinal_probe.zeta[0])
print('ptau before and after the affine map:',
      ptau_before, longitudinal_probe.ptau[0])

################################################################################
# Track a common family of entrance amplitudes
################################################################################

########################################
# Scale all transverse coordinates towards the expansion point
########################################
initial = xt.Particles(
    mass0   = xt.PROTON_MASS_EV,
    p0c     = P0C,
    x       = AMPLITUDES,
    y       = Y_OVER_X * AMPLITUDES,
    px      = PX_PER_X * AMPLITUDES,
    py      = PY_PER_Y * Y_OVER_X * AMPLITUDES,
)

########################################
# Track the reference line and both approximations
########################################
outputs     = {}
reference   = initial.copy()
# Stop at the exit marker so open-pass s is retained.
line.track(reference, ele_stop = 'end')
for label, element in [
    ('first order', first_order), ('second order', second_order)
]:
    particles = initial.copy()
    element.track(particles)
    outputs[label] = particles
    print(label, 'largest-amplitude exit px, py:',
          particles.px[-1], particles.py[-1])
print('Reference largest-amplitude exit px, py:',
      reference.px[-1], reference.py[-1])
print('Reference and map lengths [m]:', line.get_length(), second_order.length)

################################################################################
# Inspect the local approximation error
################################################################################
# Both residual components are normalized canonical momenta, so their norm
# is dimensionless. Finite-difference extraction also has a numerical floor.
errors = {}
fig, ax = plt.subplots(figsize = (8, 4))
for label, particles in outputs.items():
    error = np.hypot(particles.px - reference.px, particles.py - reference.py)
    errors[label] = error
    ax.loglog(AMPLITUDES * 1e3, error, 'o-', label = label)
ax.set_xlabel('Entrance x amplitude [mm]')
ax.set_ylabel('Normalized transverse momentum residual [1]')
ax.set_title('Taylor approximations to the same drift and multipole map')
ax.legend()
ax.grid(which = 'major')
fig.tight_layout()
plt.show()
