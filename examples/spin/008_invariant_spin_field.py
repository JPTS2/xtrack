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
# Shared electron ring and a local solenoid
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
P0C             = 10e9                  # Reference p0*c [eV]
SOLENOID_S      = 2.0                   # [m], inside an existing drift
SOLENOID_LENGTH = 0.2                   # [m]
SOLENOID_KS     = 0.05                  # Bz/(B*rho) [m^-1]

########################################
# Small orbital offsets and their amplitude control
########################################
X_OFFSET        = 2e-6                  # [m]
PX_OFFSET       = 1e-7                  # Normalized horizontal momentum
Y_OFFSET        = 1e-6                  # [m]
PY_OFFSET       = -2e-7                 # Normalized vertical momentum
DELTA_OFFSET    = 1e-5                  # Relative momentum deviation
SCALES          = np.array([1.0, 0.5])
N_TURNS         = 32

################################################################################
# Obtain the native linear invariant spin field
################################################################################

########################################
# Find the coupled periodic orbit and spin solution
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.set_particle_ref('electron', p0c = P0C)
line.insert(
    'spin.solenoid',
    xt.UniformSolenoid(
        length  = SOLENOID_LENGTH,
        ks      = SOLENOID_KS,
    ),
    at      = SOLENOID_S,
    anchor  = 'start',
)
line.configure_spin('auto')
tw      = line.twiss4d(polarization_analysis = True)
co      = tw.particle_on_co
n0      = np.array([tw.spin_x[0], tw.spin_y[0], tw.spin_z[0]])
matrix  = tw.spin_n_matrix[0]

########################################
# Inspect derivatives with respect to Xtrack phase-space coordinates
########################################
# Column order is x, px, y, py, zeta, pzeta, with pzeta = ptau/beta0.
# Position derivatives have units m^-1; momentum derivatives are dimensionless.
# With RF off, the 4D field has no zeta dependence.
xt.Table(
    data    = {
        'component': np.array(['nx', 'ny', 'nz']),
        'd_dx':      matrix[:, 0],
        'd_dpx':     matrix[:, 1],
        'd_dy':      matrix[:, 2],
        'd_dpy':     matrix[:, 3],
        'd_dzeta':   matrix[:, 4],
        'd_dpzeta':  matrix[:, 5]},
    index   = 'component').show(maxwidth = 'full')

################################################################################
# Launch spins along the field at two nearby orbital amplitudes
################################################################################

########################################
# Build orbital probes relative to the same closed orbit
########################################
particles = line.build_particles(
    particle_on_co  = co,
    mode            = 'shift',
    x               = SCALES * X_OFFSET,
    px              = SCALES * PX_OFFSET,
    y               = SCALES * Y_OFFSET,
    py              = SCALES * PY_OFFSET,
    delta           = SCALES * DELTA_OFFSET,
)
coordinates = ['x', 'px', 'y', 'py', 'zeta', 'pzeta']
offsets     = np.array([
    getattr(particles, name) - getattr(co, name)[0] for name in coordinates])

########################################
# Evaluate n(u) = n0 + N u and set unit initial spins
########################################
field               = n0[:, None] + matrix @ offsets
field              /= np.linalg.norm(field, axis = 0)
particles.spin_x    = field[0]
particles.spin_y    = field[1]
particles.spin_z    = field[2]

################################################################################
# Follow the invariant field as the orbital coordinates change
################################################################################

########################################
# Compare tracked spin with the local linear field at each return
########################################
# Each return is at the same ring entrance, so the entrance n0 and matrix
# apply throughout this comparison as the orbital phase changes.
tracked = []
fields  = []
for turn in range(N_TURNS + 1):
    if turn:
        line.track(particles)
    offsets = np.array([
        getattr(particles, name) - getattr(co, name)[0]
        for name in coordinates])
    field   = n0[:, None] + matrix @ offsets
    field  /= np.linalg.norm(field, axis = 0)
    tracked.append(np.array([
        particles.spin_x, particles.spin_y, particles.spin_z]))
    fields.append(field)
tracked = np.asarray(tracked)
fields  = np.asarray(fields)

########################################
# Inspect the accuracy of the linear field approximation
########################################
errors  = np.linalg.norm(tracked - fields, axis = 1)
for index, scale in enumerate(SCALES):
    print('Amplitude scale / maximum spin-field difference:',
          scale, np.max(errors[:, index]))
print('Final particle states:', particles.state)
# The invariant field depends on orbital phase, not only on s around the ring.
# Neglected nonlinear terms limit this first-order approximation at finite
# amplitude. Radiation is off; individual spins precess with constant length.

################################################################################
# Resolve the off-orbit spin motion and linearization error
################################################################################

########################################
# Show transverse and longitudinal spin changes for the full-amplitude probe
########################################
turns       = np.arange(N_TURNS + 1)
fig, axes   = plt.subplots(3, 1, figsize = (9, 8), sharex = True)
for axis, component, label in [(axes[0], 0, 'Sx - n0x'),
                                (axes[1], 2, 'Sz - n0z')]:
    axis.plot(turns, tracked[:, component, 0] - n0[component], 'o',
               label = 'Tracked spin')
    axis.plot(turns, fields[:, component, 0] - n0[component], '--',
               label = 'Linear invariant field')
    axis.set_ylabel(label)

########################################
# Compare the approximation error at the two orbital amplitudes
########################################
for index, scale in enumerate(SCALES):
    axes[2].plot(turns, errors[:, index], label = f'Amplitude scale {scale:g}')
axes[2].set_ylabel('|S tracked - n linear|')
axes[2].set_xlabel('Turns')
for axis in axes:
    axis.grid()
    axis.legend()
fig.tight_layout()
plt.show()
