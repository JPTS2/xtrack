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
P0C             = 1e9                   # Proton reference p0*c [eV]
LENGTH          = 1.0                   # [m]
RK4_STEPS       = [8, 16, 32, 64]
PROBE_SCALES    = [1e-4, 1e-5, 1e-6, 1e-7]
LENGTH_SCALE    = 1.0                   # Common coordinate length scale [m]

################################################################################
# Define canonical transport cases and the entrance state
################################################################################
initial = xt.Particles(
    'proton', p0c = P0C, x = 1e-3, px = 2e-3,
    y = 0.01, py = -1e-3, delta = 0.02)
cases   = {
    'Straight dipole': dict(knc = [[0.05, 0.1, -0.1]], h = 0.0),
    'Curved dipole': dict(knc = [[0.05, 0.1, -0.1]], h = 0.15),
    'Straight solenoid': dict(ksolc = [0.3, 1.0, -1.0], h = 0.0),
    'Curved solenoid': dict(ksolc = [0.3, 1.0, -1.0], h = 0.15),
}
# Canonical coordinates are (x,px,y,py,zeta,pzeta), with pzeta=ptau/beta0.
# pkin_const=False preserves the canonical convention at field interfaces.
# Radiation, acceleration, apertures and collective forces are absent.
symplectic_form = np.kron(np.eye(3), [[0.0, 1.0], [-1.0, 0.0]])
scales          = np.diag([LENGTH_SCALE, 1, LENGTH_SCALE, 1, LENGTH_SCALE, 1])

################################################################################
# Separate integration error from finite-difference error
################################################################################
fig, axes = plt.subplots(2, 2, figsize = (11, 8), layout = 'constrained')
for axis, (label, fields) in zip(axes.ravel(), cases.items()):
    element = xt.BFieldExpansion(
        length = LENGTH, num_phi = 7, pkin_const = False, **fields)
    line    = xt.Line(elements = {'field': element},
                   particle_ref = initial.copy())
    for count in RK4_STEPS:
        element.num_integration_steps   = count
        defects                         = []
        determinants                    = []
        for step in PROBE_SCALES:
            # Native central differences include the delta-to-pzeta conversion.
            matrix          = line.get_R_matrix(
                particle_on_co = initial,
                steps = dict(dx = step * LENGTH_SCALE, dpx = step,
                             dy = step * LENGTH_SCALE, dpy = step,
                             dzeta = step * LENGTH_SCALE, ddelta = step),
            )['R_matrix']
            dimensionless   = np.linalg.solve(scales, matrix @ scales)
            defect          = dimensionless.T @ symplectic_form @ dimensionless
            defects.append(np.max(np.abs(defect - symplectic_form)))
            determinants.append(np.linalg.det(matrix) - 1)
        axis.loglog(PROBE_SCALES, defects, 'o-', label = f'{count} RK4 steps')
        print(label, count, 'symplectic defects:', defects)
        print('  det(R)-1:', determinants)
    axis.set(title = label, xlabel = 'Relative finite-difference probe scale',
             ylabel = 'Max canonical symplectic defect')
    axis.grid(which = 'both')
    axis.legend(fontsize = 8)
# Every canonical pair has the same product scale, so the dimensionless matrix
# uses the same symplectic form. A determinant near one is necessary but not
# sufficient. RK4 is not symplectic; refinement should reduce its map defect
# until finite-difference truncation or roundoff limits the observation.
# These are local finite-pass Jacobians, not ring stability/lifetime evidence.
plt.show()
