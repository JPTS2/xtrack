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
LATTICE_FILE    = 'test_data/example_lattices/elena_field/lattice.py'
BEND_NAME       = 'lnr.mbhek.0135'
FIELD_SAMPLES   = 20                    # Samples per fitted segment
NEMITT          = 1e-9                  # Normalized probe emittance [m]
X_NORMALIZED    = np.array([-1.0, 0.0, 1.0])
Y_NORMALIZED    = np.array([0.5, 0.0, -0.5])

################################################################################
# Load the simplified ELENA field-expansion ring
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['elena']
tw          = line.twiss4d()
survey      = line.survey()
# The 100 MeV/c antiproton model has six 60-degree bends. Their fitted fields
# extend into straight reference sections. RF, apertures and cooler are absent.
print('Reference p0c [MeV/c]:', line.particle_ref.p0c[0] * 1e-6)
print('Circumference [m]:', line.get_length())
print('Tunes Qx, Qy:', tw.qx, tw.qy)
print('Maximum horizontal closed orbit [mm]:', np.max(np.abs(tw.x)) * 1e3)
print('Survey closure [m]:', np.linalg.norm(survey.XYZ[-1] - survey.XYZ[0]))

################################################################################
# Inspect field and reference curvature across one dipole
################################################################################
# get_field returns rigidity-normalized fields, not tesla. Curvature h defines
# the reference geometry independently of the magnetic field coefficients.
field_s     = []
field_by    = []
curvature   = []
position    = 0.0
for name in line.element_names:
    if name.startswith(BEND_NAME + '.'):
        element = line[name]
        local_s = np.linspace(0.0, element.length, FIELD_SAMPLES)
        field   = element.get_field(x = 0.0, y = 0.0, s_local = local_s)
        field_s.extend(position + local_s)
        field_by.extend(field['By'])
        curvature.extend(np.full(FIELD_SAMPLES, element.h))
        position += element.length

################################################################################
# Follow small-amplitude probes through one turn
################################################################################

########################################
# Launch around the periodic closed orbit
########################################
particles   = line.build_particles(
    method          = '4d',
    particle_on_co  = tw.particle_on_co,
    W_matrix        = tw.W_matrix[0],
    x_norm          = X_NORMALIZED,
    px_norm         = 0.0,
    y_norm          = Y_NORMALIZED,
    py_norm         = 0.0,
    nemitt_x        = NEMITT,
    nemitt_y        = NEMITT,
    delta           = 0.0,
    zeta            = 0.0,
)

########################################
# Record each element boundary
########################################
line.track(particles, turn_by_turn_monitor = 'ONE_TURN_EBE')
record  = line.record_last_track
# The middle probe starts on the closed orbit. The other two have opposite
# transverse launches; these are trajectories, not a sampled beam envelope.
print('Recorded boundary count:', record.x.shape[1])
print('Surviving probes:', np.count_nonzero(particles.state > 0))
print('Central-probe maximum |x - closed orbit| [m]:',
      np.max(np.abs(record.x[1] - tw.x)))

################################################################################
# Display the field, optics, geometry and local trajectories
################################################################################

########################################
# Compare field extent with the curved reference region
########################################
fig, axes = plt.subplots(2, 2, figsize = (12, 8), layout = 'constrained')
axes[0, 0].plot(field_s, field_by, label = r'$B_y/(B\rho)$ on axis')
axes[0, 0].plot(field_s, curvature, '--', label = 'Reference curvature h')
axes[0, 0].set(
    xlabel  = 'Distance from fitted dipole entrance [m]',
    ylabel  = r'Normalized field / curvature [m$^{-1}$]',
    title   = 'One 60-degree bend with extended fringe fields',
)
axes[0, 1].plot(tw.s, tw.betx, label = r'$\beta_x$')
axes[0, 1].plot(tw.s, tw.bety, label = r'$\beta_y$')
axes[0, 1].set(xlabel = 's [m]', ylabel = r'$\beta$ [m]')

########################################
# Distinguish the reference path from local transverse motion
########################################
axes[1, 0].plot(survey.Z, survey.X, label = 'Survey reference path')
axes[1, 0].set(xlabel = 'Z [m]', ylabel = 'X [m]', aspect = 'equal')
for index in range(len(X_NORMALIZED)):
    axes[1, 1].plot(
        record.s[index],
        (record.x[index] - tw.x) * 1e3,
        label = f'Probe {index}',
    )
axes[1, 1].set(xlabel = 's [m]', ylabel = 'x - closed orbit [mm]')
for ax in axes.ravel():
    ax.legend()
    ax.grid(alpha = 0.25)
plt.show()
