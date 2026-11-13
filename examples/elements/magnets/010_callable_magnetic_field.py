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
# Field-free entrance and exit
########################################
P0C             = 1e9                   # Proton reference p0*c [eV]
FIELD_LENGTH    = 1.0                   # [m]
PEAK_BS         = 1.0                   # Longitudinal field [T]
X_START         = 1e-3                  # [m]
Y_START         = 0.5e-3                # [m]
PX_START        = 1e-3                  # Reference-normalized momentum

########################################
# Numerical representations
########################################
BORIS_STEPS     = 256
PROFILE_SLICES  = 64

################################################################################
# Supply a three-dimensional magnetic-field callable
################################################################################
# The transverse terms make div(B)=0 to this linear transverse order.
# This near-axis model is not an exact finite-coil vacuum field solution.
def solenoid_field(x, y, z):
    inside  = (z >= 0) & (z <= FIELD_LENGTH)
    phase   = np.pi * z / FIELD_LENGTH
    bs      = inside * PEAK_BS * np.sin(phase)**2
    dbs_dz  = inside * PEAK_BS * np.pi / FIELD_LENGTH * np.sin(2 * phase)
    return -0.5 * x * dbs_dz, -0.5 * y * dbs_dz, bs

initial = xt.Particles(
    'proton', p0c = P0C, x = X_START, y = Y_START, px = PX_START)
# Both endpoints are field-free: exterior canonical and kinetic momenta agree.

################################################################################
# Track through the public spatial Boris interface
################################################################################

########################################
# Record trajectories in the callable Cartesian frame
########################################
integrator = xt.BorisSpatialIntegrator(
    fieldmap_callable = solenoid_field,
    s_start           = 0.0,
    s_end             = FIELD_LENGTH,
    n_steps           = BORIS_STEPS,
)
integrator.log_trajectories = True
boris_particles = initial.copy()
integrator.track(boris_particles)

########################################
# Check one doubled-step calculation
########################################
refined = xt.BorisSpatialIntegrator(
    fieldmap_callable = solenoid_field,
    s_start           = 0.0,
    s_end             = FIELD_LENGTH,
    n_steps           = 2 * BORIS_STEPS,
)
refined_particles = initial.copy()
refined.track(refined_particles)
print('Boris step refinement: exit x/y differences [m]:',
      boris_particles.x - refined_particles.x,
      boris_particles.y - refined_particles.y)

################################################################################
# Compare with a sampled native longitudinal profile
################################################################################

########################################
# Retain physical field units at the conversion boundary
########################################
edges   = np.linspace(0, FIELD_LENGTH, PROFILE_SLICES + 1)
bs      = solenoid_field(0.0, 0.0, edges)[2]
ks      = bs / initial.rigidity0[0]
line = xt.Line(elements = [
    xt.VariableSolenoid(
        length = edges[i + 1] - edges[i],
        ks_profile = ks[i:i + 2], num_multipole_kicks = 4)
    for i in range(PROFILE_SLICES)
])
native_particles = initial.copy()
line.track(native_particles, turn_by_turn_monitor = 'ONE_TURN_EBE')
record = line.record_last_track
print('Sampled native minus refined Boris exit x/y [m]:',
      native_particles.x - refined_particles.x,
      native_particles.y - refined_particles.y)
# These are different finite field/integration representations. Step refinement
# above does not establish convergence of the sampled profile or symplecticity.

########################################
# Compare the physical trajectories
########################################
fig, axes = plt.subplots(1, 2, figsize = (10, 4), layout = 'constrained')
for axis, coordinate in zip(axes, ['x', 'y']):
    axis.plot(integrator.z_log[:, 0],
              getattr(integrator, coordinate + '_log')[:, 0] * 1e3,
              label = 'Spatial Boris')
    axis.plot(record.s[0], getattr(record, coordinate)[0] * 1e3,
              '.', label = 'Native sampled profile')
    axis.set(xlabel = 'Cartesian field coordinate z [m]',
             ylabel = coordinate + ' [mm]')
    axis.legend()
    axis.grid()
plt.show()
