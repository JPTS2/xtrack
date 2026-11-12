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
# Fixed electron reference and bend geometry
########################################
P0C             = 10e9                  # [eV/c]
LENGTH          = 1.0                   # [m]
ANGLE           = 0.05                  # [rad]

########################################
# Mean-radiation integration resolution
########################################
KICK_COUNTS     = [1, 2, 4, 8, 16, 32, 64]
REFERENCE_KICKS = 1024

################################################################################
# Prepare identical physical launches and a radiating sector bend
################################################################################
line    = xt.Line(elements = {
    'bend': xt.Bend(
        length                  = LENGTH,
        angle                   = ANGLE,
        model                   = 'bend-kick-bend',
        integrator              = 'uniform',
        edge_entry_active       = False,
        edge_exit_active        = False,
    ),
})
line.set_particle_ref('electron', p0c = P0C)
line.configure_radiation(model = 'mean')
initial = line.build_particles(x = 0.0, delta = 0.0)
# Geometry and normalized fields stay fixed. Only the number of interleaved
# radiation/multipole kicks changes; no RF restores energy in this single pass.

################################################################################
# Refine the numerical reference before using it to measure errors
################################################################################
references  = []
for count in [REFERENCE_KICKS, 2 * REFERENCE_KICKS]:
    line['bend'].num_multipole_kicks    = count
    particles   = initial.copy()
    line.track(particles)
    references.append(particles)
reference       = references[-1]
reference_loss  = initial.energy[0] - reference.energy[0]
loss_refinement = abs(references[0].energy[0] - reference.energy[0])
orbit_refinement    = abs(references[0].x[0] - reference.x[0])
print('Reference energy loss [eV]:', reference_loss)
print('Change on doubling reference kicks:')
print('  Energy [eV]:', references[0].energy[0] - reference.energy[0])
print('  x [m], px:', references[0].x[0] - reference.x[0],
      references[0].px[0] - reference.px[0])

################################################################################
# Measure energy-loss and orbit convergence together
################################################################################
losses      = []
orbit_error = []
for count in KICK_COUNTS:
    line['bend'].num_multipole_kicks    = count
    particles   = initial.copy()
    line.track(particles)
    losses.append(initial.energy[0] - particles.energy[0])
    orbit_error.append(abs(particles.x[0] - reference.x[0]))
    print('Kicks / loss [eV] / x error [m] / px error / state:',
          count, losses[-1], orbit_error[-1],
          particles.px[0] - reference.px[0], particles.state[0])

################################################################################
# Distinguish a converged energy loss from a converged trajectory
################################################################################
loss_error  = np.abs(np.array(losses) - reference_loss)
fig, axes   = plt.subplots(1, 2, figsize = (10, 4))
axes[0].loglog(KICK_COUNTS, loss_error, 'o-')
axes[0].axhline(loss_refinement, color = 'k', linestyle = ':',
                label = 'Reference refinement change')
axes[0].set(xlabel = 'Radiation/multipole kicks',
            ylabel = '|Loss - refined reference| [eV]')
axes[1].loglog(KICK_COUNTS, orbit_error, 'o-')
axes[1].axhline(orbit_refinement, color = 'k', linestyle = ':',
                label = 'Reference refinement change')
axes[1].set(xlabel = 'Radiation/multipole kicks',
            ylabel = '|x - refined reference| [m]')
for axis in axes:
    axis.grid(which = 'both', alpha = 0.3)
    axis.legend()
fig.tight_layout()
plt.show()
# This measures convergence within one native mean-radiation model. It does
# not validate the radiation law independently or test stochastic fluctuations.
