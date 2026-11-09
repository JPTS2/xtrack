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
# Parent quadrupole and reference particle
########################################
QUAD_LENGTH     = 1.0                    # [m]
K1              = 0.7                    # [m^-2]
P0C             = 1.0e9                  # Proton reference p0*c [eV]
N_SLICES        = 3

########################################
# Common entrance coordinates
########################################
X_START         = np.linspace(-1e-3, 1e-3, 5)  # [m]
PX_START        = 2.0e-4                 # P_x / p0
Y_START         = -0.7e-3                # [m]
PY_START        = 1.0e-4                 # P_y / p0
DELTA           = 1.0e-3                 # (p - p0) / p0

################################################################################
# Make independent thin and thick representations of one parent
################################################################################

########################################
# Build the unsliced parent
########################################
env = xt.Environment()
parent = env.new_line(
    components  = [
        env.new(
            'q', xt.Quadrupole,
            length              = QUAD_LENGTH,
            k1                  = K1,
            edge_entry_active   = False,
            edge_exit_active    = False,
        ),
    ],
)
parent.set_particle_ref('proton', p0c = P0C)

########################################
# Slice independent copies
########################################
representations = {'parent': parent}
for mode in ['thin', 'thick']:
    sliced = parent.copy()
    sliced.slice_thick_elements(
        slicing_strategies  = [xt.Strategy(xt.Teapot(N_SLICES, mode = mode))],
    )
    representations[mode] = sliced

# Thick slices retain the adaptive parent model; thin slices use kicks/drifts.
# Fringe fields, RF and radiation are disabled.
print('Parent model/integrator:', env['q'].model, env['q'].integrator)

################################################################################
# Inspect parent ownership, geometric lengths and integrated strength
################################################################################
# Thin kicks carry integrated strength at zero geometric length; drift slices
# advance s. Use s_end-s_start for geometry, rather than the attribute length.
# In thin mode, kick weights and drift weights each sum to one.
for label, line in representations.items():
    table = line.get_table(attr = True)
    table['L_geom'] = table.s_end - table.s_start
    print(label, 'representation; s and L_geom [m], k1l [m^-1]:')
    table.cols['name s parent_name weight L_geom k1l'].show()
    print('Length [m]:', line.get_length(),
          'integrated k1l [m^-1]:', sum(table.k1l))

################################################################################
# Track identical entrance particles once through each representation
################################################################################

########################################
# Prepare the common entrance particles
########################################
initial = parent.build_particles(
    x       = X_START,
    px      = PX_START,
    y       = Y_START,
    py      = PY_START,
    delta   = DELTA,
)

########################################
# Track each representation
########################################
outputs = {}
for label, line in representations.items():
    particles = initial.copy()
    line.track(particles)
    outputs[label] = particles

################################################################################
# Compare transverse exit coordinates with the unsliced parent
################################################################################
# Compare maps at this slice count, using the unsliced adaptive map as the
# reference. A convergence study requires varying the numerical settings.
fig, axes = plt.subplots(2, 2, sharex = True, figsize = (10, 6))
for axis, coordinate, unit in zip(
    axes.flat, ['x', 'px', 'y', 'py'], ['um', '1e-6', 'um', '1e-6']
):
    for label in ['thin', 'thick']:
        residual = getattr(outputs[label], coordinate) - getattr(
            outputs['parent'], coordinate)
        print(label, coordinate, 'maximum absolute exit residual:',
              np.max(np.abs(residual)),
              'm' if coordinate in ['x', 'y'] else 'normalized momentum')
        axis.plot(1e3 * X_START, 1e6 * residual, '.-', label = label)
    axis.set_ylabel(coordinate + ' residual [' + unit + ']')
    axis.grid()
    axis.legend()
for axis in axes[-1]:
    axis.set_xlabel('Entrance x [mm]')
fig.suptitle(f'{N_SLICES} slices: exit coordinates minus unsliced parent')
fig.tight_layout()
plt.show()
