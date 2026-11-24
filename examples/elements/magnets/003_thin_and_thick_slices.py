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
P0C             = 1.0e9                  # Proton reference momentum [eV/c]
N_SLICES        = 3
SLICE_COUNTS    = [1, 4, 16, 64, 256]

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
        with_progress       = False,
    )
    representations[mode] = sliced

# Thick slices retain the adaptive parent model; thin slices use kicks/drifts.
# Fringe fields, RF and radiation are disabled.
print('Thin and thick representations\n------------------------------')
print('Parent model/integrator:', env['q'].model, env['q'].integrator)
print('Thin kicks have zero geometric length; drift slices advance s.')

################################################################################
# Inspect parent ownership, geometric lengths and integrated strength
################################################################################
# Thin kicks carry integrated strength at zero geometric length; drift slices
# advance s. Use s_end-s_start for geometry, rather than the attribute length.
# In thin mode, kick weights and drift weights each sum to one.
for label, line in representations.items():
    table = line.get_table(attr = True)
    table['L_geom'] = table.s_end - table.s_start
    print(f'\n{label.capitalize()} representation:')
    print('s and L_geom [m]; k1l [m^-1].')
    table.cols['name s parent_name weight L_geom k1l'].show(maxwidth = 'full')
    print(f'Length [m]: {line.get_length():.6f}; '
          f'integrated k1l [m^-1]: {sum(table.k1l):.6f}')

################################################################################
# Select slicing rules and inspect inherited attributes
################################################################################

########################################
# Combine a default, class rules and name overrides
########################################
# The last matching rule wins. A None scheme deliberately leaves a magnet
# unsliced; occurrence names distinguish two uses of the same definition.
selection_env = xt.Environment()
# Absolute integrated error arrays add to the distributed normal/skew fields.
selection_env.new(
    'q', xt.Quadrupole,
    length      = QUAD_LENGTH,
    k1          = K1,
    k1s         = 0.05,
    knl         = [0.0, 0.02, 0.03],
    ksl         = [0.0, 0.01, 0.02],
    shift_x     = 1e-3,
    rot_s_rad   = 2e-3,
)
selection_env.new('keep', xt.Quadrupole, length = QUAD_LENGTH, k1 = -K1)
selection_env.new(
    's', xt.Sextupole,
    length      = QUAD_LENGTH,
    k2          = 0.6,
    k2s         = 0.1,
    shift_y     = 2e-3,
)
selected = selection_env.new_line(components = ['q', 'q', 'keep', 's'])
print('\nSelect slicing rules\n--------------------')
print('The last matching rule wins: q::0 is thin, q::1 is thick,')
print('keep stays unsliced, and s is thin.')
selected.slice_thick_elements(
    slicing_strategies  = [
        xt.Strategy(slicing = None),
        xt.Strategy(xt.Teapot(2), element_type = xt.Quadrupole),
        xt.Strategy(xt.Teapot(2), element_type = xt.Sextupole),
        xt.Strategy(xt.Teapot(3, mode = 'thick'), name = 'q::1'),
        xt.Strategy(slicing = None, name = 'keep'),
    ],
    with_progress       = False,
)

########################################
# Inspect strengths and misalignments inherited from parents
########################################
selection_table = selected.get_table(attr = True)
magnetic = selection_table.rows[
    (selection_table.k1l != 0) | (selection_table.k2l != 0)]
print('Magnetic bodies and kicks (markers and drifts omitted):')
print('k1l/k1sl [m^-1]; k2l/k2sl [m^-2].')
magnetic.cols['name parent_name weight k1l k1sl k2l k2sl'].show(
    maxwidth = 'full')
print('\nInherited alignment: shift_x/shift_y [m]; rot_s_rad [rad].')
magnetic.cols['name shift_x shift_y rot_s_rad'].show()
print(f'Sliced sequence length [m]: {selected.get_length():.6f}')

########################################
# Compare parent settings with both sliced occurrences
########################################
parent_knl, parent_ksl = selection_env['q'].get_total_knl_ksl()
q_slices = selection_table.rows[selection_table.parent_name == 'q']
print('\nPreserve both occurrences of q\n------------------------------')
print('Distributed fields and absolute errors are weighted by each slice.')
print('Integral                     Two parents   Sum of slices')
for label, expected, actual in [
    ('Normal quadrupole [m^-1]', 2 * parent_knl[1], sum(q_slices.k1l)),
    ('Skew quadrupole [m^-1]', 2 * parent_ksl[1], sum(q_slices.k1sl)),
    ('Normal sextupole [m^-2]', 2 * parent_knl[2], sum(q_slices.k2l)),
    ('Skew sextupole [m^-2]', 2 * parent_ksl[2], sum(q_slices.k2sl)),
]:
    print(f'{label:<28} {expected:11.6f} {actual:15.6f}')

################################################################################
# Prepare identical entrance probes and track the unsliced reference
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
# Track the unsliced reference
########################################
reference = initial.copy()
parent.track(reference)

################################################################################
# Refine the slice count at fixed fields and entrance coordinates
################################################################################

########################################
# Measure maximum exit errors against the unsliced parent
########################################
# Position and normalized-momentum errors are kept separate, with no division
# by an exit coordinate that could vanish. Maxima include all entrance probes.
errors = {}
for mode in ['thin', 'thick']:
    errors[mode] = []
    for count in SLICE_COUNTS:
        sliced = parent.copy()
        sliced.slice_thick_elements(
            slicing_strategies  = [xt.Strategy(xt.Teapot(count, mode = mode))],
            with_progress       = False,
        )
        particles = initial.copy()
        sliced.track(particles)
        position_error = max(
            np.max(np.abs(particles.x - reference.x)),
            np.max(np.abs(particles.y - reference.y)),
        )
        momentum_error = max(
            np.max(np.abs(particles.px - reference.px)),
            np.max(np.abs(particles.py - reference.py)),
        )
        errors[mode].append((position_error, momentum_error))
    errors[mode] = np.array(errors[mode])

print('\nRefine the slice count\n----------------------')
print('Maximum errors over both transverse planes and all entrance probes.')
print('Reference: the same unsliced, fringe-free quadrupole.')
print('Slices      Thin position [m]  Thick position [m]'
      '    Thin momentum [1]  Thick momentum [1]')
for index, count in enumerate(SLICE_COUNTS):
    print(f'{count:6d} {errors["thin"][index, 0]:22.3e}'
          f'{errors["thick"][index, 0]:20.3e}'
          f'{errors["thin"][index, 1]:21.3e}'
          f'{errors["thick"][index, 1]:20.3e}')
print('Thin slicing approximates the body map; refinement reduces that error.')
print('Thick slices retain the uniform quadrupole map: splitting adds no')
print('accuracy here, and extra operations accumulate floating-point error.')
print('The one-slice thick errors are zero and cannot appear on a log axis.')

########################################
# Plot convergence and the round-off floor
########################################
fig, axes = plt.subplots(1, 2, figsize = (10, 4), sharex = True)
for index, (axis, label) in enumerate(zip(
    axes, ['Maximum position error [m]', 'Maximum momentum error [1]']
)):
    for mode in ['thin', 'thick']:
        axis.loglog(SLICE_COUNTS, errors[mode][:, index], '.-', label = mode)
    axis.set_xlabel('Number of slices')
    axis.set_ylabel(label)
    axis.set_xticks([1, 4, 16, 64, 256], labels = [1, 4, 16, 64, 256])
    axis.grid(which = 'both', alpha = 0.3)
    axis.legend()
fig.suptitle('Quadrupole slicing: convergence to the unsliced map')
fig.tight_layout()
plt.show()
