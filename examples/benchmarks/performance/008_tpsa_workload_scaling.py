# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from importlib.metadata import version
from pathlib import Path
import platform
from time import perf_counter

import madng_tpsa
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
import xtrack.tpsa as xtpsa

################################################################################
# User parameters
################################################################################

########################################
# Independent workload scans
########################################
MODEL_FILE          = 'test_data/example_lattices/demo_ring/lattice.py'
CELL_COUNTS         = [1, 2, 4]         # Existing 40 m arc cells
PARAMETER_COUNTS    = [0, 1, 2, 4]      # Independent quadrupole k1 parameters
FIXED_CELLS         = 4
FIXED_PARAMETERS    = 1
MAP_ORDER           = 2                # Six-coordinate Taylor order
PARAMETER_ORDER     = 1
K2LF                = 1.0              # Focusing sextupole k2*L [m^-2]
REPEATS             = 5
WARMUPS             = 2

########################################
# Same entrance point and verification scales in every case
########################################
INITIAL     = dict(
    x       = 1e-4,
    px      = 2e-5,
    y       = -2e-4,
    py      = 3e-5,
    zeta    = 1e-3,
    delta   = 2e-3,
)
# Metres for x/y/zeta; normalized momenta for px/py/delta.
SCALES      = np.full(6, 1e-3)
ORBIT_TOL   = 1e-9                      # Scaled constant-part discrepancy
FD_STEP     = 1e-5                      # Quadrupole k1 perturbation [m^-2]
FD_RTOL     = 1e-5                      # Relative derivative-vector discrepancy

################################################################################
# Load the shared model and report the execution configuration
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / MODEL_FILE)
env['k2lf'] = K2LF
reference   = env['demo'].particle_ref
reference_args  = dict(
    p0c     = reference.p0c[0],
    mass0   = reference.mass0,
    q0      = reference.q0,
)
print('Python / NumPy / Xtrack / madng_tpsa:', platform.python_version(),
      np.__version__, xt.__version__, version('madng-tpsa'))
print('Platform:', platform.platform(), platform.machine())
print('Experimental native TPSA tracking; serial CPU; radiation off')
print('Map / parameter orders:', MAP_ORDER, PARAMETER_ORDER)
print('Reference p0*c [GeV] / q0 [e]:', reference.p0c[0] * 1e-9,
      reference.q0)
print('Entrance coordinates:', INITIAL)
print('Sextupole k2*L [m^-2]:', K2LF)

################################################################################
# Measure setup and transport without changing the nominal physics
################################################################################

########################################
# Prepare one existing arc segment and an independent scalar control
########################################
def measure_case(cells, parameters):
    segment = env['demo'].select(
        start   = 'arc.s1.c1.qf.in',
        end     = f'arc.s1.c{cells}.qf.out',
    )
    line        = segment.copy()
    scalar_line = segment.copy()
    line.config.XTRACK_MULTIPOLE_NO_SYNRAD          = True
    scalar_line.config.XTRACK_MULTIPOLE_NO_SYNRAD   = True
    line.build_tracker()
    scalar_line.build_tracker()
    # Vary independent element fields, not the shared focusing-family knob.
    # qf.in is one half-quadrupole at a cell boundary; qf.out stays nominal.
    quad_names  = [f'arc.s1.c{cell}.{part}' for cell in range(1, cells + 1)
                   for part in ['qf.in', 'qd']][:parameters]
    if len(quad_names) != parameters:
        raise ValueError('Not enough independent quadrupoles in this segment')
    strengths   = [float(scalar_line[name].k1) for name in quad_names]
    scalar      = xt.Particles(**reference_args, **INITIAL)
    scalar_line.track(scalar)
    expected    = np.array([getattr(scalar, coord)[0] for coord in INITIAL])

    ########################################
    # Rebind fresh descriptor parameters before constructing each identity map
    ########################################
    setup_times = []
    track_times = []
    errors      = []
    for repeat in range(WARMUPS + REPEATS):
        before      = perf_counter()
        descriptor  = madng_tpsa.Descriptor(
            6, MAP_ORDER,
            num_params  = parameters,
            param_order = PARAMETER_ORDER,
        )
        for index, (name, strength) in enumerate(zip(quad_names, strengths)):
            line[name].k1   = descriptor.param(index + 1, strength)
        probe   = xtpsa.ParticlesTpsa(
            order       = MAP_ORDER,
            descriptor  = descriptor,
            **reference_args, **INITIAL,
        )
        setup   = perf_counter() - before
        before  = perf_counter()
        line.track(probe)
        elapsed = perf_counter() - before
        error   = np.max(np.abs((probe.const_part - expected) / SCALES))
        if not np.isfinite(error) or error > ORBIT_TOL:
            raise RuntimeError('TPSA and independent scalar orbit disagree')
        if repeat >= WARMUPS:
            setup_times.append(setup)
            track_times.append(elapsed)
            errors.append(error)

    ########################################
    # Verify that every declared parameter actually drives its magnet
    ########################################
    # Central differences are outside timing; restore the nominal scalar field
    # after each check. The parameter derivative has units coordinate*m^2.
    for index, (name, strength) in enumerate(zip(quad_names, strengths)):
        exits   = []
        for sign in [-1, 1]:
            scalar_line[name].k1    = strength + sign * FD_STEP
            particle    = xt.Particles(**reference_args, **INITIAL)
            scalar_line.track(particle)
            exits.append(np.array([getattr(particle, coord)[0]
                                   for coord in INITIAL]))
        scalar_line[name].k1    = strength
        finite_difference   = (exits[1] - exits[0]) / (2 * FD_STEP)
        sensitivity         = np.array([probe.sensitivity(coord, index)
                                        for coord in INITIAL])
        signal      = np.linalg.norm(finite_difference / SCALES)
        residual    = np.linalg.norm(
            (sensitivity - finite_difference) / SCALES)
        if (not np.isfinite(signal) or not np.isfinite(residual) or signal == 0
                or residual > FD_RTOL * signal):
            raise RuntimeError('Active-parameter sensitivity check failed')
        print(f'  {name}: scaled derivative norm {signal:.3e}; '
              f'FD relative discrepancy {residual / signal:.3e}')
    return (cells, parameters, line.get_length(), len(line.element_names),
            setup_times, track_times, max(errors))

################################################################################
# Scan segment length and active-parameter count independently
################################################################################

########################################
# Increase length at fixed map order and parameter count
########################################
print('\nCell-count scan: fixed active parameters =', FIXED_PARAMETERS)
length_rows = [measure_case(cells, FIXED_PARAMETERS) for cells in CELL_COUNTS]

########################################
# Increase parameters at fixed map order and segment length
########################################
print('\nParameter-count scan: fixed cells =', FIXED_CELLS)
parameter_rows  = [measure_case(FIXED_CELLS, count)
                   for count in PARAMETER_COUNTS]

################################################################################
# Report the measured operations and their repeat spread
################################################################################
# Loading, segment copying, tracker setup, compilation and checks are untimed.
# Setup below includes descriptor creation, field binding and fresh-map input.
# Transport includes the Python call and native map tracking, not just algebra.
# Each repeat is fresh; warmups use the same descriptor dimensions as its case.
print('\nSetup and transport: median [min, max] over', REPEATS, 'repeats [ms]')
print('Untimed warmups per case:', WARMUPS)
for rows in [length_rows, parameter_rows]:
    for cells, parameters, length, elements, setup, tracking, error in rows:
        print(f'{cells} cells, {length:g} m, {elements} elements, '
              f'{parameters} active parameters; max scaled error {error:.3e}')
        for label, samples in [('Setup', setup), ('Transport', tracking)]:
            print(f'  {label}: {np.median(samples) * 1e3:.3f} '
                  f'[{min(samples) * 1e3:.3f}, {max(samples) * 1e3:.3f}]')

################################################################################
# Plot both independent workload scans
################################################################################
fig, axes   = plt.subplots(1, 2, figsize = (11, 4), layout = 'constrained')
fig.suptitle(f'Order {MAP_ORDER}: median and min/max of {REPEATS} fresh maps')
for axis, rows, column, xlabel, title in [
    (axes[0], length_rows, 0, 'Existing arc cells',
     f'{FIXED_PARAMETERS} active parameter(s)'),
    (axes[1], parameter_rows, 1, 'Independent active quadrupole parameters',
     f'{FIXED_CELLS} cells, {parameter_rows[0][2]:g} m'),
]:
    abscissa    = [row[column] for row in rows]
    for index, label in [
        (4, 'Descriptor, binding and input'), (5, 'Transport'),
    ]:
        medians = np.array([np.median(row[index]) for row in rows]) * 1e3
        lower   = medians - np.array([min(row[index]) for row in rows]) * 1e3
        upper   = np.array([max(row[index]) for row in rows]) * 1e3 - medians
        axis.errorbar(abscissa, medians, yerr = [lower, upper], fmt = 'o-',
                      capsize = 4, label = label)
    axis.set(xlabel = xlabel, ylabel = 'Time [ms]', title = title,
             yscale = 'log', xticks = abscissa)
    axis.grid(alpha = 0.3)
    axis.legend()
plt.show()
