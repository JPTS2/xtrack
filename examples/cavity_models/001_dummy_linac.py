# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Synthetic RF cells and separate quadrupoles: tracking-derived optics.

Every case computes Twiss and tracks both 9×9 grids. P16 cases traverse the
two-cell linac with one prescribed momentum/clock programme. Uniform cases
are separate single-cavity maps and never enter downstream drifts or quads.
Continuous and averaged uniform maps are observed inside the RF field, with
native canonical Twiss functions. The frozen voltage-action faces and averaging
are distinct approximations. Radiation is off; numerical mesh differences
are not physical error bounds.
The grids share canonical launches; different boundary assumptions can give
different mechanical slopes at those same canonical coordinates.
"""

################################################################################
# Required packages
################################################################################
from time import perf_counter
example_started = perf_counter()

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt


################################################################################
# User parameters
################################################################################

########################################
# RF cells, reference and external quads
########################################
LENGTH_M          = .16
FREQUENCY_HZ      = 400e6
VOLTAGE_V         = 20e3
PHASE_RAD         = .4
GAMMA0            = 2.4
QUAD_LENGTH_M     = .06
QUAD_STRENGTH     = 3.             # 1/m²; external alternating quadrupoles
DRIFT_LENGTH_M    = .15
SOURCE_CELLS      = 16             # full/H2/longitudinal Gauss cells
AVERAGED_STEPS    = 4              # uniform quadratic body midpoint steps
BOUNDARY_STEPS    = 1              # averaged boundary midpoint steps
FROZEN_CELLS      = 0              # model-specific geometric word
GRID_SIZE         = 9
X_LIMIT_M         = .005
Y_LIMIT_M         = .005
PX_LIMIT          = .03

########################################
# Explicit source/transport assumptions
########################################
COHORTS = [
    ('p16', .25, .3, .9, [
        'phase-resolved', 'phase-resolved-paraxial',
        'phase-resolved-longitudinal']),
    ('uniform', 0., 0., 1., [
        'phase-resolved', 'phase-resolved-frozen']),
    ('uniform', .25, .3, .9, [
        'phase-resolved',
        'phase-averaged-uniform-quadratic-first-boundary',
        'phase-averaged-uniform-quadratic']),
]
INTEGRATORS = {
    'phase-resolved': 'gauss2',
    'phase-resolved-paraxial': 'gauss2',
    'phase-resolved-longitudinal': 'gauss2',
    'phase-resolved-frozen': 'geometric-kick-drift',
    'phase-averaged-uniform-quadratic-first-boundary': 'midpoint',
    'phase-averaged-uniform-quadratic': 'midpoint',
}
LABELS = {
    'phase-resolved': 'Full resolved',
    'phase-resolved-paraxial': 'Paraxial resolved',
    'phase-resolved-longitudinal': 'Longitudinal RF',
    'phase-resolved-frozen': 'Frozen coefficients and faces',
    'phase-averaged-uniform-quadratic-first-boundary': 'Quadratic averaged, W1',
    'phase-averaged-uniform-quadratic': 'Quadratic averaged, W1 + W2',
}


################################################################################
# One Line with a prescribed common momentum and RF-clock programme
################################################################################
context = xo.ContextCpu(omp_num_threads=0)
mass0 = xt.ELECTRON_MASS_EV
p0c = mass0*np.sqrt(GAMMA0**2-1.)
entrance_p0c = p0c
nominal_time = 0.
elements, names, rf_names = [], [], []
for cell in range(2):
    rf_name = f'rf{cell}'
    elements.append(xt.Cavity(model='phase-resolved', rf_profile='p16',
        length=LENGTH_M, voltage=VOLTAGE_V, frequency=FREQUENCY_HZ,
        phase=PHASE_RAD-2*np.pi*FREQUENCY_HZ*nominal_time,
        absolute_time=True, eta=.25, counter_phase=.3, calibration_beta=.9,
        num_kicks=SOURCE_CELLS))
    names.append(rf_name)
    rf_names.append(rf_name)
    nominal_time += LENGTH_M/(p0c/np.hypot(p0c, mass0)*299792458.)
    nominal_gain = VOLTAGE_V*np.sin(PHASE_RAD)
    energy_after = np.hypot(p0c, mass0)+nominal_gain
    momentum_after = np.sqrt((energy_after-mass0)*(energy_after+mass0))
    elements[-1].nominal_energy_gain = nominal_gain
    elements.append(xt.ReferenceEnergyIncrease(Delta_p0c=momentum_after-p0c))
    names.append(f'reference{cell}')
    p0c = momentum_after
    for label, element in [
            ('drift_a', xt.Drift(length=DRIFT_LENGTH_M)),
            ('quad', xt.Quadrupole(length=QUAD_LENGTH_M,
                                  k1=QUAD_STRENGTH if cell == 0 else -QUAD_STRENGTH)),
            ('drift_b', xt.Drift(length=DRIFT_LENGTH_M))]:
        elements.append(element)
        names.append(f'{label}{cell}')
        nominal_time += element.length/(p0c/np.hypot(p0c, mass0)*299792458.)
elements.append(xt.Marker())
names.append('physical_exit')
line = xt.Line(elements=elements, element_names=names)
line.particle_ref = xt.Particles(mass0=mass0, q0=1., p0c=entrance_p0c)
line.reset_s_at_end_turn = False
line.use_time_based_zeta_update = True
build_started = perf_counter()
line.build_tracker(_context=context)
build_seconds = perf_counter()-build_started


################################################################################
# Both launch grids, tracked together as 162 particles
################################################################################
x_axis = np.linspace(-X_LIMIT_M, X_LIMIT_M, GRID_SIZE)
y_axis = np.linspace(-Y_LIMIT_M, Y_LIMIT_M, GRID_SIZE)
px_axis = np.linspace(-PX_LIMIT, PX_LIMIT, GRID_SIZE)
xy_x, xy_y = np.meshgrid(x_axis, y_axis, indexing='ij')
xpx_x, xpx_px = np.meshgrid(x_axis, px_axis, indexing='ij')
grid_count = GRID_SIZE**2
launch = dict(x=np.r_[xy_x.ravel(), xpx_x.ravel()],
    y=np.r_[xy_y.ravel(), np.zeros(grid_count)],
    px=np.r_[np.zeros(grid_count), xpx_px.ravel()],
    py=np.zeros(2*grid_count), zeta=np.zeros(2*grid_count), delta=np.zeros(2*grid_count))
timings = []


################################################################################
# Tracking-derived Twiss and the two transported grids
################################################################################
for profile, eta, counter_phase, beta_cal, models in COHORTS:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    # Uniform sources have nonzero RF potentials at their observation planes.
    # Compare this interval without pretending it has field-free interfaces
    # to the rest of the linac. The already compiled Line serves both ranges.
    linac_case = profile == 'p16'
    stop = 'physical_exit' if linac_case else 'reference0'
    twiss_range = dict(start=xt.START, end=xt.END) if linac_case else dict(
        start='rf0', end='rf0')
    for model in models:
        for name in rf_names:
            cavity = line[name]
            cavity.model = model
            cavity.rf_profile = profile
            cavity.eta = eta
            cavity.counter_phase = counter_phase
            cavity.calibration_beta = beta_cal
            cavity.integrator = INTEGRATORS[model]
            cavity.num_factor_steps = BOUNDARY_STEPS
            if model == 'phase-resolved-frozen':
                cavity.num_kicks = FROZEN_CELLS
            elif model.startswith('phase-averaged'):
                cavity.num_kicks = AVERAGED_STEPS
            else:
                cavity.num_kicks = SOURCE_CELLS
        twiss_started = perf_counter()
        twiss = line.twiss(method='6d', **twiss_range,
                          betx=2., bety=3., alfx=0., alfy=0.)
        twiss_seconds = perf_counter()-twiss_started
        particles = xt.Particles(_context=context, mass0=mass0, q0=1.,
                                  p0c=entrance_p0c, **launch)
        tracking_started = perf_counter()
        line.track(particles, ele_start=0, ele_stop=stop)
        tracking_seconds = perf_counter()-tracking_started
        assert np.all(particles.state == 1)
        recovered = particles.copy()
        line.track(recovered, ele_start=0, ele_stop=stop, backtrack=True)
        assert np.all(recovered.state == 1)
        reverse_gaps = {name: float(np.max(abs(getattr(recovered, name)-values)))
                        for name, values in launch.items()}
        timings.append(dict(profile=profile, model=model,
            twiss_seconds=twiss_seconds, track_seconds=tracking_seconds,
            reverse_gaps=reverse_gaps))
        label = LABELS[model]
        colour = axes[0, 0].plot(twiss.s, twiss.betx, label=label)[0].get_color()
        axes[0, 1].plot(twiss.s, twiss.bety, color=colour, label=label)
        axes[1, 0].scatter(1e3*particles.x[:grid_count],
                           1e3*particles.y[:grid_count], s=8, color=colour, label=label)
        axes[1, 1].scatter(1e3*particles.x[grid_count:],
                           1e3*particles.px[grid_count:], s=8, color=colour, label=label)
        for axis, first, second in (
                (axes[1, 0], particles.x[:grid_count], particles.y[:grid_count]),
                (axes[1, 1], particles.x[grid_count:], particles.px[grid_count:])):
            first = 1e3*np.asarray(first).reshape(GRID_SIZE, GRID_SIZE)
            second = 1e3*np.asarray(second).reshape(GRID_SIZE, GRID_SIZE)
            axis.plot(first, second, color=colour, linewidth=.5, alpha=.4)
            axis.plot(first.T, second.T, color=colour, linewidth=.5, alpha=.4)
        print(profile, model, INTEGRATORS[model],
              'exit beta [m]:', twiss.betx[-1], twiss.bety[-1])
    axes[0, 0].set(xlabel='s [m]', ylabel=r'$\beta_x$ [m]')
    axes[0, 1].set(xlabel='s [m]', ylabel=r'$\beta_y$ [m]')
    axes[1, 0].set(xlabel='x [mm]', ylabel='y [mm]', title='Transported x-y grid')
    axes[1, 1].set(xlabel='x [mm]', ylabel=r'$10^3 p_x$ [normalized canonical]', title='Transported x-px grid')
    for axis in axes.ravel():
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    geometry = ('Two-cell linac' if linac_case else
                'Single RF interval; native canonical observation planes')
    figure.suptitle(geometry+f'\n{profile} source; eta={eta:g}, '
                    f'calibration beta={beta_cal:g}')

print('Cold build [s]:', build_seconds)
print('Twiss and tracking timings [s]:', timings)
print('Total including build and figures [s]:', perf_counter()-example_started)
plt.show()
