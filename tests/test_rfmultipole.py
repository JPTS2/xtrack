from math import factorial

import numpy as np
import pytest
from scipy.constants import c
from scipy.integrate import solve_ivp

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts
from xtrack.linear_normal_form import S


@for_all_test_contexts
@pytest.mark.parametrize('components', ['normal', 'skew', 'both'])
@pytest.mark.parametrize('q0,charge_ratio,mass_ratio', [
    (1, 1., 1.), (2, 1., 1.), (-2, 1., 1.), (82, 0.5, 1.25)])
def test_rfmultipole_kicks_and_symplecticity(
        test_context, components, q0, charge_ratio, mass_ratio):
    element = xt.RFMultipole(knl=np.zeros(6), ksl=np.zeros(6),
                            frequency=400e6, voltage=2e6, phase=0.8)
    line = xt.Line(elements=[element])
    line.build_tracker(_context=test_context)
    p0 = xt.Particles(_context=test_context, p0c=1e9, q0=q0,
                      charge_ratio=charge_ratio, mass_ratio=mass_ratio,
                      x=0.13, y=-0.09, px=1e-4, py=-2e-4,
                      zeta=0.05, delta=1e-3)
    beta0 = float(p0.beta0[0])
    chi = charge_ratio / mass_ratio
    k = 2 * np.pi * element.frequency / c
    w = 0.13 - 0.09j
    phase_n = 0.6 - k * 0.05 / beta0
    phase_s = -0.4 - k * 0.05 / beta0
    cavity_kick = (abs(q0) * chi * element.voltage / (beta0 * 1e9)
                   * np.sin(element.phase - k * 0.05 / beta0))

    for order in range(6):
        bn = 0.1 * factorial(order) if components != 'skew' else 0.
        an = -0.07 * factorial(order) if components != 'normal' else 0.
        element.knl[:] = 0
        element.ksl[:] = 0
        element.knl[order] = bn
        element.ksl[order] = an
        element.phase_n[order] = 0.6
        element.phase_s[order] = -0.4

        # All three kicks are derivatives of the same multipole potential,
        # proportional to (x + iy)**(order + 1) / (order + 1)!.
        transverse = w**order / factorial(order)
        potential = w**(order + 1) / factorial(order + 1)
        dpx = -chi * (bn * np.cos(phase_n) * transverse.real
                      - an * np.cos(phase_s) * transverse.imag)
        dpy = chi * (bn * np.cos(phase_n) * transverse.imag
                     + an * np.cos(phase_s) * transverse.real)
        dpzeta = -chi * k / beta0 * (
            bn * np.sin(phase_n) * potential.real
            - an * np.sin(phase_s) * potential.imag)

        p = p0.copy()
        line.track(p)
        xo.assert_allclose(p.px, p0.px + dpx, rtol=1e-12, atol=1e-15)
        xo.assert_allclose(p.py, p0.py + dpy, rtol=1e-12, atol=1e-15)
        xo.assert_allclose(p.pzeta, p0.pzeta + dpzeta + cavity_kick,
                           rtol=1e-12, atol=1e-15)

        matrix = line.get_R_matrix(particle_on_co=p0.copy())['R_matrix']
        xo.assert_allclose(matrix.T @ S @ matrix, S, rtol=0, atol=2e-8)

        line.track(p, backtrack=True)
        for field in ['x', 'px', 'y', 'py', 'zeta', 'pzeta']:
            xo.assert_allclose(getattr(p, field), getattr(p0, field),
                               rtol=0, atol=1e-14)


_NAMES = ('x', 'px', 'y', 'py', 'zeta', 'delta')
_LAUNCH = dict(x=2e-3, px=3e-4, y=-1e-3, py=2e-4, zeta=.05, delta=1e-3)
_THICK = dict(length=2., frequency=400e6, voltage=3e5, phase=.4,
              knl=[1e-3, .05, .8], ksl=[1e-3, .02, .1],
              phase_n=[.2, .7, -.4], phase_s=[.1, .2, .3])


def _proton(context, **coordinates):
    return xt.Particles(_context=context, p0c=1e9, mass0=xt.PROTON_MASS_EV,
                        **coordinates)


def _end(element, context, launch=_LAUNCH):
    probe = _proton(context, **launch)
    element.track(probe)
    probe.move(_context=xo.context_default)
    assert probe.state[0] == 1
    return np.array([getattr(probe, name)[0] for name in _NAMES])


@for_all_test_contexts
def test_rfmultipole_zero_length_is_the_thin_kick(test_context):
    rf = xt.RFMultipole(_context=test_context, knl=[1e-4, 2e-2], ksl=[3e-5, 0.],
                        frequency=400e6, voltage=1e5, phase=.3)
    assert rf.length == 0 and not rf.isthick
    end = _end(rf, test_context)
    # A thin kick leaves the positions unchanged.
    assert (end[0], end[2], end[4]) == (_LAUNCH['x'], _LAUNCH['y'], _LAUNCH['zeta'])
    short = xt.RFMultipole(_context=test_context, knl=[1e-4, 2e-2], ksl=[3e-5, 0.],
                           frequency=400e6, voltage=1e5, phase=.3,
                           length=1e-6, num_kicks=3)
    assert short.isthick
    np.testing.assert_allclose(_end(short, test_context), end, rtol=0, atol=1e-9)


@for_all_test_contexts
def test_rfmultipole_absolute_time_adds_the_clock_phase(test_context):
    time_s, s0, frequency, p0c = 1.3e-9, .7, 400e6, 6e9
    beta0 = xt.Particles(p0c=p0c).beta0[0]
    clock = 2 * np.pi * frequency * (time_s + s0 / (beta0 * c))
    common = dict(_context=test_context, knl=[1e-4, 2e-2], ksl=[3e-5, 1e-3],
                  frequency=frequency, voltage=1e5)
    absolute = xt.RFMultipole(absolute_time=1, phase=.3, phase_n=[.1, -.2],
                              phase_s=[.4, .5], **common)
    shifted = xt.RFMultipole(phase=.3 + clock, phase_n=[.1 + clock, -.2 + clock],
                             phase_s=[.4 + clock, .5 + clock], **common)
    probes = []
    for element in (absolute, shifted):
        probe = xt.Particles(_context=test_context, p0c=p0c, x=1e-3, y=-2e-3,
                             px=1e-5, py=2e-5, zeta=.02, delta=1e-3, s=s0)
        probe.time_s = test_context.nparray_to_context_array(np.array([time_s]))
        element.track(probe)
        probe.move(_context=xo.ContextCpu())
        probes.append(np.array([getattr(probe, name)[0] for name in _NAMES]))
    np.testing.assert_allclose(probes[0], probes[1], rtol=1e-12, atol=1e-15)


@for_all_test_contexts
def test_thick_rfmultipole_static_limit_is_a_thick_quadrupole(test_context):
    """At zero frequency the distributed kicks converge to the exact body."""
    launch = dict(x=2e-3, px=3e-4, y=-1e-3, py=2e-4)
    exact = _end(xt.Quadrupole(_context=test_context, length=1., k1=.3,
                               model='mat-kick-mat'), test_context, launch)
    errors = []
    for kicks in (16, 64):
        rf = xt.RFMultipole(_context=test_context, length=1., frequency=0.,
                            knl=[0., .3], ksl=[0., 0.], num_kicks=kicks,
                            integrator='uniform', model='drift-kick-drift-expanded')
        errors.append(np.abs((_end(rf, test_context, launch) - exact)[:4]).max())
    assert errors[0] < 2e-7
    assert 12 < errors[0]/errors[1] < 20  # second order in the kick spacing


@pytest.mark.parametrize('integrator,order', [('uniform', 2.), ('yoshida-4', 4.)])
def test_thick_rfmultipole_converges_to_the_distributed_hamiltonian(integrator, order):
    """Against an independent integration of the distributed kicks."""
    length, knl, phases = 2., [0., .05, .8], [.2, .7, -.4]
    k = 2*np.pi*400e6/c
    reference = _proton(xo.context_default)
    beta0 = reference.beta0[0]

    def rates(s, state):
        x, px, y, py, zeta, ptau = state
        pz = np.sqrt(1 + 2*ptau/beta0 + ptau**2 - px**2 - py**2)
        dpx = dpy = potential = 0.
        re, im, factorial = 1., 0., 1.
        for n, (strength, phase) in enumerate(zip(knl, phases)):
            factorial *= max(n, 1)
            per_metre = strength/length/factorial
            angle = phase - k*zeta/beta0
            dpx -= np.cos(angle)*per_metre*re
            dpy += np.cos(angle)*per_metre*im
            re, im = re*x - im*y, im*x + re*y
            potential += np.sin(angle)*per_metre*re/(n + 1)
        return [px/pz, dpx, py/pz, dpy, 1 - (1 + beta0*ptau)/pz, -k*potential]

    start = _proton(xo.context_default, **_LAUNCH)
    state = [start.x[0], start.px[0], start.y[0], start.py[0], start.zeta[0], start.ptau[0]]
    truth = solve_ivp(rates, (0., length), state, method='DOP853',
                      rtol=1e-13, atol=1e-15).y[:, -1]
    errors = []
    for kicks in (8, 16):
        rf = xt.RFMultipole(length=length, frequency=400e6, knl=knl, ksl=[0., 0., 0.],
                            phase_n=phases, phase_s=[0., 0., 0.], num_kicks=kicks,
                            integrator=integrator, model='drift-kick-drift-exact')
        probe = _proton(xo.context_default, **_LAUNCH)
        rf.track(probe)
        errors.append(np.abs(np.array([probe.x[0], probe.px[0], probe.y[0], probe.py[0],
                                       probe.zeta[0], probe.ptau[0]]) - truth).max())
    assert errors[1] < 1e-7
    assert abs(np.log2(errors[0]/errors[1]) - order) < .3


@for_all_test_contexts
def test_thick_order_zero_rfmultipole_equals_thick_crab_cavity(test_context):
    common = dict(_context=test_context, length=1., frequency=400e6, num_kicks=5,
                  integrator='uniform')
    crab = xt.CrabCavity(crab_voltage=2e6, phase=.3, **common)
    rf = xt.RFMultipole(knl=[2e6/1e9], ksl=[0.], phase_n=[.3 + np.pi/2], phase_s=[0.],
                        **common)
    np.testing.assert_array_equal(_end(crab, test_context), _end(rf, test_context))


@for_all_test_contexts
def test_misaligned_thick_rfmultipole_uses_its_length(test_context):
    misalignment = dict(shift_x=1e-3, shift_s=.01, rot_y_rad=2e-3, rot_x_rad=-1e-3,
                        rot_s_rad=.1, rot_shift_anchor=.2)
    common = dict(_context=test_context, length=.5, frequency=4e8, num_kicks=5,
                  **misalignment)
    multipole = xt.RFMultipole(knl=[1e-5], phase_n=[np.pi / 2], **common)
    crab_voltage = 1e-5 * 1e9
    crab = xt.CrabCavity(crab_voltage=crab_voltage, **common)
    np.testing.assert_allclose(_end(multipole, test_context), _end(crab, test_context),
                               rtol=0, atol=1e-14)


@for_all_test_contexts
def test_thick_rfmultipole_is_symplectic_and_reverses(test_context):
    line = xt.Line(elements=[xt.RFMultipole(_context=test_context, num_kicks=7,
        integrator='yoshida-4', model='drift-kick-drift-exact', **_THICK)])
    line.particle_ref = _proton(xo.context_default)
    line.build_tracker(_context=test_context)
    names = ('x', 'px', 'y', 'py', 'zeta', 'pzeta')
    base = dict(x=2e-3, px=3e-4, y=-1e-3, py=2e-4, zeta=.05, pzeta=1e-3)
    matrix = np.zeros((6, 6))
    for column, name in enumerate(names):
        ends = []
        for sign in (1, -1):
            launch = dict(base)
            launch[name] += sign*1e-6
            probe = _proton(test_context, **launch)
            line.track(probe)
            probe.move(_context=xo.context_default)
            ends.append(np.array([getattr(probe, n)[0] for n in names]))
        matrix[:, column] = (ends[0] - ends[1])/2e-6
    assert np.abs(matrix.T @ S @ matrix - S).max() < 1e-8
    probe = _proton(test_context, **_LAUNCH)
    line.track(probe)
    line.track(probe, backtrack=True)
    probe.move(_context=xo.context_default)
    for name in _NAMES:
        np.testing.assert_allclose(getattr(probe, name)[0], _LAUNCH[name],
                                   rtol=0, atol=1e-14)


def test_thick_rfmultipole_slices_compose_and_converge():
    def track(strategy, kicks):
        line = xt.Line(elements={'rf': xt.RFMultipole(num_kicks=kicks,
            integrator='uniform', **_THICK), 'end': xt.Marker()})
        line.particle_ref = _proton(xo.context_default)
        if strategy is not None:
            line.slice_thick_elements([xt.Strategy(strategy)])
        line.build_tracker()
        probe = _proton(xo.context_default, **_LAUNCH)
        line.track(probe)
        assert line.get_table().rows['end'].s[0] == _THICK['length']
        return np.array([getattr(probe, name)[0] for name in _NAMES])
    whole = track(None, 4)
    np.testing.assert_array_equal(track(xt.Uniform(4, mode='thick'), 4), whole)
    converged = track(None, 2048)
    errors = [np.abs(track(xt.Teapot(n, mode='thin'), 1) - converged).max()
              for n in (8, 16)]
    assert errors[1] < errors[0]/3


def test_thin_rfmultipole_slice_equivalent_element_matches():
    line = xt.Line(elements={'rf': xt.RFMultipole(**_THICK), 'end': xt.Marker()})
    line.particle_ref = _proton(xo.context_default)
    line.slice_thick_elements([xt.Strategy(xt.Teapot(1, mode='thin'))])
    line.build_tracker()
    table = line.get_table()
    thin = [name for name, kind in zip(table.name, table.element_type)
            if kind == 'ThinSliceRFMultipole']
    assert len(thin) == 1
    equivalent = line[thin[0]].get_equivalent_element()
    assert not equivalent.isthick
    np.testing.assert_array_equal(np.asarray(equivalent.knl)[:3], _THICK['knl'])
