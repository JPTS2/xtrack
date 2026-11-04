"""Every RF map on every available context, against the same map on the CPU.

These run with the contexts of XOBJECTS_TEST_CONTEXTS, so a GPU run covers the
plans, the per-particle model choice and the physical maps (TPSA and spin are
CPU only and are tested elsewhere).
"""
import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

_COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta', 's')
_MISALIGNMENT = dict(shift_x=1e-3, shift_y=-2e-4, shift_s=.01, rot_x_rad=2e-3,
                     rot_y_rad=-1e-3, rot_s_rad=.1, rot_shift_anchor=.2)
_P0C = 1e9
_CAVITY = dict(length=1., frequency=1.3e9, voltage=2e7, phase=1.4)
_ELEMENTS = {
    'longitudinal-only': (xt.Cavity, dict(_CAVITY, model='longitudinal-only')),
    'phase-averaged': (xt.Cavity, dict(_CAVITY, model='phase-averaged')),
    'phase-resolved-paraxial': (xt.Cavity, dict(_CAVITY, model='phase-resolved-paraxial', eta=.25)),
    'phase-resolved': (xt.Cavity, dict(_CAVITY, model='phase-resolved', num_kicks=8, eta=.25)),
    'phase-resolved-frozen': (xt.Cavity, dict(_CAVITY, model='phase-resolved-frozen', eta=0.)),
    'adaptive': (xt.Cavity, dict(_CAVITY, model='adaptive', eta=.25)),
    'crab': (xt.CrabCavity, dict(length=.5, frequency=4e8, crab_voltage=1e6, phase=.3)),
    'multipole': (xt.RFMultipole, dict(frequency=4e8, voltage=1e5, knl=[1e-3, 2e-2],
                                       ksl=[1e-4, 3e-3], phase_n=[.1, .2], phase_s=[.3, .4])),
    'thick multipole': (xt.RFMultipole, dict(length=.3, num_kicks=5, frequency=4e8, voltage=1e5,
                                             knl=[1e-3, 2e-2], ksl=[1e-4, 3e-3],
                                             phase_n=[.1, .2], phase_s=[.3, .4])),
}


def _launch(context, p0c=_P0C):
    ramp = np.linspace(-1., 1., 5)
    return xt.Particles(_context=context, mass0=xt.ELECTRON_MASS_EV, p0c=p0c,
                        x=1e-3 * ramp, px=1e-5 * ramp, y=-7e-4 * ramp, py=6e-6 * ramp,
                        zeta=2e-3 * ramp, delta=1e-3 * ramp[::-1])


def _end(context, name, misaligned=False, slicing=None, backtrack=False, p0c=_P0C):
    cls, controls = _ELEMENTS[name]
    controls = dict(controls, **(_MISALIGNMENT if misaligned else {}))
    line = xt.Line(elements={'start': xt.Drift(length=.2), 'rf': cls(**controls)})
    line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, p0c=_P0C)
    if slicing is not None:
        line.slice_thick_elements([xt.Strategy(slicing)])
    line.build_tracker(_context=context)
    particles = _launch(context, p0c)
    line.track(particles)
    if backtrack:
        line.track(particles, backtrack=True)
    particles.move(_context=xo.context_default)
    particles.sort(interleave_lost_particles=True)
    assert np.all(particles.state == 1)
    return np.array([getattr(particles, coord) for coord in _COORDS])


@for_all_test_contexts
@pytest.mark.parametrize('misaligned', [False, True], ids=['aligned', 'misaligned'])
@pytest.mark.parametrize('name', _ELEMENTS)
def test_map_matches_the_cpu(test_context, name, misaligned):
    end = _end(test_context, name, misaligned)
    reference = _end(xo.ContextCpu(), name, misaligned)
    xo.assert_allclose(end, reference, rtol=0, atol=2e-13)


@for_all_test_contexts
@pytest.mark.parametrize('name', ['phase-resolved', 'phase-averaged',
                                  'phase-resolved-paraxial', 'adaptive'])
def test_reversal_restores_the_launch(test_context, name):
    end = _end(test_context, name, backtrack=True)
    start = _launch(xo.context_default)
    start = np.array([getattr(start, coord) for coord in _COORDS])
    xo.assert_allclose(end, start, rtol=0, atol=1e-10)


@for_all_test_contexts
@pytest.mark.parametrize('name', ['phase-resolved', 'phase-averaged', 'adaptive'])
def test_thick_slices_match_the_whole_element(test_context, name):
    whole = _end(test_context, name)
    sliced = _end(test_context, name, slicing=xt.Teapot(4, mode='thick'))
    xo.assert_allclose(sliced, whole, rtol=0, atol=5e-11)


@for_all_test_contexts
def test_adaptive_chooses_per_particle(test_context):
    # Two particles of one batch on references that select different maps.
    def end(model, p0c):
        line = xt.Line(elements={'rf': xt.Cavity(**dict(_CAVITY, model=model, phase=np.pi / 2,
                                                        eta=0.))})
        line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, p0c=_P0C)
        line.build_tracker(_context=test_context)
        particles = xt.Particles(_context=test_context, mass0=xt.ELECTRON_MASS_EV, p0c=p0c,
                                 x=1e-3, zeta=1e-3, delta=1e-4)
        line.track(particles)
        particles.move(_context=xo.context_default)
        particles.sort(interleave_lost_particles=True)
        return np.array([getattr(particles, coord) for coord in _COORDS])
    mixed = end('adaptive', np.array([1e13, 3e6]))
    xo.assert_allclose(mixed[:, 0], end('longitudinal-only', np.array([1e13]))[:, 0],
                       rtol=0, atol=1e-14)
    xo.assert_allclose(mixed[:, 1], end('phase-resolved', np.array([3e6]))[:, 0],
                       rtol=0, atol=1e-14)
