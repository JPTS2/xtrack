"""MagnetOverlappedCavity's longitudinal-kick model: the static fields tracked as a
straight magnet with its faces, with the RF energy kick added to each of the magnet's
kicks.
"""
import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')
LENGTH, NUM_KICKS = 1.2, 8
RF = dict(voltage=3e6, frequency=400e6, phase=.4)
STATIC = dict(knl=[0., .02, .5], ksl=[0., -.01])
LAUNCH = dict(x=1e-3, px=2e-4, y=-1e-3, py=1e-4, zeta=1e-2, delta=1e-3)


@for_all_test_contexts
@pytest.mark.parametrize('ks', [0., .3])
def test_longitudinal_kick_is_the_magnet_with_a_thin_rf_kick_at_each_kick(test_context, ks):
    """With the uniform integrator each kick sits at the centre of its cell, so the
    body is exactly the chain: half cell, multipole and thin cavity, half cell."""
    end = {}
    for layout in ('element', 'chain'):
        if layout == 'element':
            elements = [xt.MagnetOverlappedCavity(length=LENGTH, num_kicks=NUM_KICKS, ks=ks,
                                                   edge_entry_active=0, edge_exit_active=0,
                                                   **STATIC, **RF)]
        else:
            elements = []
            for _ in range(NUM_KICKS):
                half_cell = (xt.UniformSolenoid(length=LENGTH / (2 * NUM_KICKS), ks=ks,
                                                edge_entry_active=0, edge_exit_active=0)
                             if ks else xt.DriftExact(length=LENGTH / (2 * NUM_KICKS)))
                elements += [half_cell,
                             xt.Multipole(knl=np.array(STATIC['knl']) / NUM_KICKS,
                                          ksl=np.array(STATIC['ksl']) / NUM_KICKS),
                             xt.Cavity(voltage=RF['voltage'] / NUM_KICKS,
                                       frequency=RF['frequency'], phase=RF['phase']),
                             half_cell.copy()]
        line = xt.Line(elements=elements)
        line.particle_ref = xt.Particles(p0c=2e9, mass0=xt.PROTON_MASS_EV)
        line.build_tracker(_context=test_context)

        particles = line.build_particles(**LAUNCH)
        line.track(particles)
        particles.move(_context=xo.ContextCpu())

        assert particles.state[0] == 1
        end[layout] = np.array([getattr(particles, name)[0] for name in COORDS])

    assert np.max(np.abs(end['element'] - np.array(list(LAUNCH.values())))) > 1e-4
    np.testing.assert_allclose(end['element'], end['chain'], rtol=0, atol=1e-15)


@for_all_test_contexts
@pytest.mark.parametrize('backtrack', [False, True])
@pytest.mark.parametrize('case', ['no static fields', 'no voltage, multipoles',
                                  'no voltage, solenoid and multipoles'])
def test_longitudinal_kick_reduces_to_the_cavity_and_to_the_magnet(test_context, case,
                                                                   backtrack):
    """Faces included: those of a Quadrupole or of a UniformSolenoid with "full" faces."""
    if case == 'no static fields':
        element = xt.MagnetOverlappedCavity(length=LENGTH, num_kicks=NUM_KICKS, **RF)
        reference = xt.Cavity(length=LENGTH, num_kicks=NUM_KICKS, **RF,
                              model='drift-kick-drift-exact')
    elif case == 'no voltage, multipoles':
        element = xt.MagnetOverlappedCavity(length=LENGTH, num_kicks=NUM_KICKS, **STATIC)
        reference = xt.Magnet(length=LENGTH, **STATIC, num_multipole_kicks=NUM_KICKS,
                              integrator='uniform', model='drift-kick-drift-exact',
                              edge_entry_model='full', edge_exit_model='full')
    else:
        element = xt.MagnetOverlappedCavity(length=LENGTH, num_kicks=NUM_KICKS, ks=.3, **STATIC)
        reference = xt.UniformSolenoid(length=LENGTH, ks=.3, **STATIC,
                                       num_multipole_kicks=NUM_KICKS, integrator='uniform',
                                       edge_entry_model='full', edge_exit_model='full')
    end = {}
    for name, chosen in (('element', element), ('reference', reference)):
        line = xt.Line(elements=[chosen])
        line.particle_ref = xt.Particles(p0c=2e9, mass0=xt.PROTON_MASS_EV)
        line.build_tracker(_context=test_context)

        particles = line.build_particles(**LAUNCH)
        line.track(particles, backtrack=backtrack)
        particles.move(_context=xo.ContextCpu())
        end[name] = np.array([getattr(particles, n)[0] for n in COORDS])

    np.testing.assert_array_equal(end['element'], end['reference'])


@for_all_test_contexts
def test_longitudinal_kick_slices_keep_the_static_fields(test_context):
    """Two thick slices split the kicks exactly, each taking the face at its end; thin
    slices converge at second order."""
    end = {}
    for key, slicing in (('whole', None), ('thick 2', xt.Teapot(2, mode='thick')),
                         ('thin 8', xt.Teapot(8, mode='thin')),
                         ('thin 32', xt.Teapot(32, mode='thin'))):
        rf = xt.MagnetOverlappedCavity(length=LENGTH, num_kicks=64, ks=.3, **STATIC, **RF)
        line = xt.Line(elements={'rf': rf, 'end': xt.Marker()})
        if slicing is not None:
            line.slice_thick_elements([xt.Strategy(slicing)])
        line.particle_ref = xt.Particles(p0c=2e9, mass0=xt.PROTON_MASS_EV)
        line.build_tracker(_context=test_context)

        particles = line.build_particles(**LAUNCH)
        line.track(particles, ele_stop='end')
        particles.move(_context=xo.ContextCpu())
        end[key] = np.array([getattr(particles, n)[0] for n in COORDS])

    np.testing.assert_array_equal(end['thick 2'], end['whole'])
    thin_8 = np.max(np.abs(end['thin 8'] - end['whole']))
    thin_32 = np.max(np.abs(end['thin 32'] - end['whole']))
    assert thin_32 < thin_8 / 10


def test_only_the_longitudinal_kick_and_the_physical_models_are_offered():
    assert xt.MagnetOverlappedCavity().model == 'adaptive'
    assert xt.MagnetOverlappedCavity.get_available_models() == [
        'adaptive', 'longitudinal-kick', '_phase-resolved', '_phase-averaged',
        '_phase-resolved-paraxial', '_phase-resolved-frozen']
    with pytest.raises(ValueError, match='Invalid MagnetOverlappedCavity model'):
        xt.MagnetOverlappedCavity(model='drift-kick-drift-exact')
    assert 'model' not in xt.MagnetOverlappedCavity(length=1.).to_dict()


def test_longitudinal_models_refuse_the_geometric_integrator():
    """The magnet integrators have no "_geometric" branch: it would track nothing."""
    with pytest.raises(ValueError, match='_geometric'):
        xt.MagnetOverlappedCavity(length=LENGTH, **RF, **STATIC, integrator='_geometric')
    rf = xt.MagnetOverlappedCavity(length=LENGTH, **RF, model='_phase-resolved',
                                   integrator='_geometric')
    with pytest.raises(ValueError, match='_geometric'):
        rf.model = 'longitudinal-kick'
    cavity = xt.Cavity(length=LENGTH, **RF, model='_phase-resolved', integrator='_geometric')
    with pytest.raises(ValueError, match='_geometric'):
        cavity.model = 'adaptive'


def test_longitudinal_kick_refuses_tpsa_knobs_on_the_static_fields():
    madng_tpsa = pytest.importorskip('madng_tpsa')
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    rf = xt.MagnetOverlappedCavity(length=LENGTH, **RF, **STATIC)
    with pytest.raises(ValueError, match='physical model'):
        rf.knl[1] = descriptor.param(1, .02)
    physical = xt.MagnetOverlappedCavity(length=LENGTH, **RF, knl=[0., .02],
                                         model='_phase-resolved')
    physical.knl[1] = descriptor.param(1, .02)
    with pytest.raises(ValueError, match='physical model'):
        physical.model = 'longitudinal-kick'
    physical.disable_tpsa()


def test_line_with_static_fields_saves_as_json(tmp_path):
    line = xt.Line(elements=[xt.MagnetOverlappedCavity(length=LENGTH, ks=.3, **STATIC, **RF)],
                   element_names=['rf'])
    line.to_json(tmp_path / 'line.json')
    restored = xt.load(tmp_path / 'line.json')
    np.testing.assert_array_equal(restored['rf'].knl, line['rf'].knl)
    np.testing.assert_array_equal(restored['rf'].ksl, line['rf'].ksl)
    assert restored['rf'].ks == .3
