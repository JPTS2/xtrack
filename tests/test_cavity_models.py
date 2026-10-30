"""Public cavity wiring: live controls, references and source-preserving slices.

Independent equations and TPSA coefficient checks live in the review/TPSA
tests. These tests exercise normal Xtrack operations rather than a private
snapshot API. New models remain experimental, without release qualification.
"""
import numpy as np
import pytest
import xtrack as xt

from test_cavity_native import native_context


MODELS = ('phase-resolved', 'phase-resolved-paraxial',
          'phase-resolved-longitudinal')
COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')


@pytest.fixture(scope='module')
def model_context(tmp_path_factory):
    directory = tmp_path_factory.mktemp('cavity-model-wiring')
    with native_context(directory) as context:
        yield context


def cavity(model='phase-resolved', **kwargs):
    controls = dict(length=.05, voltage=20e3, frequency=400e6, phase=.7,
                    eta=.25, counter_phase=.3, calibration_beta=1.,
                    model=model, integrator='gauss2', num_kicks=16)
    controls.update(kwargs)
    return xt.Cavity(**controls)


def particle(context=None):
    return xt.Particles(_context=context, mass0=xt.ELECTRON_MASS_EV,
                        q0=1., p0c=2e6, x=.0004, px=.002,
                        y=-.0003, py=-.001, zeta=.0002, delta=.003)


def coordinates(particles):
    return np.array([float(getattr(particles, name)[0]) for name in COORDS])


def physical_coordinates(particles):
    result = coordinates(particles)
    # A Line end-turn changes the shared time_s/zeta reference frame even
    # when s is retained. Compare the same laboratory clock, not raw zeta.
    result[4] = ((particles.s[0]-particles.zeta[0])/particles.beta0[0]
                 + 299792458.*particles.time_s)
    return result


def test_new_selectors_are_cavity_specific():
    for model in (*MODELS, 'phase-resolved-frozen'):
        assert cavity(model).model == model
        assert model not in xt.CrabCavity.get_available_models()
    assert xt.Cavity().model == 'adaptive'
    assert xt.Cavity().integrator == 'adaptive'
    assert 'gauss2' not in xt.Quadrupole.get_available_integrators()


def test_physical_controls_round_trip_and_carrier_origin():
    original = cavity(wave_number=None, knl=[0., .012], ksl=[0., -.003], ks=.2)
    assert original.wave_number is None
    assert original.eta == .25
    copy = xt.Cavity.from_dict(original.to_dict())
    assert copy.wave_number is None
    assert copy.model == original.model
    np.testing.assert_array_equal(copy.knl, original.knl)
    np.testing.assert_array_equal(copy.ksl, original.ksl)
    copy.wave_number = 0.
    assert copy.wave_number == 0.
    assert not copy.wave_number_from_frequency
    copy = xt.Cavity.from_dict(copy.to_dict())
    assert copy.wave_number == 0.
    copy.wave_number = None
    assert copy.wave_number_from_frequency


@pytest.mark.parametrize('model', MODELS)
def test_line_variables_update_actual_source(model_context, model):
    line = xt.Line(elements={'rf': cavity(model, _context=model_context)})
    line.particle_ref = particle(model_context)
    line.vars['rf_eta'] = .25
    line.vars['rf_phase'] = .7
    line.element_refs['rf'].eta = line.vars['rf_eta']
    line.element_refs['rf'].phase = line.vars['rf_phase']
    line.build_tracker(_context=model_context)
    line.reset_s_at_end_turn = False
    original = particle(model_context)
    initial = original.copy()
    line.track(original)
    assert original.state[0] == 1
    line.vars['rf_eta'] = .6
    line.vars['rf_phase'] = 1.1
    line.set('rf', voltage=35e3)
    changed = initial.copy()
    line.track(changed)
    assert changed.state[0] == 1
    assert not np.array_equal(coordinates(original), coordinates(changed))
    fresh = cavity(model, eta=.6, phase=1.1, voltage=35e3,
                   _context=model_context)
    expected = initial.copy()
    fresh.track(expected)
    np.testing.assert_allclose(physical_coordinates(changed),
                               physical_coordinates(expected),
                               rtol=0, atol=3e-13)
    line.vars['rf_eta'] = .25
    line.vars['rf_phase'] = .7
    line.set('rf', voltage=20e3)
    restored = initial.copy()
    line.track(restored)
    np.testing.assert_array_equal(coordinates(original), coordinates(restored))


@pytest.mark.parametrize('model', MODELS)
def test_line_tracking_twiss_and_reference_reverse(model_context, model):
    ref = xt.Particles(_context=model_context, mass0=xt.ELECTRON_MASS_EV,
                       q0=1., p0c=2e6)
    rf = cavity(model, _context=model_context)
    pilot = ref.copy()
    rf.track(pilot)
    assert pilot.state[0] == 1
    update = xt.ReferenceEnergyIncrease(_context=model_context,
                                        Delta_p0c=ref.p0c[0]*pilot.delta[0])
    line = xt.Line(elements={'rf': rf, 'reference_update': update})
    line.particle_ref = ref
    line.build_tracker(_context=model_context)
    line.reset_s_at_end_turn = False
    tw = line.twiss(method='6d', start=xt.START, end=xt.END,
                    betx=2., alfx=.3, bety=3., alfy=-.2)
    assert np.all(np.isfinite(tw.betx))
    assert np.all(np.isfinite(tw.bety))
    assert abs(tw.delta[-1]) < 1e-12
    tracked = particle(model_context)
    before = coordinates(tracked)
    p0c = tracked.p0c[0]
    line.track(tracked)
    assert tracked.state[0] == 1
    line.track(tracked, backtrack=True)
    assert tracked.state[0] == 1
    np.testing.assert_allclose(coordinates(tracked), before, rtol=0, atol=2e-12)
    np.testing.assert_allclose(tracked.p0c[0], p0c, rtol=2e-15)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('mode', ['thick', 'thin'])
@pytest.mark.parametrize('overlap', [False, True])
def test_slicing_shares_source_and_survives_serialization(model_context,
                                                         model, mode, overlap):
    controls = (dict(ks=.2, knl=[0., .01], ksl=[0., -.003],
                     shift_x=.0002, shift_y=-.0001, rot_s_rad=.2,
                     rot_y_rad=.001) if overlap else {})
    # Teapot(2) thin has lengths 1/6, 2/3, 1/6. N24 aligns the
    # original Gauss cells with both that layout and the two thick halves.
    line = xt.Line(elements={'rf': cavity(model, num_kicks=24,
                                         _context=model_context,
                                         **controls)})
    line.particle_ref = particle(model_context)
    line.reset_s_at_end_turn = False
    initial = particle(model_context)
    whole = initial.copy()
    line.build_tracker(_context=model_context)
    line.track(whole)
    assert whole.state[0] == 1
    line.discard_tracker()
    line.slice_thick_elements([xt.Strategy(slicing=xt.Teapot(2, mode=mode))])
    restored = xt.Line.from_dict(line.to_dict())
    restored.build_tracker(_context=model_context)
    restored.reset_s_at_end_turn = False
    sliced = initial.copy()
    restored.track(sliced)
    assert sliced.state[0] == 1
    np.testing.assert_allclose(physical_coordinates(sliced),
                               physical_coordinates(whole),
                               rtol=0, atol=2e-12)
    restored.optimize_for_tracking(compile=False, verbose=False)
    optimized = initial.copy()
    restored.track(optimized)
    assert optimized.state[0] == 1
    np.testing.assert_allclose(physical_coordinates(optimized),
                               physical_coordinates(sliced),
                               rtol=0, atol=2e-12)
    kind = xt.ThickSliceCavity if mode == 'thick' else xt.DriftSliceCavity
    names = [name for name in restored.element_names
             if isinstance(restored[name], kind)]
    assert len(names) == (2 if mode == 'thick' else 3)
    second = restored[names[1]]
    offset = second.slice_offset
    # Optimization deliberately disables variable management, as for other
    # Xtrack elements. Direct physical-field changes still retain parent links.
    restored['rf'].length = .1
    assert second.slice_offset == 2*offset


@pytest.mark.parametrize('kind', [xt.ThickSliceCavity, xt.DriftSliceCavity])
def test_old_serialized_slice_distance_loads(kind):
    old = dict(__class__=kind.__name__, parent_name='rf',
               weight=.5, slice_offset=.025)
    restored = kind.from_dict(old)
    restored._parent = cavity(_buffer=restored._buffer)
    assert restored.slice_offset == .025


def test_matching_uses_normal_live_phase_knob(model_context):
    line = xt.Line(elements={'rf': cavity(_context=model_context)})
    line.particle_ref = xt.Particles(_context=model_context,
        mass0=xt.ELECTRON_MASS_EV, q0=1., p0c=2e6)
    line.vars['rf_phase'] = .9
    line.element_refs['rf'].phase = line.vars['rf_phase']
    line.build_tracker(_context=model_context)
    init = dict(method='6d', start=xt.START, end=xt.END,
                betx=2., alfx=.3, bety=3., alfy=-.2)
    target = line.twiss(**init).betx[-1]
    line.vars['rf_phase'] = .7
    line.match(**init, vary=[xt.Vary('rf_phase', step=1e-4)],
               targets=[xt.Target('betx', at=xt.END, value=target, tol=1e-10)],
               verbose=False)
    np.testing.assert_allclose(line.twiss(**init).betx[-1], target,
                               rtol=0, atol=1e-10)


def test_legacy_selector_refuses_new_static_strengths(model_context):
    rf = xt.Cavity(_context=model_context, voltage=20e3, frequency=400e6,
                    phase=.7)
    initial = particle(model_context)
    allowed = initial.copy()
    rf.track(allowed)
    assert allowed.state[0] == 1
    for name, value in [('ks', .1), ('knl', [0., .01]), ('ksl', [0., .02])]:
        rf.ks = 0.
        rf.knl[:] = 0.
        rf.ksl[:] = 0.
        if name == 'ks':
            rf.ks = value
        else:
            getattr(rf, name)[:len(value)] = value
        rejected = initial.copy()
        rf.track(rejected)
        assert rejected.state[0] == -99744
        np.testing.assert_array_equal(coordinates(rejected), coordinates(initial))
