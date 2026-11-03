"""Public cavity wiring: live controls, references and source-preserving slices.

Independent equations and TPSA coefficient checks live in the review/TPSA
tests. These tests exercise normal Xtrack operations rather than a private
snapshot API. New models remain experimental, without release qualification.
"""
import numpy as np
import pytest
import xtrack as xt

from _cavity_context import kernel_context


MODELS = ('phase-resolved',)
COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')


@pytest.fixture(scope='module')
def model_context(tmp_path_factory):
    directory = tmp_path_factory.mktemp('cavity-model-wiring')
    with kernel_context(directory) as context:
        yield context


def cavity(model='phase-resolved', **kwargs):
    controls = dict(length=.05, voltage=20e3, frequency=400e6, phase=.7,
                    eta=.25, counter_phase=.3, calibration_beta=1.,
                    model=model, integrator='uniform', num_kicks=16)
    if model == 'phase-resolved-frozen':
        controls.update(eta=0., integrator='geometric')
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
    for model in (name for name in xt.Cavity.get_available_models()
                  if name.startswith('phase-')):
        assert cavity(model).model == model
        assert model not in xt.CrabCavity.get_available_models()
    assert xt.Cavity().model == 'adaptive'
    assert xt.Cavity().integrator == 'adaptive'
    assert 'gauss2' not in xt.Quadrupole.get_available_integrators()
    assert 'midpoint' not in xt.Quadrupole.get_available_integrators()


@pytest.mark.parametrize('model,profile', [(None, 'p16'),
    ('phase-resolved', 'p16'), ('phase-resolved-frozen', 'uniform')])
def test_builtin_profile_inference_is_stored(model, profile):
    rf = xt.Cavity(**({} if model is None else {'model': model}))
    assert rf.rf_profile == profile
    data = rf.to_dict()
    assert not {'rf_profile', 'rf_profile_sigma', 'rf_profile_center'} & set(data)
    restored = xt.Cavity.from_dict(data)
    assert restored.rf_profile == profile
    assert restored.rf_profile_sigma == restored.rf_profile_center == 0.


@pytest.mark.parametrize('sigma', [None, 0., -.01, np.inf, np.nan])
def test_gaussian_profile_never_guesses_a_width(sigma):
    with pytest.raises(ValueError, match='width'):
        xt.Cavity(model='phase-resolved', rf_profile='gaussian',
                   rf_profile_sigma=sigma)


def test_builtin_profile_model_changes_do_not_retarget_source():
    rf = cavity(rf_profile='uniform')
    rf.model = 'phase-resolved-frozen'
    assert rf.rf_profile == 'uniform'
    rf.model = 'phase-resolved'
    assert xt.Cavity.from_dict(rf.to_dict()).rf_profile == 'uniform'
    # Raw enum zero must not disappear and acquire a different inferred source.
    rf.rf_profile = 'p16'
    rf.model = 'phase-resolved-frozen'
    assert rf.rf_profile == rf.to_dict()['rf_profile'] == 'p16'
    with pytest.raises(ValueError, match='uniform'):
        xt.Cavity.from_dict(rf.to_dict())


def test_gaussian_profile_live_metres_and_roundtrip():
    line = xt.Line(elements={'rf': cavity('phase-resolved',
        rf_profile_sigma=.012, rf_profile_center=0.)})
    line.vars['sigma_m'] = .012
    line.vars['center_m'] = 0.
    line.set('rf', rf_profile='gaussian', rf_profile_sigma='sigma_m',
             rf_profile_center='center_m')
    line.vars['sigma_m'] = .017
    line.vars['center_m'] = -.003
    line.set('rf', frequency=800e6)
    rf = line['rf']
    assert rf.rf_profile == 'gaussian'
    assert rf.rf_profile_sigma == .017
    assert rf.rf_profile_center == -.003
    restored = xt.Cavity.from_dict(rf.to_dict())
    assert restored.rf_profile == 'gaussian'
    assert restored.rf_profile_sigma == .017
    assert restored.rf_profile_center == -.003


@pytest.mark.parametrize('model,profile', [('phase-resolved-frozen', 'p16'),
    ('phase-resolved-frozen', 'gaussian')])
def test_builtin_profile_model_mismatches_refuse(model, profile):
    with pytest.raises(ValueError):
        xt.Cavity(model=model, rf_profile=profile, rf_profile_sigma=.02)


def test_owned_ref_null_move_preserves_active_scalar_and_array_handles():
    import madng_tpsa
    import xobjects as xo

    rf = cavity()
    descriptor = madng_tpsa.Descriptor(6, 2, num_params=2, param_order=1)
    rf.phase = descriptor.param(1, .7)
    rf.knl[1] = descriptor.param(2, .01)
    scalar_owner = rf.__dict__['_tpsa_handles']
    array_owner = rf.__dict__['_tpsa_array_handles']
    bits = {name: bytes(rf._buffer.to_bytearray(rf._xobject._get_offset(name), 8))
            for name in rf._float_or_tpsa_fields}
    context = xo.ContextCpu(omp_num_threads=0)
    buffer = context.new_buffer()
    rf.move(_buffer=buffer)
    assert rf._buffer is buffer
    assert rf.__dict__['_tpsa_handles'] is scalar_owner
    assert rf.__dict__['_tpsa_array_handles'] is array_owner
    for name, raw in bits.items():
        assert bytes(rf._buffer.to_bytearray(rf._xobject._get_offset(name), 8)) == raw
    assert rf.phase.const_part == .7
    assert rf.knl[1].const_part == .01
    with pytest.raises(NotImplementedError, match='Copying TPSA'):
        rf.copy()
    rf.disable_tpsa()
    assert rf.phase == .7
    assert rf.knl[1] == .01


def test_null_owned_ref_ordinary_line_moves_to_common_buffer():
    import xobjects as xo

    context = xo.ContextCpu(omp_num_threads=0)
    rf = cavity(_context=context)
    marker = xt.Marker(_context=context)
    line = xt.Line(elements=[rf, marker])
    line.build_tracker(_context=context, compile=False)
    assert rf._buffer is marker._buffer is line._buffer
    copied = rf.copy()
    copied.move(_buffer=context.new_buffer())
    assert copied.voltage == rf.voltage


def test_serialization_writes_only_non_default_fields():
    assert xt.Cavity().to_dict() == {'__class__': 'Cavity'}
    legacy = xt.Cavity(voltage=1e6, frequency=4e8, phase=.3).to_dict()
    assert set(legacy) == {'__class__', 'voltage', 'frequency', 'phase'}
    # Explicit zeros of nonzero defaults, and explicit zero carriers, survive.
    explicit = dict(model='phase-resolved', eta=0., calibration_beta=.9,
        edge_entry_active=0, edge_exit_active=0, order=2,
        knl=[0., .1, 0.], wave_number=0., nominal_energy_gain=0.,
        rf_profile='uniform')
    data = xt.Cavity(**explicit).to_dict()
    assert set(data) == {'__class__', *explicit}
    restored = xt.Cavity.from_dict(data)
    for name, value in explicit.items():
        np.testing.assert_array_equal(getattr(restored, name), value)
    assert restored.to_dict() == data


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


def test_serialization_preserves_tiny_coefficients_and_active_order():
    rf = cavity(knl=[0., 1e-19], ksl=[0., -2e-20])
    rf.order = 0
    restored = xt.Cavity.from_dict(rf.to_dict())
    assert restored.order == 0
    np.testing.assert_array_equal(restored.knl, rf.knl)
    np.testing.assert_array_equal(restored.ksl, rf.ksl)


@pytest.mark.parametrize('gain', [None, 0., 12000.])
def test_frozen_nominal_programme_round_trip(gain):
    rf = cavity('phase-resolved-frozen', nominal_energy_gain=gain)
    restored = xt.Cavity.from_dict(rf.to_dict())
    assert restored.nominal_energy_gain == gain
    assert restored.nominal_energy_gain_from_rf == (gain is None)


def test_frozen_nominal_programme_live_expression():
    line = xt.Line(elements={'rf': cavity('phase-resolved-frozen')})
    line.vars['programme_gain'] = 0.
    line.set('rf', nominal_energy_gain='programme_gain')
    assert line['rf'].nominal_energy_gain == 0.
    assert not line['rf'].nominal_energy_gain_from_rf
    line.vars['programme_gain'] = 12000.
    assert line['rf'].nominal_energy_gain == 12000.
    assert not line['rf'].nominal_energy_gain_from_rf


@pytest.mark.parametrize('model', (*MODELS, 'phase-resolved-frozen'))
def test_thin_source_strength_readback_follows_parent_model(model_context, model):
    line = xt.Line(elements={'rf': cavity(model, voltage=20e3,
                                         _context=model_context,
                                         knl=[0., .012], ksl=[0., -.003])})
    line.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thin'))])
    line.build_tracker(_context=model_context)
    # These voltages are nominal parent-input allocations; the resolved P16
    # source's electric-crest integral over a partial interval is not weight*V.
    source = np.array([isinstance(line[name], xt.DriftSliceCavity)
                       for name in line.element_names])
    marker = np.array([isinstance(line[name], xt.ThinSliceCavity)
                       for name in line.element_names])
    np.testing.assert_allclose(line.attr['voltage'][source].sum(), 20e3)
    np.testing.assert_array_equal(line.attr['voltage'][marker], 0.)
    np.testing.assert_allclose(line.attr['k1l'][source].sum(), .012)
    np.testing.assert_allclose(line.attr['k1sl'][source].sum(), -.003)
    # Reuse the already-created attr cache through both selector changes.
    line.set('rf', model='adaptive')
    np.testing.assert_array_equal(line.attr['voltage'][source], 0.)
    np.testing.assert_allclose(line.attr['voltage'][marker].sum(), 20e3)
    line.set('rf', model=model)
    np.testing.assert_allclose(line.attr['voltage'][source].sum(), 20e3)
    np.testing.assert_array_equal(line.attr['voltage'][marker], 0.)
    line.optimize_for_tracking(compile=False, verbose=False)
    np.testing.assert_allclose(line.attr['voltage'].sum(), 20e3)
    np.testing.assert_allclose(line.attr['k1l'].sum(), .012)


def test_source_interval_alignment_readback_is_live():
    line = xt.Line(elements={'rf': cavity(shift_x=.0002, shift_y=-.0001,
                                         rot_s_rad=.2, rot_y_rad=.001)})
    line.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thin'))])
    # Metadata needs buffer offsets, not compilation of the physical maps.
    line.build_tracker(compile=False)
    intervals = np.array([isinstance(line[name], xt.DriftSliceCavity)
                          for name in line.element_names])
    for name, value in [('shift_x', .0002), ('shift_y', -.0001),
                        ('rot_s_rad', .2), ('rot_y_rad', .001)]:
        np.testing.assert_array_equal(line.attr[name][intervals], value)
    line.set('rf', model='adaptive')
    np.testing.assert_array_equal(line.attr['shift_x'][intervals], 0.)
    line.set('rf', model='phase-resolved', shift_x=.0003)
    np.testing.assert_array_equal(line.attr['shift_x'][intervals], .0003)
    # Automatic drift transforms stay off; its new source helper owns them.
    assert all(not line[name].rot_and_shift_from_parent
               for name in line.element_names
               if isinstance(line[name], xt.DriftSliceCavity))


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


@pytest.mark.parametrize('model', ['adaptive', 'phase-resolved'])
def test_cavity_slices_serialize_public_offset(model):
    """Cavity slices serialize offsets exactly as other element slices do."""
    def sliced(element):
        line = xt.Line(elements={'e': element})
        line.slice_thick_elements([xt.Strategy(xt.Teapot(3, mode='thick'))])
        data = line.to_dict()['elements']
        return [{key: value for key, value in data[name].items()
                 if key in ('slice_offset', 'weight')}
                for name in line.element_names if 'weight' in data[name]]
    cavity_slices = sliced(xt.Cavity(length=1., voltage=1e6, frequency=4e8, model=model))
    quadrupole_slices = sliced(xt.Quadrupole(length=1., k1=.1))
    assert len(cavity_slices) == len(quadrupole_slices)
    for actual, expected in zip(cavity_slices, quadrupole_slices):
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-15)


@pytest.mark.parametrize('kind', [xt.ThickSliceCavity, xt.DriftSliceCavity])
def test_old_serialized_slice_distance_loads(kind):
    old = dict(__class__=kind.__name__, parent_name='rf',
               weight=.5, slice_offset=.025)
    restored = kind.from_dict(old)
    restored._parent = cavity(_buffer=restored._buffer)
    assert restored.slice_offset == .025


@pytest.mark.parametrize('model,mode', [('phase-resolved-frozen', 'thick'),
    ('phase-resolved-frozen', 'thin'), ('phase-resolved', 'thick')])
def test_finite_parent_words_survive_layout_partition(model_context, model, mode):
    controls = dict(num_kicks=4,
                    edge_entry_active=0, edge_exit_active=1)
    rf = cavity(model, _context=model_context, **controls)
    line = xt.Line(elements={'rf': rf, 'end': xt.Marker(_context=model_context)})
    line.particle_ref = particle(model_context)
    line.build_tracker(_context=model_context)
    initial = particle(model_context)
    expected = initial.copy()
    line.track(expected, ele_start=0, ele_stop=1)
    assert expected.state[0] == 1
    line.discard_tracker()
    line.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode=mode))])
    restored = xt.Line.from_dict(line.to_dict())
    restored.build_tracker(_context=model_context)
    restored.reset_s_at_end_turn = False
    stop = restored.element_names.index('end')
    actual = initial.copy()
    restored.track(actual, ele_start=0, ele_stop=stop)
    assert actual.state[0] == 1
    np.testing.assert_allclose(coordinates(actual), coordinates(expected),
                               rtol=0., atol=3e-11)
    restored.track(actual, ele_start=0, ele_stop=stop, backtrack=True)
    assert actual.state[0] == 1
    np.testing.assert_allclose(coordinates(actual), coordinates(initial),
                               rtol=0., atol=3e-11)
    restored.optimize_for_tracking(compile=False, verbose=False)
    optimized = initial.copy()
    restored.track(optimized)
    assert optimized.state[0] == 1
    np.testing.assert_allclose(physical_coordinates(optimized),
                               physical_coordinates(expected),
                               rtol=0., atol=3e-11)


@pytest.mark.parametrize('model', ('phase-resolved', 'phase-resolved-frozen'))
def test_optimized_source_layout_keeps_live_legacy_rf_switch(model_context, model):
    source = cavity(model, _context=model_context, integrator='adaptive',
                    num_kicks=24)
    line = xt.Line(elements={'rf': source})
    line.particle_ref = particle(model_context)
    line.reset_s_at_end_turn = False
    line.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thin'))])
    line.build_tracker(_context=model_context)
    initial = particle(model_context)
    expected_source = initial.copy()
    line.track(expected_source)
    assert expected_source.state[0] == 1
    line.optimize_for_tracking(compile=False, verbose=False)

    reference = xt.Line(elements={'rf': cavity(model, _context=model_context,
                        integrator='adaptive', num_kicks=24)})
    reference['rf'].model = 'longitudinal-only'
    reference.reset_s_at_end_turn = False
    reference.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thin'))])
    reference.build_tracker(_context=model_context)
    expected_legacy = initial.copy()
    reference.track(expected_legacy)
    line['rf'].model = 'longitudinal-only'
    actual = initial.copy()
    line.track(actual)
    assert expected_legacy.state[0] == actual.state[0] == 1
    np.testing.assert_allclose(physical_coordinates(actual),
                               physical_coordinates(expected_legacy),
                               rtol=0., atol=3e-12)
    line['rf'].model = model
    restored = initial.copy()
    line.track(restored)
    assert restored.state[0] == 1
    np.testing.assert_allclose(physical_coordinates(restored),
                               physical_coordinates(expected_source),
                               rtol=0., atol=3e-12)
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


def test_thin_layout_converges_to_parent_map(model_context):
    """Thin nodes are markers; drift slices carry the cells on another mesh."""
    def endpoint(cells, sliced):
        line = xt.Line(elements={'rf': cavity(_context=model_context,
            num_kicks=cells, edge_entry_active=0), 'end': xt.Marker()})
        line.particle_ref = particle(model_context)
        if sliced:
            line.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thin'))])
        line.build_tracker(_context=model_context)
        line.reset_s_at_end_turn = False
        probe = particle(model_context)
        line.track(probe, ele_start=0, ele_stop=line.element_names.index('end'))
        assert probe.state[0] == 1
        return coordinates(probe)
    errors = [np.abs(endpoint(cells, True) - endpoint(cells, False)).max()
              for cells in (16, 64)]
    # Fourth-order Gauss cells: a fourfold refinement gains about 4**4.
    assert errors[1] < errors[0] / 200
    assert errors[1] < 1e-9


def test_legacy_models_ignore_physical_model_fields(model_context):
    plain = xt.Cavity(_context=model_context, voltage=20e3, frequency=400e6,
                      phase=.7)
    dressed = xt.Cavity(_context=model_context, voltage=20e3, frequency=400e6,
        phase=.7, ks=.1, knl=[0., .01], ksl=[0., .02], rf_profile='gaussian',
        rf_profile_sigma=.02, eta=.3)
    expected = particle(model_context)
    actual = expected.copy()
    plain.track(expected)
    dressed.track(actual)
    assert actual.state[0] == 1
    np.testing.assert_array_equal(coordinates(actual), coordinates(expected))


@pytest.mark.parametrize('mode', [None, 'thick', 'thin'])
def test_gaussian_magnetic_integrated_strength_readback(mode):
    from math import erf, sqrt, pi
    rf = cavity(ks=.4, knl=[0., .03], ksl=[0., -.02],
                magnetic_profile='gaussian', magnetic_profile_sigma=.015,
                edge_entry_active=0, edge_exit_active=0)
    data = rf.to_dict()
    restored = xt.Cavity.from_dict(data)
    assert restored.magnetic_profile == 'gaussian'
    assert restored.magnetic_profile_sigma == .015
    line = xt.Line(elements={'rf': rf})
    if mode:
        line.slice_thick_elements([xt.Strategy(xt.Teapot(3, mode=mode))])
    # Metadata uses ordinary MultiSetter access, without tracker compilation.
    for width in (.015, .03):
        rf.magnetic_profile_sigma = width
        expected_integral = sqrt(2*pi)*width*erf(rf.length/(2*sqrt(2)*width))
        np.testing.assert_allclose(sum(line.attr['ksoll']), rf.ks*expected_integral,
                                   rtol=2e-15, atol=1e-17)
        np.testing.assert_allclose(sum(line.attr['k1l']), .03, rtol=2e-15)
        np.testing.assert_allclose(sum(line.attr['k1sl']), -.02, rtol=2e-15)
        for index, element in enumerate(line.elements):
            if mode and not getattr(element, '_inherit_strengths', False):
                continue
            weight = float(element.weight) if mode else 1.
            start = float(element._slice_offset_fraction) if mode else 0.
            a, b = rf.length*(start-.5), rf.length*(start+weight-.5)
            partial = sqrt(pi/2)*width*(erf(b/(sqrt(2)*width))
                                         -erf(a/(sqrt(2)*width)))
            np.testing.assert_allclose(line.attr['ks'][index], rf.ks, rtol=0.)
            np.testing.assert_allclose(line.attr['ksoll'][index], rf.ks*partial,
                                       rtol=2e-15, atol=1e-17)
            np.testing.assert_allclose(line.attr['k1l'][index],
                                       .03*partial/expected_integral, rtol=2e-15)
            np.testing.assert_allclose(line.attr['k1sl'][index],
                                       -.02*partial/expected_integral, rtol=2e-15)
        if mode:
            actual = line.attr['ksoll']
            # A nonuniform profile must not reduce to equal weight allocation.
            assert np.ptp(actual[actual != 0.]) > 0.
        np.testing.assert_allclose(sum(line.attr['voltage']), rf.voltage,
                                   rtol=2e-15)


@pytest.mark.parametrize('width', [None, 0., -.01, np.inf, np.nan])
def test_gaussian_magnetic_width_is_explicit(width):
    with pytest.raises(ValueError, match='width'):
        cavity(magnetic_profile='gaussian', magnetic_profile_sigma=width,
               edge_entry_active=0, edge_exit_active=0)




@pytest.mark.parametrize('faces', [0, 1])
def test_frozen_model_is_symplectic_with_canonical_faces(model_context, faces):
    """The SAD face rescales momenta by p_after/p; the canonical face does not."""
    names = ('x', 'px', 'y', 'py', 'zeta', 'pzeta')
    base = dict(x=1e-3, px=2e-4, y=-1e-3, py=1e-4, zeta=1e-3, pzeta=1e-4)
    p0c = xt.ELECTRON_MASS_EV * np.sqrt(3.2**2 - 1)
    rf = xt.Cavity(_context=model_context, length=1., voltage=2e6,
                   frequency=1.3e9, phase=.4, model='phase-resolved-frozen',
                   edge_entry_active=faces, edge_exit_active=faces)
    step = 1e-6
    matrix = np.zeros((6, 6))
    for column, name in enumerate(names):
        ends = []
        for sign in (1, -1):
            launch = dict(base)
            launch[name] += sign * step
            probe = xt.Particles(_context=model_context, p0c=p0c,
                                 mass0=xt.ELECTRON_MASS_EV, **launch)
            rf.track(probe)
            assert probe.state[0] == 1
            ends.append(np.array([getattr(probe, n)[0] for n in names]))
        matrix[:, column] = (ends[0] - ends[1]) / (2 * step)
    form = np.zeros((6, 6))
    for i in range(3):
        form[2*i, 2*i+1], form[2*i+1, 2*i] = 1, -1
    # The SAD face fails this at 4e-6; finite differences give about 1e-9.
    assert np.abs(matrix.T @ form @ matrix - form).max() < 1e-7


@pytest.mark.parametrize('old,new', [
    ('drift-kick-drift-expanded', 'longitudinal-only'),
    ('drift-kick-drift-exact', 'longitudinal-only-exact'),
    ('rot-kick-rot-low-order', 'longitudinal-only-exact'),
    ('rot-kick-rot-high-order', 'longitudinal-only-exact')])
def test_legacy_model_names_are_deprecated_aliases(model_context, old, new):
    with pytest.warns(FutureWarning, match='deprecated'):
        aliased = xt.Cavity(_context=model_context, length=1., voltage=20e6,
                            frequency=1.3e9, phase=.3, num_kicks=4, model=old)
    assert aliased.model == new
    direct = xt.Cavity(_context=model_context, length=1., voltage=20e6,
                       frequency=1.3e9, phase=.3, num_kicks=4, model=new)
    tracked = []
    for element in (aliased, direct):
        probe = particle(model_context)
        element.track(probe)
        tracked.append(coordinates(probe))
    np.testing.assert_array_equal(tracked[0], tracked[1])


def test_old_integrator_names_are_deprecated_aliases():
    with pytest.warns(FutureWarning, match='deprecated'):
        rf = xt.Cavity(model='phase-resolved', length=.1, integrator='gauss2')
    assert rf.integrator == 'uniform'
    with pytest.warns(FutureWarning, match='deprecated'):
        rf = xt.Cavity(model='phase-resolved-frozen', length=.1,
                       integrator='geometric-kick-drift')
    assert rf.integrator == 'geometric'


def _resolved_endpoint(integrator, cells):
    rf = xt.Cavity(length=.5, voltage=5e6, frequency=1.3e9, phase=.6, eta=.3,
                   model='phase-resolved', integrator=integrator, num_kicks=cells)
    probe = xt.Particles(p0c=xt.ELECTRON_MASS_EV*np.sqrt(20.**2 - 1),
        mass0=xt.ELECTRON_MASS_EV, x=1e-3, px=2e-3, y=-5e-4, zeta=1e-3,
        delta=1e-3)
    rf.track(probe)
    assert probe.state[0] == 1
    return np.array([probe.x[0], probe.px[0], probe.zeta[0], probe.delta[0]])


@pytest.mark.parametrize('integrator,cells,order', [
    ('uniform', 32, 4), ('geometric', 32, 4), ('yoshida-6', 16, 6),
    ('yoshida-8', 8, 7.5)])
def test_resolved_integrators_reach_their_order(integrator, cells, order):
    """Gauss cells are fourth order; Yoshida compositions raise the order."""
    reference = _resolved_endpoint('yoshida-8', 512)
    coarse, fine = (np.abs(_resolved_endpoint(integrator, n) - reference).max()
                    for n in (cells, 2*cells))
    assert np.log2(coarse/fine) > order - .5


def test_yoshida_4_equals_uniform_for_the_fourth_order_cell():
    np.testing.assert_array_equal(_resolved_endpoint('yoshida-4', 16),
                                  _resolved_endpoint('uniform', 16))


def test_teapot_refused_for_phase_resolved():
    with pytest.raises(ValueError, match='teapot'):
        xt.Cavity(model='phase-resolved', length=.1, integrator='teapot')


def test_frozen_uniform_and_geometric_meshes_share_their_limit():
    def endpoint(integrator, cells):
        rf = xt.Cavity(length=.5, voltage=5e6, frequency=1.3e9, phase=.6,
                       model='phase-resolved-frozen', integrator=integrator,
                       num_kicks=cells)
        probe = xt.Particles(p0c=xt.ELECTRON_MASS_EV*np.sqrt(20.**2 - 1),
            mass0=xt.ELECTRON_MASS_EV, x=1e-3, px=2e-3, zeta=1e-3)
        rf.track(probe)
        return np.array([probe.x[0], probe.px[0], probe.zeta[0], probe.delta[0]])
    gaps = [np.abs(endpoint('uniform', n) - endpoint('geometric', n)).max()
            for n in (32, 128)]
    assert gaps[1] < gaps[0]/3


def _averaged_cavity(model='phase-averaged', **changes):
    controls = dict(length=1.15, voltage=20e6, frequency=1.3e9, phase=1.2,
                    model=model, num_kicks=32)
    controls.update(changes)
    return xt.Cavity(**controls)


def _launch(gamma, **changes):
    values = dict(x=1e-3, px=2e-4, y=-1e-3, py=1e-4, zeta=1e-3, delta=1e-3)
    values.update(changes)
    return xt.Particles(p0c=xt.ELECTRON_MASS_EV*np.sqrt(gamma**2 - 1),
                        mass0=xt.ELECTRON_MASS_EV, **values)


def test_phase_averaged_converges_to_phase_resolved_with_energy():
    """The averaging error falls with g/p (D17)."""
    gaps = []
    for gamma in (200., 2000.):
        ends = []
        for model in ('phase-averaged', 'phase-resolved'):
            probe = _launch(gamma)
            _averaged_cavity(model, num_kicks=64).track(probe)
            assert probe.state[0] == 1
            ends.append(np.array([probe.x[0], probe.px[0], probe.zeta[0]]))
        gaps.append(np.abs(ends[0] - ends[1]).max())
    assert gaps[0] < 3e-8
    assert gaps[1] < gaps[0] / 30


@pytest.mark.parametrize('eta', [0., 1.])
def test_phase_averaged_is_symplectic(eta):
    names = ('x', 'px', 'y', 'py', 'zeta', 'pzeta')
    rf = _averaged_cavity(eta=eta)
    step = 1e-6
    matrix = np.zeros((6, 6))
    for column, name in enumerate(names):
        ends = []
        for sign in (1, -1):
            probe = xt.Particles(p0c=xt.ELECTRON_MASS_EV*np.sqrt(20.**2 - 1),
                mass0=xt.ELECTRON_MASS_EV, x=1e-3, px=2e-4, y=-1e-3, py=1e-4,
                zeta=1e-3, pzeta=1e-3)
            setattr(probe, name, getattr(probe, name) + sign*step)
            rf.track(probe)
            ends.append(np.array([getattr(probe, n)[0] for n in names]))
        matrix[:, column] = (ends[0] - ends[1]) / (2*step)
    form = np.zeros((6, 6))
    for i in range(3):
        form[2*i, 2*i+1], form[2*i+1, 2*i] = 1, -1
    assert np.abs(matrix.T @ form @ matrix - form).max() < 1e-7


def test_phase_averaged_reverses_and_slices_exactly():
    names = ('x', 'px', 'y', 'py', 'zeta', 'delta')
    def line(sliced):
        out = xt.Line(elements={'rf': _averaged_cavity(), 'end': xt.Marker()})
        out.particle_ref = _launch(20., x=0., px=0., y=0., py=0., zeta=0., delta=0.)
        if sliced:
            # Two slices split the 32 parent steps exactly; an uneven split
            # (ceil per slice) gives a different staircase.
            out.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thick'))])
        out.build_tracker()
        out.reset_s_at_end_turn = False
        return out
    whole, sliced = line(False), line(True)
    start = _launch(20.)
    a, b = start.copy(), start.copy()
    whole.track(a, ele_start=0, ele_stop=1)
    sliced.track(b, ele_start=0, ele_stop=sliced.element_names.index('end'))
    for name in names:
        assert getattr(a, name)[0] == getattr(b, name)[0]
    whole.track(a, ele_start=0, ele_stop=1, backtrack=True)
    for name in names:
        np.testing.assert_allclose(getattr(a, name)[0], getattr(start, name)[0],
                                   rtol=0, atol=1e-14)


def test_phase_averaged_tpsa_matches_finite_differences():
    import xtrack.tpsa as xtpsa
    gamma = 20.
    p0c = xt.ELECTRON_MASS_EV*np.sqrt(gamma**2 - 1)
    base = dict(x=1e-3, px=2e-4, y=-1e-3, py=1e-4, zeta=1e-3, delta=1e-3)
    names = ('x', 'px', 'y', 'py', 'zeta', 'delta')
    line = xt.Line(elements={'rf': _averaged_cavity()})
    line.particle_ref = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV)
    line.build_tracker()
    line.reset_s_at_end_turn = False
    series = xtpsa.ParticlesTpsa(order=2, p0c=p0c, mass0=xt.ELECTRON_MASS_EV, **base)
    line.track(series)
    finite = np.zeros((6, 6))
    for column, name in enumerate(names):
        ends = []
        for sign in (1, -1):
            launch = dict(base)
            launch[name] += sign*1e-6
            probe = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV, **launch)
            line.track(probe)
            ends.append(np.array([getattr(probe, n)[0] for n in names]))
        finite[:, column] = (ends[0] - ends[1]) / 2e-6
    np.testing.assert_allclose(series.jacobian(), finite, rtol=0, atol=2e-8)


def test_phase_averaged_refuses_static_fields_and_other_integrators():
    with pytest.raises(ValueError, match='static'):
        _averaged_cavity(ks=.1)
    with pytest.raises(ValueError, match='uniform'):
        _averaged_cavity(integrator='yoshida-6')


@pytest.mark.parametrize('phase', [np.pi/2, 1.])
def test_phase_averaged_spin_follows_phase_resolved(phase):
    """Thomas-BMT rotations of the averaged factors (PLAN P4, D15)."""
    gamma, anomaly = 200., 0.00115965218
    p0c = xt.ELECTRON_MASS_EV*np.sqrt(gamma**2 - 1)
    turned = {}
    for model in ('phase-averaged', 'phase-resolved'):
        line = xt.Line(elements={'rf': _averaged_cavity(model, phase=phase,
                                                        num_kicks=64)})
        line.particle_ref = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV)
        line.build_tracker()
        line.configure_spin(True)
        probe = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV, x=2e-3, y=-1e-3,
            px=1e-4, anomalous_magnetic_moment=anomaly, spin_z=1.)
        line.track(probe)
        assert probe.state[0] == 1
        turned[model] = np.array([probe.spin_x[0], probe.spin_y[0],
                                  probe.spin_z[0] - 1.])
    gap = np.linalg.norm(turned['phase-averaged'] - turned['phase-resolved'])
    assert gap < .03*np.linalg.norm(turned['phase-resolved'])
