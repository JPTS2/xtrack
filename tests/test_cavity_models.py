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
    if model == 'phase-resolved-frozen':
        controls.update(eta=0., integrator='geometric-kick-drift')
    if model == 'phase-averaged-magnetic':
        controls.update(rf_profile='gaussian', rf_profile_sigma=.02,
                        integrator='midpoint')
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
        if model=='phase-averaged-envelope':
            with pytest.raises(ValueError,match='envelope_table'):
                xt.Cavity(model=model,rf_profile='gaussian',rf_profile_sigma=.02)
        else:
            assert cavity(model).model == model
        assert model not in xt.CrabCavity.get_available_models()
    assert xt.Cavity().model == 'adaptive'
    assert xt.Cavity().integrator == 'adaptive'
    assert 'gauss2' not in xt.Quadrupole.get_available_integrators()
    assert 'midpoint' not in xt.Quadrupole.get_available_integrators()


@pytest.mark.parametrize('model,profile', [(None, 'p16'),
    ('phase-resolved', 'p16'), ('phase-averaged-full-slow', 'p16'),
    ('phase-resolved-frozen', 'uniform'),
    ('phase-averaged-uniform-quadratic-first-boundary', 'uniform'),
    ('phase-averaged-uniform-quadratic', 'uniform')])
def test_builtin_profile_inference_is_stored(model, profile):
    rf = xt.Cavity(**({} if model is None else {'model': model}))
    assert rf.rf_profile == profile
    data = rf.to_dict()
    assert data['rf_profile'] == profile
    assert data['rf_profile_sigma'] == 0.
    assert data['rf_profile_center'] == 0.
    restored = xt.Cavity.from_dict(data)
    assert restored.rf_profile == profile
    assert restored.rf_profile_sigma == restored.rf_profile_center == 0.


@pytest.mark.parametrize('sigma', [None, 0., -.01, np.inf, np.nan])
def test_gaussian_profile_never_guesses_a_width(sigma):
    with pytest.raises(ValueError, match='width'):
        xt.Cavity(model='phase-averaged-magnetic', rf_profile='gaussian',
                   rf_profile_sigma=sigma)


def test_builtin_profile_model_changes_do_not_retarget_source():
    rf = cavity(rf_profile='uniform')
    rf.model = 'phase-averaged-full-slow'
    assert rf.rf_profile == 'uniform'
    assert xt.Cavity.from_dict(rf.to_dict()).rf_profile == 'uniform'
    # Raw enum zero must not disappear and acquire a different inferred source.
    rf.rf_profile = 'p16'
    rf.model = 'phase-resolved-frozen'
    assert rf.rf_profile == rf.to_dict()['rf_profile'] == 'p16'
    with pytest.raises(ValueError, match='uniform'):
        xt.Cavity.from_dict(rf.to_dict())


def test_gaussian_profile_live_metres_and_roundtrip():
    line = xt.Line(elements={'rf': cavity('phase-averaged-magnetic',
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
    ('phase-averaged-uniform-quadratic', 'gaussian'),
    ('phase-averaged-magnetic', 'p16'), ('adaptive', 'gaussian')])
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
    assert rf._buffer is buffer and rf.envelope_table is None
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
    assert copied.envelope_table is None


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


@pytest.mark.parametrize('controls', [dict(ks=.1), dict(knl=[0., .1]),
    dict(ksl=[0., .1]), dict(wave_number=0.), dict(rf_field_order=0),
    dict(radiation_flag=1), dict(spin_tracking=1)])
def test_private_diagnostic_does_not_ignore_new_public_controls(controls):
    from xtrack.beam_elements.cavity_native import prepare_p16_native_data

    rf = xt.Cavity(length=.05, frequency=400e6, voltage=2e4,
                    absolute_time=True, **controls)
    with pytest.raises(ValueError, match='private native diagnostic'):
        prepare_p16_native_data(rf, particle(), eta=.25, radius_m=.001)


@pytest.mark.parametrize('model', (*MODELS, 'phase-averaged-second-order',
                                  'phase-averaged-cubic',
                                  'phase-averaged-quartic'))
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
def test_saved_static_taper_is_physical_field_setting(model_context, model):
    controls = dict(voltage=0., frequency=0., ks=.2,
                    knl=[.0001, .012], ksl=[-.0002, -.003])
    prepared = cavity(model, _context=model_context, delta_taper=.07, **controls)
    explicit = cavity(model, _context=model_context, delta_taper=0.,
        **dict(controls, ks=.2*1.07,
               knl=np.array(controls['knl'])*1.07,
               ksl=np.array(controls['ksl'])*1.07))
    initial = particle(model_context)
    for no_sr in (True, False):
        line = xt.Line(elements={'rf': prepared})
        line.build_tracker(_context=model_context)
        line.config.XTRACK_MULTIPOLE_NO_SYNRAD = no_sr
        a = initial.copy(); b = initial.copy()
        line.track(a, ele_start=0, ele_stop=1)
        explicit.track(b)
        assert a.state[0] == b.state[0] == 1
        np.testing.assert_allclose(coordinates(a), coordinates(b),
                                   rtol=0., atol=2e-13)


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


@pytest.mark.parametrize('model', ('phase-resolved-frozen',
                                  'phase-averaged-cubic',
                                  'phase-averaged-quartic'))
@pytest.mark.parametrize('mode', ('thick', 'thin'))
def test_finite_parent_words_survive_layout_partition(model_context, model, mode):
    rf = cavity(model, _context=model_context, num_kicks=4,
                num_factor_steps=1, edge_entry_active=0, edge_exit_active=1)
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
    reference['rf'].model = 'drift-kick-drift-expanded'
    reference.reset_s_at_end_turn = False
    reference.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thin'))])
    reference.build_tracker(_context=model_context)
    expected_legacy = initial.copy()
    reference.track(expected_legacy)
    line['rf'].model = 'drift-kick-drift-expanded'
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


def test_taper_saved_profile_uses_restored_actual_map(model_context):
    from test_cavity_integration_review import taper_ring

    line = taper_ring(model_context, 'phase-resolved', strong=True)
    line.compensate_radiation_energy_loss(delta0=0., max_iter=10,
                                         rtol_eneloss=1e-12, verbose=False)
    probe = line.particle_ref.copy()
    energy0 = float(probe.energy[0])
    line.track(probe, turn_by_turn_monitor='ONE_TURN_EBE')
    assert probe.state[0] == 1
    monitor = line.record_last_track
    means = .5*(monitor.delta[0, :-1]+monitor.delta[0, 1:])
    item = line.attr._cache['delta_taper']
    tolerance = 1e-12*energy0/float(probe.beta0[0]*probe.p0c[0])
    np.testing.assert_allclose(item.multisetter.get_values(), means[item.mask],
                               rtol=0., atol=tolerance)
    assert abs(float(probe.energy[0])-energy0) <= 1e-12*energy0


def test_default_taper_zero_mean_uses_restored_actual_map(model_context):
    from test_cavity_integration_review import taper_ring

    line = taper_ring(model_context, 'phase-resolved', strong=True)
    line.compensate_radiation_energy_loss(max_iter=10,
                                         rtol_eneloss=1e-12, verbose=False)
    entrance_delta = float(line.record_last_track.delta[0, 0])
    probe = line.particle_ref.copy()
    probe.delta = entrance_delta
    energy0 = float(probe.energy[0])
    delta_tolerance = 1e-12*energy0/float(
        probe.p0c[0]*probe.beta0[0]*probe.rvv[0])
    line.track(probe, turn_by_turn_monitor='ONE_TURN_EBE')
    assert probe.state[0] == 1
    monitor = line.record_last_track
    means = .5*(monitor.delta[0, :-1]+monitor.delta[0, 1:])
    positions = np.r_[line._get_s_elements(), line.get_length()]
    mean_delta = np.sum(means*np.diff(positions))/positions[-1]
    assert abs(mean_delta) <= delta_tolerance
    item = line.attr._cache['delta_taper']
    np.testing.assert_allclose(item.multisetter.get_values(), means[item.mask],
                               rtol=0., atol=delta_tolerance)
    assert abs(float(probe.energy[0])-energy0) <= 1e-12*energy0


def test_taper_refuses_slice_before_inspecting_numeric_storage(model_context):
    line = xt.Line(elements={'rf': cavity(_context=model_context)})
    line.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thin'))])
    line.build_tracker(_context=model_context)
    assert line.tracker._tracker_data_base.cache.get('attr') is None
    with pytest.raises(ValueError, match='Cavity slices'):
        line.compensate_radiation_energy_loss(verbose=False)
    assert line.tracker._tracker_data_base.cache.get('attr') is None
