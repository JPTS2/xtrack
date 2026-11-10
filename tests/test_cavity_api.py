"""Public Cavity interface: selectors, profiles, serialization, live controls and
slices. The physics of the maps is tested in test_cavity_lorentz.py, test_cavity_limits.py
and test_cavity_maps.py.
"""
import numpy as np
import pytest
from scipy.constants import c as clight
import xobjects as xo
import xtrack as xt
from _cavity_context import kernel_context


MODELS = ('_phase-resolved',)


COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')


@pytest.fixture(scope='module')
def model_context(tmp_path_factory):
    directory = tmp_path_factory.mktemp('cavity-model-wiring')
    with kernel_context(directory) as context:
        yield context


def cavity(model='_phase-resolved', **kwargs):
    controls = dict(length=.05, voltage=20e3, frequency=400e6, phase=.7,
                    counter_wave_amplitude=.5, counter_wave_phase=.3,
                    model=model, integrator='uniform', num_kicks=16)
    if model == '_phase-resolved-frozen':
        controls.update(counter_wave_amplitude=0., integrator='_geometric')
    controls.update(kwargs)
    return xt.MagnetOverlappedCavity(**controls)


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
    explicit = dict(model='_phase-resolved', counter_wave_amplitude=0., n_cells=3,
        edge_entry_active=0, edge_exit_active=0, order=2,
        knl=[0., .1, 0.], nominal_energy_gain=0.)
    data = xt.MagnetOverlappedCavity(**explicit).to_dict()
    assert set(data) == {'__class__', *explicit}
    restored = xt.MagnetOverlappedCavity.from_dict(data)
    for name, value in explicit.items():
        np.testing.assert_array_equal(getattr(restored, name), value)
    assert restored.to_dict() == data


def test_physical_controls_round_trip():
    original = cavity(knl=[0., .012], ksl=[0., -.003], ks=.2)
    assert original.n_cells is None
    assert original.counter_wave_amplitude == .5
    copy = xt.MagnetOverlappedCavity.from_dict(original.to_dict())
    assert copy.n_cells is None
    assert copy.model == original.model
    np.testing.assert_array_equal(copy.knl, original.knl)
    np.testing.assert_array_equal(copy.ksl, original.ksl)
    copy.n_cells = 2
    assert xt.MagnetOverlappedCavity.from_dict(copy.to_dict()).n_cells == 2
    copy.n_cells = None
    assert 'n_cells' not in copy.to_dict()


def test_cells_fix_the_wave_number_whatever_the_frequency():
    rf = xt.Cavity(length=1.2, frequency=704.4e6, n_cells=6)
    np.testing.assert_allclose(rf.cell_length, .2, rtol=1e-15)
    np.testing.assert_allclose(rf.wave_number, np.pi/.2, rtol=1e-15)
    np.testing.assert_allclose(rf.synchronous_beta, 2*704.4e6*.2/clight, rtol=1e-15)
    rf.frequency = 650e6
    np.testing.assert_allclose(rf.cell_length, .2, rtol=1e-15)
    np.testing.assert_allclose(rf.wave_number, np.pi/.2, rtol=1e-15)


def test_without_cells_the_waves_travel_at_c():
    rf = xt.Cavity(length=1.2, frequency=704.4e6)
    assert rf.cell_length is None
    assert rf.synchronous_beta == 1.
    np.testing.assert_allclose(rf.wave_number, 2*np.pi*704.4e6/clight, rtol=1e-15)


def test_serialization_preserves_tiny_coefficients_and_active_order():
    rf = cavity(knl=[0., 1e-19], ksl=[0., -2e-20])
    rf.order = 0
    restored = xt.MagnetOverlappedCavity.from_dict(rf.to_dict())
    assert restored.order == 0
    np.testing.assert_array_equal(restored.knl, rf.knl)
    np.testing.assert_array_equal(restored.ksl, rf.ksl)


@pytest.mark.parametrize('gain', [None, 0., 12000.])
def test_frozen_nominal_programme_round_trip(gain):
    rf = cavity('_phase-resolved-frozen', nominal_energy_gain=gain)
    restored = xt.Cavity.from_dict(rf.to_dict())
    assert restored.nominal_energy_gain == gain
    assert restored.nominal_energy_gain_from_rf == (gain is None)


def test_frozen_nominal_programme_live_expression():
    line = xt.Line(elements={'rf': cavity('_phase-resolved-frozen')})
    line.vars['programme_gain'] = 0.
    line.set('rf', nominal_energy_gain='programme_gain')
    assert line['rf'].nominal_energy_gain == 0.
    assert not line['rf'].nominal_energy_gain_from_rf
    line.vars['programme_gain'] = 12000.
    assert line['rf'].nominal_energy_gain == 12000.
    assert not line['rf'].nominal_energy_gain_from_rf


@pytest.mark.parametrize('model', (*MODELS, '_phase-resolved-frozen'))
def test_thin_source_strength_readback_follows_parent_model(model_context, model):
    line = xt.Line(elements={'rf': cavity(model, voltage=20e3,
                                         _context=model_context,
                                         knl=[0., .012], ksl=[0., -.003])})
    line.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thin'))])
    line.build_tracker(_context=model_context)
    # These voltages are nominal parent-input allocations; the resolved
    # source's electric-crest integral over a partial interval is not weight*V.
    source = np.array([isinstance(line[name], xt.DriftSliceMagnetOverlappedCavity)
                       for name in line.element_names])
    marker = np.array([isinstance(line[name], xt.ThinSliceMagnetOverlappedCavity)
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
    intervals = np.array([isinstance(line[name], xt.DriftSliceMagnetOverlappedCavity)
                          for name in line.element_names])
    for name, value in [('shift_x', .0002), ('shift_y', -.0001),
                        ('rot_s_rad', .2), ('rot_y_rad', .001)]:
        np.testing.assert_array_equal(line.attr[name][intervals], value)
    line.set('rf', model='adaptive')
    np.testing.assert_array_equal(line.attr['shift_x'][intervals], 0.)
    line.set('rf', model='_phase-resolved', shift_x=.0003)
    np.testing.assert_array_equal(line.attr['shift_x'][intervals], .0003)
    # Automatic drift transforms stay off; its new source helper owns them.
    assert all(not line[name].rot_and_shift_from_parent
               for name in line.element_names
               if isinstance(line[name], xt.DriftSliceMagnetOverlappedCavity))


@pytest.mark.parametrize('model', MODELS)
def test_line_variables_update_actual_source(model_context, model):
    line = xt.Line(elements={'rf': cavity(model, _context=model_context)})
    line.particle_ref = particle(model_context)
    line.vars['rf_amplitude'] = .5
    line.vars['rf_phase'] = .7
    line.element_refs['rf'].counter_wave_amplitude = line.vars['rf_amplitude']
    line.element_refs['rf'].phase = line.vars['rf_phase']
    line.build_tracker(_context=model_context)
    line.reset_s_at_end_turn = False
    original = particle(model_context)
    initial = original.copy()
    line.track(original)
    assert original.state[0] == 1
    line.vars['rf_amplitude'] = .8
    line.vars['rf_phase'] = 1.1
    line.set('rf', voltage=35e3)
    changed = initial.copy()
    line.track(changed)
    assert changed.state[0] == 1
    assert not np.array_equal(coordinates(original), coordinates(changed))
    fresh = cavity(model, counter_wave_amplitude=.8, phase=1.1, voltage=35e3,
                   _context=model_context)
    expected = initial.copy()
    fresh.track(expected)
    np.testing.assert_allclose(physical_coordinates(changed),
                               physical_coordinates(expected),
                               rtol=0, atol=3e-13)
    line.vars['rf_amplitude'] = .5
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
    kind = (xt.ThickSliceMagnetOverlappedCavity if mode == 'thick'
            else xt.DriftSliceMagnetOverlappedCavity)
    names = [name for name in restored.element_names
             if isinstance(restored[name], kind)]
    assert len(names) == (2 if mode == 'thick' else 3)
    second = restored[names[1]]
    offset = second.slice_offset
    # Optimization deliberately disables variable management, as for other
    # Xtrack elements. Direct physical-field changes still retain parent links.
    restored['rf'].length = .1
    assert second.slice_offset == 2*offset


@pytest.mark.parametrize('model', ['adaptive', '_phase-resolved'])
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
    restored._parent = xt.Cavity(length=.05, _buffer=restored._buffer)
    assert restored.slice_offset == .025


@pytest.mark.parametrize('model,mode', [('_phase-resolved-frozen', 'thick'),
    ('_phase-resolved-frozen', 'thin'), ('_phase-resolved', 'thick')])
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


@pytest.mark.parametrize('model', ('_phase-resolved', '_phase-resolved-frozen'))
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


def test_legacy_models_ignore_physical_model_fields(model_context):
    plain = xt.Cavity(_context=model_context, voltage=20e3, frequency=400e6,
                      phase=.7)
    dressed = xt.MagnetOverlappedCavity(_context=model_context, voltage=20e3, frequency=400e6,
        phase=.7, ks=.1, knl=[0., .01], ksl=[0., .02], n_cells=1,
        counter_wave_amplitude=.55)
    expected = particle(model_context)
    actual = expected.copy()
    plain.track(expected)
    dressed.track(actual)
    assert actual.state[0] == 1
    np.testing.assert_array_equal(coordinates(actual), coordinates(expected))


def test_teapot_refused_for_phase_resolved():
    with pytest.raises(ValueError, match='teapot'):
        xt.Cavity(model='_phase-resolved', length=.1, integrator='teapot')


def _averaged_cavity(model='_phase-averaged', **changes):
    controls = dict(length=1.15, voltage=20e6, frequency=1.3e9, phase=1.2,
                    model=model, num_kicks=32)
    controls.update(changes)
    return xt.MagnetOverlappedCavity(**controls)


def _launch(gamma, **changes):
    values = dict(x=1e-3, px=2e-4, y=-1e-3, py=1e-4, zeta=1e-3, delta=1e-3)
    values.update(changes)
    return xt.Particles(p0c=xt.ELECTRON_MASS_EV*np.sqrt(gamma**2 - 1),
                        mass0=xt.ELECTRON_MASS_EV, **values)


@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved-paraxial'])
def test_phase_averaged_refuses_unsupported_static_fields_and_integrators(model):
    with pytest.raises(ValueError, match='static'):
        _averaged_cavity(model, ks=.1)                     # solenoid with active faces
    with pytest.raises(ValueError, match='static'):
        _averaged_cavity(model, knl=[0., 0., .1], order=2)  # sextupole
    for integrator in ('teapot', 'yoshida-6'):
        with pytest.raises(ValueError, match='_geometric'):
            _averaged_cavity(model, integrator=integrator)


def test_phase_averaged_with_static_fields_loses_a_spinning_particle():
    line = xt.Line(elements={'rf': _averaged_cavity(knl=[0., .05])})
    line.particle_ref = _launch(200., x=0., px=0., y=0., py=0., zeta=0., delta=0.)
    line.build_tracker()
    line.configure_spin(True)
    particles = _launch(200., spin_z=1.)
    line.track(particles)
    assert particles.state[0] == -99744


def _adaptive_line(model_context, **controls):
    line = xt.Line(elements={'rf': xt.Cavity(**controls)})
    line.particle_ref = xt.Particles(p0c=1e9)
    line.build_tracker(_context=model_context)
    return line


def test_plan_follows_element_changes_between_track_calls(model_context):
    line = _adaptive_line(model_context, model='drift-kick-drift-expanded', length=1.,
                          frequency=1e9, voltage=1e5, phase=np.pi / 2)
    def gain():
        probe = line.build_particles(zeta=0.)
        energy = probe.energy[0]
        line.track(probe)
        return probe.energy[0] - energy
    first = gain()
    line['rf'].voltage = 2e5
    np.testing.assert_allclose(gain(), 2 * first, rtol=1e-12)
    line['rf'].model = '_phase-resolved'
    assert gain() != 2 * first

