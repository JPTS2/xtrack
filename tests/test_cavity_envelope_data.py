"""Fixed prepared-table storage checks, independent of scientific preparation."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import xobjects as xo

from xtrack.beam_elements.cavity_envelope_data import EnvelopeTableData
from test_cavity_native import native_context


def sample_table():
    rng = np.random.default_rng(7321)
    modes = np.array([[0, 0], [0, 1], [0, -1], [1, -1], [-1, 1]])
    shape = (5, len(modes), 4, 3, 4)
    body = rng.normal(size=shape)+1j*rng.normal(size=shape)
    boundary = rng.normal(size=shape)+1j*rng.normal(size=shape)
    for values in (body, boundary):
        values[:, 0].imag = 0.
        values[:, 2] = values[:, 1].conj()
        values[:, 4] = values[:, 3].conj()
        values[0, 0, 0, 0, 0] = complex(-0., -0.)
    return SimpleNamespace(energy_interval=(1.2, 4.),
        position_interval=(-.4, .4), nu=.9, anchor=-.3,
        profile_width=.2, profile_centre=.025, modes=modes,
        body_coefficients=body, boundary_coefficients=boundary)


def test_table_payload_preserves_bytes_and_json():
    source = sample_table()
    payload = EnvelopeTableData.from_table(source)
    restored = EnvelopeTableData.from_dict(json.loads(json.dumps(payload.to_dict())))
    for result in (payload, restored):
        assert np.array_equal(result.modes, source.modes)
        assert result.energy_interval == source.energy_interval
        assert result.position_interval == source.position_interval
        for name in ('body_coefficients', 'boundary_coefficients'):
            assert getattr(result, name).tobytes() == getattr(source, name).tobytes()
    copied = payload.copy()
    assert copied.body_coefficients.tobytes() == source.body_coefficients.tobytes()
    # Changing an inspection copy must not mutate the native payload.
    payload.body_coefficients[:] = 123
    assert payload.body_coefficients.tobytes() == source.body_coefficients.tobytes()
    with pytest.raises(ValueError, match='read-only'):
        payload.body_real[0] = 123.
    with pytest.raises(AttributeError, match='fixed'):
        payload.anchor = -.2


@pytest.mark.parametrize('field,value', [
    ('energy_interval', (1., 4.)), ('position_interval', (.4, -.4)),
    ('anchor', .5), ('profile_width', 0.), ('nu', np.nan),
    ('modes', [[0, 1]]),
])
def test_table_rejects_invalid_fixed_hardware(field, value):
    source = sample_table()
    setattr(source, field, value)
    with pytest.raises(ValueError):
        EnvelopeTableData.from_table(source)


def test_table_rejects_nonreal_fourier_and_mismatched_degrees():
    source = sample_table()
    source.body_coefficients[0, 1, 0, 0, 0] += .1j
    with pytest.raises(ValueError, match='conjugacy'):
        EnvelopeTableData.from_table(source)
    source = sample_table()
    source.boundary_coefficients = source.boundary_coefficients[..., :-1]
    with pytest.raises(ValueError, match='degrees'):
        EnvelopeTableData.from_table(source)


def test_bound_normalization_metadata_survives_copy_and_json():
    original = bound_table()
    for result in (EnvelopeTableData.from_table(original),
                   EnvelopeTableData.from_dict(
                       json.loads(json.dumps(original.to_dict())))):
        assert result.normalization_frequency == original.normalization_frequency
        assert result.normalization_wave_number == original.normalization_wave_number
    with pytest.raises(ValueError, match='bound together'):
        EnvelopeTableData.from_table(sample_table(), normalization_frequency=400e6)
    with pytest.raises(ValueError, match='must equal'):
        EnvelopeTableData.from_table(sample_table(), normalization_frequency=400e6,
                                    normalization_wave_number=1.)


def bound_table(*, vacuum=False):
    """An explicit polynomial payload isolates wiring from offline physics."""
    from scipy.constants import c
    source = sample_table()
    # Low strengths keep this synthetic scalar comfortably inside its domain.
    source.body_coefficients *= 1e-3
    source.boundary_coefficients *= 1e-3
    if vacuum:
        source.body_coefficients[:] = 0.
        source.boundary_coefficients[:] = 0.
    frequency = 400e6
    return EnvelopeTableData.from_table(source,
        normalization_frequency=frequency,
        normalization_wave_number=2*np.pi*frequency/c)


def envelope_controls():
    from scipy.constants import c
    k = 2*np.pi*400e6/c
    return dict(model='phase-averaged-envelope', length=.8/k,
        frequency=400e6, voltage=2e3, eta=.25, counter_phase=.3,
        phase=.4, calibration_beta=.9, rf_profile='gaussian',
        rf_profile_sigma=.2/k, rf_profile_center=.025/k,
        wave_number=.9*k, num_kicks=8, num_factor_steps=2,
        absolute_time=True)


def test_public_attachment_metadata_roundtrip():
    """The table is a typed owned payload, including after normal JSON load."""
    import xtrack as xt
    source = bound_table()
    original = xt.Cavity(envelope_table=source, **envelope_controls())
    restored = xt.Cavity.from_dict(json.loads(json.dumps(original.to_dict(), cls=xo.JEncoder)))
    for result in (original, original.copy(), restored):
        assert result.envelope_table._buffer is result._buffer
        assert result.envelope_table.normalization_frequency == 400e6
        assert result.envelope_table.body_coefficients.tobytes() == \
               source.body_coefficients.tobytes()
        assert result.model == 'phase-averaged-envelope'
    assert original.envelope_table._buffer is not source._buffer


@pytest.mark.parametrize('model', ('adaptive', 'phase-resolved'))
def test_cavity_without_table_remains_movable(model):
    """Adding an optional Ref must not block every existing Cavity Line build."""
    import xtrack as xt
    element = xt.Cavity(model=model, length=.1, voltage=1000., frequency=400e6)
    destination = xo.ContextCpu().new_buffer()
    element.move(_buffer=destination)
    assert element._buffer is destination
    assert element.envelope_table is None
    assert element.voltage == 1000.
    assert element.model == model


def launch(context, **changes):
    import xtrack as xt
    parameters = dict(_context=context, mass0=xt.ELECTRON_MASS_EV,
        p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1.),
        x=2e-5, px=3e-4, y=-3e-5, py=-2e-4,
        zeta=4e-4, delta=1e-3, time_s=1.9e-9)
    parameters.update(changes)
    return xt.Particles(**parameters)


def coordinates(particles):
    assert particles.state[0] == 1
    return np.array([getattr(particles, name)[0] for name in
                     ('x', 'px', 'y', 'py', 'zeta', 'delta')])


@pytest.fixture(scope='module')
def envelope_context(tmp_path_factory):
    with native_context(tmp_path_factory.mktemp('envelope-native')) as context:
        yield context


def test_public_envelope_copy_json_uses_identical_payload(envelope_context):
    import xtrack as xt
    original = xt.Cavity(_context=envelope_context,
        envelope_table=bound_table(), **envelope_controls())
    reconstructed = xt.Cavity.from_dict(
        json.loads(json.dumps(original.to_dict(), cls=xo.JEncoder)),
        _context=envelope_context)
    expected = None
    for element in (original, original.copy(_context=envelope_context), reconstructed):
        line = xt.Line(elements=[element, xt.Marker()],
                       element_names=['rf', 'end'])
        line.reset_s_at_end_turn = False
        line.build_tracker(_context=envelope_context)
        probe = launch(envelope_context)
        line.track(probe, ele_start='rf', ele_stop='end')
        actual = coordinates(probe)
        if expected is None:
            expected = actual
        else:
            np.testing.assert_array_equal(actual, expected)
        line.track(probe, ele_start='rf', ele_stop='end', backtrack=True)
        np.testing.assert_allclose(coordinates(probe),
            coordinates(launch(envelope_context)), atol=3e-12, rtol=0.)
        assert element.envelope_table.body_coefficients.tobytes() == \
               original.envelope_table.body_coefficients.tobytes()


def test_public_envelope_zero_rf_is_h2_vacuum(envelope_context):
    import xtrack as xt
    controls = envelope_controls()
    controls['voltage'] = 0.
    prepared = xt.Cavity(_context=envelope_context,
        envelope_table=bound_table(), **controls)
    ordinary = xt.Cavity(_context=envelope_context,
        **dict(controls, model='phase-resolved-paraxial', integrator='gauss2'))
    actual, expected = launch(envelope_context), launch(envelope_context)
    prepared.track(actual)
    ordinary.track(expected)
    np.testing.assert_allclose(coordinates(actual), coordinates(expected),
                               atol=3e-13, rtol=0.)


@pytest.mark.parametrize('field,factor', [
    ('frequency', 1.01), ('length', 1.01), ('wave_number', 1.01),
    ('rf_profile_sigma', 1.01), ('rf_profile_center', 1.01),
])
def test_public_envelope_refuses_changed_hardware(envelope_context, field, factor):
    import xtrack as xt
    element = xt.Cavity(_context=envelope_context,
        envelope_table=bound_table(), **envelope_controls())
    setattr(element, field, getattr(element, field)*factor)
    probe = launch(envelope_context)
    before = coordinates(probe)
    element.track(probe)
    assert probe.state[0] < 0
    after = np.array([getattr(probe, name)[0] for name in
                      ('x', 'px', 'y', 'py', 'zeta', 'delta')])
    np.testing.assert_array_equal(after, before)


def test_public_envelope_live_rf_parameters_equal_fresh(envelope_context):
    import xtrack as xt
    controls = envelope_controls()
    element = xt.Cavity(_context=envelope_context,
        envelope_table=bound_table(), **controls)
    for changes in (dict(eta=.49, phase=.7),
                    dict(counter_phase=-.2, calibration_beta=.8, voltage=4e3)):
        controls.update(changes)
        for name, value in changes.items():
            setattr(element, name, value)
        fresh = xt.Cavity(_context=envelope_context,
            envelope_table=bound_table(), **controls)
        actual, expected = launch(envelope_context), launch(envelope_context)
        element.track(actual)
        fresh.track(expected)
        np.testing.assert_array_equal(coordinates(actual), coordinates(expected))


@pytest.mark.parametrize('field', ('frequency', 'wave_number',
                                   'rf_profile_sigma', 'rf_profile_center'))
def test_public_envelope_refuses_seeded_fixed_hardware(envelope_context, field):
    """Matching base points must not conceal unsupported hardware derivatives."""
    import madng_tpsa
    import xtrack as xt
    from xtrack.tpsa import ParticlesTpsa
    controls = envelope_controls()
    element = xt.Cavity(_context=envelope_context,
        envelope_table=bound_table(), **controls)
    desc = madng_tpsa.Descriptor(6, 2, num_params=1, param_order=2)
    scalar = launch(envelope_context)
    tracked = ParticlesTpsa(order=2, descriptor=desc, _context=envelope_context,
        mass0=float(scalar.mass0), p0c=float(scalar.p0c[0]),
        x=float(scalar.x[0]), px=float(scalar.px[0]),
        y=float(scalar.y[0]), py=float(scalar.py[0]),
        zeta=float(scalar.zeta[0]), delta=float(scalar.delta[0]),
        time_s=float(scalar.time_s))
    initial = tracked.const_part.copy()
    try:
        setattr(element, field, desc.param(1, controls[field]))
        with pytest.raises(RuntimeError, match='TPSA map lost'):
            element.track(tracked)
        assert tracked._xobject.state < 0
        np.testing.assert_array_equal(tracked.const_part, initial)
    finally:
        element.disable_tpsa()


def test_public_envelope_phase_eta_calibration_gradients(envelope_context):
    """Native TPSA differentiates the executed fixed-table finite map."""
    import madng_tpsa
    import xtrack as xt
    from xtrack.tpsa import ParticlesTpsa
    controls = envelope_controls()
    element = xt.Cavity(_context=envelope_context,
        envelope_table=bound_table(), **controls)
    desc = madng_tpsa.Descriptor(6, 2, num_params=3, param_order=2)
    scalar = launch(envelope_context)
    element.phase = desc.param(1, controls['phase'])
    element.eta = desc.param(2, controls['eta'])
    element.calibration_beta = desc.param(3, controls['calibration_beta'])
    tracked = ParticlesTpsa(order=2, descriptor=desc, _context=envelope_context,
        mass0=float(scalar.mass0), p0c=float(scalar.p0c[0]),
        x=float(scalar.x[0]), px=float(scalar.px[0]),
        y=float(scalar.y[0]), py=float(scalar.py[0]),
        zeta=float(scalar.zeta[0]), delta=float(scalar.delta[0]),
        time_s=float(scalar.time_s))
    try:
        element.track(tracked)
        assert tracked._xobject.state == 1
        derivatives = tracked.param_jacobian().copy()
        constant = tracked.const_part.copy()
    finally:
        element.disable_tpsa()
    for name in ('phase', 'eta', 'calibration_beta'):
        setattr(element, name, controls[name])
    element.track(scalar)
    np.testing.assert_allclose(constant, coordinates(scalar), atol=3e-13, rtol=0.)
    for column, name in enumerate(('phase', 'eta', 'calibration_beta')):
        endpoints = []
        for sign in (-1, 1):
            setattr(element, name, controls[name]+sign*1e-5)
            probe = launch(envelope_context)
            element.track(probe)
            endpoints.append(coordinates(probe))
        setattr(element, name, controls[name])
        expected = (endpoints[1]-endpoints[0])/2e-5
        np.testing.assert_allclose(derivatives[:, column], expected,
                                   rtol=2e-5, atol=2e-10)


def test_prepared_physical_table_taper_closes_restored_map(envelope_context):
    """Known loss checks orchestration and saved hardware, not radiation law."""
    import gzip
    from pathlib import Path
    import xtrack as xt
    path = (Path(__file__).resolve().parents[1]/'examples'/'cavity_models'/
            '005_prepared_envelope.json.gz')
    with gzip.open(path, 'rt') as stream:
        table = EnvelopeTableData.from_dict(json.load(stream)['table'])
    k = table.normalization_wave_number
    rf = xt.Cavity(model='phase-averaged-envelope', envelope_table=table,
        frequency=table.normalization_frequency,
        length=(table.position_max-table.position_min)/k,
        wave_number=table.nu*k, rf_profile='gaussian',
        rf_profile_sigma=table.profile_width/k,
        rf_profile_center=table.profile_centre/k,
        voltage=2e4, eta=.25, counter_phase=.3, calibration_beta=.9,
        phase=.2, num_kicks=8, num_factor_steps=2)
    line = xt.Line(elements=[
        xt.LineSegmentMap(length=1., qx=.23, qy=.31, betx=2., bety=3.,
            longitudinal_mode='frozen', energy_increment=-100.),
        xt.Multipole(knl=[0.]), rf],
        element_names=['sink', 'taper_slot', 'rf'])
    line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1.,
        p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1.))
    line.build_tracker(_context=envelope_context)
    hardware = {name:getattr(rf, name) for name in
                ('voltage', 'frequency', 'harmonic', 'phase', 'lag')}
    original_coefficients = rf.envelope_table.body_coefficients.tobytes()
    line.compensate_radiation_energy_loss(delta0=0., rtol_eneloss=1e-12,
                                         verbose=False)
    for name, value in hardware.items():
        assert getattr(rf, name) == value
    assert rf.envelope_table.body_coefficients.tobytes() == original_coefficients
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
