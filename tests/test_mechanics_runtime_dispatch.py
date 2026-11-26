"""Consumed-source dispatch retains receiver identity and ordered rotations."""

import numpy as np
import pytest
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt
from xtrack.mechanics.runtime import PreparedMechanics, FIELDS
from xtrack.mechanics.model import MechanicalTimeSeries
from xtrack.mechanics.local_deformations import rotation_increment


@for_all_test_contexts
def test_mixed_native_kernel_reduces_interleaved_rows_in_declared_order(test_context):
    from xtrack.mechanics.runtime import _compile
    from xtrack.mechanics.context_arrays import to_context

    _compile(test_context)
    starts = np.array([0, 3, 4, 4], dtype=np.int64)
    rows = np.array([-1, 0, -1, 1], dtype=np.int64)
    data = np.array(
        [
            [3.0, 0.2, 0.0, np.pi / 2, 0.0, 0.0, 0.0, 0.0],
            np.zeros(8),
            [5.0, 0.4, 0.0, np.pi / 2, 0.0, 0.0, 0.0, 0.0],
            np.zeros(8),
        ]
    )
    samples = np.array(
        [[1e-9, 2e-9, 0.0, 0.0, 0.3, 0.0], [0.0, -1e-9, 0.0, 1e-12, -2e-12, 3e-12]]
    )
    output = test_context.zeros(3 * 12, dtype=np.float64)
    upload = lambda values: to_context(test_context, values)
    test_context.kernels.mechanics_sources_mixed(
        n=3,
        source_starts=upload(starts),
        source_rows=upload(rows),
        source_data=upload(data.ravel()),
        sample_values=upload(samples.ravel()),
        time_s=0.2,
        dynamic_local=output,
    )
    actual = test_context.nparray_from_context_array(output).reshape(3, 12)
    expected = np.zeros((3, 12))
    contributions = [
        [
            np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),
            samples[0],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.4]),
        ],
        [samples[1]],
        [],
    ]
    for row, values in zip(expected, contributions):
        for value in values:
            row[:3] += value[:3]
            left, right = rotation_increment(*value[3:]), row[3:].reshape(3, 3)
            row[3:] = (left + right + left @ right).ravel()
    np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=1e-25, rtol=2e-14)
    np.testing.assert_allclose(actual[:, 3:], expected[:, 3:], atol=3e-16, rtol=3e-14)


def make_line(context):
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=0.5, k1=0.1, shift_x=0.003, rot_s_rad=0.03)
    env.new('d', xt.Drift, length=0.5)
    line = env.new_line(name='beam', components=['q', 'd'])
    line.set_particle_ref('proton', p0c=7e12)
    line.build_tracker(_context=context)
    env.mechanics.new_girder('floor', anchor=xt.Frame())
    line.mechanics.new_girder('g', elements=['q'], parent='floor')
    return env, line


def alignment(line):
    return np.array([getattr(line['q'], field) for field in FIELDS])


@for_all_test_contexts
def test_consumed_closure_skips_unrelated_expired_motion_and_static_settings(
    test_context,
):
    env, line = make_line(test_context)
    other = env.mechanics.new_girder('other', anchor=(100.0, 0.0, 0.0))
    env.mechanics.add_source(
        'expired',
        xt.mechanics.RigidMotion([0.0, 0.1], np.zeros((2, 1, 6))),
        targets='other',
    )
    env.mechanics.add_source(
        'active', xt.mechanics.Harmonic(2e-8, 3.0), targets='floor'
    )
    # An unused body cannot affect the consumer's prepared static settings.
    other.installation.shift_y = 1j
    runtime = PreparedMechanics(line.mechanics)
    assert runtime.names == ('floor', 'g')
    assert [item.name for item in runtime.sources] == ['active']
    runtime.refresh_static()
    runtime.apply(0.5)
    runtime.finish()
    line.mechanics.apply(time_s=0.5)
    expected = alignment(line)
    runtime.apply(0.5)
    runtime.finish()
    np.testing.assert_allclose(alignment(line), expected, atol=3e-17, rtol=2e-14)
    record = MechanicalTimeSeries(
        env.mechanics, [0.0, 0.1], {'expired': np.zeros((2, 1, 6))}
    )
    runtime.apply(0.5, record)
    runtime.finish()
    np.testing.assert_allclose(alignment(line), expected, atol=3e-17, rtol=2e-14)


@for_all_test_contexts
def test_relevant_custom_source_keeps_full_receiver_order_before_selection(
    test_context,
):
    env, line = make_line(test_context)
    env.mechanics.new_girder('other', anchor=(100.0, 0.0, 0.0))

    class JointReceivers:
        frame = 'body_local'
        unit = 'm'

        def __init__(self):
            self.receivers = []

        def evaluate(self, time_s, positions_m):
            self.receivers.append(positions_m.copy())
            # The second receiver's realization explicitly depends on total size.
            return np.broadcast_to(
                (1 + np.arange(len(positions_m))) * len(positions_m) * 1e-8,
                (len(time_s), len(positions_m)),
            )

    source = JointReceivers()
    env.mechanics.add_source('joint', source, targets=['other', 'g'])
    line.mechanics.apply(time_s=0.2)
    expected = alignment(line)
    runtime = PreparedMechanics(line.mechanics)
    runtime.apply(0.2)
    runtime.finish()
    assert source.receivers[-1].shape == (2, 3)
    np.testing.assert_array_equal(source.receivers[-1], source.receivers[0])
    assert runtime.raw_values.shape == (1, 6)
    assert runtime.raw_values[0, 1] == 4e-8
    np.testing.assert_allclose(alignment(line), expected, atol=3e-17, rtol=2e-14)


@pytest.mark.parametrize('budget', [0, 650, 32*1024**2])
@pytest.mark.parametrize('recorded', [False, True])
def test_composed_continuous_sources_cache_receivers_with_bounded_memory(
    monkeypatch, budget, recorded,
):
    from xtrack.mechanics import runtime as runtime_module

    monkeypatch.setattr(runtime_module, '_SPECTRAL_RECEIVER_BUDGET', budget)
    env, line = make_line(xo.ContextCpu())
    env.mechanics.new_girder('other', anchor=(100., 0., 0.))
    def spectral(seed):
        return xt.mechanics.SpectralGroundMotion(
            [0., 3., 6., 7.], [0., 1e-12, 1e-12, 0.],
            seed=seed, correlation_length_m=30., n_modes=16,
        )
    regional = xt.mechanics.RegionalGroundMotion(
        spectral(4),
        (xt.mechanics.GaussianRegion(spectral(5), (0., 0., 0.), 20.),),
    )
    floor = xt.mechanics.VectorGroundMotion((None, regional, None))
    pump = xt.mechanics.LocatedSource(spectral(6), (0., 1., 2.), 10., 300.)
    for name, source in (('floor', floor), ('pump', pump)):
        env.mechanics.add_source(name, source, targets=('other', 'floor'))
    record = (env.mechanics.compute_time_series(time_s=np.linspace(0., 1., 51))
              if recorded else None)
    times = (.2, .37, .91)
    expected = []
    for time in times:
        line.mechanics.apply(time_s=time, time_series=record)
        expected.append(alignment(line).copy())

    calls = {4: 0, 5: 0, 6: 0}
    original = xt.mechanics.SpectralGroundMotion._prepare_receivers
    def counted(self, positions):
        calls[self.seed] += 1
        # The two world-field receivers retain original target identity, even
        # though the first is not consumed by this line. The pump has one port.
        assert len(positions) == (1 if self.seed == 6 else 2)
        return original(self, positions)
    monkeypatch.setattr(xt.mechanics.SpectralGroundMotion, '_prepare_receivers', counted)

    runtime = PreparedMechanics(line.mechanics)
    for time, reference in zip(times, expected):
        runtime.apply(time, record)
        runtime.finish()
        np.testing.assert_allclose(alignment(line), reference, atol=3e-18, rtol=0.)
    assert runtime._spectral_receiver_bytes <= budget
    if budget == 0:
        # LocatedSource evaluates a delayed waveform for each receiver.
        assert calls == {4: 3, 5: 3, 6: 6}
    elif budget == 650:
        assert calls == {4: 1, 5: 3, 6: 6}
        assert runtime._spectral_receiver_bytes == 640
    else:
        assert calls == {4: 1, 5: 1, 6: 1}
        assert runtime._spectral_receiver_bytes == 1664


def test_cached_composite_exact_provider_changes_with_the_prescribed_record():
    env, line = make_line(xo.ContextCpu())
    psd = xt.mechanics.DisplacementPSD(
        [0., 3., 6., 7.], [0., 1e-12, 1e-12, 0.], seed=4,
    )
    spectral = xt.mechanics.SpectralGroundMotion(
        [0., 3., 6., 7.], [0., 1e-12, 1e-12, 0.],
        seed=5, correlation_length_m=30., n_modes=16,
    )
    source = xt.mechanics.VectorGroundMotion((psd, spectral, None))
    env.mechanics.add_source('mixed_field', source, targets='floor')
    first = env.mechanics.compute_time_series(time_s=np.arange(64)/32)
    second = env.mechanics.compute_time_series(time_s=np.arange(128)/64)
    sampled = env.mechanics.compute_time_series(time_s=np.arange(64)/32,
                                               interpolation='linear')
    runtime = PreparedMechanics(line.mechanics)
    values = []
    for record in (first, second, sampled, first):
        line.mechanics.apply(time_s=.37, time_series=record)
        reference = alignment(line).copy()
        runtime.apply(.37, record)
        runtime.finish()
        np.testing.assert_allclose(alignment(line), reference, atol=3e-18, rtol=0.)
        values.append(reference)
        assert bool(runtime._spectral_receiver_bytes) == (record is not sampled)
    assert abs(values[0][0]-values[1][0]) > 1e-12
    np.testing.assert_array_equal(values[0], values[-1])


def test_over_budget_composite_uses_bounded_uncached_receiver_blocks(monkeypatch):
    from xtrack.mechanics import sources, runtime as runtime_module

    env, line = make_line(xo.ContextCpu())
    env.mechanics.new_girder('other', anchor=(100., 0., 0.))
    spectral = xt.mechanics.SpectralGroundMotion(
        [0., 3., 6., 7.], [0., 1e-12, 1e-12, 0.],
        seed=956, correlation_length_m=30., n_modes=16,
    )
    field = xt.mechanics.VectorGroundMotion((None, spectral, None))
    env.mechanics.add_source('floor', field, targets=('other', 'floor'))
    line.mechanics.apply(time_s=.37)
    reference = alignment(line)
    monkeypatch.setattr(runtime_module, '_SPECTRAL_RECEIVER_BUDGET', 0)
    monkeypatch.setattr(sources, '_SPECTRAL_RECEIVER_SCRATCH_BYTES', 8*16*3)
    original = xt.mechanics.SpectralGroundMotion._prepare_receivers
    lengths = []
    def capture(self, receivers):
        lengths.append(len(receivers))
        return original(self, receivers)
    monkeypatch.setattr(xt.mechanics.SpectralGroundMotion, '_prepare_receivers', capture)
    runtime = PreparedMechanics(line.mechanics)
    runtime.apply(.37)
    runtime.finish()
    assert runtime._spectral_receiver_bytes == 0
    assert lengths == [1, 1]
    np.testing.assert_allclose(alignment(line), reference, atol=3e-18, rtol=0.)


@pytest.mark.parametrize('reverse', [False, True])
@for_all_test_contexts
def test_mixed_sources_keep_native_evaluation_and_noncommuting_order(
    test_context, monkeypatch, reverse
):
    env, line = make_line(test_context)

    class CustomYaw:
        frame = 'body_local_rigid'

        def evaluate(self, time_s, positions_m):
            values = np.zeros((len(time_s), len(positions_m), 6))
            values[..., 4] = 0.3
            values[..., 0] = 2e-5
            return values

    sources = [
        ('pitch', xt.mechanics.Harmonic(0.2, 0.0, np.pi / 2, unit='rad'), 'pitch'),
        ('custom', CustomYaw(), 'y'),
        ('roll', xt.mechanics.Harmonic(0.4, 0.0, np.pi / 2, unit='rad'), 'roll'),
    ]
    for name, source, component in reversed(sources) if reverse else sources:
        env.mechanics.add_source(name, source, targets='g', component=component)
    line.mechanics.apply(time_s=0.2)
    expected = alignment(line)
    runtime = PreparedMechanics(line.mechanics)
    assert [item.name for item in runtime.host_sources] == ['custom']
    assert {item.name for item in runtime.native_sources} == {'pitch', 'roll'}

    def forbidden(*args, **kwargs):
        raise AssertionError('A mixed binding moved an analytic source onto the host')

    monkeypatch.setattr(xt.mechanics.Harmonic, 'evaluate', forbidden)
    runtime.apply(0.2)
    runtime.finish()
    np.testing.assert_allclose(alignment(line), expected, atol=3e-16, rtol=3e-14)
    payload = runtime.sample_values
    rows = runtime.source_rows
    runtime.apply(0.3)
    runtime.finish()
    assert runtime.sample_values is payload
    assert runtime.source_rows is rows


@for_all_test_contexts
def test_partial_record_mixes_interpolated_native_and_live_sources(test_context):
    env, line = make_line(test_context)

    class LiveOffset:
        frame = 'body_local'
        unit = 'm'
        value = 2e-8

        def evaluate(self, time_s, positions_m):
            return np.full((len(time_s), len(positions_m)), self.value)

    live = LiveOffset()
    env.mechanics.add_source(
        'recorded', xt.mechanics.Harmonic(1e-8, 2.0), targets='floor'
    )
    env.mechanics.add_source(
        'native', xt.mechanics.Harmonic(3e-8, 3.0), targets='g', component='x'
    )
    env.mechanics.add_source('control', live, targets='g', live=True, role='control')
    values = np.zeros((2, 1, 6))
    values[:, 0, 1] = [0.0, 4e-8]
    record = MechanicalTimeSeries(env.mechanics, [0.0, 1.0], {'recorded': values})
    runtime = PreparedMechanics(line.mechanics)
    runtime.apply(0.25, record)
    runtime.finish()
    assert [item.name for item in runtime.native_sources] == ['native']
    assert [item.name for item in runtime.host_sources] == ['recorded', 'control']
    line.mechanics.apply(time_s=0.25, time_series=record)
    expected = alignment(line)
    runtime.apply(0.25, record)
    runtime.finish()
    np.testing.assert_allclose(alignment(line), expected, atol=3e-17, rtol=2e-14)
    live.value = 7e-8
    runtime.apply(0.25, record)
    runtime.finish()
    assert alignment(line)[1] - expected[1] == pytest.approx(5e-8, abs=3e-17)


def test_live_builtin_is_captured_and_mode_changes_fail_before_lowering():
    env, line = make_line(xo.ContextCpu())
    env.mechanics.add_source(
        'live',
        xt.mechanics.Harmonic(2e-8, 0.0, np.pi / 2),
        targets='g',
        live=True,
        role='control',
    )
    runtime = PreparedMechanics(line.mechanics)
    assert not runtime.native_sources
    assert [item.name for item in runtime.host_sources] == ['live']
    runtime.apply(0.2)
    runtime.finish()
    np.testing.assert_allclose(
        line.mechanics._last_control_values['live'],
        [[0.0, 2e-8, 0.0, 0.0, 0.0, 0.0]],
        atol=1e-25,
    )
    before = alignment(line).copy()
    line.mechanics.freeze_updates()
    with pytest.raises(ValueError, match='source modes changed'):
        runtime.apply(0.3)
    np.testing.assert_array_equal(alignment(line), before)


@for_all_test_contexts
def test_multiline_subset_mapping_keeps_each_beam_clock_and_origin(test_context):
    env = xt.Environment()
    for index in range(2):
        env.new(f'q{index}', xt.Quadrupole, length=0.5, k1=0.1)
    lines = [
        env.new_line(name=f'beam{index}', components=[f'q{index}'])
        for index in range(2)
    ]
    buffer = None
    for index, heading in enumerate((0.0, np.pi)):
        name = f'q{index}'
        line = lines[index]
        line.set_particle_ref('proton', p0c=7e12)
        line.build_tracker(_context=test_context, _buffer=buffer)
        buffer = line.tracker._buffer
        line.mechanics.set_reference_frame(
            xt.Frame.from_survey_angles(X=10.0 * index, theta=heading)
        )
        line.mechanics.new_girder(f'g{index}', elements=[name])
    env.mechanics.add_source(
        'wave',
        xt.mechanics.PlaneWave(
            1e-8, 9.0, (1.0, 0.0, 0.0), 300.0, polarization=(1.0, 1.0, 0.0)
        ),
        targets=['g0', 'g1'],
    )
    values = np.zeros((2, 2, 6))
    values[:, :, 1] = [[1e-9, 2e-9], [3e-9, 4e-9]]
    values[:, :, 3] = [[1e-7, 2e-7], [3e-7, 4e-7]]
    env.mechanics.add_source(
        'record',
        xt.mechanics.RigidMotion([0.0, 1.0], values, target_names=['g1', 'g0']),
        targets=['g1', 'g0'],
    )
    for index, line in enumerate(lines):
        line.mechanics.apply(time_s=0.2)
        expected = np.array([getattr(line[f'q{index}'], f) for f in FIELDS])
        runtime = PreparedMechanics(line.mechanics)
        assert runtime.names == (f'g{index}',)
        runtime.apply(0.2)
        runtime.finish()
        actual = [getattr(line[f'q{index}'], f) for f in FIELDS]
        np.testing.assert_allclose(actual, expected, atol=3e-17, rtol=3e-14)
        line.mechanics.enable_vibrations(
            backend='compiled', time_origin_s=0.2 + 0.2 * index
        )
    first, second = [line.build_particles(x=1e-7, y=2e-8) for line in lines]
    lines[0].track(first, num_turns=3)
    first_time = first.time_s
    lines[1].track(second, num_turns=3)
    assert first.time_s == first_time == second.time_s
    assert first.reference_turns == second.reference_turns == 3.0
    assert lines[0].mechanics.last_applied_time_s == pytest.approx(
        0.2 + 2 * first.t_sim
    )
    assert lines[1].mechanics.last_applied_time_s == pytest.approx(
        0.4 + 2 * second.t_sim
    )


def test_recorder_preflight_failure_precedes_public_alignment_writes(monkeypatch):
    env, line = make_line(xo.ContextCpu())
    env.mechanics.add_source(
        'live',
        xt.mechanics.Harmonic(2e-8, 0.0, np.pi / 2),
        targets='g',
        role='control',
        live=True,
    )
    runtime = PreparedMechanics(line.mechanics)
    before = alignment(line).copy()

    def reject(time_s, source_values, **kwargs):
        assert source_values['live'][0, 1] == 2e-8
        raise ValueError('Recorder rejected this freeze')

    monkeypatch.setattr(line.mechanics, '_preflight_applied', reject)
    with pytest.raises(ValueError, match='Recorder rejected'):
        runtime.apply(0.2)
    np.testing.assert_array_equal(alignment(line), before)
    assert not runtime.applied


def test_live_bindings_own_shared_provider_staging_before_next_evaluation():
    env, line = make_line(xo.ContextCpu())

    class ReusedBuffer:
        frame = 'body_local_rigid'

        def __init__(self):
            self.values = np.zeros((1, 1, 6))

        def evaluate(self, time_s, positions_m):
            self.values[:] = 0.0
            self.values[0, :, 1] = 1e-8 + 4e-8 * positions_m[:, 2]
            return self.values

    source = ReusedBuffer()
    for name, target in [('first', 'floor'), ('second', 'g')]:
        env.mechanics.add_source(
            name, source, targets=target, role='control', live=True
        )
    line.mechanics.enable_vibrations(backend='compiled')
    runtime = PreparedMechanics(line.mechanics)
    runtime.apply(0.0)
    runtime.finish()
    assert line.mechanics._last_control_values['first'][0, 1] == 1e-8
    assert line.mechanics._last_control_values['second'][0, 1] == 2e-8
    before = alignment(line).copy()
    line.mechanics.disable_vibrations()
    np.testing.assert_allclose(alignment(line), before, atol=3e-17, rtol=2e-14)
