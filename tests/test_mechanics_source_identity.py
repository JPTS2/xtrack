"""Finite FFT receiver sets must not silently acquire perfect correlation."""

import numpy as np
import pytest

import xtrack as xt


def make_manager():
    env = xt.Environment()
    for name, position in [('a', 0.0), ('b', 10.0)]:
        env.mechanics.new_body(name, anchor=xt.Frame.from_survey_angles(X=position))
    return env, env.mechanics


def fft_field(seed=246, *, length=1.0):
    return xt.mechanics.DisplacementPSD(
        [0.0, 1.0, 10.0, 11.0],
        [0.0, 1e-18, 1e-18, 0.0],
        seed,
        correlation_length_m=length,
    )


@pytest.mark.parametrize('same_instance', [False, True])
@pytest.mark.parametrize('length', [None, 1.0])
def test_repeated_finite_fft_field_rejects_before_binding(same_instance, length):
    env, manager = make_manager()
    source = fft_field(length=length)
    manager.add_source('first', source, targets='a')
    revision = manager._revision
    other = source if same_instance else fft_field(length=length)
    with pytest.raises(ValueError, match='joint receiver binding'):
        manager.add_source('second', other, targets='b')
    assert tuple(manager.sources) == ('first',)
    assert manager._revision == revision


def test_joint_fft_record_and_copy_preserve_receiver_realization():
    env, manager = make_manager()
    manager.add_source('floor', fft_field(), targets=['a', 'b'])
    times = np.arange(256) / 128
    record = manager.compute_time_series(time_s=times)
    first = record.get_table('a').shift_y
    second = record.get_table('b').shift_y
    assert not np.array_equal(first, second)
    restored = xt.Environment()
    restored.mechanics.from_dict(manager.to_dict())
    replay = restored.mechanics.compute_time_series(time_s=times)
    np.testing.assert_array_equal(replay.values['floor'], record.values['floor'])


def test_independent_fft_seeds_remain_supported():
    env, manager = make_manager()
    manager.add_source('first', fft_field(246), targets='a')
    manager.add_source('second', fft_field(247), targets='b')
    assert tuple(manager.sources) == ('first', 'second')


@pytest.mark.parametrize('wrapper', ['regional', 'vector'])
def test_repeated_fft_field_inside_wrapper_is_rejected(wrapper):
    env, manager = make_manager()
    if wrapper == 'regional':
        source = xt.mechanics.RegionalGroundMotion(fft_field(), ())
    else:
        source = xt.mechanics.VectorGroundMotion((None, fft_field(), None))
    manager.add_source('first', source, targets='a')
    with pytest.raises(ValueError, match='joint receiver binding'):
        manager.add_source('second', source, targets='b')


def test_continuous_field_is_stable_across_separate_bindings():
    env, manager = make_manager()
    source = xt.mechanics.SpectralGroundMotion(
        [0.0, 1.0, 10.0, 11.0],
        [0.0, 1e-18, 1e-18, 0.0],
        246,
        1.0,
        n_modes=32,
    )
    manager.add_source('first', source, targets='a')
    manager.add_source('second', source, targets='b')
    times = np.arange(32) / 128
    record = manager.compute_time_series(time_s=times)
    expected = source.evaluate(times, [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        record.values['first'][:, 0, 1], expected[:, 0], atol=1e-24
    )
    np.testing.assert_allclose(
        record.values['second'][:, 0, 1], expected[:, 1], atol=1e-24
    )


def test_located_fft_waveform_remains_a_reusable_temporal_excitation():
    env, manager = make_manager()
    source = xt.mechanics.LocatedSource(
        fft_field(),
        position_m=(0.0, 0.0, 0.0),
        decay_length_m=20.0,
        speed_m_s=100.0,
    )
    manager.add_source('first_path', source, targets='a')
    manager.add_source('second_path', source, targets='b')
    times = np.arange(128) / 128
    record = manager.compute_time_series(time_s=times)
    expected = source.evaluate(times, np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    np.testing.assert_allclose(
        record.values['first_path'][:, 0, :3], expected[:, 0], atol=1e-24
    )
    np.testing.assert_allclose(
        record.values['second_path'][:, 0, :3], expected[:, 1], atol=1e-24
    )
