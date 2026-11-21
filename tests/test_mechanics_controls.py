"""Declared live controls remain applied independently of vibration lifecycle."""

import numpy as np
import pytest

import xtrack as xt


class HeldPosition:
    unit = 'm'

    def __init__(self, value=2e-6):
        self.value = value
        self.calls = 0

    def evaluate(self, time_s, positions_m):
        self.calls += 1
        return np.full((len(time_s), len(positions_m)), self.value)


def make_controlled_line():
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=0.3, k1=0.1, shift_y=3e-6)
    line = env.new_line(name='ring', components=['q'])
    line.set_particle_ref('proton', p0c=7e12)
    body = line.mechanics.new_girder('g', elements=['q'])
    body.installation.shift_y = 20e-6
    control = HeldPosition()
    env.mechanics.add_source('control', control, targets='g', role='control', live=True)
    env.mechanics.add_source(
        'noise', xt.mechanics.Harmonic(1e-6, 0.0, np.pi / 2), targets='g'
    )
    return env, line, control


@pytest.mark.parametrize('owner', ['manager', 'line'])
def test_manual_apply_retains_last_applied_control_on_vibration_disable(owner):
    env, line, control = make_controlled_line()
    target = env.mechanics if owner == 'manager' else line.mechanics
    target.apply(time_s=0.25)
    assert line['q'].shift_y == pytest.approx(26e-6)
    control.value = 7e-6
    calls = control.calls
    line.mechanics.disable_vibrations()
    assert control.calls == calls
    assert line['q'].shift_y == pytest.approx(25e-6)
    assert not line.mechanics.vibrations_enabled
    assert not line.mechanics.controls_enabled
    assert not line._mechanics_updates_enabled
    line.mechanics.freeze_updates()
    assert not line._mechanics_updates_enabled
    assert line['q'].shift_y == pytest.approx(25e-6)


@pytest.mark.parametrize('backend', ['reference', 'compiled'])
def test_live_control_ignores_record_and_survives_disabled_expired_disturbance(backend):
    env, line, control = make_controlled_line()
    record = env.mechanics.compute_time_series(time_s=[0.0, 0.01])
    assert control.calls == 0
    assert set(record.values) == {'noise'}
    assert record.omitted_live_sources == ('control',)
    with pytest.raises(ValueError, match='live source history'):
        record.get_table('g')
    line.build_tracker()
    line.mechanics.enable_vibrations(time_series=record, backend=backend)
    particles = line.build_particles()
    line.track(particles, num_turns=1)
    assert line['q'].shift_y == pytest.approx(26e-6)
    control.value = 7e-6
    calls = control.calls
    line.mechanics.disable_vibrations()
    assert control.calls == calls
    assert line['q'].shift_y == pytest.approx(25e-6)
    assert line.mechanics.controls_enabled
    assert line._mechanics_updates_enabled
    particles.time_s = 0.2
    line.track(particles, num_turns=1)
    assert line['q'].shift_y == pytest.approx(30e-6)
    assert control.calls == calls + 1
    line.mechanics.freeze_updates()
    assert line['q'].shift_y == pytest.approx(30e-6)
    line.mechanics.disable_vibrations()
    assert not line._mechanics_updates_enabled
    env.mechanics.clear()
    assert line['q'].shift_y == 3e-6


def test_explicit_disabled_control_updates_held_pose_to_zero():
    env, line, control = make_controlled_line()
    line.mechanics.apply(time_s=0.2)
    line.mechanics.apply(time_s=0.3, disabled_sources=('control',))
    control.value = 8e-6
    calls = control.calls
    line.mechanics.disable_vibrations()
    assert control.calls == calls
    assert line['q'].shift_y == pytest.approx(23e-6)


@pytest.mark.parametrize('control_first', [False, True])
def test_absolute_descendant_cannot_silently_mask_ancestor_control(control_first):
    env = xt.Environment()
    env.mechanics.new_girder('parent', anchor=(0.0, 0.0, 0.0))
    env.mechanics.new_body('child', parent='parent', anchor=(0.0, 0.0, 1.0))
    operations = [
        lambda: env.mechanics.add_source(
            'control', HeldPosition(), targets='parent', role='control', live=True
        ),
        lambda: env.mechanics.add_source(
            'plant',
            xt.mechanics.RigidMotion([0.0, 1.0], np.zeros((2, 6))),
            targets='child',
            reference='absolute',
        ),
    ]
    if not control_first:
        operations.reverse()
    operations[0]()
    with pytest.raises(ValueError, match='mask ancestor control'):
        operations[1]()
    assert len(env.mechanics.sources) == 1


def test_downstream_control_mover_composes_with_absolute_plant():
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=0.3, k1=0.1, shift_y=3e-6)
    line = env.new_line(name='ring', components=['q'])
    env.mechanics.new_girder('plant', anchor=(0.0, 0.0, 0.0))
    env.mechanics.new_body(
        'mover', parent='plant', anchor=(0.0, 0.0, 0.15), line=line, elements=['q']
    )
    values = np.zeros((2, 6))
    values[:, 1] = 1e-6
    env.mechanics.add_source(
        'plant_motion',
        xt.mechanics.RigidMotion([0.0, 1.0], values),
        targets='plant',
        reference='absolute',
    )
    env.mechanics.add_source(
        'control', HeldPosition(), targets='mover', role='control', live=True
    )
    env.mechanics.apply()
    assert line['q'].shift_y == pytest.approx(6e-6)


def test_binding_capabilities_are_inspectable_and_custom_live_export_is_clear():
    env, line, control = make_controlled_line()
    table = env.mechanics.get_source_table()
    assert table['role', 'control'] == 'control'
    assert table['live', 'control']
    assert table['capability', 'noise'] == 'prescribed'
    with pytest.raises(TypeError, match='Live source.*cannot be serialized'):
        env.to_dict()


def test_builtin_control_metadata_roundtrips_without_enabling_runtime():
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=0.3, k1=0.1)
    line = env.new_line(name='ring', components=['q'])
    line.mechanics.new_girder('g', elements=['q'])
    env.mechanics.add_source(
        'control',
        xt.mechanics.Harmonic(1e-6, 2.0),
        targets='g',
        role='control',
        live=True,
    )
    copied = env.copy(with_progress=False)
    binding = copied.mechanics.sources['control']
    assert binding.role == 'control' and binding.live
    assert not copied.ring.mechanics.automatic_updates_enabled


def test_computed_record_rejects_complex_times_and_invalid_manual_channels():
    env, line, control = make_controlled_line()
    with pytest.raises(ValueError, match='real'):
        env.mechanics.compute_time_series(time_s=np.array([0.0, 1.0 + 1j]))
    with pytest.raises(ValueError, match='prescribed source'):
        xt.mechanics.MechanicalTimeSeries(
            env.mechanics, [0.0, 1.0], {'control': np.zeros((2, 1, 6))}
        )
    with pytest.raises(ValueError, match='shape'):
        xt.mechanics.MechanicalTimeSeries(
            env.mechanics, [0.0, 1.0], {'noise': np.zeros((2, 6))}
        )
    with pytest.raises(ValueError, match='real'):
        xt.mechanics.MechanicalTimeSeries(
            env.mechanics, [0.0, 1.0], {'noise': np.ones((2, 1, 6), dtype=complex)}
        )
    assert control.calls == 0


def test_disabling_retains_actual_coarse_prescribed_control_from_manual_apply():
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=0.3, k1=0.1)
    line = env.new_line(name='ring', components=['q'])
    line.mechanics.new_girder('g', elements=['q'])
    env.mechanics.add_source(
        'control', xt.mechanics.Harmonic(1e-6, 0.5), targets='g', role='control'
    )
    record = env.mechanics.compute_time_series(time_s=[0.0, 1.0])
    env.mechanics.apply(time_s=0.5, time_series=record)
    actual = line['q'].shift_y
    assert actual == pytest.approx(0.0, abs=1e-20)
    line.mechanics.disable_vibrations()
    assert line['q'].shift_y == actual
    assert not line._mechanics_updates_enabled


def test_record_metadata_cannot_be_rebound_or_deleted():
    env, _, _ = make_controlled_line()
    record = env.mechanics.compute_time_series(time_s=[0.0, 1.0])
    for name, value in [('values', {}), ('time_s', np.array([0.0, 2.0]))]:
        with pytest.raises(AttributeError, match='read-only'):
            setattr(record, name, value)
        with pytest.raises(AttributeError, match='read-only'):
            delattr(record, name)


def test_active_recorder_rejects_manual_passage_with_disabled_live_input():
    env, line, control = make_controlled_line()
    line.mechanics.record_applied_sources(sources='control', max_samples=4)
    before = line['q'].shift_y
    with pytest.raises(ValueError, match='turn freezes only'):
        line.mechanics.apply(
            time_s=0.1, sampling='passage', disabled_sources=('control',)
        )
    assert line['q'].shift_y == before
    assert control.calls == 0


@pytest.mark.parametrize('live', [False, True])
def test_shared_rigid_provider_buffers_are_owned_before_next_binding(live):
    class ReusedRigid:
        frame = 'body_local_rigid'

        def evaluate(self, time_s, positions_m):
            shape = (len(time_s), len(positions_m), 6)
            if not hasattr(self, 'buffer') or self.buffer.shape != shape:
                self.buffer = np.zeros(shape)
            self.buffer[...] = 0.0
            self.buffer[..., 1] = positions_m[:, 2] * 1e-8
            return self.buffer

    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=0.3)
    line = env.new_line(name='ring', components=['q'])
    env.mechanics.new_girder('floor', anchor=(0.0, 0.0, 1.0))
    env.mechanics.new_body(
        'g', parent='floor', anchor=(0.0, 0.0, 2.0), elements=['q'], line=line
    )
    source = ReusedRigid()
    for name, target in [('a', 'floor'), ('b', 'g')]:
        env.mechanics.add_source(
            name, source, targets=target, role='control', live=live
        )
    if live:
        env.mechanics.apply()
        assert line['q'].shift_y == pytest.approx(3e-8)
        line.mechanics.disable_vibrations()
        assert line['q'].shift_y == pytest.approx(3e-8)
    else:
        record = env.mechanics.compute_time_series(time_s=[0.0, 1.0])
        np.testing.assert_allclose(record.values['a'][..., 1], 1e-8)
        np.testing.assert_allclose(record.values['b'][..., 1], 2e-8)
