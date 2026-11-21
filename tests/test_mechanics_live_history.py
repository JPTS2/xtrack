"""Held live commands must not be retrospectively presented as a trajectory."""

import numpy as np
import pytest

import xtrack as xt


class HeldInput:
    frame = 'body_local'
    unit = 'm'

    def __init__(self):
        self.value = 20e-9
        self.calls = 0

    def evaluate(self, time_s, positions_m):
        self.calls += 1
        return np.full((len(time_s), len(positions_m)), self.value)


def model(*, role='control', live=True, target='support.live'):
    env = xt.Environment()
    env.new('q.live', xt.Quadrupole, length=0.5, k1=0.1, shift_x=1e-3)
    env.new('d', xt.Drift, length=1.0)
    env.new('q.quiet', xt.Quadrupole, length=0.5, k1=-0.1)
    line = env.new_line(name='ring', components=['q.live', 'd', 'q.quiet'])
    line.mechanics.adopt_reference()
    root = env.mechanics.new_girder('support.live', anchor=xt.Frame())
    env.mechanics.new_body(
        'magnet.live',
        kind='magnet',
        parent=root,
        mount=xt.Frame().translate_s(0.25),
        elements=['q.live'],
        line=line,
    )
    line.mechanics.new_girder('support.quiet', elements=['q.quiet'])
    env.mechanics.add_source(
        'ground',
        xt.mechanics.Harmonic(2e-9, 2.0, 0.4),
        targets=['support.live', 'support.quiet'],
    )
    held = HeldInput()
    env.mechanics.add_source('mover', held, targets=target, role=role, live=live)
    return env, line, held


@pytest.mark.parametrize('body_table', [False, True])
@pytest.mark.parametrize('posed', [False, True])
@pytest.mark.parametrize('with_record', [False, True])
def test_retained_live_ancestor_history_rejects_before_and_after_command_change(
    body_table, posed, with_record
):
    env, line, held = model()
    record = (
        env.mechanics.compute_time_series(time_s=[0.0, 0.25, 0.5])
        if with_record
        else None
    )
    assert held.calls == 0  # A source record is not an applied live-control log.
    if record is not None:
        assert 'mover' not in record.values
    owner = env.mechanics if body_table else line.mechanics
    name = 'magnet.live' if body_table else 'q.live'
    table = owner.get_table(
        **({'time_s': 0.125, 'time_series': record} if posed else {})
    )
    selected = table.rows[name].cols['name total_Y' if posed else 'name']
    assert selected.live_source_names == ('mover',)
    assert selected.live_source_roles == {'mover': 'control'}
    assert selected.query_kind == ('current_held_pose' if posed else 'configuration')
    assert selected.live_state == ('current_held' if posed else 'unevaluated')
    snapshot = selected.total_Y.copy() if posed else selected.name.copy()
    for command in (20e-9, -7e-9):
        held.value = command
        calls = held.calls
        with pytest.raises(ValueError, match='history of live mechanics inputs.*mover'):
            selected.get_time_series(time_s=[0.0, 0.25], time_series=record)
        assert held.calls == calls
        np.testing.assert_array_equal(
            selected.total_Y if posed else selected.name, snapshot
        )


@pytest.mark.parametrize('body_table', [False, True])
def test_unrelated_live_branch_does_not_block_prescribed_history(body_table):
    env, line, held = model()
    record = env.mechanics.compute_time_series(time_s=[0.0, 0.25, 0.5])
    owner = env.mechanics if body_table else line.mechanics
    name = 'support.quiet' if body_table else 'q.quiet'
    selected = owner.get_table().rows[name]
    assert selected.live_source_names == ()
    assert selected.live_state == 'not_live'
    history = selected.get_time_series(time_s=[0.0, 0.125, 0.25], time_series=record)
    assert held.calls == 0
    assert history.history_kind == 'prescribed_inputs'
    assert np.all(np.isfinite(history.total_Y))


def test_point_query_is_explicitly_current_held_and_old_snapshot_remains_readable():
    env, line, held = model()
    first = line.mechanics.get_table(time_s=0.125).rows['q.live']
    original = first.dynamic_Y.copy()
    held.value += 7e-9
    second = line.mechanics.get_table(time_s=0.125).rows['q.live']
    np.testing.assert_allclose(second.dynamic_Y - original, 7e-9, rtol=0, atol=1e-22)
    np.testing.assert_array_equal(first.dynamic_Y, original)
    assert first.query_kind == second.query_kind == 'current_held_pose'
    assert first.query_time_s == second.query_time_s == 0.125
    # Live object access is not a historical query or another provider sample.
    calls = held.calls
    assert first.get_element() is line.element_dict['q.live']
    assert first.get_body() is env.mechanics['magnet.live']
    assert held.calls == calls


def test_selection_and_export_retain_honest_live_input_labels(tmp_path):
    env, line, held = model(role='disturbance')
    table = line.mechanics.get_table(time_s=0.125)
    calls = held.calls
    live = table.rows['q.live'].cols['total_Y']
    quiet = table.rows['q.quiet'].cols['total_Y']
    assert live.live_source_roles == {'mover': 'disturbance'}
    assert live.query_kind == 'current_held_pose'
    assert quiet.query_kind == 'prescribed_pose'
    assert quiet.live_source_names == ()
    destination = tmp_path / 'snapshot.json'
    table.to_json(destination)
    loaded = xt.mechanics.MechanicsTable.from_json(destination)
    assert loaded.rows['q.live'].live_state == 'current_held'
    assert loaded.rows['q.quiet'].live_state == 'not_live'
    np.testing.assert_array_equal(loaded.total_matrix, table.total_matrix)
    assert held.calls == calls
    with pytest.raises(ValueError, match='No live mechanics context'):
        loaded.rows['q.live'].get_time_series(time_s=[0.0, 0.25])


def test_live_descendant_does_not_change_ancestor_history_contract():
    env, _, held = model(target='magnet.live')
    root = env.mechanics.get_table().rows['support.live']
    history = root.get_time_series(time_s=[0.0, 0.25])
    assert root.live_source_names == ()
    assert held.calls == 0
    assert np.all(np.isfinite(history.total_Y))


def test_explicit_binding_capability_is_not_inferred_from_provider_attribute():
    env, line, held = model(live=False)
    held.live = True
    selected = line.mechanics.get_table().rows['q.live']
    assert selected.live_state == 'not_live'
    selected.get_time_series(time_s=[0.0, 0.25])
    assert held.calls > 0


def test_known_recorded_command_timeline_can_be_replayed_as_nonlive_input():
    env = xt.Environment()
    env.mechanics.new_body('mover', anchor=xt.Frame())
    env.mechanics.add_source(
        'recorded_command',
        xt.mechanics.MeasuredMotion(
            [0.0, 0.25, 0.5], [1e-9, 3e-9, -2e-9], interpolation='previous'
        ),
        targets='mover',
        role='control',
        live=False,
    )
    selected = env.mechanics.get_table().rows['mover']
    history = selected.get_time_series(time_s=[0.0, 0.125, 0.25, 0.5])
    np.testing.assert_array_equal(history.dynamic_Y, [1e-9, 1e-9, 3e-9, -2e-9])
    assert selected.live_source_names == ()


def test_static_case_guard_still_precedes_live_history_query():
    env, line, held = model()
    selected = line.mechanics.get_table().rows['q.live']
    env.mechanics['support.live'].installation.shift_y = 3e-6
    with pytest.raises(ValueError, match='static/reference case changed'):
        selected.get_time_series(time_s=[0.0, 0.25])
    assert held.calls == 0


def test_animation_cannot_resample_past_live_commands():
    pytest.importorskip('matplotlib')
    env, line, held = model()
    selected = line.mechanics.get_table().rows['q.live']
    with pytest.raises(ValueError, match='history of live mechanics inputs'):
        selected.animate(time_s=[0.0, 0.25], size_m=(0.3, 0.3, 0.5))
    assert held.calls == 0


def record_three_freezes(
    env, line, held, *, disabled_sources=(), record=None, max_samples=8
):
    recorder = line.mechanics.record_applied_sources(
        sources=['mover'], max_samples=max_samples
    )
    for time, value in zip((0.0, 0.25, 0.5), (20e-9, 40e-9, -10e-9)):
        held.value = value
        line.mechanics.apply(
            time_s=time, disabled_sources=disabled_sources, time_series=record
        )
    return recorder


@pytest.mark.parametrize('body_table', [False, True])
def test_applied_live_history_replays_captured_commands_without_provider_queries(
    body_table,
):
    env, line, held = model()
    recorder = record_three_freezes(env, line, held)
    applied = recorder.snapshot()
    held.value = 99e-9
    calls = held.calls
    owner = env.mechanics if body_table else line.mechanics
    name = 'magnet.live' if body_table else 'q.live'
    selected = owner.get_table().rows[name]
    times = np.array([0.0, 0.125, 0.25, 0.5])
    history = selected.get_time_series(time_s=times, applied_history=applied)
    expected = 2e-9 * np.sin(4 * np.pi * times + 0.4) + [20e-9, 20e-9, 40e-9, -10e-9]
    np.testing.assert_allclose(history.dynamic_Y, expected, rtol=0, atol=1e-22)
    assert history.history_kind == 'applied_live_inputs'
    assert held.calls == calls
    for array in (applied.time_s, applied.values['mover']):
        with pytest.raises(ValueError):
            array.flat[0] = 0.0
    with pytest.raises(TypeError):
        applied.values['extra'] = np.zeros((3, 1, 6))
    before = applied.values['mover'].copy()
    line.mechanics.apply(time_s=0.75)
    np.testing.assert_array_equal(applied.values['mover'], before)
    assert len(recorder.snapshot().time_s) == 4


def test_applied_history_bounds_static_case_and_missing_live_coverage():
    env, line, held = model()
    another = HeldInput()
    env.mechanics.add_source(
        'second_mover', another, targets='magnet.live', role='control', live=True
    )
    recorder = record_three_freezes(env, line, held)
    applied = recorder.snapshot()
    selected = line.mechanics.get_table().rows['q.live']
    calls = held.calls, another.calls
    with pytest.raises(ValueError, match='cover all selected live'):
        selected.get_time_series(applied_history=applied)
    assert (held.calls, another.calls) == calls
    # Recording is scoped to affected consumers, not an unrelated machine cache.
    with pytest.raises(ValueError, match='static/reference case'):
        line.mechanics.get_table().rows['q.quiet'].get_time_series(
            applied_history=applied
        )
    line.mechanics.get_table().rows['q.quiet'].get_time_series(time_s=[0.0, 0.25])
    with pytest.raises(ValueError, match='outside the captured'):
        applied.at(0.75)
    env.mechanics['support.live'].installation.shift_y = 1e-6
    with pytest.raises(ValueError, match='static/reference case'):
        line.mechanics.get_table().rows['q.live'].get_time_series(
            applied_history=applied
        )


def test_applied_history_retains_disabled_disturbances_and_prescribed_record_identity():
    env, line, held = model()
    record = env.mechanics.compute_time_series(time_s=[0.0, 0.25, 0.5, 0.75])
    recorder = record_three_freezes(
        env, line, held, disabled_sources=('ground',), record=record
    )
    applied = recorder.snapshot()
    selected = line.mechanics.get_table().rows['q.live']
    history = selected.get_time_series(applied_history=applied)
    np.testing.assert_allclose(
        history.dynamic_Y, [20e-9, 40e-9, -10e-9], rtol=0, atol=1e-22
    )
    assert applied.disabled_sources == ('ground',)
    assert applied.prescribed_record is record
    alternative = env.mechanics.compute_time_series(time_s=[0.0, 0.25, 0.5, 0.75])
    with pytest.raises(ValueError, match='prescribed record differs'):
        selected.get_time_series(applied_history=applied, time_series=alternative)
    before = float(line['q.live'].shift_y)
    with pytest.raises(ValueError, match='mode/record changed'):
        line.mechanics.apply(time_s=0.75, disabled_sources=(), time_series=record)
    assert line['q.live'].shift_y == before


def test_recorder_overflow_stops_recording_without_stopping_motion(tmp_path):
    env, line, held = model()
    recorder = record_three_freezes(env, line, held, max_samples=2)
    assert recorder.overflowed and not recorder.active
    applied = recorder.snapshot()
    np.testing.assert_array_equal(applied.time_s, [0.0, 0.25])
    assert applied.overflow_time_s == 0.5
    assert line['q.live'].shift_y < 0.0  # Third command was applied despite capacity.
    with pytest.raises(ValueError, match='outside the captured'):
        line.mechanics.get_table().rows['q.live'].get_time_series(
            time_s=[0.0, 0.5], applied_history=applied
        )
    destination = tmp_path / 'applied.json'
    applied.to_json(destination)
    assert 'AppliedSourceHistory' in destination.read_text()


def test_recorder_rejects_same_time_conflict_and_static_change_before_writes():
    env, line, held = model()
    recorder = line.mechanics.record_applied_sources(sources='mover', max_samples=8)
    line.mechanics.apply(time_s=0.0)
    before = float(line['q.live'].shift_y)
    line.mechanics.apply(time_s=0.0)  # Exact repeated application is a no-op sample.
    assert len(recorder.snapshot().time_s) == 1
    held.value += 1e-9
    with pytest.raises(ValueError, match='same-time freezes'):
        line.mechanics.apply(time_s=0.0)
    assert line['q.live'].shift_y == before
    env.mechanics['support.live'].installation.shift_y = 1e-6
    with pytest.raises(ValueError, match='static/reference case changed'):
        line.mechanics.apply(time_s=0.25)
    assert line['q.live'].shift_y == before


def test_compiled_freezes_capture_live_inputs_and_snapshot_queries_do_not_read_provider():
    env, line, held = model()
    line.set_particle_ref('proton', p0c=7e12)
    line.build_tracker()
    recorder = line.mechanics.record_applied_sources(sources='mover', max_samples=8)
    line.mechanics.enable_vibrations(backend='compiled')
    line.track(line.build_particles(), num_turns=3)
    applied = recorder.snapshot()
    assert len(applied.time_s) == 3
    held.value = 1e-3
    calls = held.calls
    line.mechanics.get_table().rows['q.live'].get_time_series(applied_history=applied)
    assert held.calls == calls
    # Restarting numerical time while this recorder is active must reject
    # before the newly requested command reaches any element or particle.
    before = float(line['q.live'].shift_y)
    fresh = line.build_particles()
    with pytest.raises(ValueError, match='same-time freezes'):
        line.track(fresh, num_turns=1)
    assert fresh.at_turn[0] == 0
    assert line['q.live'].shift_y == before


def test_applied_six_coordinate_multitarget_inputs_match_actual_element_frames():
    env, line, _ = model()

    class RigidCommand:
        frame = 'body_local_rigid'

        def __init__(self):
            self.values = np.array(
                [
                    [1e-9, 2e-9, 3e-9, 0.001, 0.002, 0.003],
                    [-2e-9, 1e-9, 4e-9, -0.002, 0.001, -0.004],
                ]
            )

        def evaluate(self, time_s, positions_m):
            return np.broadcast_to(self.values, (len(time_s), 2, 6)).copy()

    command = RigidCommand()
    env.mechanics.add_source(
        'sixaxis',
        command,
        targets=['support.live', 'support.quiet'],
        role='control',
        live=True,
    )
    recorder = line.mechanics.record_applied_sources(
        sources=['mover', 'sixaxis'], max_samples=4
    )
    actual = []
    for time, scale in ((0.0, 1.0), (0.25, 2.0)):
        command.values *= scale
        line.mechanics.apply(time_s=time)
        actual.append(
            line.survey(include_element_frames=True)
            .get_frame('q.live', which='elem_start')
            .matrix
        )
    snapshot = recorder.snapshot()
    command.values[:] = 0.0
    history = (
        line.mechanics.get_table()
        .rows['q.live']
        .get_time_series(applied_history=snapshot)
    )
    np.testing.assert_allclose(history.total_matrix, actual, rtol=0, atol=3e-15)
    assert snapshot.values['sixaxis'].shape == (2, 2, 6)


def test_recorder_rejects_unbounded_or_unrelated_requests():
    env, line, _ = model()
    with pytest.raises(MemoryError, match='max_memory_bytes'):
        line.mechanics.record_applied_sources(
            sources='mover', max_samples=100, max_memory_bytes=1
        )
    with pytest.raises(ValueError, match='explicitly live'):
        line.mechanics.record_applied_sources(sources='ground')


def test_replay_does_not_require_or_evaluate_disabled_live_disturbance():
    env, line, external_floor = model(role='disturbance')
    command = HeldInput()
    env.mechanics.add_source(
        'control', command, targets='support.live', role='control', live=True
    )
    recorder = line.mechanics.record_applied_sources(sources='control', max_samples=4)
    for time, value in ((0.0, 20e-9), (0.25, 40e-9)):
        command.value = value
        line.mechanics.apply(time_s=time, disabled_sources=('ground', 'mover'))
    assert external_floor.calls == 0
    calls = command.calls
    snapshot = recorder.snapshot()
    history = (
        line.mechanics.get_table()
        .rows['q.live']
        .get_time_series(applied_history=snapshot)
    )
    np.testing.assert_allclose(history.dynamic_Y, [20e-9, 40e-9], rtol=0, atol=1e-22)
    assert command.calls == calls and external_floor.calls == 0
    other_env, other_line, _ = model()
    with pytest.raises(ValueError, match='another or changed mechanics model'):
        other_line.mechanics.get_table().rows['q.live'].get_time_series(
            applied_history=snapshot
        )
