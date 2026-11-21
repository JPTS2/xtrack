"""Retained query rows must not silently compose another installation case."""

import numpy as np
import pytest

import xtrack as xt


def model():
    env = xt.Environment()
    env.new('q1', xt.Quadrupole, length=.4, k1=.2,
            shift_x=1e-3, rot_s_rad=.12)
    env.new('d', xt.Drift, length=1.2)
    env.new('q2', xt.Quadrupole, length=.4, k1=-.2)
    line = env.new_line(name='ring', components=['q1', 'd', 'q2'])
    line.mechanics.adopt_reference()
    root = env.mechanics.new_girder('root', anchor=xt.Frame())
    for name, s in [('q1', .2), ('q2', 1.8)]:
        env.mechanics.new_body('m.' + name, parent=root, kind='magnet',
            mount=xt.Frame().translate_s(s), elements=[name], line=line)
    root.installation.shift_y = 2e-6
    env.mechanics.add_source('vertical', xt.mechanics.Harmonic(10e-9, 3.),
                             targets=['root'])
    return env, line


@pytest.mark.parametrize('owner_kind', ['line', 'body'])
@pytest.mark.parametrize('posed', [False, True])
@pytest.mark.parametrize('layer', ['reference', 'installation'])
def test_selected_rows_reject_changed_static_case(owner_kind, posed, layer):
    env, line = model()
    owner = line.mechanics if owner_kind == 'line' else env.mechanics
    name = 'q1' if owner_kind == 'line' else 'm.q1'
    table = owner.get_table(**({'time_s': .125} if posed else {}))
    # Context must survive both row and column selection, without a body column.
    selected = table.rows[name].cols['name']
    selected.get_time_series(time_s=[0., .125])
    target = env.mechanics['m.q1']
    if layer == 'installation':
        target.installation.shift_y = 7e-6
    else:
        target.shift_y = 7e-6
    assert selected.name[0] == name  # Numerical snapshot access remains valid.
    for query in (lambda: selected.get_time_series(time_s=[0., .125]),
                  selected.get_body):
        with pytest.raises(ValueError, match='static/reference case changed.*fresh table'):
            query()
    if owner_kind == 'line':
        with pytest.raises(ValueError, match='static/reference case changed.*fresh table'):
            selected.get_element()
    fresh = owner.get_table().rows[name]
    fresh.get_time_series(time_s=[0., .125])
    assert fresh.get_body() is target


@pytest.mark.parametrize('layer', ['reference', 'installation'])
def test_ancestor_changes_reject_history_without_altering_snapshot(layer):
    env, line = model()
    table = line.mechanics.get_table(time_s=0.).rows['q1']
    before = table.static_Y.copy()
    root = env.mechanics['root']
    target = root if layer == 'reference' else root.installation
    target.shift_y += 3e-6
    with pytest.raises(ValueError, match='static/reference case changed'):
        table.get_time_series(time_s=[0., .125])
    np.testing.assert_array_equal(table.static_Y, before)
    fresh = line.mechanics.get_table().rows['q1'].get_time_series(time_s=[0., .125])
    np.testing.assert_allclose(fresh.static_Y, np.full(2, before[0] + 3e-6), atol=1e-18)


def test_unrelated_branch_change_does_not_invalidate_selected_row():
    env, line = model()
    elements = line.mechanics.get_table(time_s=0.)
    bodies = env.mechanics.get_table()
    env.mechanics['m.q2'].installation.rot_s_rad = .01
    elements.rows['q1'].get_time_series(time_s=[0., .125])
    bodies.rows['m.q1'].get_time_series(time_s=[0., .125])
    assert elements.rows['q1'].get_element() is line.element_dict['q1']
    with pytest.raises(ValueError, match='static/reference case changed'):
        elements.rows['q2'].get_time_series(time_s=[0., .125])


def test_case_reapplication_and_A_B_A_use_values_not_mutation_counter():
    env, line = model()
    case_a = env.mechanics.generate_installation_errors(
        seed=956, bodies=['root', 'm.q1'], translation_sigma_m=(0., 1e-6, 0.))
    case_b = env.mechanics.generate_installation_errors(
        seed=957, bodies=['root', 'm.q1'], translation_sigma_m=(0., 1e-6, 0.))
    case_a.apply(env.mechanics)
    selected = line.mechanics.get_table(time_s=0.).rows['q1']
    expected = selected.get_time_series(time_s=[0., .125]).total_matrix.copy()
    case_a.apply(env.mechanics)
    np.testing.assert_array_equal(selected.get_time_series(time_s=[0., .125]).total_matrix, expected)
    case_b.apply(env.mechanics)
    with pytest.raises(ValueError, match='static/reference case changed'):
        selected.get_time_series(time_s=[0., .125])
    case_a.apply(env.mechanics)
    np.testing.assert_array_equal(selected.get_time_series(time_s=[0., .125]).total_matrix, expected)


def test_live_element_access_does_not_freeze_time_or_strength():
    env, line = model()
    selected = line.mechanics.get_table(time_s=0.).rows['q1']
    snapshot = selected.total_Y.copy()
    env.mechanics.apply(time_s=.125)
    line['q1'].k1 = .3
    element = selected.get_element()
    assert element is line.element_dict['q1']
    assert element.k1 == .3
    assert element.shift_y != snapshot[0]
    np.testing.assert_array_equal(selected.total_Y, snapshot)
    selected.get_time_series(time_s=[0., .125])


@pytest.mark.parametrize('change', ['source', 'topology', 'clear'])
def test_existing_definition_guards_remain_fail_closed(change):
    env, line = model()
    selected = line.mechanics.get_table().rows['q1']
    if change == 'source':
        env.mechanics.add_source('extra', xt.mechanics.Harmonic(1e-9, 1.), targets=['root'])
    elif change == 'topology':
        env.mechanics.new_body('unrelated', anchor=xt.Frame())
    else:
        env.mechanics.clear()
    for query in (selected.get_element, selected.get_body,
                  lambda: selected.get_time_series(time_s=[0., .125])):
        with pytest.raises(ValueError, match='definitions changed.*fresh table'):
            query()


def test_copy_uses_independent_context_and_rejects_foreign_source_record():
    env, line = model()
    record = env.mechanics.compute_time_series(time_s=[0., .125])
    selected = line.mechanics.get_table(time_s=0., time_series=record).rows['q1']
    clone = env.copy(with_progress=False)
    clone.mechanics['root'].installation.shift_y += 2e-6
    selected.get_time_series()
    copied = clone['ring'].mechanics.get_table().rows['q1']
    with pytest.raises(ValueError, match='another or changed mechanics model'):
        copied.get_time_series(time_series=record)
    copied.get_time_series(time_s=[0., .125])


def test_export_preserves_numbers_and_drops_static_query_context(tmp_path):
    env, line = model()
    selected = line.mechanics.get_table(time_s=0.).rows['q1']
    destination = tmp_path / 'pose.json'
    selected.to_json(destination)
    assert '_mechanics_static_case' not in destination.read_text()
    loaded = xt.mechanics.MechanicsTable.from_json(destination)
    np.testing.assert_array_equal(loaded.static_Y, selected.static_Y)
    with pytest.raises(ValueError, match='No live mechanics context'):
        loaded.get_time_series(time_s=[0., .125])


def test_element_reference_edits_keep_existing_alignment_guard():
    env, line = model()
    selected = line.mechanics.get_table(time_s=0.).rows['q1']
    line['q1'].rot_s_rad += .01
    for query in (selected.get_element,
                  lambda: selected.get_time_series(time_s=[0., .125])):
        with pytest.raises(ValueError):
            query()
