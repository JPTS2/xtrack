"""Bulk mechanics preparation preserves live optics and assignment ownership."""
import numpy as np
import pytest

import xtrack as xt


def make_line():
    env = xt.Environment()
    env['lq'] = .4
    env['lb'] = 1.5
    env['bend_angle'] = .03
    env['kq'] = .8
    env.new('q1', xt.Quadrupole, length='lq', k1='kq', shift_x=3e-5)
    env.new('d', xt.Drift, length=.5)
    env.new('b', xt.Bend, length='lb', angle='bend_angle')
    env.new('q2', xt.Quadrupole, length='lq', k1='-kq')
    line = env.new_line(name='ring', components=['q1', 'd', 'b', 'q2'])
    return env, line


def test_bulk_resolves_one_survey_and_geometry_signature(monkeypatch):
    env, line = make_line()
    calls = {'survey': 0, 'signature': 0}
    survey = line.survey
    signature = line.mechanics._geometry_signature
    def measured_survey(*args, **kwargs):
        calls['survey'] += 1
        return survey(*args, **kwargs)
    def measured_signature():
        calls['signature'] += 1
        return signature()
    monkeypatch.setattr(line, 'survey', measured_survey)
    monkeypatch.setattr(line.mechanics, '_geometry_signature', measured_signature)
    line.mechanics.adopt_reference()
    bodies = line.mechanics.new_girders({
        'first': {'elements': ['b', 'q1'], 'support_class': 'combined'},
        'second': ['q2'],
    }, support_class='quad')
    assert calls == {'survey': 1, 'signature': 1}
    assert list(bodies) == ['first', 'second']
    assert bodies['first'].kind == 'girder'
    assert bodies['first'].support_class == 'combined'
    assert bodies['second'].support_class == 'quad'
    # Each group's centre uses entrance/exit in line order, not input order.
    reference = survey(include_element_frames=True)
    expected = (reference.get_frame('q1', which='ref_start').XYZ
                + reference.get_frame('b', which='ref_end').XYZ)/2
    np.testing.assert_array_equal(bodies['first'].nominal_frame.XYZ, expected)
    assert list(line.mechanics._attachments) == ['q1', 'b', 'q2']
    assert list(env.mechanics.get_table().rows.match(support_class='combined').name) == ['first']
    assert list(line.mechanics.get_table().rows.match(support_class='combined').name) == ['q1', 'b']
    assert list(env.mechanics.get_table(time_s=0.).rows.match(support_class='quad').name) == ['second']


def test_geometry_expressions_remain_live_but_geometry_changes_invalidate():
    env, line = make_line()
    line.mechanics.new_girders({'g1': ['q1'], 'g2': ['b', 'q2']})
    env['kq'] = 1.3
    line.mechanics.validate()
    assert line['q1'].k1 == 1.3
    assert line['q2'].k1 == -1.3
    assert env.ref['q1'].length._expr is not None
    assert env.ref['b'].angle._expr is not None
    env.mechanics.apply()
    assert env.ref['q1'].length._expr is not None
    assert env.ref['b'].angle._expr is not None
    env['lq'] = .41
    with pytest.raises(ValueError, match='reference geometry changed'):
        line.mechanics.validate()
    # Geometry ownership stays with Xdeps even after invalidation.
    assert line['q1'].length == .41
    assert env.ref['q1'].length._expr is not None


@pytest.mark.parametrize('assignments', [
    {'a': ['q1'], 'b': ['q1']},
    {'a': ['q1'], 'b': ['q2', 'missing']},
    {'a': ['q1'], 'b': {'elements': ['q2'], 'anchor': 'bad'}},
    {'a': ['q1'], 'b': {'elements': ['q2'], 'parent': 'missing'}},
    {'a': ['q1'], 'b': {'elements': ['q2'], 'support_class': 42}},
])
def test_invalid_batch_is_atomic(assignments):
    env, line = make_line()
    with pytest.raises(ValueError):
        line.mechanics.new_girders(assignments)
    assert not env.mechanics.bodies
    assert not line.mechanics._attachments


def test_alignment_expressions_stay_owned_and_aliases_remain_rejected():
    env, line = make_line()
    env['offset'] = 1e-5
    env.ref['q1'].shift_y = env.ref['offset']
    with pytest.raises(ValueError, match='expression-driven alignment'):
        line.mechanics.new_girders({'a': ['q1']})
    assert not env.mechanics.bodies
    env['offset'] = 2e-5
    assert line['q1'].shift_y == 2e-5
    # Repetition is still disallowed by the scalar-writing reference backend.
    other = env.new_line(name='other', components=['q2', 'q2'])
    other.mechanics.set_reference_frame(xt.Frame())
    with pytest.raises(ValueError, match='repeated'):
        other.mechanics.new_girders({'a': ['q2']})


def test_support_class_roundtrip_and_existing_parent():
    env, line = make_line()
    parent = env.mechanics.new_girder('floor', anchor=xt.Frame(), support_class='floor')
    env.mechanics.new_girders({'g': {'elements': ['q1', 'b'], 'parent': parent,
        'feet_local_m': [[-.3, -.5, -.5], [.3, -.5, .5]], 'support_class': 'arc'}}, line=line)
    copied = env.copy()
    assert copied.mechanics['g'].parent == 'floor'
    assert copied.mechanics['g'].support_class == 'arc'
    assert copied.mechanics['floor'].support_class == 'floor'
    assert set(copied.lines['ring'].mechanics.get_table().support_class) == {'arc'}
    assert copied.ref['q1'].length._expr is not None


def test_bulk_magnet_children_have_one_survey_and_preserve_body_kinds(monkeypatch):
    env, line = make_line()
    support = env.mechanics.new_girder('support', anchor=xt.Frame(), support_class='arc')
    calls = []
    survey = line.survey
    def measured_survey(*args, **kwargs):
        calls.append(None)
        return survey(*args, **kwargs)
    monkeypatch.setattr(line, 'survey', measured_survey)
    magnets = env.mechanics.new_bodies({
        'm1': dict(elements=['q1'], parent=support, support_class='arc'),
        'm2': dict(elements=['q2'], parent=support, support_class='arc'),
    }, line=line, kind='magnet')
    assert len(calls) == 1
    assert all(body.kind == 'magnet' and body.parent == support.name for body in magnets.values())
    assert list(env.mechanics.get_table().rows.match(kind='girder').name) == ['support']
    assert list(env.mechanics.get_table().rows.match(kind='magnet').name) == ['m1', 'm2']
    support.shift_y = 2e-6
    magnets['m2'].installation.shift_y = 3e-6
    env.mechanics.apply()
    assert line['q1'].shift_y == pytest.approx(2e-6)
    assert line['q2'].shift_y == pytest.approx(5e-6)
    copied = env.copy(with_progress=False)
    assert copied.mechanics['m1'].kind == 'magnet'
    assert copied.mechanics['m1'].parent == 'support'
    copied.mechanics.clear()
    assert copied['q1'].shift_x == 3e-5
    assert copied['q2'].shift_y == 0.


def test_invalid_bulk_body_kind_does_not_install_any_attachment():
    env, line = make_line()
    with pytest.raises(ValueError, match='body kind'):
        line.mechanics.new_bodies({'m1': ['q1'], 'm2': ['q2']}, kind='')
    assert not env.mechanics.bodies
    assert not line.mechanics._attachments
