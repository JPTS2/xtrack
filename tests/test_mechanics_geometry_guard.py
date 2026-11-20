"""Context-native geometry ownership guard, with no scalar entry transfers."""
import pytest
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt
from xtrack.mechanics.geometry_guard import GeometryGuard


def prepared(test_context):
    env = xt.Environment()
    env['lq'] = .4
    env['angle'] = .03
    env['kq'] = .7
    env.new('q', xt.Quadrupole, length='lq', k1='kq', shift_x=3e-6)
    env.new('d', xt.Drift, length=.5)
    env.new('b', xt.Bend, length=1.5, angle='angle')
    line = env.new_line(name='ring', components=['q', 'd', 'b'])
    line.build_tracker(_context=test_context)
    line.mechanics.new_girders({'gq': ['q'], 'gb': ['b']})
    line.mechanics.validate()
    return env, line, GeometryGuard(line.mechanics)


@for_all_test_contexts
def test_guard_accepts_strength_changes_and_tracks_current_alignment(test_context, monkeypatch):
    env, line, guard = prepared(test_context)
    env['kq'] = 1.1
    guard.check()
    env.mechanics['gq'].shift_y = 2e-5
    env.mechanics.apply()
    guard.check()
    # The prepared guard must not invoke per-scalar buffer reads on any context.
    buffer_type = type(guard.buffer)
    def forbidden(*args, **kwargs):
        raise AssertionError('A prepared entry guard attempted a scalar read')
    monkeypatch.setattr(buffer_type, 'to_bytearray', forbidden)
    guard.check()


@pytest.mark.parametrize('change', ['length_knob', 'angle_knob', 'unmounted_drift', 'alignment'])
@for_all_test_contexts
def test_guard_catches_geometry_and_alignment_changes(test_context, change):
    env, line, guard = prepared(test_context)
    if change == 'length_knob':
        env['lq'] = .41
    elif change == 'angle_knob':
        env['angle'] = .031
    elif change == 'unmounted_drift':
        env['d'].length = .6
    else:
        env['q'].shift_y = 1e-6
    with pytest.raises(ValueError, match='geometry changed|alignment edited'):
        guard.check()


@for_all_test_contexts
def test_guard_rejects_new_equal_value_alignment_expression(test_context):
    env, line, guard = prepared(test_context)
    env['offset'] = 3e-6
    env.ref['q'].shift_x = env.ref['offset']
    assert line['q'].shift_x == 3e-6
    with pytest.raises(ValueError, match='expression-driven alignment'):
        guard.check()
    env['offset'] = 4e-6
    assert line['q'].shift_x == 4e-6


@pytest.mark.parametrize('change', ['object', 'alias'])
@for_all_test_contexts
def test_guard_rejects_object_replacement_and_added_alias(test_context, change):
    env, line, guard = prepared(test_context)
    if change == 'alias':
        env.new_line(name='alias', components=['q'])
    else:
        line.element_dict['d'] = xt.Drift(length=.5, _context=test_context)
    with pytest.raises(ValueError, match='shared/repeated|object changed'):
        guard.check()


def test_guard_rejects_untracked_array_geometry_and_restores_buffer_methods(monkeypatch):
    import xobjects as xo
    _, line, previous = prepared(xo.ContextCpu())
    buffer = previous.buffer
    original = buffer.to_nplike
    # A custom geometry property may read an array rather than a scalar. Fail
    # closed so auto can retain the reference backend for that geometry.
    monkeypatch.setattr(line.mechanics, '_geometry_signature',
                        lambda: line.element_dict['q'].knl[0])
    with pytest.raises(NotImplementedError, match='array-based geometry'):
        GeometryGuard(line.mechanics)
    assert buffer.to_nplike == original


def test_guard_checks_python_collective_geometry():
    import xobjects as xo
    from types import SimpleNamespace
    _, line, previous = prepared(xo.ContextCpu())
    # Python collective devices may own plain mutable placement attributes and
    # remain outside the shared non-collective Xobject buffer.
    device = SimpleNamespace(length=.5, iscollective=True)
    line.element_dict['d'] = device
    guard = GeometryGuard(line.mechanics)
    guard.check()
    device.length = .6
    with pytest.raises(ValueError, match='Custom element reference geometry changed'):
        guard.check()
