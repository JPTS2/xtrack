"""Ownership-safe frozen snapshots and removal failures for mechanics."""
from copy import deepcopy
import gc
import weakref

import numpy as np
import pytest
import xobjects as xo

import xtrack as xt


FIELDS = ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad',
          'rot_s_rad_no_frame', 'rot_shift_anchor')


def make_snapshot_case():
    env = xt.Environment()
    env.new('q1', xt.Quadrupole, length=.3, k1=.2, shift_x=.001,
            rot_s_rad=.17, rot_s_rad_no_frame=.03, rot_shift_anchor=.4)
    env.new('q2', xt.Quadrupole, length=.4, k1=-.2, shift_y=3e-6)
    line = env.new_line(name='ring', components=['q1', 'q2'])
    line.mechanics.adopt_reference()
    body = line.mechanics.new_girder('g', elements=['q1', 'q2'])
    body.shift_y = 2e-6
    body.installation.shift_y = 20e-6
    env.mechanics.add_source('tone', xt.mechanics.Harmonic(50e-9, 5.),
                             targets='g')
    env.mechanics.apply(time_s=.05)
    return env, line


def values(line):
    return np.array([[getattr(line[name], field) for field in FIELDS]
                     for name in ('q1', 'q2')])


@pytest.mark.parametrize('change, message', [
    ('alignment', 'alignment edited'),
    ('geometry', 'reference geometry changed'),
    ('alias', 'shared/repeated'),
])
def test_snapshot_does_not_launder_conflicting_ownership(change, message):
    env, line = make_snapshot_case()
    if change == 'alignment':
        line['q2'].shift_y = 99e-6
    elif change == 'geometry':
        line['q2'].length = .5
    else:
        env.new_line(name='alias', components=['q1'])
    before = values(line)
    # Both serialization entry points must reject an inconsistent snapshot.
    # In particular, copying must not adopt a raw edit as a legitimate output
    # and then silently overwrite it when the copied model is applied.
    with pytest.raises(ValueError, match=message):
        env.to_dict()
    with pytest.raises(ValueError, match=message):
        env.copy(with_progress=False)
    np.testing.assert_array_equal(values(line), before)


def test_copy_preserves_frozen_pose_and_all_three_layers_without_enabling():
    env, line = make_snapshot_case()
    line.mechanics.enable_vibrations(time_origin_s=.2)
    before = values(line)
    copied = env.copy(with_progress=False)
    other = copied.lines['ring']
    np.testing.assert_array_equal(values(other), before)
    assert not other.mechanics.vibrations_enabled
    assert copied.mechanics['g'].shift_y == 2e-6
    assert copied.mechanics['g'].installation.shift_y == 20e-6
    assert copied.mechanics.sources['tone'].source.frequency_hz == 5.
    copied.mechanics.apply(time_s=.05)
    np.testing.assert_array_equal(values(other), before)
    copied.mechanics.clear()
    assert other['q1'].shift_x == .001
    assert other['q1'].rot_s_rad == .17
    assert other['q1'].rot_s_rad_no_frame == .03
    assert other['q1'].rot_shift_anchor == .4
    assert other['q2'].shift_y == 3e-6
    np.testing.assert_array_equal(values(line), before)
    assert line.mechanics.vibrations_enabled


def test_snapshot_synchronizes_prepared_output_without_changing_the_pose():
    env, line = make_snapshot_case()
    line.set_particle_ref('proton', p0c=7e12)
    line.build_tracker(_context=xo.ContextCpu())
    line.mechanics.enable_vibrations(backend='compiled')
    # A snapshot may be requested during an explicit compiled callback. Its
    # ownership check must first acknowledge the kernel's own field writes.
    line.mechanics._prepare_for_tracking()
    runtime = line.mechanics._runtime
    runtime.apply(.025)
    assert runtime.applied
    before = values(line)
    copied = env.copy(with_progress=False)
    assert not runtime.applied
    np.testing.assert_array_equal(values(line), before)
    np.testing.assert_array_equal(values(copied.lines['ring']), before)
    copied.mechanics.apply(time_s=.025)
    np.testing.assert_array_equal(values(copied.lines['ring']), before)
    assert line.mechanics.vibrations_enabled
    line.mechanics.validate()


def test_frozen_aperture_snapshot_is_allowed_but_motion_still_requires_discard():
    env, line = make_snapshot_case()
    profile = env.aperture.new_profile('round', 'Circle', radius=.01)
    pipe = env.aperture.new_pipe('pipe', positions=[profile.place(),
                                                   profile.place(at=.2)])
    line.aperture.place_pipe('installed', pipe, at='q1')
    line.aperture.build_model()
    copied = env.copy(with_progress=False)
    other = copied.lines['ring']
    # Aperture definitions are serialized; derived survey models are rebuilt.
    assert not other.aperture.is_built
    assert copied.aperture.pipes['pipe'].name == 'pipe'
    np.testing.assert_array_equal(values(other), values(line))
    other.aperture.build_model()
    with pytest.raises(ValueError, match='aperture survey model'):
        copied.mechanics.apply(time_s=.1)
    other.aperture.discard_model()
    copied.mechanics.apply(time_s=.1)


def test_cleared_manager_can_reload_without_reactivating_old_motion_records():
    env, line = make_snapshot_case()
    view = line.mechanics
    before = values(line)
    record = env.mechanics.compute_time_series(time_s=[0., .05, .1])
    data = env.mechanics.to_dict()
    env.mechanics.clear()
    env.mechanics.from_dict(data)
    assert line.mechanics is view
    assert not view.vibrations_enabled
    env.mechanics.apply(time_s=.05)
    np.testing.assert_array_equal(values(line), before)
    with pytest.raises(ValueError, match='changed mechanics model'):
        env.mechanics.apply(time_s=.05, time_series=record)


def test_loading_does_not_overwrite_unattached_reference_configuration():
    source, _ = make_snapshot_case()
    data = source.mechanics.to_dict()
    env = xt.Environment()
    env.new('q1', xt.Quadrupole, length=.3, k1=.2)
    env.new('q2', xt.Quadrupole, length=.4, k1=-.2)
    line = env.new_line(name='ring', components=['q1', 'q2'])
    line.mechanics.set_reference_frame(xt.Frame.from_survey_angles(X=1.))
    with pytest.raises(ValueError, match='empty manager'):
        env.mechanics.from_dict(data)
    assert line.mechanics.reference_frame.X == 1.
    assert not env.mechanics.bodies


@pytest.mark.parametrize('invalid', ['source', 'attachment'])
def test_failed_loading_leaves_empty_manager_and_can_be_retried(invalid):
    env, line = make_snapshot_case()
    data = env.mechanics.to_dict()
    env.mechanics.clear()
    view = line.mechanics
    before = values(line)
    broken = deepcopy(data)
    if invalid == 'source':
        broken['sources'][0]['source']['type'] = 'UnknownMotion'
    else:
        broken['lines'][0]['attachments'][0]['body'] = 'unknown_body'
    with pytest.raises(KeyError):
        env.mechanics.from_dict(broken)
    assert line.mechanics is view
    assert not env.mechanics._has_configuration
    np.testing.assert_array_equal(values(line), before)
    env.mechanics.from_dict(data)
    assert line.mechanics is view
    assert not view.vibrations_enabled
    assert env.mechanics.to_dict() == data


@pytest.mark.parametrize('layer', ['reference', 'installation'])
def test_snapshot_rejects_nonfinite_static_body_settings(layer):
    env, line = make_snapshot_case()
    body = env.mechanics['g']
    offset = body if layer == 'reference' else body.installation
    offset.shift_y = np.nan
    before = values(line)
    with pytest.raises(ValueError, match='finite'):
        env.to_dict()
    np.testing.assert_array_equal(values(line), before)


@pytest.mark.parametrize('operation', ['restore', 'clear', 'disable'])
def test_failed_removal_does_not_partially_reset_alignments_or_enable_state(operation):
    env, line = make_snapshot_case()
    line.mechanics.enable_vibrations()
    line['q2'].shift_y = 99e-6
    before = values(line)
    remove = {'restore': env.mechanics.restore,
              'clear': env.mechanics.clear,
              'disable': line.mechanics.disable_vibrations}[operation]
    with pytest.raises(ValueError, match='edited'):
        remove()
    np.testing.assert_array_equal(values(line), before)
    assert line.mechanics.vibrations_enabled
    assert tuple(env.mechanics.bodies) == ('g',)
    assert tuple(env.mechanics.sources) == ('tone',)


@pytest.mark.parametrize('have_body', [False, True])
@pytest.mark.parametrize('explicit_frame', [False, True])
def test_copy_preserves_reference_configuration_before_attachment(have_body, explicit_frame):
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.3, k1=.2, shift_x=.001)
    line = env.new_line(name='ring', components=['q'])
    line.mechanics.adopt_reference()
    if explicit_frame:
        line.mechanics.set_reference_frame(
            xt.Frame.from_survey_angles(X=123., theta=.2))
    if have_body:
        env.mechanics.new_girder('unattached', anchor=(1., 2., 3.))
    expected_frame = line.mechanics.reference_frame.matrix.copy()
    expected_inputs = line.mechanics.get_reference_table().shift_x.copy()

    for copied in (env.copy(with_progress=False),
                   xt.Environment.from_dict(env.to_dict(), with_progress=False)):
        other = copied.lines['ring']
        np.testing.assert_array_equal(other.mechanics.reference_frame.matrix, expected_frame)
        np.testing.assert_array_equal(other.mechanics.get_reference_table().shift_x, expected_inputs)
        assert other.mechanics._reference_is_explicit == explicit_frame
        assert not other.mechanics._attachments
        assert not other.mechanics.vibrations_enabled
        body = other.mechanics.new_girder('bound', elements=['q'])
        expected_center = xt.Frame(expected_frame).translate_s(.15)
        np.testing.assert_allclose(body.nominal_frame.matrix, expected_center.matrix,
                                   rtol=0., atol=1e-14)


def test_pending_reference_copy_does_not_bypass_second_line_placement_guard():
    env = xt.Environment()
    env.new('q1', xt.Quadrupole, length=.3, k1=.2)
    env.new('q2', xt.Quadrupole, length=.3, k1=.2)
    first = env.new_line(name='first', components=['q1'])
    second = env.new_line(name='second', components=['q2'])
    second.mechanics.adopt_reference()
    first.mechanics.new_girder('g1', elements=['q1'])
    copied = env.copy(with_progress=False)
    with pytest.raises(ValueError, match='explicit survey reference frame'):
        copied.second.mechanics.new_girder('g2', elements=['q2'])


def test_resolved_world_can_copy_and_extend_independent_of_view_access_order():
    env = xt.Environment()
    for name in ('q1', 'q2', 'q3'):
        env.new(name, xt.Quadrupole, length=.3, k1=.2)
    first = env.new_line(name='first', components=['q1', 'q3'])
    second = env.new_line(name='second', components=['q2'])
    # Access order differs from attachment order. Both placements are already
    # resolved when saved, even though the first line used its default frame.
    second.mechanics.set_reference_frame(xt.Frame.from_survey_angles(X=10.))
    first.mechanics.new_girder('g1', elements=['q1'])
    second.mechanics.new_girder('g2', elements=['q2'])
    copied = env.copy(with_progress=False)
    for model in (env, copied):
        assert not model.first.mechanics._reference_is_explicit
        assert model.second.mechanics._reference_is_explicit
        model.first.mechanics.new_girder('g3', elements=['q3'])
        model.mechanics.apply()
        assert model.mechanics['g3'].nominal_frame.Z == pytest.approx(.45)


def test_unattached_configuration_uses_environment_snapshot_not_line_copy_or_pickle():
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.3, k1=.2)
    line = env.new_line(name='ring', components=['q'])
    # Access alone is not a model and does not change ordinary serialization.
    _ = line.mechanics
    assert 'mechanics' not in env.to_dict()
    assert '_mechanics' not in env.__getstate__()

    line.mechanics.set_reference_frame(xt.Frame.from_survey_angles(X=123.))
    with pytest.raises(NotImplementedError, match='complete environment'):
        line.to_dict()
    with pytest.raises(NotImplementedError, match='env.copy'):
        line.copy()
    with pytest.raises(NotImplementedError, match='to_dict'):
        env.__getstate__()


@pytest.mark.parametrize('query', ['poses', 'bpm'])
@pytest.mark.parametrize('change', ['geometry', 'alignment'])
def test_element_pose_queries_reject_invalidated_bindings(query, change):
    env = xt.Environment()
    env.new('d', xt.Drift, length=1.)
    env.new('q', xt.Quadrupole, length=.3, k1=.2)
    line = env.new_line(name='ring', components=['d', 'q'])
    body = line.mechanics.new_girder('g', elements=['q'])
    body.shift_x = 1e-6
    if change == 'geometry':
        env['d'].length = 2.
        message = 'reference geometry changed'
    else:
        env['q'].shift_x = .001
        message = 'alignment edited'
    before = env['q'].shift_x
    call = (line.mechanics.get_element_poses if query == 'poses'
            else lambda: line.mechanics.read_bpm('q', x=0.))
    with pytest.raises(ValueError, match=message):
        call()
    assert env['q'].shift_x == before


@pytest.mark.parametrize('value', [1e-6 + 1e-6j, np.complex128(1e-6 + 1e-6j)])
@pytest.mark.parametrize('operation', ['apply', 'poses', 'table', 'save'])
def test_complex_body_reference_is_rejected_before_casting_or_writing(value, operation):
    env, line = make_snapshot_case()
    env.mechanics['g'].shift_y = value
    before = values(line)
    call = {'apply': env.mechanics.apply,
            'poses': env.mechanics.get_body_poses,
            'table': env.mechanics.get_table,
            'save': env.mechanics.to_dict}[operation]
    with pytest.raises(ValueError, match='must be real'):
        call()
    np.testing.assert_array_equal(values(line), before)


def test_world_angular_source_is_not_reinterpreted_as_displacement():
    class AngularWorldSource:
        frame = 'world'
        unit = 'rad'

        def evaluate(self, time_s, positions_m):
            raise AssertionError('Invalid source units must fail before evaluation')

    env = xt.Environment()
    env.mechanics.new_girder('g', anchor=(0., 0., 0.))
    with pytest.raises(ValueError, match='displacement in metres'):
        env.mechanics.add_source('angular', AngularWorldSource(), targets='g')
    assert not env.mechanics.sources


def test_anonymous_attached_line_stays_owned_until_offsets_are_restored():
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.3, k1=.2, shift_x=3e-6)
    line = env.new_line(components=['q'])
    body = line.mechanics.new_girder('g', elements=['q'])
    body.shift_x = 1e-4
    env.mechanics.apply()
    line_ref = weakref.ref(line)
    del line
    gc.collect()
    assert line_ref() is not None
    assert env['q'].shift_x != 3e-6
    env.mechanics.clear()
    gc.collect()
    assert env['q'].shift_x == 3e-6
    assert line_ref() is None


def test_lazy_mechanics_accessor_does_not_retain_anonymous_line():
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.3, k1=.2)
    line = env.new_line(components=['q'])
    view = line.mechanics
    line_ref = weakref.ref(line)
    del line
    gc.collect()
    assert line_ref() is None
    assert not view._has_configuration


@pytest.mark.parametrize('compiled', [False, True])
def test_upstream_thickness_change_invalidates_reference_geometry(compiled):
    line = xt.Line(elements={
        'upstream': xt.Multipole(length=1., knl=[0.], isthick=False),
        'q': xt.Quadrupole(length=.3, k1=.2),
    })
    line.set_particle_ref('proton', p0c=7e12)
    line.mechanics.new_girder('g', elements=['q'])
    if compiled:
        line.mechanics.enable_vibrations(backend='compiled')
        line.mechanics._prepare_for_tracking()
    line['upstream'].isthick = True
    if compiled:
        call = lambda: line.track(line.build_particles(), num_turns=1)
    else:
        call = line.env.mechanics.apply
    with pytest.raises(ValueError, match='reference geometry changed'):
        call()


@pytest.mark.parametrize('upstream', ['replica', 'slice'])
def test_unsupported_upstream_geometry_is_rejected_before_binding(upstream):
    parent = xt.Quadrupole(length=1., k1=.2)
    if upstream == 'replica':
        element = xt.Replica(parent_name='parent')
    else:
        element = xt.DriftSliceQuadrupole(_parent=parent, weight=1., _buffer=parent._buffer)
    line = xt.Line(elements={'parent': parent, 'upstream': element,
                             'q': xt.Quadrupole(length=.3, k1=.2)},
                   element_names=['upstream', 'q'])
    with pytest.raises(ValueError, match='reference geometry'):
        line.mechanics.new_girder('g', elements=['q'])
    assert not line.env.mechanics.bodies
    assert not line.mechanics._attachments


def make_line_with_unrelated_slices():
    line = xt.Line(elements={
        'bend': xt.RBend(length_straight=1., angle=.02, rot_s_rad=.03),
        'q': xt.Quadrupole(length=.3, k1=.2),
    })
    line.env.lines['ring'] = line
    line.set_particle_ref('proton', p0c=7e12)
    line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=None),
            xt.Strategy(slicing=xt.Teapot(2, mode='thick'), name='bend'),
        ],
        with_progress=False,
    )
    return line


def test_unrelated_native_slices_can_share_a_line_with_mechanics():
    line = make_line_with_unrelated_slices()
    view = line.mechanics
    with pytest.raises(ValueError, match='unsliced alignment-capable'):
        view.new_girder('bad', elements=['bend..0'])
    assert not line.env.mechanics.bodies
    girder = view.new_girder('g', elements=['q'])
    girder.shift_y = 1e-6
    line.env.mechanics.add_source(
        'wave', xt.mechanics.Harmonic(2e-6, 100., .4, unit='rad'),
        targets='g', component='roll',
    )
    initial = line.build_particles(x=2e-6, y=3e-6)
    results = []
    for backend in ('reference', 'compiled'):
        view.enable_vibrations(backend=backend)
        particles = initial.copy()
        line.track(particles, num_turns=4)
        results.append(particles)
        view.disable_vibrations()
    for coordinate in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
        np.testing.assert_allclose(
            getattr(results[0], coordinate), getattr(results[1], coordinate),
            rtol=2e-13, atol=2e-17,
        )
    assert line['bend'].shift_y == 0.
    assert line['bend..0'].weight == .5
    assert line.env.copy(with_progress=False).mechanics.bodies['g'].shift_y == 1e-6


@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('change', [
    'length', 'angle', 'roll', 'model', 'weight', 'offset',
    'parent_name', 'parent_object', 'raw_pointer',
])
def test_unrelated_slice_geometry_is_guarded_before_updates(prepared, change):
    line = make_line_with_unrelated_slices()
    view = line.mechanics
    # Create another compatible parent in the same buffer for pointer tests.
    line.element_dict['other_bend'] = xt.RBend(length_straight=1., angle=.02)
    view.new_girder('g', elements=['q']).shift_y = 2e-6
    parent = line.element_dict['bend']
    other = line.element_dict['other_bend']
    slc = line.element_dict['bend..0']
    other.move(_buffer=parent._buffer)
    if prepared:
        from xtrack.mechanics.geometry_guard import GeometryGuard
        guard = GeometryGuard(view)
        guard.check()
        check = guard.check
    else:
        check = line.env.mechanics.apply
    if change == 'length':
        parent.length_straight += .01
    elif change == 'angle':
        parent.angle += .01
    elif change == 'roll':
        parent.rot_s_rad += .01
    elif change == 'model':
        parent.rbend_model = 'straight-body'
    elif change == 'weight':
        slc.weight += .1
    elif change == 'offset':
        slc.slice_offset += .01
    elif change == 'parent_name':
        slc.parent_name = 'other_bend'
    elif change == 'parent_object':
        line.element_dict['bend'] = other
    elif change == 'raw_pointer':
        slc._xobject._parent = other._xobject
    with pytest.raises(ValueError, match='reference geometry changed'):
        check()
    assert line['q'].shift_y == 0.
