"""Owned pipe snapshots intersect full 3D geometry in fixed survey planes."""

import json

import numpy as np
import pytest

import xtrack as xt


def pipe_case(*, origin=(0., 0., 0.), angles=(0., 0., 0.)):
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=2., k1=.1, shift_x=.001,
            rot_s_rad_no_frame=.02)
    line = env.new_line(name='line', components=['q'])
    line.set_particle_ref('proton', p0c=7e12)
    line.mechanics.set_reference_frame(xt.Frame.from_survey_angles(
        X=origin[0], Y=origin[1], Z=origin[2],
        phi=angles[0], theta=angles[1], psi=angles[2]))
    line.mechanics.adopt_reference()
    girder = line.mechanics.new_girder('g', elements=['q'])
    profile = env.aperture.new_profile('rectangle', 'Rectangle',
                                      max_x=.01, max_y=.015)
    pipe = env.aperture.new_pipe('tube', positions=[
        profile.place(at=.1), profile.place(at=1.9)])
    line.aperture.place_pipe('installed', pipe, at='q', shift_x=.002)
    line.aperture._with_progress = False
    return env, line, girder


@pytest.mark.parametrize('owned', [False, True])
def test_fixed_or_owned_pipe_preserves_placement_and_magnetic_reference(owned):
    env, line, girder = pipe_case()
    line.mechanics.attach_pipe('installed', body=girder if owned else None)
    adopted = line.aperture.pipe_positions['installed'].transformation.copy()
    girder.installation.shift_y = 3e-6
    env.mechanics.add_source('vibration', xt.mechanics.Harmonic(50e-9, 1.),
                            targets='g', component='y')
    env.mechanics.apply(time_s=.25)
    assert line['q'].shift_x == .001
    assert line['q'].shift_y == pytest.approx(3.05e-6)
    assert line['q'].rot_s_rad_no_frame == .02
    for disabled, expected in [((), 3.05e-6), (('vibration',), 3e-6)]:
        aperture = line.mechanics.get_aperture(time_s=.25, disabled_sources=disabled)
        actual = aperture.pipe_positions['installed'].shift_y
        assert actual == pytest.approx(expected if owned else 0., abs=1e-20)
        # Each request owns fresh placements/bounds, and reference stays unchanged.
        np.testing.assert_array_equal(
            line.aperture.pipe_positions['installed'].transformation, adopted)
    aperture.pipe_positions['installed'].shift_y = .1
    assert line.aperture.pipe_positions['installed'].shift_y == 0.
    env.mechanics.clear()
    assert line['q'].shift_y == 0.
    assert line['q'].shift_x == .001


@pytest.mark.parametrize('coordinate', ['pitch', 'yaw', 'roll'])
def test_finite_pipe_rotation_and_mount_composition_match_independent_world_map(coordinate):
    env, line, girder = pipe_case(origin=(20000., -30000., 50000.),
                                  angles=(.2, -.3, .1))
    child = env.mechanics.new_body('chamber', parent=girder,
        mount=xt.Frame.from_survey_angles(Y=.2, Z=.3, psi=.03), kind='pipe')
    child.installation.shift_y = .0003
    line.mechanics.attach_pipe('installed', body=child)
    env.mechanics.add_source('angle', xt.mechanics.Harmonic(.03, 0., np.pi/2,
                            unit='rad'), targets='g', component=coordinate)
    baseline = line.mechanics._pipes.references['installed'][1]
    poses = env.mechanics.get_body_poses()
    expected = (poses['chamber'] @ child.nominal_frame.inverse()
                @ xt.Frame(baseline)).matrix
    result = line.mechanics.get_pipe_table(time_s=0.)
    assert result.query_kind[0] == 'prescribed_pose'
    np.testing.assert_allclose(result.pose[0], expected, atol=2e-11, rtol=0)
    aperture = line.mechanics.get_aperture(time_s=0.)
    actual = (line.mechanics.reference_frame @ aperture.survey.get_frame('q')
              @ xt.Frame(aperture.pipe_positions['installed'].transformation)).matrix
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)


def test_yawed_pipe_changes_fixed_plane_width_not_just_polygon_pose():
    env, line, girder = pipe_case()
    # Remove the reference horizontal placement for an independent analytic oracle.
    line.aperture.pipe_positions['installed'].shift_x = 0.
    line.mechanics.attach_pipe('installed', body=girder)
    angle = .15
    girder.rot_y_rad = angle
    aperture = line.mechanics.get_aperture(time_s=0.)
    positions = np.array([.4, 1., 1.6])
    sections = aperture.get_cross_sections_at_s(positions)
    low = np.array([section[:, 0].min() for section in sections.cross_section])
    high = np.array([section[:, 0].max() for section in sections.cross_section])
    np.testing.assert_allclose((low + high)/2, (positions - 1.)*np.tan(angle),
                               atol=2e-10, rtol=0)
    np.testing.assert_allclose((high - low)/2, .01/np.cos(angle), atol=2e-10, rtol=0)


def test_frozen_aperture_keeps_the_shape_it_was_built_with():
    env, line, girder = pipe_case()
    unused = env.aperture.new_profile('unused', 'Circle', radius=.1)
    used = env.aperture.new_profile('inner', 'Rectangle', max_x=.008, max_y=.01)
    env.aperture.new_pipe('unused_pipe', positions=[unused.place(at=0.)])
    env.aperture.new_pipe('inner_pipe', positions=[used.place(at=.1), used.place(at=1.9)])
    line.aperture.from_dict(dict(pipe_positions=[]))
    line.aperture.place_pipe('installed', 'inner_pipe', at='q')
    line.mechanics.attach_pipe('installed', body=girder)
    frozen = line.mechanics.get_aperture(time_s=0.)
    assert list(frozen.profiles.keys()) == ['inner']
    assert frozen.pipe_positions['installed'].pipe.name == 'inner_pipe'
    expected = frozen.get_cross_sections_at_s([1.]).cross_section[0]
    env.aperture.profiles['inner'].raw.shape.max_x = .004
    frozen.build_model()
    np.testing.assert_array_equal(frozen.get_cross_sections_at_s([1.]).cross_section[0], expected)
    assert frozen.profiles['inner'].raw.shape.max_x == .008
    with pytest.raises(ValueError, match='Adopted pipe geometry changed'):
        line.mechanics.get_aperture(time_s=0.)


@pytest.mark.parametrize('keep_built_registry', [False, True])
def test_failed_import_restores_newly_built_pipe_models(keep_built_registry):
    env, line, girder = pipe_case()
    line.mechanics.attach_pipe('installed', body=girder)
    env.new_line(name='second', components=[])
    data = env.mechanics.to_dict()
    env.mechanics.clear()
    if not keep_built_registry:
        env.aperture.discard_model()
    assert not line.aperture.is_built
    broken = {**data, 'lines': [*data['lines'], dict(
        line='second', reference_frame=np.zeros((4, 4)), attachments=[],
    )]}
    with pytest.raises(ValueError, match='proper rigid transform'):
        env.mechanics.from_dict(broken)
    assert not env.mechanics._has_configuration
    assert not line.aperture.is_built
    assert env.aperture.is_built == keep_built_registry
    env.mechanics.from_dict(data)
    assert line.mechanics.get_pipe_table()['body', 'installed'] == 'g'


def test_small_local_pipe_translation_survives_large_world_origin():
    env, line, girder = pipe_case(origin=(1e9, -1e9, 1e9))
    line.mechanics.attach_pipe('installed', body=girder)
    env.mechanics.add_source('tiny', xt.mechanics.Harmonic(1e-13, 0., np.pi/2),
                            targets='g', component='y')
    table = line.mechanics.get_pipe_table(time_s=0.)
    assert table.change_Y[0] == pytest.approx(1e-13, rel=1e-15)
    aperture = line.mechanics.get_aperture(time_s=0.)
    assert aperture.pipe_positions['installed'].shift_y == pytest.approx(1e-13, rel=1e-15)


def test_pipe_definitions_are_pinned_and_missing_ownership_rejects_before_writes():
    env, line, girder = pipe_case()
    profile = env.aperture.new_profile('other', 'Circle', radius=.01)
    other = env.aperture.new_pipe('other_pipe', positions=[
        profile.place(at=.1), profile.place(at=.9)])
    # Adjacent placements, rather than overlapping independently owned pipes.
    line.aperture.pipes['tube'][1].shift_s = .9
    line.aperture.place_pipe('second', other, at='q', shift_z=1.)
    line.mechanics.attach_pipe('installed', body=girder)
    girder.shift_y = .001
    with pytest.raises(ValueError, match='every installed pipe'):
        env.mechanics.apply()
    assert line['q'].shift_y == 0.
    line.mechanics.attach_pipe('second', body=None)
    line.aperture.pipe_positions['installed'].shift_y = .002
    with pytest.raises(ValueError, match='Adopted pipe geometry changed'):
        env.mechanics.apply()
    assert line['q'].shift_y == 0.


def test_frozen_tracking_limits_observe_requested_pipe_not_reference_pipe():
    env, line, girder = pipe_case()
    line.mechanics.attach_pipe('installed', body=girder)
    girder.shift_x = .01
    snapshot = line.mechanics.get_aperture(time_s=0.)
    for aperture, expected in [(line.aperture, [1, -1]), (snapshot, [-1, 1])]:
        limit = aperture.get_limit_elements_at_s([1.])[1.]
        probe = xt.Line(elements=[limit])
        probe.particle_ref = xt.Particles(p0c=7e12)
        particles = probe.build_particles(x=[-.005, .02])
        probe.track(particles)
        state_by_id = np.array(particles.state)[np.argsort(particles.particle_id)]
        assert list(state_by_id > 0) == [value > 0 for value in expected]
        assert limit.installed_pipe_name == 'installed'


def test_round_trip_retains_pipe_ownership_and_frozen_view():
    env, line, girder = pipe_case()
    line.mechanics.attach_pipe('installed', body=girder)
    girder.installation.shift_y = 2e-6
    env.mechanics.apply()
    from xobjects import JEncoder
    with pytest.raises(ValueError, match='include_aperture=True'):
        env.to_dict()
    data = json.loads(json.dumps(env.to_dict(include_aperture=True), cls=JEncoder))
    copied = xt.Environment.from_dict(data)
    other = copied.lines['line']
    assert other.mechanics.get_pipe_table().body[0] == 'g'
    assert other.mechanics.get_aperture(time_s=0.).pipe_positions['installed'].shift_y == pytest.approx(2e-6)
    copied.mechanics.add_source('moving', xt.mechanics.Harmonic(1e-6, 1.), targets='g')
    with pytest.raises(NotImplementedError, match='Automatic moving-pipe'):
        other.mechanics.enable_vibrations()


def test_fixed_pipe_allows_compiled_magnet_motion_without_aperture_rebuild(monkeypatch):
    env, line, girder = pipe_case()
    line.mechanics.attach_pipe('installed', body=None)
    env.mechanics.add_source('tone', xt.mechanics.Harmonic(2e-6, 1.),
                            targets='g', component='y')
    line.mechanics.enable_vibrations(backend='compiled')
    monkeypatch.setattr(line.aperture, 'build_model', lambda *a, **k:
                        pytest.fail('Tracking rebuilt reference aperture'))
    particles = line.build_particles()
    line.track(particles, num_turns=3)
    line.track(particles, num_turns=2)
    assert particles.state[0] > 0
    assert line.aperture.pipe_positions['installed'].shift_y == 0.
    line.aperture.pipe_positions['installed'].shift_y = .001
    with pytest.raises(ValueError, match='Adopted pipe geometry changed'):
        line.track(particles)


@pytest.mark.parametrize('backend', ['compiled', 'reference'])
def test_static_pipe_owner_allows_unrelated_vibration_and_rechecks_sources(backend):
    env, line, _ = pipe_case()
    mech = env.mechanics
    base = mech.new_body('base', anchor=xt.Frame())
    owner = mech.new_body('chamber', anchor=xt.Frame(), parent=base)
    owner.installation.shift_y = .001
    line.mechanics.attach_pipe('installed', body=owner)
    mech.add_source('magnet', xt.mechanics.Harmonic(1e-6, 1.), targets='g')
    line.mechanics.enable_vibrations(backend=backend)
    aperture = line.mechanics.get_aperture(time_s=0.)
    assert aperture.pipe_positions['installed'].shift_y == pytest.approx(.001)
    particles = line.build_particles()
    line.track(particles)
    mech.add_source('pipe_parent', xt.mechanics.Harmonic(1e-6, 1.), targets='base')
    with pytest.raises(NotImplementedError, match='Automatic moving-pipe'):
        line.track(particles)
    assert particles.at_turn[0] == 1


def test_unused_aperture_shapes_do_not_invalidate_ownership():
    env, line, _ = pipe_case()
    profile = env.aperture.new_profile('unused_profile', 'Circle', radius=.01)
    env.aperture.new_pipe('unused', positions=[profile.place(at=.2), profile.place(at=.4)])
    line.mechanics.attach_pipe('installed', body=None)
    env.aperture.pipes['unused'][1].shift_s = .6
    assert line.mechanics.get_pipe_table().name[0] == 'installed'
    env.aperture.pipes['tube'][1].shift_s = 1.8
    with pytest.raises(ValueError, match='Adopted pipe geometry changed'):
        line.mechanics.get_pipe_table()


def test_stale_built_survey_is_rejected_at_pipe_only_attachment():
    _, line, _ = pipe_case()
    line.mechanics.clear()
    line.aperture.build_model()
    line['q'].length = 2.1
    with pytest.raises(ValueError, match='Aperture survey is stale'):
        line.mechanics.attach_pipe('installed', body=None)


def test_pipe_only_history_replays_live_and_prescribed_inputs_at_recorded_freeze():
    class HeldMotion:
        value = 0.

        def evaluate(self, *, time_s, positions_m):
            return np.full((len(time_s), len(positions_m)), self.value)

    env, line, _ = pipe_case()
    line.mechanics.clear()
    mech = env.mechanics
    owner = mech.new_body('pipe_owner', anchor=xt.Frame())
    line.mechanics.attach_pipe('installed', body=owner)
    held = HeldMotion()
    mech.add_source('live', held, targets=owner.name, live=True, role='control')
    mech.add_source('tone', xt.mechanics.Harmonic(1e-3, 1.), targets=owner.name)
    recorder = line.mechanics.record_applied_sources(sources=['live'])
    for time, value in [(0., .001), (.25, .002)]:
        held.value = value
        line.mechanics.apply(time_s=time)
    history = recorder.snapshot()
    recorder.stop()
    held.value = .1
    with pytest.raises(ValueError, match='applied history'):
        line.mechanics.get_aperture(time_s=.125)
    table = line.mechanics.get_pipe_table(time_s=.125, applied_history=history)
    assert table.query_kind[0] == 'applied_history'
    assert table.pose_time_s[0] == 0.
    assert table.change_Y[0] == pytest.approx(.001)
    aperture = line.mechanics.get_aperture(time_s=.25, applied_history=history)
    assert aperture.pipe_positions['installed'].shift_y == pytest.approx(.003)


def test_coverage_search_also_accepts_placements_far_from_their_reference():
    env = xt.Environment()
    names = [f'd{i}' for i in range(12)]
    for name in names:
        env.new(name, xt.Drift, length=1.)
    line = env.new_line(name='line', components=names)
    profile = env.aperture.new_profile('circle', 'Circle', radius=.01)
    pipe = env.aperture.new_pipe('pipe', positions=[profile.place(at=8.2), profile.place(at=8.8)])
    line.aperture.place_pipe('installed', pipe, at=names[0])
    line.aperture._with_progress = False
    line.mechanics.attach_pipe('installed', body=None)
    assert line.mechanics.get_aperture(time_s=0.).get_limit_elements_at_s([8.5])[8.5].installed_pipe_name == 'installed'


def test_axial_motion_outside_reference_path_rejects_before_geometry_kernel():
    env, line, girder = pipe_case()
    line.mechanics.attach_pipe('installed', body=girder)
    before = line.aperture.to_dict()
    girder.shift_s = 2.
    with pytest.raises(ValueError, match='no bracketed survey intersection'):
        line.mechanics.get_aperture(time_s=0.)
    np.testing.assert_array_equal(
        line.aperture.to_dict()['pipe_positions'][0]['transformation'],
        before['pipe_positions'][0]['transformation'])


def test_sample_outside_finite_pipe_rejects_instead_of_creating_unbounded_limit():
    env, line, girder = pipe_case()
    line.mechanics.attach_pipe('installed', body=girder)
    girder.shift_s = .05
    aperture = line.mechanics.get_aperture(time_s=0.)
    with pytest.raises(ValueError, match='No installed pipe covers'):
        aperture.get_limit_elements_at_s([.125])
    assert aperture.get_limit_elements_at_s([.2])[.2].installed_pipe_name == 'installed'


def test_bulk_ownership_invalid_batch_has_no_partial_declarations():
    env, line, girder = pipe_case()
    revision = env.mechanics._revision
    with pytest.raises(KeyError, match='Unknown installed pipe'):
        line.mechanics.attach_pipes({'installed': girder, 'missing': None})
    assert line.mechanics._pipes is None
    assert not line.aperture.is_built
    assert env.mechanics._revision == revision


def test_pipe_only_line_captures_geometry_and_round_trips_with_another_line():
    env, line, girder = pipe_case()
    line.mechanics.clear()
    line.mechanics.attach_pipe('installed', body=None)
    env.new('other_d', xt.Drift, length=2.)
    other = env.new_line(name='other', components=['other_d'])
    other.aperture.place_pipe('other_installed', 'tube', at='other_d')
    other.aperture._with_progress = False
    with pytest.raises(ValueError, match='explicit survey reference frame'):
        other.mechanics.attach_pipe('other_installed', body=None)
    other.mechanics.set_reference_frame(xt.Frame())
    other.mechanics.attach_pipe('other_installed', body=None)
    line.mechanics.get_pipe_table()
    copied = xt.Environment.from_dict(env.to_dict(include_aperture=True))
    assert copied.lines['line'].mechanics.get_pipe_table().motion[0] == 'fixed'
    assert copied.lines['other'].mechanics.get_pipe_table().motion[0] == 'fixed'
    line['q'].length = 2.1
    with pytest.raises(ValueError, match='reference geometry changed'):
        line.mechanics.get_aperture(time_s=0.)


def test_pipe_only_export_rejects_changed_reference_geometry():
    env, line, girder = pipe_case()
    line.mechanics.clear()
    line.mechanics.attach_pipe('installed', body=None)
    line.aperture.pipe_positions['installed'].shift_x += .001
    with pytest.raises(ValueError, match='Adopted pipe geometry changed'):
        env.to_dict(include_aperture=True)


def test_envelope_settings_are_not_pinned_mechanical_geometry():
    env, line, girder = pipe_case()
    line.mechanics.attach_pipe('installed', body=girder)
    line.aperture.halo_params['nemitt_x'] = 2e-6
    aperture = line.mechanics.get_aperture(time_s=0.)
    assert aperture.halo_params['nemitt_x'] == 2e-6
    with pytest.raises(ValueError, match='real seconds'):
        line.mechanics.get_aperture(time_s=1j)
    with pytest.raises(ValueError, match='must be real'):
        aperture.get_limit_elements_at_s(np.array([1. + 1j]))


def test_independent_pipe_owners_retain_explicit_gap_interpolation():
    env, line, girder = pipe_case()
    line.aperture.pipes['tube'][1].shift_s = .9
    profile = env.aperture.profiles['rectangle']
    right_pipe = env.aperture.new_pipe('right_shape', positions=[
        profile.place(at=.1), profile.place(at=.9)])
    line.aperture.place_pipe('right', right_pipe, at='q', shift_x=.002, shift_z=1.)
    right = env.mechanics.new_body('right_body', anchor=xt.Frame())
    line.mechanics.attach_pipes({'installed': girder, 'right': right})
    girder.shift_y = .0003
    right.shift_y = -.0002
    aperture = line.mechanics.get_aperture(time_s=0.)
    sections = aperture.get_cross_sections_at_s([.5, 1., 1.5]).cross_section
    centres = [(points[:, 1].min() + points[:, 1].max())/2 for points in sections]
    np.testing.assert_allclose(centres, [.0003, .00005, -.0002], atol=1e-12, rtol=0)
    # This bridge is an aperture interpolation rule, not a structural bellows.
    assert aperture.get_limit_elements_at_s([1.])[1.].installed_pipe_name == 'installed,right'


def test_pipe_tagged_frozen_limits_cannot_be_mechanically_attached():
    env = xt.Environment()
    env.new('d', xt.Drift, length=2.)
    env.new('ap', xt.LimitRect, min_x=-.01, max_x=.01,
            min_y=-.01, max_y=.01, installed_pipe_name='installed')
    line = env.new_line(name='line', components=['d', 'ap'])
    profile = env.aperture.new_profile('round', 'Circle', radius=.01)
    pipe = env.aperture.new_pipe('pipe', positions=[profile.place(at=.1), profile.place(at=1.9)])
    line.aperture.place_pipe('installed', pipe, at='d')
    line.aperture._with_progress = False
    line.mechanics.attach_pipe('installed', body=None)
    with pytest.raises(ValueError, match='Pipe-derived limits are frozen'):
        line.mechanics.new_girder('bad', elements=['ap'])
    assert 'bad' not in env.mechanics.bodies


@pytest.mark.parametrize('num_points', [129, 513])
def test_shared_pipe_world_sections_agree_with_polygon_sampling_error(num_points):
    env, forward, girder = pipe_case()
    env.new('q_back', xt.Quadrupole, length=2., k1=.1)
    backward = env.new_line(name='backward', components=['q_back'])
    reverse = xt.Frame.from_survey_angles(Z=2., theta=np.pi)
    backward.mechanics.set_reference_frame(reverse)
    backward.mechanics.attach(girder, elements=['q_back'])
    backward.aperture.place_pipe('back_pipe', 'tube', at='q_back', shift_x=-.002)
    backward.aperture._with_progress = False
    forward.aperture.num_profile_points = num_points
    backward.aperture.num_profile_points = num_points
    forward.mechanics.attach_pipe('installed', body=girder)
    backward.mechanics.attach_pipe('back_pipe', body=girder)
    girder.rot_x_rad, girder.rot_y_rad, girder.rot_s_rad = .02, -.03, .04
    world_sections = []
    for line in (forward, backward):
        aperture = line.mechanics.get_aperture(time_s=0.)
        section = aperture.get_cross_sections_at_s([1.]).cross_section[0][:-1]
        reference = line.mechanics.reference_frame @ aperture.survey.get_frame(0)
        local = np.column_stack((section, np.ones(len(section))))
        world = local @ reference.E_matrix.T + reference.XYZ
        world_sections.append(world)
    # Native section resampling changes its cyclic phase on beam reversal.
    # Compare world point sets within one perimeter sample, rather than demand
    # identical point indices. Both the mesh and this error bound refine with N.
    distance = np.linalg.norm(world_sections[0][:, None] - world_sections[1], axis=-1)
    hausdorff = max(distance.min(axis=0).max(), distance.min(axis=1).max())
    assert hausdorff < 4*(.01 + .015)/(num_points - 1)


def test_matched_freeze_applies_tilted_pipe_and_magnet_from_one_live_read():
    class HeldMotion:
        calls = 0

        def evaluate(self, *, time_s, positions_m):
            self.calls += 1
            return np.full((len(time_s), len(positions_m)), self.calls * .001)

    env, line, girder = pipe_case()
    line.mechanics.attach_pipe('installed', body=girder)
    girder.installation.shift_y = 3e-4
    held = HeldMotion()
    env.mechanics.add_source('drive', held, targets='g', component='x', live=True)
    env.mechanics.add_source('tilt', xt.mechanics.Harmonic(.025, 1., unit='rad'),
                            targets='g', component='yaw')
    view = line.mechanics
    check_s = 1.6
    frozen = view.freeze_at(time_s=.25, aperture_at_s=[check_s])
    assert held.calls == 1
    assert frozen.time_s == frozen.pose_time_s == view.last_applied_time_s == .25
    assert tuple(frozen.source_names) == ('drive', 'tilt')
    assert frozen.live_values['drive'][0, 0] == .001
    assert frozen.static_case['g'][7] == 3e-4
    assert frozen.design_alignment['q']['shift_x'] == .001
    assert line['q'].shift_x == pytest.approx(.001 + .001 * np.cos(.025) - np.sin(.025))
    expected = line.survey(include_element_frames=True).get_frame('q', which='elem_start')
    np.testing.assert_allclose(frozen.element_poses['q'].matrix, expected.matrix,
                               atol=1e-15, rtol=0)
    assert frozen.pipe_table.query_kind[0] == 'fresh_freeze'
    assert frozen.pipe_table.pose_time_s[0] == .25
    assert frozen.pipe_table.change_Y[0] == pytest.approx(3e-4)
    limit = frozen.limits[check_s]
    # Independent fixed-plane rectangle intersection, including the adopted
    # .002 m horizontal offset in the tilted frame and the live translation.
    centre = .001 + .002 / np.cos(.025) + (check_s - 1.) * np.tan(.025)
    np.testing.assert_allclose([min(limit.x_vertices), max(limit.x_vertices)],
                               centre + np.array([-.01, .01]) / np.cos(.025),
                               atol=1e-10, rtol=0)
    reference_at_check = line.aperture.get_limit_elements_at_s([check_s])[check_s]
    for check, expected_alive in [(reference_at_check, [True, False]), (limit, [False, True])]:
        probe = xt.Line(elements=[check])
        probe.particle_ref = line.particle_ref.copy()
        probe.build_tracker()
        particles = probe.build_particles(x=[.002, centre])
        probe.track(particles)
        order = np.argsort(particles.particle_id)
        np.testing.assert_array_equal(particles.state[order] > 0, expected_alive)
    before = (line['q'].shift_x, line['q'].rot_y_rad, view.last_applied_time_s)
    with pytest.raises(ValueError, match='No installed pipe'):
        view.freeze_at(time_s=.5, aperture_at_s=[9.])
    assert (line['q'].shift_x, line['q'].rot_y_rad, view.last_applied_time_s) == before
    np.testing.assert_array_equal(frozen.live_values['drive'][0, 0], .001)


def test_matched_freeze_history_replays_the_whole_line_at_the_held_time():
    class HeldMotion:
        value = 0.
        calls = 0

        def evaluate(self, *, time_s, positions_m):
            self.calls += 1
            return np.full((len(time_s), len(positions_m)), self.value)

    env, line, girder = pipe_case()
    # The pipe-only owner falls outside the live magnet's affected case.
    quiet = env.mechanics.new_body('quiet', anchor=xt.Frame())
    line.mechanics.attach_pipe('installed', body=quiet)
    held = HeldMotion()
    mech = env.mechanics
    mech.add_source('live', held, targets=girder.name, live=True)
    mech.add_source('tone', xt.mechanics.Harmonic(.001, 1.), targets=quiet.name)
    mech.add_source('disabled', xt.mechanics.Harmonic(.01, 1.), targets=girder.name)
    record = mech.compute_time_series(time_s=[0., .25, .5])
    view = line.mechanics
    recorder = view.record_applied_sources(sources='live', case_scope='line')
    for time_s, value in [(0., .001), (.25, .002), (.5, .003)]:
        held.value = value
        view.apply(time_s=time_s, time_series=record, disabled_sources=['disabled'])
    history = recorder.snapshot()
    recorder.stop()
    calls = held.calls
    frozen = view.freeze_at(time_s=.375, applied_history=history, aperture_at_s=[1.])
    assert held.calls == calls
    assert frozen.time_s == .375 and frozen.pose_time_s == view.last_applied_time_s == .25
    assert frozen.disabled_sources == ('disabled',) and frozen.prescribed_record is record
    assert line['q'].shift_y == pytest.approx(.002)
    assert frozen.pipe_table.change_Y[0] == pytest.approx(.001)
    assert frozen.aperture.mechanics_pose_time_s == .25
    quiet.installation.shift_y = 1e-4
    with pytest.raises(ValueError, match='static/reference case'):
        view.freeze_at(time_s=.375, applied_history=history)
    assert line['q'].shift_y == pytest.approx(.002)
