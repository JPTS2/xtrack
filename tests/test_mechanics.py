import json
from io import StringIO

import numpy as np
import pytest

import xtrack as xt
from xtrack.mechanics import Harmonic, PlaneWave, MeasuredMotion, DisplacementPSD


def make_line():
    env = xt.Environment()
    env.new('q1', xt.Quadrupole, length=.3, k1=.8, shift_y=3e-6)
    env.new('d', xt.Drift, length=1.)
    env.new('q2', xt.Quadrupole, length=.4, k1=-.6, rot_s_rad_no_frame=.001)
    line = env.new_line(name='ring', components=['q1', 'd', 'q2'])
    line.set_particle_ref('proton', p0c=7e12)
    return env, line


def test_native_accessor_and_finite_nested_survey_pose():
    env, line = make_line()
    assert line.mechanics.manager is env.mechanics
    assert line.mechanics is env.mechanics.for_line('ring')
    baseline = line.survey(include_element_frames=True)
    parent = env.mechanics.new_girder('support', anchor=xt.Frame())
    girder = line.mechanics.new_girder('g', elements=['q1', 'q2'], parent=parent)
    parent.rot_s_rad = .02
    girder.shift_y = 2e-4
    girder.rot_y_rad = .003
    targets = line.mechanics.get_element_poses()
    env.mechanics.apply()
    actual = line.survey(include_element_frames=True)
    for name in ('q1', 'q2'):
        np.testing.assert_allclose(actual.get_frame(name, which='elem_start').matrix,
                                   targets[name].matrix, atol=2e-14, rtol=0)
    fields = line['q1'].shift_y
    env.mechanics.apply()
    assert line['q1'].shift_y == fields
    env.mechanics.restore()
    restored = line.survey(include_element_frames=True)
    for name in ('q1', 'q2'):
        np.testing.assert_array_equal(restored.get_frame(name, which='elem_start').matrix,
                                      baseline.get_frame(name, which='elem_start').matrix)


def test_curved_nominal_path_preserves_both_end_frames():
    env = xt.Environment()
    env.new('b', xt.Bend, length=2, angle=.4, rot_s_rad=.15,
            shift_x=.001, rot_x_rad=.003, rot_shift_anchor=.7)
    line = env.new_line(name='l', components=['b'])
    before = line.survey(include_element_frames=True)
    body = line.mechanics.new_girder('g', elements=['b'])
    body.shift_x, body.shift_y, body.rot_y_rad = .005, -.002, .02
    move = env.mechanics.get_body_poses()['g'] @ body.nominal_frame.inverse()
    env.mechanics.apply()
    after = line.survey(include_element_frames=True)
    for which in ('elem_start', 'elem_end'):
        np.testing.assert_allclose(after.get_frame('b', which=which).matrix,
            (move @ before.get_frame('b', which=which)).matrix, atol=2e-14, rtol=0)


def test_shared_body_two_survey_placements_and_opposite_traversal():
    env = xt.Environment()
    env.new('q1', xt.Quadrupole, length=1, k1=.1)
    env.new('q2', xt.Quadrupole, length=1, k1=.1)
    first = env.new_line(name='first', components=['q1'])
    second = env.new_line(name='second', components=['q2'])
    second.mechanics.set_reference_frame(xt.Frame.from_survey_angles(Z=1, theta=np.pi))
    g = first.mechanics.new_girder('shared', elements=['q1'])
    second.mechanics.attach(g, elements=['q2'])
    assert second.mechanics.manager is first.mechanics.manager
    env.mechanics.add_source('tone', Harmonic(50e-9, 5, np.pi/2), targets='shared', component='x')
    env.mechanics.apply(time_s=0)
    assert first['q1'].shift_x == pytest.approx(50e-9)
    assert second['q2'].shift_x == pytest.approx(-50e-9)
    with pytest.raises(NotImplementedError, match='multi-line'):
        first.mechanics.enable_vibrations()
    with pytest.raises(ValueError, match='Specify line='):
        env.mechanics.new_girder('ambiguous', elements=['q1'])


def test_sources_time_series_replay_and_local_static_changes():
    env, line = make_line()
    g1 = line.mechanics.new_girder('g1', elements=['q1'])
    line.mechanics.new_girder('g2', elements=['q2'])
    env.mechanics.add_source('wave', PlaneWave(50e-9, 5, (0, 0, 1), 100), targets=['g1', 'g2'])
    times = np.arange(1000)*.001
    series = env.mechanics.compute_time_series(time_s=times)
    for t in times[[0, 50, 999]]:
        direct = env.mechanics.get_body_poses(time_s=t)
        stored = env.mechanics.get_body_poses(time_s=t, time_series=series)
        for name in direct:
            np.testing.assert_array_equal(direct[name].matrix, stored[name].matrix)
    g1.shift_y = 20e-6
    env.mechanics.apply(time_s=.05, time_series=series)
    with pytest.raises(ValueError):
        series.values['wave'][0, 0, 0] = 5
    with pytest.raises(ValueError, match='outside'):
        env.mechanics.get_body_poses(time_s=2., time_series=series)
    assert len(series.get_table('g1').time_s) == 1000
    env.mechanics.add_source('extra', Harmonic(1e-9, 3), targets='g1')
    with pytest.raises(ValueError, match='changed'):
        env.mechanics.get_body_poses(time_s=0, time_series=series)


def test_psd_seed_normalization_and_colocated_correlation():
    dt, count = .001, 32768
    source = DisplacementPSD([0, 1, 2, 80, 81], [0, 0, 1e-17, 1e-17, 0],
                             seed=956, correlation_length_m=10)
    times = np.arange(count)*dt
    positions = np.zeros((2, 3))
    values = source.evaluate(times, positions)
    np.testing.assert_array_equal(values, source.evaluate(times, positions))
    np.testing.assert_allclose(values[:, 0], values[:, 1], atol=1e-20)
    f = np.fft.rfftfreq(count, dt)
    expected = np.interp(f, source.frequency_hz, source.density, left=0, right=0).sum()/(count*dt)
    assert np.mean(values[:, 0]**2) == pytest.approx(expected, rel=.06)
    with pytest.raises(ValueError, match='Nyquist'):
        source.evaluate(np.arange(100)*.1, positions)
    with pytest.raises(ValueError, match='uniform'):
        source.evaluate(np.array([0., .1, .3]), positions)


def test_psd_window_uses_the_same_physical_phase_origin():
    count, dt, offset = 4096, 1/1024, 128
    source = DisplacementPSD([0, 1, 2, 80, 81], [0, 0, 1e-17, 1e-17, 0], seed=956)
    time = np.arange(count)*dt
    original = source.evaluate(time, np.zeros((1, 3)))
    shifted = source.evaluate(time + offset*dt, np.zeros((1, 3)))
    np.testing.assert_allclose(shifted, np.roll(original, -offset, axis=0), atol=1e-21, rtol=0)


def test_measured_values_and_projection_to_rotated_world_frame():
    source = MeasuredMotion([0, 1, 2], [0, 2e-9, 0], interpolation='previous')
    np.testing.assert_array_equal(source.evaluate(np.array([.5, 1.5]), np.zeros((1, 3))), [[0], [2e-9]])
    with pytest.raises(ValueError, match='outside'):
        source.evaluate(np.array([3.]), np.zeros((1, 3)))
    env, line = make_line()
    anchor = xt.Frame.from_survey_angles(psi=.3)
    line.mechanics.new_girder('g', elements=['q1'], anchor=anchor)
    env.mechanics.add_source('world', PlaneWave(1e-6, 0, (1, 0, 0), 100,
                                             polarization=(0, 1, 0), phase_rad=np.pi/2), targets='g')
    position = env.mechanics.get_body_poses()['g'].XYZ
    np.testing.assert_allclose(position, [0, 1e-6, 0], atol=1e-20)


def test_environment_roundtrip_and_copy_do_not_double_apply():
    env, line = make_line()
    g = line.mechanics.new_girder('g', elements=['q1', 'q2'])
    g.shift_y = 2e-5
    env.mechanics.add_source('tone', Harmonic(5e-9, 2), targets='g')
    env.mechanics.apply(time_s=.1)
    before = line['q1'].shift_y
    stream = StringIO()
    xt.json.dump(env.to_dict(), stream)
    data = json.loads(stream.getvalue())
    for copied in (xt.Environment.from_dict(data, with_progress=False), env.copy(with_progress=False)):
        assert copied.mechanics['g'] is not g
        assert copied['q1'].shift_y == before
        copied.mechanics.apply(time_s=.1)
        assert copied['q1'].shift_y == pytest.approx(before, abs=1e-18)
        copied.mechanics.restore()
        assert copied['q1'].shift_y == 3e-6
    with pytest.raises(NotImplementedError, match='complete environment'):
        line.to_dict()
    with pytest.raises(NotImplementedError, match='env.copy'):
        line.copy()


def test_conflicting_edits_aliases_and_topology_fail_before_writes():
    env, line = make_line()
    g = line.mechanics.new_girder('g', elements=['q1', 'q2'])
    g.shift_y = 1e-4
    env['q2'].shift_y = 9e-5
    with pytest.raises(ValueError, match='edited'):
        env.mechanics.apply()
    assert env['q1'].shift_y == 3e-6
    env['q2'].shift_y = 0
    env.new_line(name='alias', components=['q1'])
    with pytest.raises(ValueError, match='shared/repeated'):
        env.mechanics.apply()
    assert env['q1'].shift_y == 3e-6


def test_source_overlap_and_memory_budget_are_explicit():
    env, line = make_line()
    env.mechanics.new_girder('s', anchor=(0, 0, 0))
    line.mechanics.new_girder('g', elements=['q1'], parent='s')
    with pytest.raises(ValueError, match='ancestor'):
        env.mechanics.add_source('bad', Harmonic(1e-9, 5), targets=['s', 'g'])
    env.mechanics.add_source('good', Harmonic(1e-9, 5), targets='s')
    with pytest.raises(ValueError, match='max_memory_bytes'):
        env.mechanics.compute_time_series(time_s=np.arange(100)*.01, max_memory_bytes=1)


def test_unmounted_reference_geometry_changes_invalidate_survey_bindings():
    env, line = make_line()
    line.mechanics.new_girder('g', elements=['q2'])
    env['d'].length = 2
    with pytest.raises(ValueError, match='reference geometry changed'):
        env.mechanics.apply()


def test_second_line_requires_explicit_placement_and_expressions_stay_owned():
    env, line = make_line()
    line.mechanics.new_girder('g', elements=['q1'])
    env.new('other', xt.Quadrupole, length=1, k1=.1)
    other = env.new_line(name='other_line', components=['other'])
    with pytest.raises(ValueError, match='explicit survey'):
        other.mechanics.new_girder('other_g', elements=['other'])
    assert 'other_g' not in env.mechanics.bodies
    env['offset'] = 3e-6
    env.ref['q1'].shift_y = env.ref['offset']
    with pytest.raises(ValueError, match='expression-driven'):
        env.mechanics.apply()
    env['offset'] = 7e-6
    assert env['q1'].shift_y == 7e-6


@pytest.mark.parametrize('factory', [
    lambda: Harmonic(1j, 5),
    lambda: PlaneWave(1, 5, (1j, 0, 0), 100),
    lambda: MeasuredMotion([0, 1], [0, 1j]),
    lambda: DisplacementPSD([0, 1], [0, 1j], 4),
])
def test_complex_displacements_are_not_silently_cast(factory):
    with pytest.raises(ValueError, match='real'):
        factory()


def test_custom_world_source_and_recorded_motion_can_drive_the_same_model():
    class Pump:
        frame = 'world'

        def evaluate(self, *, time_s, positions_m):
            distance = np.linalg.norm(positions_m - [0, 0, 0], axis=1)
            amplitude = 1e-8*np.exp(-distance/10)
            return (np.cos(2*np.pi*5*time_s)[:, None, None]
                    * amplitude[None, :, None]*np.array([0, 1, 0]))

    env, line = make_line()
    line.mechanics.new_girder('g', elements=['q1'])
    env.mechanics.add_source('pump', Pump(), targets='g')
    expected = Pump().evaluate(time_s=np.array([0.]), positions_m=np.array([[0, 0, .15]]))[0, 0, 1]
    env.mechanics.apply(time_s=0.)
    assert env['q1'].shift_y == pytest.approx(3e-6 + expected)
    with pytest.raises(TypeError, match='custom source serialization'):
        env.to_dict()


def test_vibration_enable_disable_and_split_tracking_use_physical_time():
    env, line = make_line()
    g = line.mechanics.new_girder('g', elements=['q1'])
    g.shift_y = 10e-6
    env.mechanics.add_source('tone', Harmonic(2e-6, 1e5), targets='g')
    line.build_tracker()
    particle = line.build_particles()
    times, values = [], []
    line.mechanics.enable_vibrations()
    assert line.mechanics.vibrations_enabled
    line.track(particle, num_turns=3, log=xt.Log(y=lambda line, p: (
        times.append(p.time_s), values.append(line['q1'].shift_y), p.time_s)[-1]))
    line.track(particle, num_turns=2, log=xt.Log(y=lambda line, p: (
        times.append(p.time_s), values.append(line['q1'].shift_y), p.time_s)[-1]))
    expected = 3e-6 + 10e-6 + 2e-6*np.sin(2*np.pi*1e5*np.asarray(times))
    np.testing.assert_allclose(values, expected, atol=1e-18, rtol=0)
    assert np.all(np.diff(times) > 0)
    with pytest.raises(ValueError, match='Disable vibrations'):
        line.twiss()
    with pytest.raises(NotImplementedError, match='complete turns'):
        line.track(particle, ele_start=1, num_turns=1)
    line.mechanics.disable_vibrations()
    assert not line.mechanics.vibrations_enabled
    assert line['q1'].shift_y == pytest.approx(13e-6)


def test_psd_enable_requires_computed_record():
    env, line = make_line()
    line.mechanics.new_girder('g', elements=['q1'])
    env.mechanics.add_source('noise', DisplacementPSD([0, 1, 10, 20], [0, 1e-18, 1e-18, 0], 5), targets='g')
    with pytest.raises(ValueError, match='compute_time_series'):
        line.mechanics.enable_vibrations()
    record = env.mechanics.compute_time_series(time_s=np.arange(1000)*.001)
    line.mechanics.enable_vibrations(time_series=record)
    line.mechanics.disable_vibrations()


def test_clear_removes_all_mechanics_but_preserves_deliberate_rolls_and_offsets():
    env, line = make_line()
    env['q1'].rot_s_rad = .17
    env['q1'].rot_s_rad_no_frame = .03
    local = {name: [getattr(env[name], field) for field in (
        'shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad',
        'rot_s_rad_no_frame', 'rot_shift_anchor', 'rot_s_rad')] for name in ('q1', 'q2')}
    g = line.mechanics.new_girder('g', elements=['q1', 'q2'])
    g.shift_y, g.rot_s_rad = 20e-6, .002
    env.mechanics.add_source('tone', Harmonic(50e-9, 5, np.pi/2), targets='g')
    record = env.mechanics.compute_time_series(time_s=np.arange(100)*.001)
    env.mechanics.apply(time_s=.01, time_series=record)
    table = line.mechanics.get_alignment_table()
    assert table['local_shift_y', 'q1'] == 3e-6
    assert table['design_rot_s_rad', 'q1'] == .17
    line.mechanics.enable_vibrations(time_series=record)
    # A deliberate design change made during exploration must survive removal.
    env['q1'].rot_s_rad = .18
    env.mechanics.clear()
    assert not env.mechanics.bodies and not env.mechanics.sources
    assert not line.mechanics.vibrations_enabled
    assert len(line.mechanics.get_table()) == 0
    for name in ('q1', 'q2'):
        fields = ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad',
                  'rot_s_rad_no_frame', 'rot_shift_anchor', 'rot_s_rad')
        actual = [getattr(env[name], field) for field in fields]
        expected = local[name].copy()
        if name == 'q1':
            expected[-1] = .18
        np.testing.assert_array_equal(actual, expected)
    assert 'mechanics' not in env.to_dict()
    # Retain the old scenario's input data for comparisons, without allowing it
    # to be applied silently to a cleared/reconfigured physical model.
    assert len(record.get_table('g')) == 100
    with pytest.raises(ValueError, match='changed'):
        env.mechanics.get_body_poses(time_s=.01, time_series=record)


def test_line_clear_detaches_only_that_line_and_conflicts_do_not_get_erased():
    env, first = make_line()
    g = first.mechanics.new_girder('shared', elements=['q1'])
    env.new('other', xt.Quadrupole, length=1, k1=.1)
    second = env.new_line(name='second', components=['other'])
    second.mechanics.set_reference_frame(xt.Frame())
    second.mechanics.attach(g, elements=['other'])
    g.shift_y = 10e-6
    env.mechanics.apply()
    first.mechanics.clear()
    assert env['q1'].shift_y == 3e-6
    assert env['other'].shift_y == 10e-6
    assert 'shared' in env.mechanics.bodies
    env['other'].shift_y = 99e-6
    with pytest.raises(ValueError, match='edited'):
        env.mechanics.clear()
    assert env['other'].shift_y == 99e-6
    assert 'shared' in env.mechanics.bodies


def test_clear_invalidates_derived_aperture_survey_without_removing_pipe_definitions():
    env, line = make_line()
    body = line.mechanics.new_girder('g', elements=['q1'])
    body.shift_y = 10e-6
    env.mechanics.apply()
    profile = env.aperture.new_profile('round', 'Circle', radius=.01)
    pipe = env.aperture.new_pipe('pipe', positions=[profile.place(), profile.place(at=.2)])
    line.aperture.place_pipe('installed', pipe, at='q1')
    line.aperture.build_model()
    assert line.aperture.is_built
    env.mechanics.clear()
    assert not line.aperture.is_built
    assert env.aperture.profiles['round'].name == 'round'
    assert env.aperture.pipes['pipe'].name == 'pipe'
    assert line['q1'].shift_y == 3e-6
