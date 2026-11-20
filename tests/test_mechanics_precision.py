"""Independent local-increment checks, not agreement of two world surveys."""
import numpy as np
import pytest

import xtrack as xt
from xtrack.mechanics import Harmonic, RigidMotion
from xtrack.mechanics.local_deformations import rotation_increment


def placed_quad(origin, *, angles=(0., 0., 0.), **kwargs):
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.5, k1=.2, **kwargs)
    line = env.new_line(name='ring', components=['q'])
    line.mechanics.set_reference_frame(xt.Frame.from_survey_angles(
        X=origin[0], Y=origin[1], Z=origin[2],
        phi=angles[0], theta=angles[1], psi=angles[2]))
    return env, line


@pytest.mark.parametrize('origin', [(0., 0., 0.), (20000., -30000., 50000.),
                                  (1e9, -1e9, 1e9)])
@pytest.mark.parametrize('amplitude', [1e-9, 1e-11, 1e-13])
def test_small_local_translation_does_not_round_trip_through_world(origin, amplitude):
    env, line = placed_quad(origin, angles=(.23, -.71, .12), shift_x=.003,
                            rot_s_rad=.2, rot_s_rad_no_frame=.1)
    line.mechanics.new_girder('g', elements=['q'])
    env.mechanics.add_source('translation', Harmonic(amplitude, 0., np.pi/2),
                            targets='g', component='x')
    env.mechanics.apply()
    assert line['q'].shift_x == pytest.approx(.003+amplitude, rel=0., abs=1e-18)
    assert line['q'].shift_y == pytest.approx(0., rel=0., abs=1e-24)
    assert line['q'].shift_s == pytest.approx(0., rel=0., abs=1e-24)
    assert line['q'].rot_s_rad_no_frame == pytest.approx(.1, rel=0., abs=1e-16)


@pytest.mark.parametrize('origin', [(0., 0., 0.), (20000., -30000., 50000.),
                                  (1e9, -1e9, 1e9)])
def test_nanoradian_rotation_keeps_small_second_order_longitudinal_shift(origin):
    env, line = placed_quad(origin)
    line.mechanics.new_girder('g', elements=['q'])
    angle = 1e-9
    env.mechanics.add_source('yaw', Harmonic(angle, 0., np.pi/2, unit='rad'),
                            targets='g', component='yaw')
    env.mechanics.apply()
    # The nominal body is the centre of the 0.5 m quadrupole. A positive
    # survey yaw moves its entrance in -x and +s. Use stable analytic values.
    assert line['q'].shift_x == pytest.approx(-.25*np.sin(angle), rel=1e-15, abs=0.)
    assert line['q'].shift_s == pytest.approx(.5*np.sin(angle/2)**2, rel=1e-15, abs=0.)
    assert line['q'].rot_y_rad == pytest.approx(angle, rel=1e-15, abs=0.)


def test_zero_motion_preserves_deliberate_fields_and_rotation_anchor_exactly():
    env = xt.Environment()
    env.new('b', xt.Bend, length=1.5, angle=.1, shift_x=.003,
            shift_y=-.002, shift_s=.0004, rot_x_rad=.02, rot_y_rad=-.04,
            rot_s_rad_no_frame=.07, rot_s_rad=.3, rot_shift_anchor=.8)
    line = env.new_line(name='ring', components=['b'])
    line.mechanics.set_reference_frame(xt.Frame.from_survey_angles(X=50000., Z=-30000.))
    fields = ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad',
              'rot_s_rad_no_frame', 'rot_shift_anchor', 'rot_s_rad')
    original = [getattr(line['b'], field) for field in fields]
    line.mechanics.adopt_reference()
    line.mechanics.new_girder('g', elements=['b'])
    env.mechanics.apply()
    assert [getattr(line['b'], field) for field in fields] == original


def hierarchical_case(origin):
    env, line = placed_quad(origin)
    reference = xt.Frame.from_survey_angles(X=origin[0]+2., Y=origin[1]+1., Z=origin[2]-3.,
                                           phi=.2, theta=-.3, psi=.1)
    parent = env.mechanics.new_girder('floor', anchor=reference)
    child = line.mechanics.new_girder('g', elements=['q'], parent=parent)
    parent.shift_y = 2e-6
    parent.installation.rot_s_rad = 3e-5
    child.installation.shift_x = 1e-6
    env.mechanics.add_source('floor_yaw', Harmonic(1e-9, 0., np.pi/2, unit='rad'),
                            targets='floor', component='yaw')
    env.mechanics.add_source('local', Harmonic(1e-12, 0., np.pi/2),
                            targets='g', component='x')
    return env, line


def test_nested_motion_and_installation_are_invariant_to_world_origin():
    values = []
    for origin in ((0., 0., 0.), (20000., -30000., 50000.), (1e9, -1e9, 1e9)):
        env, line = hierarchical_case(origin)
        env.mechanics.apply()
        values.append([getattr(line['q'], name) for name in
            ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad', 'rot_s_rad_no_frame')])
    np.testing.assert_array_equal(values[0], values[1])
    np.testing.assert_array_equal(values[0], values[2])


def test_absolute_child_keeps_static_mount_but_rejects_parent_vibration():
    env, line = placed_quad((20000., 0., 30000.))
    parent = env.mechanics.new_girder('floor', anchor=(20000., 0., 30000.))
    parent.installation.shift_y = 2e-6
    child = line.mechanics.new_girder('g', elements=['q'], parent=parent)
    env.mechanics.add_source('floor_motion', Harmonic(1e-6, 0., np.pi/2),
                            targets='floor', component='y')
    coordinates = np.zeros((2, 1, 6))
    coordinates[:, 0, 0] = 1e-12
    coordinates[:, 0, 4] = 1e-9
    motion = RigidMotion([0., 1.], coordinates, target_names=('g',))
    env.mechanics.add_source('absolute_child', motion, targets='g', reference='absolute')
    env.mechanics.apply(time_s=.5)
    assert line['q'].shift_y == pytest.approx(2e-6, rel=0., abs=1e-21)
    assert line['q'].shift_x == pytest.approx(1e-12-.25e-9, rel=0., abs=1e-24)


def test_increment_rotation_agrees_with_finite_survey_convention():
    for angles in ((.2, -.3, .4), (-1.2, .9, -.7), (1e-9, -2e-9, 3e-9)):
        expected = xt.Frame.from_survey_angles(phi=angles[0], theta=angles[1], psi=angles[2])
        np.testing.assert_allclose(np.eye(3)+rotation_increment(*angles), expected.E_matrix,
                                   rtol=0., atol=3e-16)


def test_pose_tables_and_selected_history_preserve_sub_world_ulp_motion():
    env, line = placed_quad((1e9, -1e9, 1e9))
    line.mechanics.new_girder('g', elements=['q'])
    amplitude = 1e-13
    env.mechanics.add_source('tone', Harmonic(amplitude, 1.), targets='g', component='x')
    for table in (env.mechanics.get_table(time_s=.25), line.mechanics.get_table(time_s=.25)):
        # The world display pose cannot encode 100 fm at a 1e9 m origin.
        assert table.total_X[0] == table.static_X[0]
        assert table.dynamic_X[0] == pytest.approx(amplitude, rel=1e-15, abs=0.)
        history = table.get_time_series(time_s=[0., .25, .5])
        assert history.dynamic_X[1] == pytest.approx(amplitude, rel=1e-15, abs=0.)


def test_bpm_reading_uses_local_pose_and_preserves_small_beam_coordinates():
    from xtrack.mechanics.observation import project_to_bpm
    env, line = placed_quad((1e9, -1e9, 1e9))
    line.mechanics.new_girder('g', elements=['q'])
    amplitude = 1e-13
    env.mechanics.add_source('tone', Harmonic(amplitude, 0., np.pi/2), targets='g', component='x')
    reading = line.mechanics.read_bpm('q', x=3e-13)
    assert reading.x == pytest.approx(2e-13, rel=1e-15, abs=0.)
    frame = xt.Frame.from_survey_angles(X=1e9, Y=-1e9, Z=1e9, theta=.2)
    # Two explicitly equal world frames require no world displacement sum.
    generic = project_to_bpm(beam_frame=frame, bpm_frame=frame, x=amplitude, y=-amplitude)
    assert generic.x == pytest.approx(amplitude, rel=1e-15, abs=0.)
    assert generic.y == pytest.approx(-amplitude, rel=1e-15, abs=0.)


def test_transport_preserves_zero_and_nanoradian_lever_terms():
    source_frame = xt.Frame.from_survey_angles(X=1e9, Y=-1e9, Z=1e9,
                                              phi=.2, theta=-.7, psi=.1)
    target_frame = source_frame @ xt.Frame.from_survey_angles(X=2., Z=3., theta=.3)
    zero = RigidMotion([0.,1.], np.zeros((2,1,6)))
    transported = zero.transport(from_frames=[source_frame],to_frames=[target_frame])
    np.testing.assert_array_equal(transported.values, np.zeros((2,1,6)))
    angle = 1e-9
    values = np.zeros((2,1,6))
    values[...,4] = angle
    source_frame = xt.Frame.from_survey_angles(X=1e9, Y=-1e9, Z=1e9)
    target_frame = xt.Frame.from_survey_angles(X=1e9, Y=-1e9, Z=1e9+2.)
    transported = RigidMotion([0.,1.],values).transport(from_frames=[source_frame],to_frames=[target_frame])
    assert transported.values[0,0,0] == pytest.approx(2*np.sin(angle), rel=1e-15,abs=0.)
    assert transported.values[0,0,2] == pytest.approx(-4*np.sin(angle/2)**2, rel=1e-15,abs=0.)
    assert transported.values[0,0,4] == pytest.approx(angle, rel=1e-15,abs=0.)


def test_selected_query_composes_ancestor_closure_and_preserves_joint_sources():
    class JointSource:
        frame = 'body_local'
        unit = 'm'
        def evaluate(self, *, time_s, positions_m):
            # A receiver-set-dependent source must retain its joint contract.
            return np.full((len(time_s),len(positions_m)),len(positions_m)*1e-9)
    env = xt.Environment()
    root = env.mechanics.new_girder('floor',anchor=(20000.,0.,30000.))
    child = env.mechanics.new_girder('g',anchor=(20000.,0.,30001.),parent=root)
    unused = env.mechanics.new_girder('unused',anchor=(0.,0.,0.))
    root.installation.shift_x = 1e-6
    env.mechanics.add_source('joint',JointSource(),targets=['floor','unused'],component='x')
    all_poses = env.mechanics.get_body_poses()
    selected = env.mechanics.get_body_poses(bodies=['g'])
    assert tuple(selected) == ('g',)
    np.testing.assert_array_equal(selected['g'].matrix,all_poses['g'].matrix)
    assert selected._deformations['g'][0][0] == pytest.approx(1e-6+2e-9,rel=0.,abs=1e-21)
    # A selected history need not compose an unrelated body's broken settings.
    unused.shift_y = np.nan
    env.mechanics.get_body_poses(bodies=['g'])
    with pytest.raises(ValueError,match='local alignment must be finite'):
        env.mechanics.get_body_poses()


def test_scaled_display_uses_precise_dynamic_columns_before_world_addition():
    from xtrack.mechanics.plotting import _display_matrices, _table_display_motion
    env, line = placed_quad((20000.,-30000.,50000.))
    body = line.mechanics.new_girder('g', elements=['q'])
    body.installation.shift_y = 1e-5
    env.mechanics.add_source('tone',Harmonic(1e-13,0.,np.pi/2),targets='g',component='x')
    table = env.mechanics.get_table(time_s=0.)
    assert table.total_X[0] == table.static_X[0]
    shown = _display_matrices(table.static_matrix,table.total_matrix,1e10,1.,
                              **_table_display_motion(table,'static'))
    assert shown[0,0,3]-table.static_X[0] == pytest.approx(.001,rel=0.,abs=5e-12)
    # A static installation must not be magnified into a dynamic display.
    assert shown[0,1,3] == table.static_Y[0]


@pytest.mark.parametrize('pitch', [np.pi/2,np.pi/2-1e-12,-np.pi/2,
                                  -np.pi/2+1e-12,np.pi/2+.001,-np.pi/2-.001])
def test_total_rotation_at_and_near_euler_lock_preserves_mapping(pitch):
    from xtrack.mechanics.motion import angles_from_matrix
    expected = xt.Frame.from_survey_angles(phi=pitch,theta=.3,psi=-.7).E_matrix
    increment = rotation_increment(pitch,.3,-.7)
    phi,theta,psi = angles_from_matrix(np.eye(3)+increment)
    actual = xt.Frame.from_survey_angles(phi=phi,theta=theta,psi=psi).E_matrix
    np.testing.assert_allclose(actual,expected,rtol=0.,atol=1e-15)
    env,line = placed_quad((20000.,0.,30000.))
    body = line.mechanics.new_girder('g',elements=['q'])
    body.rot_x_rad,body.rot_y_rad,body.rot_s_rad = pitch,.3,-.7
    env.mechanics.apply()
    reconstructed = xt.Frame.from_survey_angles(phi=line['q'].rot_x_rad,
        theta=line['q'].rot_y_rad,psi=line['q'].rot_s_rad_no_frame).E_matrix
    np.testing.assert_allclose(reconstructed,expected,rtol=0.,atol=1e-15)


def test_survey_euler_extraction_reconstructs_random_finite_rotations():
    from xtrack.mechanics.motion import angles_from_matrix
    rng = np.random.default_rng(956)
    values = rng.uniform(-np.pi,np.pi,(256,3))
    rotations = np.eye(3)+rotation_increment(*values.T)
    angles = angles_from_matrix(rotations)
    rebuilt = np.eye(3)+rotation_increment(*angles.T)
    np.testing.assert_allclose(rebuilt,rotations,rtol=0.,atol=3e-14)


@pytest.mark.parametrize('pitch', [np.pi/2-1e-8, -np.pi/2+1e-8, .3])
def test_finite_rotation_near_euler_singularity_retains_rigid_mapping(pitch):
    env, line = placed_quad((0., 0., 0.), shift_x=.003, shift_y=-.002,
                            rot_x_rad=.13, rot_y_rad=-.2, rot_s_rad_no_frame=.17,
                            rot_s_rad=.4, rot_shift_anchor=.2)
    original = line.survey(include_element_frames=True).get_frame('q', which='elem_start')
    body = line.mechanics.new_girder('g', elements=['q'])
    body.rot_x_rad = pitch
    body.rot_y_rad = np.pi-1e-9
    body.rot_s_rad = -.7
    body.shift_x = .002
    delta = xt.Frame.from_survey_angles(X=.002, phi=pitch, theta=np.pi-1e-9, psi=-.7)
    expected = body.nominal_frame @ delta @ body.nominal_frame.inverse() @ original
    env.mechanics.apply()
    actual = line.survey(include_element_frames=True).get_frame('q', which='elem_start')
    np.testing.assert_allclose(actual.matrix, expected.matrix, rtol=0., atol=2e-14)
