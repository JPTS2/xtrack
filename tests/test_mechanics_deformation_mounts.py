"""A prescribed flexible support carries rigid magnets at coherent mount ports."""

import numpy as np
import pytest
from scipy.constants import c
from scipy.spatial.transform import Rotation
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt


LENGTH = 6.0
STATIONS = np.array([1.0, 3.0, 5.0])
TIMES = np.arange(41) * 0.001
PORT_HEIGHT = 0.2
PARENT_VALUES = np.array([20e-6, -30e-6, 10e-6, 0.04, -0.03, 0.05])
INSTALLATION = np.array([3e-6, 7e-6, -2e-6, 0.001, -0.002, 0.003])
ALIGNMENT = ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad',
             'rot_s_rad_no_frame', 'rot_shift_anchor', 'rot_s_rad')


def affine(rotation=np.eye(3), translation=(0.0, 0.0, 0.0)):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, translation
    return result


def survey_rotation(angles):
    pitch, yaw, roll = angles
    return Rotation.from_euler('YXZ', [yaw, -pitch, roll]).as_matrix()


def amplitudes(time):
    return 2e-6 * np.sin(2 * np.pi * 5 * time), 8e-6 * np.cos(2 * np.pi * 7 * time)


def prescribed_ports():
    """One shared field: bending slope sets pitch; torsion rotates the section."""
    values = np.zeros((len(TIMES), len(STATIONS), 6))
    for row, time in enumerate(TIMES):
        bending, torsion = amplitudes(time)
        for column, station in enumerate(STATIONS):
            phase = np.pi * station / LENGTH
            displacement = bending * np.sin(phase)
            pitch = np.arctan(bending * np.pi / LENGTH * np.cos(phase))
            roll = torsion * np.cos(phase)
            rotation = survey_rotation((pitch, 0.0, roll))
            values[row, column, :3] = (
                np.array([0.0, displacement, 0.0])
                + (rotation - np.eye(3)) @ [0.0, PORT_HEIGHT, 0.0]
            )
            values[row, column, 3:] = pitch, 0.0, roll
    return xt.mechanics.RigidMotion(TIMES, values, ('port.q1', 'port.q2', 'port.q3'))


def make_case(flexural=None, *, context=None):
    env = xt.Environment()
    for name, strength in [('q1', 0.12), ('q2', -0.15), ('q3', 0.1)]:
        env.new(name, xt.Quadrupole, length=0.4, k1=strength)
    env['q1'].shift_x = 0.3e-3
    env['q1'].shift_y = -0.2e-6
    env['q1'].rot_s_rad = 0.04
    env['q2'].rot_x_rad = 0.002
    env['q2'].rot_shift_anchor = 0.3
    env.new('entrance', xt.Drift, length=0.8)
    env.new('gap1', xt.Drift, length=1.6)
    env.new('gap2', xt.Drift, length=1.6)
    env.new('exit', xt.Drift, length=0.8)
    reference = xt.Particles(p0c=7e12)
    # The map supplies a stable toy return and a 1 ms reference period.
    env.elements['return'] = xt.LineSegmentMap(
        length=float(reference.beta0[0]) * c * 0.001 - LENGTH,
        qx=0.23, qy=0.31, betx=8.0, bety=9.0,
        damping_rate_x=0.02, damping_rate_px=0.02,
        damping_rate_y=0.02, damping_rate_py=0.02,
    )
    line = env.new_line(name='ring', components=[
        'entrance', 'q1', 'gap1', 'q2', 'gap2', 'q3', 'exit', 'return',
    ])
    line.particle_ref = reference
    line.build_tracker(_context=context)
    original = {name: np.array([getattr(line[name], field) for field in ALIGNMENT])
                for name in ('q1', 'q2', 'q3')}
    baseline = line.survey(include_element_frames=True)
    mech = env.mechanics
    line.mechanics.adopt_reference()
    girder = mech.new_girder('reference', anchor=xt.Frame.from_survey_angles(Y=-0.6))
    for name, value in zip(ALIGNMENT[:6], INSTALLATION):
        # Body roll uses rot_s_rad rather than element rot_s_rad_no_frame.
        setattr(girder.installation, 'rot_s_rad' if name == 'rot_s_rad_no_frame' else name, value)
    for name, station in zip(('q1', 'q2', 'q3'), STATIONS):
        port = mech.new_body('port.' + name, parent=girder,
                             mount=xt.Frame.from_survey_angles(Y=PORT_HEIGHT, Z=station),
                             kind='mount')
        mech.new_body('magnet.' + name, parent=port,
                      mount=xt.Frame.from_survey_angles(Y=0.2),
                      elements=[name], line=line, kind='magnet')
    mech.add_source('common_parent', xt.mechanics.RigidMotion(
        TIMES, np.tile(PARENT_VALUES, (len(TIMES), 1)), ('reference',)),
        targets='reference')
    if flexural is None:
        flexural = prescribed_ports()
    mech.add_source('flexural_mounts', flexural, targets=flexural.target_names)
    return env, line, baseline, original


def expected_parent(dynamic=True):
    installation_rotation = survey_rotation(INSTALLATION[3:])
    rotation = survey_rotation(PARENT_VALUES[3:]) @ installation_rotation if dynamic else installation_rotation
    translation = np.array([0.0, -0.6, 0.0]) + INSTALLATION[:3]
    if dynamic:
        translation += PARENT_VALUES[:3]
    return affine(rotation, translation)


def independent_section(station, time):
    """Build axes from the centreline tangent, independently of survey Euler code."""
    bending, torsion = amplitudes(time)
    phase = np.pi * station / LENGTH
    slope = bending * np.pi / LENGTH * np.cos(phase)
    tangent = np.array([0.0, slope, 1.0])
    tangent /= np.linalg.norm(tangent)
    cross_x = np.array([1.0, 0.0, 0.0])
    cross_y = np.cross(tangent, cross_x)
    untwisted = np.column_stack((cross_x, cross_y, tangent))
    twist = Rotation.from_rotvec([0.0, 0.0, torsion * np.cos(phase)]).as_matrix()
    return affine(untwisted @ twist, [0.0, bending * np.sin(phase), station])


@pytest.mark.parametrize('backend', ['reference', 'compiled'])
def test_deformed_mount_geometry_and_cleanup(backend):
    env, line, baseline, original = make_case()
    line.mechanics.enable_vibrations(backend=backend)
    for time in TIMES[[0, 10, 25, 40]]:
        particles = line.build_particles(x=1e-4, y=-2e-4)
        particles.time_s = float(time)
        line.track(particles, num_turns=1)
        assert np.all(particles.state > 0)
        assert (line.mechanics._runtime is not None) == (backend == 'compiled')
        actual = line.survey(include_element_frames=True)
        for name, station in zip(('q1', 'q2', 'q3'), STATIONS):
            # Nominal magnet-body base is 0.2 m below the magnetic axis.
            nominal_base = affine(translation=[0.0, -0.2, station])
            section = independent_section(station, float(time))
            # Section neutral axis → mount (0.2 m) → stand/base (0.2 m).
            moved_base = expected_parent() @ section @ affine(translation=[0.0, 0.4, 0.0])
            for which in ('elem_start', 'elem_end'):
                expected = moved_base @ np.linalg.inv(nominal_base) @ baseline.get_frame(name, which=which).matrix
                np.testing.assert_allclose(actual.get_frame(name, which=which).matrix,
                                           expected, atol=2e-13, rtol=0)
    # Neither design fields nor the installation input have been rewritten.
    assert line['q1'].rot_s_rad == 0.04
    np.testing.assert_array_equal(env.mechanics['reference'].installation.values(), INSTALLATION)
    # Removing either prescribed contribution leaves the other in its declared
    # physical frame. An absolute port record would incorrectly mask the parent.
    for disabled, parent_dynamic, flex_dynamic in [
        (('flexural_mounts',), True, False),
        (('common_parent',), False, True),
    ]:
        poses = line.mechanics.get_element_poses(time_s=0.025, disabled_sources=disabled)
        for name, station in zip(('q1', 'q2', 'q3'), STATIONS):
            section = independent_section(station, 0.025) if flex_dynamic else affine(
                translation=[0.0, 0.0, station])
            nominal_base = affine(translation=[0.0, -0.2, station])
            expected = (expected_parent(dynamic=parent_dynamic) @ section
                        @ affine(translation=[0.0, 0.4, 0.0])
                        @ np.linalg.inv(nominal_base)
                        @ baseline.get_frame(name, which='elem_start').matrix)
            np.testing.assert_allclose(poses[name].matrix, expected, atol=2e-13, rtol=0)
    line.mechanics.disable_vibrations()
    assert not line.mechanics.automatic_updates_enabled
    actual = line.survey(include_element_frames=True)
    for name in ('q1', 'q2', 'q3'):
        expected = expected_parent(dynamic=False) @ affine(translation=[0.0, 0.6, 0.0])
        # This conjugation rotates/translates the entire original line geometry.
        expected = expected @ baseline.get_frame(name, which='elem_start').matrix
        np.testing.assert_allclose(actual.get_frame(name, which='elem_start').matrix,
                                   expected, atol=2e-13, rtol=0)
    env.mechanics.clear()
    assert not env.mechanics.bodies and not env.mechanics.sources
    for name, values in original.items():
        np.testing.assert_array_equal([getattr(line[name], field) for field in ALIGNMENT], values)


def test_reference_compiled_and_split_tracking_agree():
    cases = []
    for backend, blocks in [('reference', [12]), ('compiled', [12]), ('compiled', [5, 7])]:
        _, line, _, _ = make_case()
        line.mechanics.enable_vibrations(backend=backend)
        particles = line.build_particles(x=1e-4, y=-2e-4)
        for turns in blocks:
            line.track(particles, num_turns=turns)
        assert np.all(particles.state > 0)
        assert int(particles.at_turn[0]) == 12
        cases.append((particles, line))
    reference, _ = cases[0]
    for particles, line in cases[1:]:
        assert line.mechanics._runtime is not None
        for coordinate in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
            np.testing.assert_allclose(getattr(particles, coordinate), getattr(reference, coordinate),
                                       atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(float(particles.time_s), float(reference.time_s), atol=2e-17, rtol=0)


def test_sections_are_rigid_but_stations_have_different_deformations():
    time = 0.025
    cross_section = np.array([[-0.15, 0.2, 0.0], [0.15, 0.2, 0.0], [0.0, 0.6, 0.0]])
    pitches, twists = [], []
    for station in STATIONS:
        frame = independent_section(station, time)
        moved = cross_section @ frame[:3, :3].T + frame[:3, 3]
        np.testing.assert_allclose(np.linalg.norm(moved[1] - moved[0]), 0.3, atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(moved[2] - moved[0]),
                                   np.linalg.norm(cross_section[2] - cross_section[0]), atol=1e-15)
        pitches.append(frame[1, 2])
        twists.append(frame[0, 1])
    assert pitches[0] > 0 and pitches[-1] < 0
    assert twists[0] < 0 and twists[-1] > 0
    assert abs(pitches[1]) < 1e-20 and abs(twists[1]) < 1e-20


def test_grouped_modal_shapes_and_snapshot_round_trip():
    targets = ('port.q1', 'port.q2', 'port.q3')
    modes = ('bend', 'twist')
    shapes = np.arange(36.).reshape(2, 3, 6)
    amplitudes = np.array([[[1., 2.], [3., 4.]], [[2., 4.], [-3., 5.]]]) * 1e-8
    groups = [1, 0, 1]
    motion = xt.mechanics.ModalMotion([0., 1.], amplitudes, shapes, targets, modes, groups)
    samples = motion.evaluate([0., .25, 1.], STATIONS)
    expected = np.empty_like(samples)
    for row, time in enumerate([0., .25, 1.]):
        for col, group in enumerate(groups):
            for coordinate in range(6):
                modal_amplitude = (1. - time) * amplitudes[0, group] + time * amplitudes[1, group]
                expected[row, col, coordinate] = sum(modal_amplitude * shapes[:, col, coordinate])
    np.testing.assert_allclose(samples, expected, rtol=2e-15, atol=0.)
    for array in (motion.time_s, motion.amplitudes, motion.mode_shapes, motion.amplitude_groups):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(ValueError, match='outside'):
        motion.evaluate([1.1], STATIONS)
    with pytest.raises(ValueError, match='group index'):
        xt.mechanics.ModalMotion([0., 1.], amplitudes, shapes, targets, modes, [1, 0, 2])

    env = xt.Environment()
    for index, name in enumerate(targets):
        env.mechanics.new_body(name, anchor=(0., 0., STATIONS[index]))
    env.mechanics.add_source('flexural_mounts', motion, targets=targets)
    copied = xt.Environment.from_dict(env.to_dict())
    restored = copied.mechanics.sources['flexural_mounts'].source
    assert restored.target_names == targets and restored.mode_names == modes
    np.testing.assert_array_equal(restored.evaluate([0., .25, 1.], STATIONS), samples)
    series = env.mechanics.compute_time_series(time_s=[0., .25, 1.])
    np.testing.assert_array_equal(series.values['flexural_mounts'], samples)


@for_all_test_contexts
def test_modal_compiled_tracking_matches_direct_and_zero_modes(test_context):
    targets = ('port.q1', 'port.q2', 'port.q3')
    shapes = np.zeros((2, len(targets), 6))
    shapes[0, :, 1] = [.3, 1., .6]
    shapes[0, :, 3] = [.6, .0, -.6]
    shapes[1, :, 5] = [1., .0, -1.]
    amplitude = np.array([amplitudes(time) for time in TIMES])
    for mask in ([1., 0.], [1., 1.], [0., 0.]):
        modal = xt.mechanics.ModalMotion(TIMES, amplitude * mask, shapes,
                                       targets, ('bend', 'twist'))
        direct = xt.mechanics.RigidMotion(
            TIMES, np.einsum('tm,mbc->tbc', amplitude * mask, shapes), targets,
        )
        cases = []
        for source in (modal, direct):
            _, line, _, _ = make_case(source, context=test_context)
            line.mechanics.enable_vibrations(backend='compiled')
            particles = line.build_particles(x=1e-4, y=-2e-4)
            particles.time_s = .0005  # Halfway between samples
            line.track(particles, num_turns=8)
            cases.append((particles, line.survey(include_element_frames=True)))
            assert test_context.nparray_from_context_array(particles.at_turn)[0] == 8
            if not any(mask):
                moving = line.mechanics.get_element_poses(time_s=.005)
                rigid = line.mechanics.get_element_poses(time_s=.005, disabled_sources=['flexural_mounts'])
                for name in moving:
                    np.testing.assert_array_equal(moving[name].matrix, rigid[name].matrix)
        for coordinate in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
            np.testing.assert_allclose(
                *(test_context.nparray_from_context_array(getattr(p, coordinate)) for p, _ in cases),
                atol=2e-14, rtol=0)
        np.testing.assert_allclose(cases[0][1].XYZ_elem_start, cases[1][1].XYZ_elem_start,
                                   atol=1e-16, rtol=0)
