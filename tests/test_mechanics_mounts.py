"""Parent-local mounting has the same geometry in scalar and batch builders."""

import numpy as np
import pytest

import xtrack as xt


def mounted_line():
    env = xt.Environment()
    env.new('q1', xt.Quadrupole, length=.5, k1=.2, shift_x=.001)
    env.new('q2', xt.Quadrupole, length=.5, k1=-.2, rot_s_rad=.03)
    env.new('d', xt.Drift, length=.5)
    line = env.new_line(name='line', components=['q1', 'd', 'q2'])
    line.mechanics.adopt_reference()
    parent = env.mechanics.new_girder('g', anchor=xt.Frame.from_survey_angles(
        X=10., Y=-.7, Z=3., theta=.2, psi=-.03))
    return env, line, parent


def test_batch_magnet_and_pickup_mounts_match_scalar_nested_geometry():
    mounts = [xt.Frame.from_survey_angles(Y=.7, Z=.25, psi=.01),
              xt.Frame.from_survey_angles(Y=.7, Z=1.25, phi=-.02)]
    cases = []
    for batch in (False, True):
        env, line, parent = mounted_line()
        assignments = {f'm.{name}': dict(elements=[name], parent=parent, mount=mount)
                       for name, mount in zip(('q1', 'q2'), mounts)}
        if batch:
            bodies = line.mechanics.new_bodies(assignments, kind='magnet')
        else:
            bodies = {name: env.mechanics.new_body(name, line=line, kind='magnet', **options)
                      for name, options in assignments.items()}
        for name, mount in zip(('m.q1', 'm.q2'), mounts):
            np.testing.assert_array_equal(bodies[name].nominal_frame.matrix,
                                          (parent.nominal_frame @ mount).matrix)
        # A pickup is another physical child, even without a beam-element binding.
        pickup = env.mechanics.new_body('bpm', parent=bodies['m.q1'],
            mount=xt.Frame.from_survey_angles(Y=.1, Z=.2), kind='pickup')
        parent.installation.shift_y = 3e-6
        bodies['m.q1'].installation.rot_s_rad = .002
        env.mechanics.add_source('girder', xt.mechanics.Harmonic(
            .003, 0., np.pi/2, unit='rad'), targets=['g'], component='roll')
        env.mechanics.add_source('magnet', xt.mechanics.Harmonic(
            2e-8, 0., np.pi/2), targets=['m.q1'], component='x')
        env.mechanics.add_source('pickup', xt.mechanics.Harmonic(
            1e-8, 0., np.pi/2), targets=[pickup.name], component='y')
        cases.append((env, line, env.mechanics.get_body_poses(),
                      line.mechanics.get_element_poses()))
    for name in ('g', 'm.q1', 'm.q2', 'bpm'):
        np.testing.assert_array_equal(cases[0][2][name].matrix, cases[1][2][name].matrix)
    for name in ('q1', 'q2'):
        np.testing.assert_array_equal(cases[0][3][name].matrix, cases[1][3][name].matrix)
    for env, line, *_ in cases:
        env.mechanics.apply()
        env.mechanics.clear()
        assert line['q1'].shift_x == .001
        assert line['q2'].rot_s_rad == .03


def test_batch_mounts_use_one_survey(monkeypatch):
    env, line, parent = mounted_line()
    calls = []
    original = line.survey

    def survey(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(line, 'survey', survey)
    env.mechanics.new_bodies({
        'm1': dict(elements=['q1'], parent=parent, mount=(0., .7, .25)),
        'm2': dict(elements=['q2'], parent='g', mount=(0., .7, 1.25)),
    }, line=line, kind='magnet')
    assert len(calls) == 1


@pytest.mark.parametrize('invalid', [
    dict(mount=(0., .7, 0.)),
    dict(parent='g', mount=(0., .7, 0.), anchor=(0., 0., 0.)),
    dict(parent='unknown', mount=(0., .7, 0.)),
    dict(parent='foreign', mount=(0., .7, 0.)),
    dict(parent='g', mount=(0., .7j, 0.)),
])
def test_invalid_batch_mount_is_rejected_before_any_attachment(invalid):
    env, line, parent = mounted_line()
    options = dict(invalid)
    if options.get('parent') == 'foreign':
        options['parent'] = xt.Environment().mechanics.new_girder('g', anchor=xt.Frame())
    revision = env.mechanics._revision
    with pytest.raises(ValueError):
        line.mechanics.new_bodies({
            'valid': dict(elements=['q1'], parent=parent, mount=(0., .7, .25)),
            'invalid': dict(elements=['q2'], **options),
        }, kind='magnet')
    assert tuple(env.mechanics.bodies) == ('g',)
    assert not line.mechanics._attachments
    assert env.mechanics._revision == revision
    assert line['q1'].shift_x == .001


def test_scalar_and_batch_girders_accept_the_same_local_mount():
    matrices = []
    for batch in (False, True):
        env, line, parent = mounted_line()
        mount = xt.Frame.from_survey_angles(Y=.4, Z=.8, theta=.03)
        if batch:
            body = line.mechanics.new_girders({
                'upper': dict(elements=['q1', 'q2'], parent=parent, mount=mount)
            })['upper']
        else:
            body = line.mechanics.new_girder('upper', elements=['q1', 'q2'],
                                             parent=parent, mount=mount)
        matrices.append(body.nominal_frame.matrix)
        copied = env.copy()
        np.testing.assert_array_equal(copied.mechanics['upper'].nominal_frame.matrix,
                                      body.nominal_frame.matrix)
    np.testing.assert_array_equal(matrices[0], matrices[1])
