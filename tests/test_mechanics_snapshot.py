"""Ownership-safe frozen snapshots and removal failures for mechanics."""
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
