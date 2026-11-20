"""Prepared mechanics shares Tracker's buffers, flags and failure lifecycle."""
import numpy as np
import pytest
import xobjects as xo
import xtrack as xt


def make_line():
    line = xt.Line(elements={
        'q': xt.Quadrupole(length=.5, k1=.1),
        'd': xt.Drift(length=1.),
    })
    line.set_particle_ref('proton', p0c=7e12)
    line.build_tracker(_context=xo.ContextCpu())
    return line


def assert_particles_equal(first, second):
    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's',
                 'at_turn', 'at_element', 'state'):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.reference_turns == second.reference_turns
    assert first.t_sim == second.t_sim
    assert first.time_s == pytest.approx(second.time_s, rel=0., abs=1e-22)


def test_prepared_and_reference_share_both_monitor_buffers():
    line = make_line()
    line.mechanics.new_girder('g', elements=['q'])
    line.env.mechanics.add_source('h', xt.mechanics.Harmonic(1e-8, 20000.), targets='g')
    records = []
    for log in (None, xt.Log()):
        line.mechanics.enable_vibrations(backend='compiled')
        particles = line.build_particles(x=[1e-5, -2e-5], y=[2e-6, -1e-6])
        line.track(particles, num_turns=4, turn_by_turn_monitor=True,
                   multi_element_monitor_at=['q', 'd'], log=log)
        records.append((particles.copy(), line.record_last_track,
                        line.record_multi_element_last_track))
        line.mechanics.disable_vibrations()
    assert_particles_equal(records[0][0], records[1][0])
    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
        np.testing.assert_array_equal(getattr(records[0][1], name),
                                      getattr(records[1][1], name))
    np.testing.assert_array_equal(records[0][2].data, records[1][2].data)


@pytest.mark.parametrize('tracking_options', [
    dict(ele_start=1, ele_stop=1, num_turns=2),
    dict(num_turns=2, freeze_longitudinal=True),
    dict(num_turns=2, backtrack=True),
])
def test_unsupported_mechanics_schedules_fail_before_moving_particles(tracking_options):
    line = make_line()
    initial = line.build_particles(x=[1e-5, -2e-5], y=[2e-6, -1e-6], delta=1e-4)
    expected = initial.copy()
    line.track(expected, **tracking_options)
    line.mechanics.new_girder('g', elements=['q'])
    line.mechanics.enable_vibrations()
    actual = initial.copy()
    with pytest.raises(NotImplementedError):
        line.track(actual, **tracking_options)
    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's', 'at_turn', 'at_element'):
        np.testing.assert_array_equal(getattr(actual, name), getattr(initial, name))
    line.mechanics.disable_vibrations()
    line.track(actual, **tracking_options)
    assert_particles_equal(actual, expected)


def test_ebe_monitor_uses_fallback_with_identical_element_records():
    line = make_line()
    initial = line.build_particles(x=1e-5, y=2e-6)
    expected = initial.copy()
    line.track(expected, turn_by_turn_monitor='ONE_TURN_EBE')
    expected_monitor = line.record_last_track
    line.mechanics.new_girder('g', elements=['q'])
    line.mechanics.enable_vibrations()
    actual = initial.copy()
    line.track(actual, turn_by_turn_monitor='ONE_TURN_EBE')
    assert_particles_equal(actual, expected)
    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 'at_element'):
        np.testing.assert_array_equal(getattr(line.record_last_track, name),
                                      getattr(expected_monitor, name))


def test_failed_source_preserves_completed_turn_and_can_resume():
    class InterruptibleMotion(xt.mechanics.Harmonic):
        fail = True

        def evaluate(self, time_s, positions_m):
            if self.fail and np.any(np.asarray(time_s) > 0.):
                raise ValueError('source unavailable')
            return super().evaluate(time_s, positions_m)

    line = make_line()
    line.mechanics.new_girder('g', elements=['q'])
    source = InterruptibleMotion(1e-8, 20000., np.pi/3)
    line.env.mechanics.add_source('h', source, targets='g')
    line.mechanics.enable_vibrations()
    particles = line.build_particles(x=1e-5, y=2e-6)
    with pytest.raises(ValueError, match='source unavailable'):
        line.track(particles, num_turns=3)
    assert particles.at_turn[0] == 1
    assert particles.reference_turns == 1.
    assert particles.time_s == particles.t_sim
    # Cleanup copied back the last successful applied fields: the conflict
    # guard must permit continuation, rather than reject its own kernel output.
    line.mechanics.validate()
    source.fail = False
    line.track(particles, num_turns=2)
    uninterrupted = line.build_particles(x=1e-5, y=2e-6)
    line.track(uninterrupted, num_turns=3)
    assert_particles_equal(particles, uninterrupted)


def test_tracker_restores_config_even_if_mechanics_cleanup_raises(monkeypatch):
    line = make_line()
    line.mechanics.new_girder('g', elements=['q'])
    line.mechanics.enable_vibrations()
    original = line.config.XTRACK_TPSA_TRACK = True

    def fail_cleanup(self):
        raise RuntimeError('cleanup failed')

    monkeypatch.setattr(type(line.mechanics), '_finish_tracking', fail_cleanup)
    with pytest.raises(RuntimeError, match='cleanup failed'):
        line.track(line.build_particles())
    assert line.config.XTRACK_TPSA_TRACK == original
