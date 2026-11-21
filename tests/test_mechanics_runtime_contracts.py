"""Prepared motion retains validation and logging contracts on nested supports."""
import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts


def nested_line(context):
    line = xt.Line(elements={
        'q': xt.Quadrupole(length=.5, k1=.1, shift_x=3e-6),
        'd': xt.Drift(length=.7),
    })
    line.set_particle_ref('proton', p0c=7e12)
    line.build_tracker(_context=context)
    manager = line.env.mechanics
    manager.new_girder('support', anchor=[0., 0., .25])
    manager.new_body('magnet', line=line, elements=['q'], parent='support', kind='magnet')
    manager.new_body('pickup', anchor=[0., 0., .3], parent='magnet', kind='pickup')
    manager.add_source('ground', xt.mechanics.Harmonic(2e-8, 100., .3),
                       targets='support', component='y')
    manager.add_source('local', xt.mechanics.Harmonic(1e-9, 300., .4, unit='rad'),
                       targets='magnet', component='roll')
    return line


def coordinates(context, particles):
    return {name: context.nparray_from_context_array(getattr(particles, name))
            for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 'at_turn')}


@for_all_test_contexts
def test_empty_log_auto_keeps_prepared_motion_and_log_reset(test_context):
    line = nested_line(test_context)
    results = []
    for backend in ('reference', 'auto'):
        line.mechanics.enable_vibrations(backend=backend)
        particles = line.build_particles(x=1e-6, y=2e-6)
        line.log_last_track = {'stale': [1.]}
        line.track(particles, num_turns=3, log=xt.Log())
        assert line.log_last_track == {}
        assert (line.mechanics._runtime is not None) == (backend == 'auto')
        results.append(coordinates(test_context, particles))
        line.mechanics.disable_vibrations()
    for name in results[0]:
        np.testing.assert_allclose(results[0][name], results[1][name], rtol=2e-13, atol=2e-17)


@pytest.mark.parametrize('log_kind', ['callback', 'subclass'])
def test_auto_does_not_trust_arbitrary_log_execution(log_kind):
    line = nested_line(xo.ContextCpu())
    line.mechanics.enable_vibrations()
    calls = []
    def mutate(line, particles):
        calls.append(int(particles.at_turn[0]))
        line['q'].length += .01
        return 0.
    class MutatingEmptyLog(xt.Log):
        def __iter__(self):
            mutate(line, particles)
            return super().__iter__()
    particles = line.build_particles()
    log = xt.Log(change=mutate) if log_kind == 'callback' else MutatingEmptyLog()
    with pytest.raises(ValueError, match='reference geometry changed'):
        line.track(particles, num_turns=2, log=log)
    assert calls == [0]
    assert line.mechanics._runtime is None


def test_alias_filter_checks_live_attached_objects_across_all_physical_lines():
    line = nested_line(xo.ContextCpu())
    env = line.env
    env.new('unrelated', xt.Quadrupole, length=.1)
    unrelated = env.new_line(name='unrelated_line', components=['unrelated', 'unrelated'])
    line.mechanics.enable_vibrations()
    line.track(line.build_particles(), num_turns=1)
    # Unconsumed aliases are permitted; an attached object under a new name is not.
    env.element_dict['renamed_q'] = env.element_dict['q']
    aliased = env.new_line(name='other_physical_line', components=['renamed_q'])
    with pytest.raises(ValueError, match='shared/repeated'):
        line.track(line.build_particles(), num_turns=1)
    assert unrelated.env is aliased.env is env


def test_empty_log_source_failure_keeps_completed_turn_and_can_resume():
    class InterruptibleMotion(xt.mechanics.Harmonic):
        fail = False

        def evaluate(self, time_s, positions_m):
            if self.fail and np.any(np.asarray(time_s) > 0.):
                raise ValueError('source unavailable')
            return super().evaluate(time_s, positions_m)

    line = nested_line(xo.ContextCpu())
    source = InterruptibleMotion(1e-8, 20000., .5)
    line.env.mechanics.add_source('pump', source, targets='support')
    line.mechanics.enable_vibrations()
    initial = line.build_particles(x=1e-6, y=2e-6)
    expected, actual = initial.copy(), initial.copy()
    line.track(expected, num_turns=2, log=xt.Log())
    source.fail = True
    with pytest.raises(ValueError, match='source unavailable'):
        line.track(actual, num_turns=2, log=xt.Log())
    assert actual.at_turn[0] == 1
    assert actual.reference_turns == 1.
    assert actual.time_s == actual.t_sim
    source.fail = False
    line.track(actual, num_turns=1, log=xt.Log())
    for name, values in coordinates(line._context, expected).items():
        np.testing.assert_array_equal(values, getattr(actual, name))
    assert actual.time_s == expected.time_s


@pytest.mark.parametrize('finish_during_evaluation', [False, True])
def test_source_cannot_add_callbacks_to_an_admitted_empty_log(finish_during_evaluation):
    line = nested_line(xo.ContextCpu())
    log = xt.Log()
    callbacks = []

    def mutate(line, particles):
        callbacks.append(True)
        line['q'].length += .01

    class LogChangingMotion(xt.mechanics.Harmonic):
        change_log = True

        def evaluate(self, time_s, positions_m):
            if self.change_log:
                if finish_during_evaluation:
                    # Exercise bookkeeping being cleared during user source
                    # code; the freeze must retain its admitted log locally.
                    line.mechanics._finish_tracking()
                log['change'] = mutate
            return super().evaluate(time_s, positions_m)

    source = LogChangingMotion(1e-9, 10.)
    line.env.mechanics.add_source('custom', source, targets='support')
    line.mechanics.enable_vibrations()
    particles = line.build_particles(x=1e-6, y=2e-6)
    initial = {name: values.copy()
               for name, values in coordinates(line._context, particles).items()}
    with pytest.raises(ValueError, match='initially empty Log changed'):
        line.track(particles, num_turns=2, log=log)
    assert not callbacks
    assert line['q'].length == .5
    assert particles.at_turn[0] == 0
    assert particles.time_s == 0.
    for name, values in initial.items():
        np.testing.assert_array_equal(values, getattr(particles, name))
    assert line.mechanics._prepared_empty_log is None
    # Failure does not retain the old mutable log or contaminate a later call.
    source.change_log = False
    log.clear()
    line.track(particles, num_turns=1, log=log)
    assert particles.at_turn[0] == 1
    assert line.mechanics._prepared_empty_log is None
