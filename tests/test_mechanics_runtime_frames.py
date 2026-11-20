"""Portable shared-support frames, independent clocks and line lifecycle."""
import numpy as np
import pytest
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt


@for_all_test_contexts
def test_prepared_shared_support_opposite_axes_and_independent_clocks(test_context):
    env = xt.Environment()
    for suffix, length in (('forward', 100.), ('backward', 180.)):
        env.new('q.'+suffix, xt.Quadrupole, length=1., k1=.1)
        env.new('d.'+suffix, xt.Drift, length=length)
        line = env.new_line(name=suffix, components=['q.'+suffix, 'd.'+suffix])
        line.set_particle_ref('proton', p0c=7e12)
        line.build_tracker(_context=test_context)
    forward, backward = env.forward, env.backward
    backward.mechanics.set_reference_frame(xt.Frame.from_survey_angles(Z=1., theta=np.pi))
    support = forward.mechanics.new_girder('shared', elements=['q.forward'])
    backward.mechanics.attach(support, elements=['q.backward'])
    support.installation.shift_y = 20e-6
    amplitude, frequency = 50e-9, 1e5
    env.mechanics.add_source('shared_tone', xt.mechanics.Harmonic(
        amplitude, frequency, phase_rad=np.pi/2), targets='shared', component='x')
    for line in (forward, backward):
        line.mechanics.enable_vibrations(backend='compiled', time_origin_s=0.)

    # Same physical translation has opposite local x in the two beam frames.
    for line, name, sign in ((forward, 'q.forward', 1.), (backward, 'q.backward', -1.)):
        particle = line.build_particles(x=1e-7)
        line.track(particle, num_turns=1)
        assert line[name].shift_x == pytest.approx(sign*amplitude, abs=1e-22, rel=0.)

    first, second = forward.build_particles(), backward.build_particles()
    forward.track(first, num_turns=3)
    first_time, first_x = first.time_s, forward['q.forward'].shift_x
    backward.track(second, num_turns=5)
    assert first.time_s == first_time
    assert first.time_s != second.time_s
    assert first.reference_turns == 3.
    assert second.reference_turns == 5.
    assert forward['q.forward'].shift_x == first_x
    # Environment elements can share one Xobjects storage buffer. Prepared
    # occurrence output/state is line-owned and writes distinct element offsets.
    assert forward['q.forward']._xobject._offset != backward['q.backward']._xobject._offset
    assert forward.mechanics._runtime.bodyout is not backward.mechanics._runtime.bodyout
    assert forward.mechanics._runtime.values is not backward.mechanics._runtime.values
    assert forward.mechanics._runtime._clocks is not backward.mechanics._runtime._clocks
    for line, name, sign, particle, count in (
            (forward, 'q.forward', 1., first, 3),
            (backward, 'q.backward', -1., second, 5)):
        expected = sign*amplitude*np.cos(2*np.pi*frequency*(count-1)*particle.t_sim)
        assert line[name].shift_x == pytest.approx(expected, abs=1e-21, rel=0.)

    # Disabling one line removes only its dynamic placement. The shared source,
    # other line's current pose and both particle clocks remain untouched.
    second_time, second_x = second.time_s, backward['q.backward'].shift_x
    forward.mechanics.disable_vibrations()
    assert forward['q.forward'].shift_x == 0.
    assert forward['q.forward'].shift_y == 20e-6
    assert backward['q.backward'].shift_x == second_x
    assert backward.mechanics.vibrations_enabled
    assert second.time_s == second_time
    assert first.time_s == first_time
    assert 'shared_tone' in env.mechanics.sources
    backward.track(second, num_turns=2)
    expected = -amplitude*np.cos(2*np.pi*frequency*6*second.t_sim)
    assert backward['q.backward'].shift_x == pytest.approx(expected, abs=1e-21, rel=0.)
    assert first.time_s == first_time
    backward.mechanics.disable_vibrations()
