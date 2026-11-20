"""Owned alignment outputs must not silently leave xdeps consumers stale."""

import numpy as np
import pytest

import xtrack as xt


def make_line():
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.3, k1=.1, shift_y=3e-6)
    env.new('corr', xt.Multipole, knl=[0.])
    line = env.new_line(name='ring', components=['q', 'corr'])
    line.set_particle_ref('proton', p0c=7e12)
    return env, line


def add_consumer(env, target):
    if target == 'element':
        env.ref['corr'].knl[0] = env.ref['q'].shift_y
    else:
        env['derived_y'] = env.ref['q'].shift_y


@pytest.mark.parametrize('target', ['element', 'variable'])
def test_alignment_consumers_rejected_before_binding(target):
    env, line = make_line()
    add_consumer(env, target)
    with pytest.raises(ValueError, match='cannot drive xdeps expressions'):
        line.mechanics.new_girder('g', elements=['q'])
    assert not env.mechanics.bodies
    assert not line.mechanics._attachments
    assert env['q'].shift_y == 3e-6


def test_late_consumer_rejected_before_host_writes_and_copy():
    env, line = make_line()
    body = line.mechanics.new_girder('g', elements=['q'])
    body.installation.shift_y = 20e-6
    add_consumer(env, 'variable')
    for call in (env.mechanics.apply, env.mechanics.clear,
                 lambda: env.copy(with_progress=False)):
        with pytest.raises(ValueError, match='cannot drive xdeps expressions'):
            call()
        assert env['q'].shift_y == 3e-6
        assert env['derived_y'] == 3e-6
    assert 'g' in env.mechanics.bodies


def test_late_consumer_rejected_at_warm_compiled_entry_before_particle_motion():
    env, line = make_line()
    body = line.mechanics.new_girder('g', elements=['q'])
    body.shift_y = 20e-6
    line.build_tracker()
    line.mechanics.enable_vibrations(backend='compiled')
    line.mechanics._prepare_for_tracking()
    runtime = line.mechanics._runtime
    particles = line.build_particles(x=1e-7, y=2e-8)
    before = particles.to_dict()
    add_consumer(env, 'element')
    with pytest.raises(ValueError, match='cannot drive xdeps expressions'):
        line.track(particles, num_turns=1)
    assert line.mechanics._runtime is runtime
    assert env['q'].shift_y == 3e-6
    assert env['corr'].knl[0] == 3e-6
    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 'at_turn', 'at_element'):
        np.testing.assert_array_equal(getattr(particles, name), before[name])


def test_unrelated_strength_consumers_remain_supported():
    env, line = make_line()
    env['derived_k1'] = env.ref['q'].k1
    body = line.mechanics.new_girder('g', elements=['q'])
    body.shift_y = 20e-6
    env.mechanics.apply()
    assert env['q'].shift_y == pytest.approx(23e-6)
    assert env['derived_k1'] == .1
