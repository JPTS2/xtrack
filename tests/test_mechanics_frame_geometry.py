"""Frame-changing survey elements must not invalidate adopted world geometry silently."""

import numpy as np
import pytest
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt
from xtrack.mechanics.geometry_guard import GeometryGuard


CHANGES = [
    ('translation', 'shift_x', .01),
    ('translation', 'shift_y', .02),
    ('translation', 'shift_s', .03),
    ('rotation', 'rot_x_rad', .04),
    ('rotation', 'rot_y_rad', .05),
    ('rotation', 'rot_s_rad', .06),
    ('rotation', 'seq', 'sxy'),
]


def make_case(kind):
    env = xt.Environment()
    if kind == 'translation':
        env.new('frame', xt.Translation)
    else:
        env.new('frame', xt.Rotation, rot_x_rad=.01, rot_y_rad=.02,
                rot_s_rad=.03)
    env.new('q', xt.Quadrupole, length=.4, k1=.2)
    line = env.new_line(name='line', components=['frame', 'q'])
    line.set_particle_ref('proton', p0c=7e12)
    line.mechanics.new_girder('g', elements=['q'])
    return env, line


@pytest.mark.parametrize('kind,field,value', CHANGES)
def test_reference_frame_changes_reject_before_numeric_writes(kind, field, value):
    env, line = make_case(kind)
    env.mechanics['g'].shift_y = 1e-6
    setattr(line['frame'], field, value)
    with pytest.raises(ValueError, match='reference geometry changed'):
        line.mechanics.validate()
    with pytest.raises(ValueError, match='reference geometry changed'):
        env.mechanics.apply()
    assert line['q'].shift_y == 0.


@pytest.mark.parametrize('kind,field,value', CHANGES)
def test_prepared_guard_traces_frame_transform_dependencies(kind, field, value, monkeypatch):
    _, line = make_case(kind)
    line.build_tracker(_context=xo.ContextCpu())
    guard = GeometryGuard(line.mechanics)
    guard.check()
    # Native entry validation must keep its batched gather, including the
    # dependencies discovered by the new survey-protocol trace.
    def forbidden(*args, **kwargs):
        raise AssertionError('Prepared guard attempted a scalar geometry read')

    monkeypatch.setattr(type(guard.buffer), 'to_bytearray', forbidden)
    guard.check()
    setattr(line['frame'], field, value)
    with pytest.raises(ValueError, match='geometry changed'):
        guard.check()


@for_all_test_contexts
@pytest.mark.parametrize('backend', ['reference', 'compiled'])
@pytest.mark.parametrize('kind,field,value', [CHANGES[0], CHANGES[-1]])
def test_tracking_rejects_changed_frame_before_advancing_particles(
    test_context, backend, kind, field, value,
):
    env, line = make_case(kind)
    env.mechanics.add_source('tone', xt.mechanics.Harmonic(1e-6, 2., .3),
                             targets='g', component='y')
    line.build_tracker(_context=test_context)
    line.mechanics.enable_vibrations(backend=backend)
    particles = line.build_particles(x=1e-4)
    line.track(particles, num_turns=1)
    before = np.array([line['q'].shift_x, line['q'].shift_y])
    setattr(line['frame'], field, value)
    with pytest.raises(ValueError, match='geometry changed'):
        line.track(particles, num_turns=1)
    assert test_context.nparray_from_context_array(particles.at_turn)[0] == 1
    np.testing.assert_array_equal([line['q'].shift_x, line['q'].shift_y], before)


def test_custom_survey_frame_dependency_is_protected():
    class CustomStep(xt.BeamElement):
        _xofields = {'horizontal_step': xo.Float64}
        allow_rot_and_shift = False
        iscollective = True

        def track(self, particles):
            pass

        def track_frame(self, frame, backtrack=False):
            frame.translate_x((-1 if backtrack else 1) * self.horizontal_step)

    env = xt.Environment()
    env.elements['step'] = CustomStep(horizontal_step=.001)
    env.new('q', xt.Quadrupole, length=.4, k1=.2)
    line = env.new_line(name='line', components=['step', 'q'])
    line.mechanics.new_girder('g', elements=['q'])
    line['step'].horizontal_step = .002
    with pytest.raises(ValueError, match='reference geometry changed'):
        line.mechanics.validate()
