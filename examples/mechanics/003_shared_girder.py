# Two line views of one physical girder, including opposite traversal directions.
import numpy as np
import xtrack as xt

env = xt.Environment()
env.new('q.forward', xt.Quadrupole, length=1., k1=.1)
env.new('q.backward', xt.Quadrupole, length=1., k1=.1)
forward = env.new_line(name='forward', components=['q.forward'])
backward = env.new_line(name='backward', components=['q.backward'])

# Place BOTH surveys in the same physical world. The reverse line starts at Z=1
# and travels towards Z=0. This is geometry, not a second mechanical world.
backward.mechanics.set_reference_frame(xt.Frame.from_survey_angles(Z=1., theta=np.pi))

mech = env.mechanics
girder = forward.mechanics.new_girder('shared', elements=['q.forward'])
backward.mechanics.attach(girder, elements=['q.backward'])
mech.add_source('surface_wave', xt.mechanics.PlaneWave(
    amplitude=50e-9, frequency_hz=5., speed_m_s=200.,
    direction=(0, 0, 1), polarization=(1, 0, 0), phase_rad=np.pi/2),
    targets=['shared'])

# Explicit global time is unambiguous even when beam turn lengths differ.
time_s = .01
mech.apply(time_s=time_s)
print('World body pose at t [s]:', time_s, mech.get_body_poses(time_s=time_s)['shared'])
print('Local horizontal offsets forward/backward [nm]:',
      env['q.forward'].shift_x*1e9, env['q.backward'].shift_x*1e9)
np.testing.assert_allclose(env['q.forward'].shift_x, -env['q.backward'].shift_x, atol=1e-20)

# Automatic shared-body scheduling needs a common event clock. The exploration
# currently rejects enable_vibrations() for multi-line bindings rather than
# assigning a global meaning to either line's turn counter.
