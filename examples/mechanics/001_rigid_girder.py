# Attach a rigid girder after constructing an ordinary Xtrack line.
# Edit the names, anchor, displacement and yaw, then inspect the native tables.
import numpy as np
import xobjects as xo
import xtrack as xt

context = xo.ContextCpu()
# context = xo.ContextCupy()

env = xt.Environment()
env.new('q1', xt.Quadrupole, length=.2, k1=.2, shift_y=3e-6)
env.new('gap', xt.Drift, length=2.)
env.new('q2', xt.Quadrupole, length=.2, k1=-.2, rot_s_rad=.1)  # Deliberate magnet roll
line = env.new_line(name='line', components=['q1', 'gap', 'q2'])
line.set_particle_ref('proton', p0c=7e12)

mech = env.mechanics
girder = line.mechanics.new_girder('girder', elements=['q1', 'q2'], anchor='center')
girder.shift_y = 20e-6                  # [m], relative to the girder's nominal frame
girder.rot_y_rad = 2e-6                 # [rad], about its surveyed anchor
mech.apply()

print('Body-local settings')
mech.get_table().show()
print('Line attachments')
line.mechanics.get_table().show()
print('Original local alignment, current effective fields, deliberate design roll')
line.mechanics.get_alignment_table().cols[
    'name local_shift_y effective_shift_y design_rot_s_rad'].show()
print('Applied q1/q2 vertical offsets [µm]:', env['q1'].shift_y*1e6, env['q2'].shift_y*1e6)
np.testing.assert_allclose(env['q1'].shift_y, 23e-6, atol=1e-15)

line.build_tracker(_context=context)
particles = line.build_particles(x=0, y=0)
line.track(particles)
print('Particle at line exit x/y [µm]:', particles.x[0]*1e6, particles.y[0]*1e6)
# mech.restore() removes the girder contribution and recovers the local errors.
# mech.clear() also removes body/source definitions and all mechanics bindings.
