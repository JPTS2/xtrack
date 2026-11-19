# One named source interface for a tone, a measured trace or a seeded PSD.
# The editable model is the point of this example; no install()/realize()/snapshot().
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

env = xt.Environment()
components = []
for i in range(4):
    env.new(f'q{i}', xt.Quadrupole, length=1., k1=.005*(-1)**i)
    env.new(f'd{i}', xt.Drift, length=100.)
    components += [f'q{i}', f'd{i}']
line = env.new_line(name='ring', components=components)
line.set_particle_ref('proton', p0c=7e12)
# A periodic focusing toy with a straight survey, not a closed surveyed ring.

mech = env.mechanics
girder = line.mechanics.new_girder('g1', elements=['q0', 'q1'])
girder.shift_y = 20e-6

# Choose ONE displacement-source model. Harmonic amplitude is peak, not RMS.
source = xt.mechanics.Harmonic(amplitude=50e-6, frequency_hz=500.)
# source = xt.mechanics.MeasuredMotion(
#     time_s=[0., .002, .004], displacement=[0., 50e-6, 0.])
# source = xt.mechanics.DisplacementPSD(
#     frequency_hz=[0., 1., 2., 40., 41.],
#     density=[0., 0., 1e-17, 1e-17, 0.], seed=956,
#     correlation_length_m=20.)
mech.add_source('floor', source, targets=['g1'], component='y')

# All times are physical seconds. Reuse the record for matched comparisons.
motion = mech.compute_time_series(time_s=np.arange(8192)*1e-4)
table = motion.get_table('g1')           # Source-induced local y, excludes static alignment
plt.plot(table.time_s[:1000], table.shift_y[:1000]*1e6)
plt.xlabel('Time [s]')
plt.ylabel('Source displacement [µm]')

line.build_tracker()
particles = line.build_particles()
line.mechanics.enable_vibrations(time_series=motion)
line.track(particles, num_turns=256)
print('Tracked physical time [s]:', particles.time_s)
line.mechanics.disable_vibrations()      # Static 20 µm and magnet-local errors remain
print('q0 shift after disabling vibrations [µm]:', env['q0'].shift_y*1e6)

# For a frozen Twiss state, apply the selected time explicitly with tracking off.
mech.apply(time_s=.0005, time_series=motion)
twiss = line.twiss4d()
print('Frozen closed orbit before q0 [µm]:', twiss.y[0]*1e6)
plt.show()
