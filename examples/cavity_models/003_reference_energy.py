"""Reference momentum updates change the chart, not the physical particle.

Both runs use the same RF field and incoming particles. The update is fixed
from the nominal on-axis pilot, not fitted to each tracked particle. The
absolute-time zeta convention preserves the physical event when beta0 changes.
Twiss uses the same native tracking map; its canonical beta values depend on
the reference chart. Radiation and spin are off.
"""

################################################################################
# Imports
################################################################################
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################
LENGTH_M        = 0.05
FREQUENCY_HZ    = 400E6
VOLTAGE_V       = 20E3
PHASE_RAD       = 0.7
ETA             = 0.25
P0C_EV          = 2E6
NUM_GAUSS_CELLS = 16

################################################################################
# Line and reference particle
################################################################################
line = xt.Line(elements={
    'rf': xt.Cavity(model='phase-resolved', length=LENGTH_M,
        frequency=FREQUENCY_HZ, voltage=VOLTAGE_V, phase=PHASE_RAD,
        eta=ETA, num_kicks=NUM_GAUSS_CELLS, absolute_time=True),
    'reference_update': xt.ReferenceEnergyIncrease(Delta_p0c=0.),
    'exit': xt.Marker()})
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1., p0c=P0C_EV)
line.reset_s_at_end_turn = False
line.use_time_based_zeta_update = True
line.build_tracker()

########################################
# Fix the common nominal programme
########################################
pilot = line.particle_ref.copy()
line.track(pilot, ele_start='rf', ele_stop='reference_update')
assert pilot.state[0] == 1
nominal_momentum_increase = P0C_EV * float(pilot.delta[0])
incoming = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1., p0c=P0C_EV,
    x=[-3E-4, 0., 3E-4], y=[2E-4, 0., -2E-4],
    px=[-1E-4, 0., 1E-4], py=0., zeta=[-1E-4, 0., 1E-4], delta=0.002)

################################################################################
# Fixed reference and updating reference
################################################################################
tracked = {}
optics = {}
for label, momentum_increase in (
        ('fixed', 0.), ('updated', nominal_momentum_increase)):
    line['reference_update'].Delta_p0c = momentum_increase
    particles = incoming.copy()
    line.track(particles, ele_start='rf', ele_stop='exit')
    assert np.all(particles.state == 1)
    tracked[label] = particles
    optics[label] = line.twiss(method='6d', start=xt.START, end=xt.END,
        betx=2., alfx=0.3, bety=3., alfy=-0.2)
    print(label, 'p0c [eV/c]:', particles.p0c)
    print(label, 'delta:', particles.delta)
    print(label, 'end canonical beta [m]:',
          optics[label].betx[-1], optics[label].bety[-1])

################################################################################
# Compare the same physical coordinates, momenta, energy and event
################################################################################
def physical_readbacks(particles):
    ct = 299792458. * particles.time_s + (particles.s-particles.zeta)/particles.beta0
    return dict(x_m=particles.x, y_m=particles.y,
        px_eV_c=particles.p0c*particles.px,
        py_eV_c=particles.p0c*particles.py,
        energy_eV=particles.energy, ct_m=ct)

fixed = physical_readbacks(tracked['fixed'])
updated = physical_readbacks(tracked['updated'])
for name in fixed:
    print('Updated minus fixed', name, ':', np.asarray(updated[name]-fixed[name]))
