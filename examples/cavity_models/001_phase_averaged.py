"""Gaussian RF with a uniform solenoid and normal quadrupole.

Run this script or execute its sections in IPython. The chosen RF2/P2 model
uses nonzero-field observation planes: the Twiss functions below are native
canonical optics. Source and factor meshes are separate numerical controls;
this demonstration supplies no physical or numerical error certificate.
"""

################################################################################
# IMPORTS
################################################################################

import numpy as np
import xobjects as xo
import xtrack as xt


################################################################################
# USER PARAMETERS
################################################################################

model             = 'phase-averaged-magnetic'
length            = .05          # m
frequency         = 400e6        # Hz
voltage           = 2e3          # V, fixed-beta electric-crest calibration
phase             = .4           # rad
eta               = .25          # squared counter/co-wave amplitude ratio
counter_phase     = .3           # rad
calibration_beta  = .9           # fixed calibration ray, not evolving beta
rf_sigma          = .0125        # m
rf_centre         = .003         # m from the physical parent midpoint
ks                = .2           # 1/m, reference-rigidity solenoid strength
knl               = [0., .01]    # integrated normal multipoles
source_steps      = 16           # RF-source RK4 mesh
factor_steps      = 16           # body AND boundary midpoint meshes
mass0             = xt.ELECTRON_MASS_EV
q0                = 1.           # positron
gamma0            = 2.4


########################################
# Native coordinate inspection
########################################

coordinate_names = ('x', 'px', 'y', 'py', 'zeta', 'delta')

def coordinates(particles):
    return np.array([float(getattr(particles, name)[0])
                     for name in coordinate_names])


################################################################################
# LINE AND REFERENCE PARTICLE
################################################################################

context = xo.ContextCpu()
rf = xt.Cavity(model=model, integrator='midpoint', length=length,
    frequency=frequency, voltage=voltage, phase=phase, eta=eta,
    counter_phase=counter_phase, calibration_beta=calibration_beta,
    rf_profile='gaussian', rf_profile_sigma=rf_sigma,
    rf_profile_center=rf_centre, ks=ks, knl=knl,
    num_kicks=source_steps, num_factor_steps=factor_steps,
    edge_entry_active=0, edge_exit_active=0, absolute_time=True)
line = xt.Line(elements=[rf, xt.ReferenceEnergyIncrease(Delta_p0c=0.),
                         xt.Marker()],
    element_names=['rf', 'reference_update', 'end'])
line.particle_ref = xt.Particles(mass0=mass0, q0=q0,
    p0c=mass0*np.sqrt(gamma0**2-1.))
line.reset_s_at_end_turn = False
line.use_time_based_zeta_update = True
line.vars['rf_phase'] = phase
line.vars['rf_eta'] = eta
line.set('rf', phase='rf_phase', eta='rf_eta')
line.build_tracker(_context=context)


################################################################################
# TRACK AND BACKTRACK THE SAME MAP
################################################################################

initial = line.particle_ref.copy()
for name, value in dict(x=3e-4, px=2e-4, y=-2e-4, py=-1e-4,
                       zeta=1e-4, delta=.002).items():
    setattr(initial, name, value)
tracked = initial.copy()
line.track(tracked, ele_start='rf', ele_stop='reference_update')
assert tracked.state[0] == 1
endpoint = coordinates(tracked)
energy_gain = float(tracked.energy[0]-initial.energy[0])
line.track(tracked, ele_start='rf', ele_stop='reference_update', backtrack=True)
assert tracked.state[0] == 1
reverse_gaps = coordinates(tracked)-coordinates(initial)


################################################################################
# SINGLE-PASS TWISS IN THE NATIVE CANONICAL PLANES
################################################################################

line['reference_update'].Delta_p0c = initial.p0c[0]*endpoint[5]
twiss = line.twiss(method='6d', start=xt.START, end=xt.END,
    **dict(zip(coordinate_names, coordinates(initial))),
    betx=2., alfx=.1, bety=3., alfy=-.2)
line['reference_update'].Delta_p0c = 0.


print('RF energy gain [eV]:', energy_gain)
print('Signed-reverse gaps:', dict(zip(coordinate_names, reverse_gaps)))
print('End canonical beta [m]:', twiss.betx[-1], twiss.bety[-1])
