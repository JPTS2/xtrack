"""Rest-spin tracking through a full phase-resolved cavity with optional overlap.

BMT carries the Cartesian rest-spin vector in fixed lab axes on the accepted
orbital stages. Spin is passive: there is no Stern--Gerlach force, spin-flip
sampling or polarization model. The first implementation is CPU scalar,
M2/P16 with optional uniform solenoid and normal/skew quadrupole fields.
The overlap case observes two planes INSIDE the static field, with canonical
input momenta: it does not model hard-edge spin fringes. Mean radiation is
off in this reversible example. Magnetic fringes, GPU and TPSA effects remain
unsupported.
"""

################################################################################
# Imports
################################################################################
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################
LENGTH_M          = 0.1
FREQUENCY_HZ      = 3E9
VOLTAGE_V         = 20E6
PHASE_RAD         = 0.7
ETA               = 0.25
COUNTER_PHASE_RAD = 0.3
GAMMA0            = 1000.
NUM_GAUSS_CELLS   = 128
ANOMALY           = 0.00115965218128
INITIAL_SPIN      = np.array([0.3, 0.4, np.sqrt(0.75)])

########################################
# Optional uniform static overlap
########################################
KS                = 0.            # 1/m, normalized solenoid field
KNL1              = 0.            # 1/m, integrated normal quadrupole
KSL1              = 0.            # 1/m, integrated skew quadrupole
# The reference remains a positron. For another reference momentum/charge,
# convert these strengths if the same SI magnetic hardware is intended.

################################################################################
# Line and incoming positron
################################################################################
k = 2 * np.pi * FREQUENCY_HZ / 299792458.
kappa = np.sqrt(GAMMA0**2 - 1.)
line = xt.Line(elements={
    'rf': xt.Cavity(model='phase-resolved', rf_profile='p16',
        length=LENGTH_M, frequency=FREQUENCY_HZ, voltage=VOLTAGE_V,
        phase=PHASE_RAD, eta=ETA, counter_phase=COUNTER_PHASE_RAD,
        calibration_beta=1., num_kicks=NUM_GAUSS_CELLS, absolute_time=True,
        ks=KS, knl=[0., KNL1], ksl=[0., KSL1],
        edge_entry_active=0, edge_exit_active=0),
    'exit': xt.Marker()})
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1.,
    p0c=xt.ELECTRON_MASS_EV * kappa)
line.reset_s_at_end_turn = False
line.use_time_based_zeta_update = True
line.build_tracker()
line.configure_spin(True)
# Mechanical u_x=2, u_y=-1 are in units of mc. P16 RF A vanishes here;
# canonical momenta include the uniform solenoid's transverse potential.
# Setting native ax/ay consistently makes before/after readbacks explicit.
x0, y0 = 0.1/k, -0.07/k
ax0, ay0 = -KS*y0/2, KS*x0/2
initial = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1.,
    p0c=xt.ELECTRON_MASS_EV * kappa, x=x0, y=y0,
    px=2./kappa+ax0, py=-1./kappa+ay0, ax=ax0, ay=ay0,
    spin_x=INITIAL_SPIN[0],
    spin_y=INITIAL_SPIN[1], spin_z=INITIAL_SPIN[2],
    anomalous_magnetic_moment=ANOMALY)

################################################################################
# Forward tracking and signed reverse through the same finite map
################################################################################
tracked = initial.copy()
line.track(tracked, ele_start='rf', ele_stop='exit')
assert tracked.state[0] == 1
final_spin = np.array([tracked.spin_x[0], tracked.spin_y[0], tracked.spin_z[0]])
print('Initial spin:', INITIAL_SPIN)
print('Final spin:  ', final_spin)
print('Spin change: ', final_spin - INITIAL_SPIN)
print('Spin norm change:', np.linalg.norm(final_spin) - np.linalg.norm(INITIAL_SPIN))

line.track(tracked, ele_start='rf', ele_stop='exit', backtrack=True)
assert tracked.state[0] == 1
returned_spin = np.array([tracked.spin_x[0], tracked.spin_y[0], tracked.spin_z[0]])
print('Forward/reverse spin gap:', returned_spin - INITIAL_SPIN)
print('Forward/reverse orbit gaps:', {
    name: float(getattr(tracked, name)[0] - getattr(initial, name)[0])
    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta')})
line.configure_spin(False)
