"""Deterministic mean radiation reaction in a full phase-resolved cavity.

This CPU implementation supports M2/P16 RF and optional uniform solenoid
and normal/skew quadrupole overlap. It applies the Landau--Lifshitz force on
the reacted stages. With static overlap both planes are INSIDE the field:
there is no hard-edge spin or radiation prescription. Other reductions,
magnetic fringes, quantum photons, GPU and TPSA effects are not supported.
The mechanical-energy difference below includes the changed RF trajectory;
it is not an emitted-photon-energy measurement or an accuracy certificate.
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
MASS0_EV          = xt.ELECTRON_MASS_EV
CHARGE_E          = 1.             # Positron reference in this example

########################################
# Optional uniform static overlap
########################################
KS                = 0.            # 1/m, normalized solenoid field
KNL1              = 0.            # 1/m, integrated normal quadrupole
KSL1              = 0.            # 1/m, integrated skew quadrupole
# These are reference-rigidity strengths. Changing the reference momentum
# or charge while keeping them fixed does NOT retain the same SI magnet.

################################################################################
# Line and incoming particle
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
line.particle_ref = xt.Particles(mass0=MASS0_EV, q0=CHARGE_E,
    p0c=MASS0_EV * kappa)
line.reset_s_at_end_turn = False
line.use_time_based_zeta_update = True
line.build_tracker()
# Specify mechanical transverse momenta u_x=2, u_y=-1 in units of mc.
# At this P16 entrance the RF potential vanishes. Inside the uniform
# solenoid A gives native ax=-ks*y/2, ay=ks*x/2 for this reference species;
# canonical px/py include those contributions. Quadrupole A_z does not
# change the transverse canonical lift. Native ax/ay are derived readback.
x0, y0 = 0.1/k, -0.07/k
ax0, ay0 = -KS*y0/2, KS*x0/2
initial = xt.Particles(mass0=MASS0_EV, q0=CHARGE_E,
    p0c=MASS0_EV * kappa, x=x0, y=y0,
    px=2./kappa+ax0, py=-1./kappa+ay0, ax=ax0, ay=ay0)

################################################################################
# Conservative and reacted tracking from the same input
################################################################################
conservative = initial.copy()
line.track(conservative, ele_start='rf', ele_stop='exit')
line.configure_radiation(model='mean')
reacted = initial.copy()
line.track(reacted, ele_start='rf', ele_stop='exit')
assert conservative.state[0] == reacted.state[0] == 1

########################################
# Compare physical readbacks
########################################
print('Conservative final energy [eV]:', float(conservative.energy[0]))
print('Reacted final energy [eV]:     ', float(reacted.energy[0]))
print('Mean minus conservative [eV]: ',
      float(reacted.energy[0] - conservative.energy[0]))
print('Mean minus conservative x [m]:', float(reacted.x[0] - conservative.x[0]))
print('Mean minus conservative y [m]:', float(reacted.y[0] - conservative.y[0]))
line.configure_radiation(model=None)
