"""Deterministic mean radiation reaction in a full phase-resolved cavity.

This first CPU implementation supports RF-only M2/P16 fields. It applies the
Landau--Lifshitz force on the reacted stages. Other cavity reductions, static
fields, quantum photons, GPU and TPSA effects are not yet supported.
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
CHARGE_E          = 1.             # Positron; hardware calibration stays fixed

################################################################################
# Line and incoming particle
################################################################################
k = 2 * np.pi * FREQUENCY_HZ / 299792458.
kappa = np.sqrt(GAMMA0**2 - 1.)
line = xt.Line(elements={
    'rf': xt.Cavity(model='phase-resolved', rf_profile='p16',
        length=LENGTH_M, frequency=FREQUENCY_HZ, voltage=VOLTAGE_V,
        phase=PHASE_RAD, eta=ETA, counter_phase=COUNTER_PHASE_RAD,
        calibration_beta=1., num_kicks=NUM_GAUSS_CELLS, absolute_time=True),
    'exit': xt.Marker()})
line.particle_ref = xt.Particles(mass0=MASS0_EV, q0=CHARGE_E,
    p0c=MASS0_EV * kappa)
line.reset_s_at_end_turn = False
line.use_time_based_zeta_update = True
line.build_tracker()
initial = xt.Particles(mass0=MASS0_EV, q0=CHARGE_E,
    p0c=MASS0_EV * kappa, x=0.1/k, y=-0.07/k,
    px=2./kappa, py=-1./kappa)

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
