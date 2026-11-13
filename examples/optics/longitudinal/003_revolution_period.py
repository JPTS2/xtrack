# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
from scipy.constants import c
import xtrack as xt

################################################################################
# User parameters
################################################################################
LATTICE_FILE        = 'test_data/example_lattices/demo_ring/lattice.py'
RF_VOLTAGE          = 1e5                # Thin-cavity voltage [V]
NUM_SEARCH_TURNS    = 4
DELTA_SCAN          = np.array([-1e-4, 0.0, 1e-4])

################################################################################
# Load independent RF-on and RF-off configurations
################################################################################
repo_root           = Path(__file__).resolve().parents[3]
env                 = xt.load(repo_root / LATTICE_FILE)
rf_ring             = env['demo']
env['voltage_rf']   = RF_VOLTAGE
magnetic_env        = xt.load(repo_root / LATTICE_FILE)
magnetic            = magnetic_env['demo']

################################################################################
# Compare reference and tracked synchronous revolution periods
################################################################################

########################################
# Compute the reference period from length and reference speed
########################################
ordinary    = rf_ring.twiss6d()
beta0       = rf_ring.particle_ref.beta0[0]
from_length = rf_ring.get_length() / (beta0 * c)
print('Length/(beta0*c) [s]:', from_length)
print('Twiss reference t_rev0 [s]:', ordinary.t_rev0)

########################################
# Request a revolution-period search
########################################
# The search varies six entrance coordinates and enforces repeated x, px, y,
# py and delta across sampled turns. zeta can slip; that slip determines t_rev.
# Use lowercase t_rev0 and t_rev.
searched = rf_ring.twiss6d(
    search_for_t_rev        = True,
    num_turns_search_t_rev   = NUM_SEARCH_TURNS)
print('Searched orbit period t_rev [s]:', searched.t_rev)
print('Period minus reference [s]:', searched.t_rev - searched.t_rev0)

################################################################################
# Measure off-momentum periods without RF confinement
################################################################################

########################################
# Track transverse closed orbits at fixed momentum
########################################
# The fresh RF-off load lets delta remain fixed. Over one turn,
# Delta_zeta = length - beta0*c*T, hence T = t_rev0 - Delta_zeta/(beta0*c).
nominal = magnetic.twiss4d()
periods = np.zeros(len(DELTA_SCAN))
for i, delta in enumerate(DELTA_SCAN):
    optics  = magnetic.twiss4d(delta0 = delta, chrom = False)
    initial = optics.particle_on_co.copy()
    tracked = initial.copy()
    magnetic.track(tracked, num_turns = 1)
    dzeta       = tracked.zeta[0] - initial.zeta[0]
    periods[i]  = nominal.t_rev0 - dzeta / (beta0 * c)

########################################
# Compare with the first-order slip-factor prediction
########################################
# eta = alpha_c - 1/gamma0^2; positive eta means increasing period with delta.
# Both path-length and speed changes enter this derivative at fixed p0.
relative = periods / nominal.t_rev0 - 1
xt.Table(
    data    = {
        'delta':            DELTA_SCAN,
        'period_ns':        periods * 1e9,
        'relative_change':  relative,
        'eta_delta':        nominal.slip_factor * DELTA_SCAN},
    index   = 'delta').show(maxwidth = 'full')

# These small offsets sample the bare lattice near its integer vertical tune;
# they do not establish a momentum-acceptance interval.
