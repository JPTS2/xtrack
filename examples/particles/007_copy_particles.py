# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################
P0C         = 1e9                         # Proton reference p0*c [eV]
X           = np.array([-1e-3, 0, 1e-3])  # Launch positions [m]
K1L_A       = 0.2                         # Integrated k1 [m^-1]
K1L_B       = 0.4                         # [m^-1]

################################################################################
# Keep an initial launch and make independent tracking copies
################################################################################

########################################
# Construct the initial coordinates
########################################
initial = xt.Particles(pdg_id_0 = 'proton', p0c = P0C, x = X)

########################################
# Copy the stored particle data
########################################
# Assignment alone would give another name for the same object. copy gives
# independent coordinates while preserving IDs, reference and tracking state.
case_a = initial.copy()
case_b = initial.copy()

################################################################################
# Apply two strengths to the same launch
################################################################################

########################################
# Track the copies independently
########################################
xt.Multipole(knl = [0, K1L_A]).track(case_a)
xt.Multipole(knl = [0, K1L_B]).track(case_b)

########################################
# Inspect the unchanged launch and the two responses
########################################
xt.Table(
    data    = {
        'particle_id': initial.particle_id,
        'x_m':         initial.x,
        'px_initial':  initial.px,
        'px_case_a':   case_a.px,
        'px_case_b':   case_b.px},
    index   = 'particle_id').show(maxwidth = 'full')
