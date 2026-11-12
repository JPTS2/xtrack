# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Common launches and dimensionless error definition
########################################
P0C             = 1e9                    # Electron reference momentum [eV/c]
COORDINATES     = ['x', 'px', 'y', 'py', 'zeta']
SCALES          = np.array([1e-2, 1e-3, 1e-2, 1e-3, 1e-2])
# Position scales are metres; momentum scales are relative to reference p0.
N_SLICES        = np.array([1, 2, 4, 8, 16, 32, 64])
STAGES          = {'uniform': 1, 'teapot': 1, 'yoshida-4': 3,
                  'yoshida-6': 7, 'yoshida-8': 15}
REFERENCE_N     = 256                    # Yoshida-6 slices, then doubled

########################################
# Pure bend and combined straight magnet
########################################
LENGTH          = 1.0                    # [m]
ANGLE           = np.pi / 10             # Pure-bend deflection [rad]
K1              = 0.37                   # Straight normal quadrupole [m^-2]
K1S             = 0.08                   # Straight skew quadrupole [m^-2]
K2              = 2.0                    # Straight sextupole [m^-3]

################################################################################
# Establish an exact bend control and refine the nontrivial reference
################################################################################

########################################
# Use a pure-bend map with no multipole kick to integrate
########################################
launch      = xt.Particles(
    'electron',
    p0c     = P0C,
    x       = [3e-3, -4e-3],
    px      = [1e-3, -2e-3],
    y       = [2e-3, -3e-3],
    py      = [-1e-3, 2e-3],
    delta   = [0.02, -0.03],
)
bend        = xt.Magnet(
    length              = LENGTH,
    angle               = ANGLE,
    model               = 'bend-kick-bend',
    edge_entry_active   = False,
    edge_exit_active    = False,
)
exact_bend  = launch.copy()
bend.track(exact_bend)
on_axis = xt.Particles('electron', p0c = P0C)
bend.track(on_axis)
print('Analytic on-axis bend control (x,px,y,py,zeta):',
      [getattr(on_axis, name)[0] for name in COORDINATES])

########################################
# Refine one fixed exact-drift Hamiltonian, including skew and sextupole fields
########################################
combined    = xt.Magnet(
    length      = LENGTH,
    k1          = K1,
    k1s         = K1S,
    k2          = K2,
    model       = 'drift-kick-drift-exact',
    integrator  = 'yoshida-6',
)
references  = []
for count in [REFERENCE_N, 2 * REFERENCE_N]:
    combined.num_multipole_kicks    = 7 * count
    particles                       = launch.copy()
    combined.track(particles)
    references.append(np.array([getattr(particles, name).copy()
                                for name in COORDINATES]))
reference_change    = np.max(
    np.abs(references[1] - references[0]) / SCALES[:, None])
print('Combined-magnet reference refinement, scaled maximum:', reference_change)

################################################################################
# Refine integration without changing the chosen field or map model
################################################################################

########################################
# Compare native schemes at their actual kick counts
########################################
fig, axes   = plt.subplots(1, 2, figsize = (12, 5), constrained_layout = True)
errors      = {}
for case, magnet, model, reference in [
    ('Pure bend', bend, 'rot-kick-rot-low-order',
     np.array([getattr(exact_bend, name) for name in COORDINATES])),
    ('Combined straight magnet', combined,
     'drift-kick-drift-exact', references[1]),
]:
    axis    = axes[0] if case == 'Pure bend' else axes[1]
    for integrator, stages in STAGES.items():
        values  = []
        for count in N_SLICES:
            candidate                       = magnet.copy()
            candidate.model                 = model
            candidate.integrator            = integrator
            candidate.num_multipole_kicks   = int(stages * count)
            particles                       = launch.copy()
            candidate.track(particles)
            result  = np.array([getattr(particles, name)
                                for name in COORDINATES])
            values.append(np.max(np.abs(result - reference) / SCALES[:, None]))
        errors[case, integrator]    = np.array(values)
        axis.loglog(stages * N_SLICES, values, 'o-', label = integrator)
        print(case, integrator, 'scaled errors:', values)
    axis.set(title = case, xlabel = 'Multipole kicks',
             ylabel = 'Max scaled error')
    axis.grid(which = 'both', alpha = 0.3)
    axis.legend()

########################################
# Mark reference uncertainty and avoid fitting across a numerical floor
########################################
axes[1].axhline(reference_change, color = 'k', linestyle = ':',
               label = 'Reference refinement')
axes[1].legend()
# The low-order bend rotation becomes exact as its integration is refined.
# This does not test the paraxial approximation of an expanded-drift model.
# No single fitted order is reported: high-order curves reach roundoff early.
plt.show()
