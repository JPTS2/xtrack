# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Shared geometry and reference
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
P0C             = 10e9                  # Reference p0*c [eV]

########################################
# Independent radiation-source changes
########################################
RF_VOLTAGE      = 50e6                  # [V], above the mean loss per turn
RF_HARMONIC     = 200
WIGGLER_LENGTH  = 1.0                   # Each of four straight dipoles [m]
WIGGLER_K0      = 0.05                  # Field curvature magnitude [m^-1]
WIGGLER_SLICES  = 16                    # Finite-length slices per dipole
GRADIENT_BEND   = 'arc.s1.c1.b1'
BEND_K1         = 1e-4                  # Added normal gradient [m^-2]

################################################################################
# Load the common geometry and reference
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
env['p0c']  = P0C
line.set_particle_ref('electron', p0c = P0C)
################################################################################
# Add a closed vertical field wiggler to one independent copy
################################################################################

########################################
# Preserve the surveyed reference geometry while bending the actual orbit
########################################
cases           = {
    'Planar':            line.copy(),
    'Vertical wiggler':  line.copy(),
    'Bend gradient':     line.copy(),
}
wiggler_line    = cases['Vertical wiggler']
wiggler_start   = wiggler_line.get_table()['s', 'straight.s4.l.d7'] + 1.0
for index, sign in enumerate([1, -1, -1, 1]):
    wiggler_line.insert(
        f'wiggler.{index}',
        xt.Bend(
            length      = WIGGLER_LENGTH,
            angle       = 0.0,
            k0          = sign * WIGGLER_K0,
            rot_s_rad   = np.pi / 2,
        ),
        at      = wiggler_start + index * WIGGLER_LENGTH,
        anchor  = 'start',
    )
# angle=0 defines a straight reference; k0 bends the actual particle path.
# At this roll positive k0 bends toward negative y; native signed kappa_y
# has the opposite sign to the vertical trajectory second derivative.
# The +--+ field pattern closes vertical displacement and direction. Its
# physical path-length change is retained by tracking, not drawn by survey.
wiggler_line.slice_thick_elements([
    xt.Strategy(slicing = None),
    xt.Strategy(
        slicing = xt.Uniform(WIGGLER_SLICES, mode = 'thick'),
        name    = 'wiggler.*',
    ),
])

########################################
# Check actual orbit closure before enabling radiation
########################################
wiggler_optics  = wiggler_line.twiss4d(radiation_integrals = True)
wiggler_end     = wiggler_start + 4 * WIGGLER_LENGTH
outside         = ((wiggler_optics.s < wiggler_start)
                   | (wiggler_optics.s >= wiggler_end))
print('Radiation-free wiggler max outside |y| [m], |py|:',
      np.max(np.abs(wiggler_optics.y[outside])),
      np.max(np.abs(wiggler_optics.py[outside])))
print('Peak vertical orbit excursion [m]:', np.max(np.abs(wiggler_optics.y)))
reference_survey    = line.survey()
wiggler_survey      = wiggler_line.survey()
print('Survey endpoint change X/Y/Z [m]:',
      wiggler_survey.X[-1] - reference_survey.X[-1],
      wiggler_survey.Y[-1] - reference_survey.Y[-1],
      wiggler_survey.Z[-1] - reference_survey.Z[-1])

################################################################################
# Compare radiation integrals with full mean-map linearization
################################################################################

########################################
# Give a separate horizontal bend a small normal gradient
########################################
cases['Bend gradient'].env.set(GRADIENT_BEND, k1 = BEND_K1)
# This independent case isolates a combined-function contribution to I4x.
# It does not change the wiggler or add an arbitrary skew-field excitation.

########################################
# Compensate and taper each case, then compare on the same local optics
########################################
comparisons = {}
for label, case in cases.items():
    case.vars['frequency_rf']   = 0.0
    case['rf.1'].harmonic       = RF_HARMONIC
    case.vars['voltage_rf']     = RF_VOLTAGE
    case.configure_radiation(model = 'mean')
    case.compensate_radiation_energy_loss()
    result              = case.twiss6d(
        chrom               = False,
        radiation_analysis  = True,
        radiation_integrals = True,
        radiation_method    = 'full',
    )
    comparisons[label]  = result
    beta_gamma          = (result.particle_on_co.beta0[0]
                  * result.particle_on_co.gamma0[0])
    integral_values     = np.array([
        result.rad_int_energy_loss,
        result.rad_int_eq_gemitt_x,
        result.rad_int_eq_gemitt_y,
        result.rad_int_eq_gemitt_x * beta_gamma,
        result.rad_int_eq_gemitt_y * beta_gamma,
        result.rad_int_sigma_delta,
        result.rad_int_damping_constant_x_s,
        result.rad_int_damping_constant_y_s,
        result.rad_int_damping_constant_zeta_s,
    ])
    full_sigma_delta    = (result.eq_beam_covariance_matrix.sigma_pzeta[0]
                         / result.particle_on_co.rvv[0])
    map_values          = np.array([
        result.energy_loss,
        result.eq_gemitt_x,
        result.eq_gemitt_y,
        result.eq_nemitt_x,
        result.eq_nemitt_y,
        full_sigma_delta,
        *result.damping_constants_s,
    ])
    # pzeta=ptau/beta0: the entrance sigma_delta is sigma_pzeta/rvv
    # to first order. Use the full covariance, including modal correlations.
    print(label)
    xt.Table(
        data    = {
            'quantity':  np.array(['U0 [eV]', 'eps_x [m]', 'eps_y [m]',
                                  'epsn_x [m]', 'epsn_y [m]',
                                  'sigma_delta (entrance)',
                                  'damping_x [s^-1]',
                                  'damping_y [s^-1]', 'damping_zeta [s^-1]']),
            'integrals': integral_values,
            'mean_map':  map_values,
        },
        index   = 'quantity',
    ).show(maxwidth = 'full')
    print('Full-map entrance sigma_delta / integral estimate - 1:',
          full_sigma_delta / result.rad_int_sigma_delta - 1)
# The integral momentum spread is a smooth longitudinal estimate. The full
# entrance covariance includes longitudinal alpha and cross-mode correlations;
# its local RMS need not equal that scalar estimate.
# Full analysis uses the tracked mean-radiation map and diffusion prediction,
# not a stochastic bunch equilibrated over many damping times. Integral and
# full-map predictions need not agree exactly at finite element resolution.
# Both are now evaluated on each compensated, tapered operating point, rather
# than mixing its optics with the nominal radiation-free baseline above.

########################################
# Inspect signed curvature and nonnegative excitation inside the wiggler
########################################
wiggler_result  = comparisons['Vertical wiggler']
wiggler_mask    = ((wiggler_result.s >= wiggler_start)
                & (wiggler_result.s < wiggler_end)
                & (wiggler_result.length > 0))
wiggler_rows    = wiggler_result.rows[wiggler_mask]
wiggler_rows.rows[::WIGGLER_SLICES].cols[
    'name s length rad_int_kappa_y rad_int_i3_integrand '
    'rad_int_curly_hy rad_int_i5y_integrand'].show(maxwidth = 'full')
# Native kappa_y retains its sign. I3 and I5 use total curvature magnitude,
# so opposite poles do not cancel emitted energy or quantum excitation.

########################################
# Separate the signed gradient contribution to I4x in the horizontal bend
########################################
gradient_result = comparisons['Bend gradient']
gradient_term   = (2 * gradient_result['dx', GRADIENT_BEND]
                 * gradient_result['k1l', GRADIENT_BEND]
                 * gradient_result['rad_int_kappa_x', GRADIENT_BEND])
total_i4_bend   = (gradient_result['rad_int_i4x_integrand', GRADIENT_BEND]
                 * gradient_result['length', GRADIENT_BEND])
print('Combined-function bend integrated I4x: total / gradient part [m^-1]:',
      total_i4_bend, gradient_term)
print('Planar / gradient-case total I4x [m^-1]:',
      comparisons['Planar'].rad_int_i4x, gradient_result.rad_int_i4x)
# In native conventions the normal-gradient term is 2*Dx*k1*kappa_x.
# Unlike I3 and I5, this signed term can increase or decrease a partition.

################################################################################
# Resolve the vertical orbit and radiation sources along the insertion
################################################################################
fig, axes = plt.subplots(3, 1, figsize = (8, 8), sharex = True)
axes[0].plot(wiggler_rows.s, wiggler_rows.y, label = 'Closed orbit y')
axes[0].plot(wiggler_rows.s, wiggler_rows.dy, label = 'Dispersion Dy')
axes[0].set_ylabel('y and Dy [m]')
axes[0].legend()
axes[1].step(wiggler_rows.s, wiggler_rows.rad_int_kappa_y, where = 'post')
axes[1].set_ylabel('Signed curvature [m^-1]')
axes[2].plot(wiggler_rows.s, wiggler_rows.rad_int_i5y_integrand)
axes[2].set_ylabel('I5y integrand [m^-2]')
axes[2].set_xlabel('s [m]')
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
