# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Shared electron ring with weak finite-amplitude tune shifts
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
QX_TARGET       = 8.28                  # Full tune, not fractional part
QY_TARGET       = 10.19
K3L_F           = 1000.0                # Existing oct.f integral [m^-3]
K3L_D           = 1000.0                # Existing oct.d integral [m^-3]

########################################
# MAD-NG normal form and native tracking probes
########################################
MAP_ORDER       = 4
MADNG_SLICES    = 8
NEMITT          = 1e-6                  # Normalized launch-emittance scale [m]
N_TURNS         = 512                   # The smaller launches use twice this
A0_SIGMAS       = 0.1                   # Small amplitude in the other plane
A1_SIGMAS       = 1.0                   # First varied launch amplitude
A2_SIGMAS       = 2.0                   # Second varied launch amplitude

########################################
# Native momentum scans around delta = 0
########################################
DELTA_MAX       = 1e-4                  # (p-p0)/p0; also compare half-range
NUM_DELTA       = 13
FIT_ORDER       = 4

################################################################################
# Evaluate MAD-NG normal-form derivatives
################################################################################

########################################
# Set the working point before exporting the shared geometry
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.match(
    method  = '4d',
    vary    = xt.VaryList(['kqf', 'kqd'], step = 1e-5),
    targets = xt.TargetSet(qx = QX_TARGET, qy = QY_TARGET, tol = 1e-8),
)
env['k3lf'] = K3L_F
env['k3ld'] = K3L_D
# The reference is a 1 GeV/c electron; RF, radiation and sextupoles are off.
# Fractional tunes below 0.5 avoid real-signal frequency folding in the native
# tracking helper. This is a local nonlinear comparison, not an acceptance scan.

########################################
# Request the supported normal-form fields through the public interface
########################################
model       = line.build_madng_model()
normal_form = line.madng_twiss(
    normal_form         = True,
    mapdef_normal_form  = MAP_ORDER,
    nslice              = MADNG_SLICES,
)
print('Native full tunes / MAD-NG normal-form fractional tunes:',
      normal_form.qx, normal_form.qy,
      normal_form.q1_nf_ng, normal_form.q2_nf_ng)
# Normal-form derivatives describe the exported MAD-NG map near zero action.
# Export does not preserve every Xtrack integration/model setting; agreement
# between these two descriptions is measured rather than imposed.

################################################################################
# Compare all four amplitude-detuning derivatives with native tracking
################################################################################

########################################
# Measure finite-action slopes with the public NAFF-based helper
########################################
tracked_detuning= {}
for label, turns, amplitude_scale in [
    ('Nominal launches', N_TURNS, 1.0),
    ('Half launches, longer record', 2 * N_TURNS, 0.5),
]:
    tracked_detuning[label] = line.get_amplitude_detuning_coefficients(
        nemitt_x    = NEMITT,
        nemitt_y    = NEMITT,
        num_turns   = turns,
        a0_sigmas   = amplitude_scale * A0_SIGMAS,
        a1_sigmas   = amplitude_scale * A1_SIGMAS,
        a2_sigmas   = amplitude_scale * A2_SIGMAS,
    )
# Each call tracks four probes; optional nafflib supplies the tune estimator.
# J=(x_norm^2+px_norm^2)*nemitt/(2*beta0*gamma0), in metres, and similarly Jy.
# MAD-NG's internal anharmonicity convention differentiates with respect to
# 2J. The public dqxdjx_nf_ng etc. already convert to dQ/dJ: do not double them.

########################################
# Print the finite-action estimates beside the zero-action normal form
########################################
derivatives             = [
    ('dQx/dJx', 'det_xx', 'dqxdjx_nf_ng'),
    ('dQy/dJy', 'det_yy', 'dqydjy_nf_ng'),
    ('dQx/dJy', 'det_xy', 'dqxdjy_nf_ng'),
    ('dQy/dJx', 'det_yx', 'dqydjx_nf_ng'),
]
normal_form_detuning    = np.array([
    getattr(normal_form, field) for label, key, field in derivatives])
for label, values in tracked_detuning.items():
    native_values       = np.array(
        [values[key] for name, key, field in derivatives])
    print(label, ': derivatives per geometric action [m^-1]')
    xt.Table(
        data    = {
            'derivative': np.array([name for name, key, field in derivatives]),
            'MADNG_nf':   normal_form_detuning,
            'tracking':   native_values,
            'diff_pct':   100 * (native_values / normal_form_detuning - 1),
        },
        index   = 'derivative',
    ).show(maxwidth = 'full')
# The vertical self-detuning differs by about 3% in this configuration, even
# after reducing the launches and extending the record. The comparison does
# not identify all sources of that offset or establish identical nonlinear maps.

################################################################################
# Compare first and second momentum derivatives with native Twiss scans
################################################################################
chromatic_fits              = {}
normal_form_chromaticity    = np.array([
    normal_form.dq1_nf_ng, normal_form.dq2_nf_ng,
    normal_form.d2q1_nf_ng, normal_form.d2q2_nf_ng,
])
for half_range in [DELTA_MAX, DELTA_MAX / 2]:
    fitted                      = line.get_non_linear_chromaticity(
        delta0_range    = (-half_range, half_range),
        num_delta       = NUM_DELTA,
        fit_order       = FIT_ORDER,
        chrom           = False,
    )
    chromatic_fits[half_range]  = fitted
    native_derivatives          = np.array([
        fitted.dnqx[1], fitted.dnqy[1], fitted.dnqx[2], fitted.dnqy[2]])
    print('Momentum half-range:', half_range)
    xt.Table(
        data    = {
            'derivative': np.array(['dQx/ddelta', 'dQy/ddelta',
                                    'd2Qx/ddelta2', 'd2Qy/ddelta2']),
            'MADNG_nf':   normal_form_chromaticity,
            'native_fit': native_derivatives,
            'diff_pct':   100 * (native_derivatives
                                / normal_form_chromaticity - 1),
        },
        index   = 'derivative',
    ).show(maxwidth = 'full')
# dnqx[n] and the normal-form d2q1/d2q2 fields are derivatives, not Taylor
# coefficients: the quadratic tune term is (d2Q/ddelta2)*delta^2/2.
# Here delta is relative momentum at fixed reference and fields, not ptau.
# This is an ultrarelativistic 4D comparison at fixed reference and fields.

model.close()
