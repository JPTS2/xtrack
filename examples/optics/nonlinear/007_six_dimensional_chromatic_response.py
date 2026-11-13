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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
RF_VOLTAGE      = 1e5                   # [V]
DELTA_SCALE     = 1e-5                  # Estimated scan half-width
N_SAMPLES       = 7
OBSERVATION     = 'bpm.s1.d'
CHROM_STEP      = 1e-5

################################################################################
# Load stable 6D optics and derive the slippage scale
################################################################################
repo_root           = Path(__file__).resolve().parents[3]
env                 = xt.load(repo_root / LATTICE_FILE)
line                = env['demo']
env['voltage_rf']   = RF_VOLTAGE
reference           = line.twiss6d()
length              = line.get_length()
zeta_step           = reference.slip_factor * length * DELTA_SCALE
# zeta_shift requests a one-turn zeta advance in the closed-orbit search.
# It is an auxiliary closure condition, not a change of physical RF frequency
# or an accelerating reference. Positive shift corresponds to negative delta
# for this above-transition model: Delta_zeta approximately -eta*C*delta.

################################################################################
# Scan the imposed slippage and measure the actual momentum offsets
################################################################################

########################################
# Keep field strengths and reference momentum fixed
########################################
rows = []
for shift in np.linspace(-zeta_step, zeta_step, N_SAMPLES):
    tw          = line.twiss6d(zeta_shift = shift, chrom = False)
    mean_delta  = np.trapezoid(tw.delta, tw.s) / length
    rows.append([shift, mean_delta, tw.qx, tw.qy,
                 tw.zeta[-1] - tw.zeta[0], tw['delta', OBSERVATION],
                 tw['x', OBSERVATION], tw['px', OBSERVATION]])
rows = np.array(rows)
xt.Table(
    data    = {
        'zeta_shift_m': rows[:, 0],
        'mean_delta':   rows[:, 1],
        'qx':           rows[:, 2],
        'qy':           rows[:, 3],
        'tracked_slip': rows[:, 4],
        'delta_at_bpm': rows[:, 5],
        'x_at_bpm_m':   rows[:, 6],
        'px_at_bpm':    rows[:, 7],
    },
    index   = 'mean_delta',
).show(maxwidth = 'full')
# The RF is active, so scan coordinates are the measured mean delta rather
# than an assumed fixed input delta or the entrance coordinate alone.

################################################################################
# Compare 6D response with fixed-momentum 4D optics
################################################################################

########################################
# Fit first and second derivatives over the stated small window
########################################
qx_coeff    = np.polynomial.polynomial.polyfit(rows[:, 1], rows[:, 2], 2)
qy_coeff    = np.polynomial.polynomial.polyfit(rows[:, 1], rows[:, 3], 2)
magnetic    = line.copy()
magnetic['voltage_rf'] = 0.0
fixed_delta = magnetic.twiss4d(chrom = True, delta_chrom = CHROM_STEP)
print('6D scan / fixed-delta4D first tune derivatives:')
print('x:', qx_coeff[1], fixed_delta.dqx)
print('y:', qy_coeff[1], fixed_delta.dqy)
print('6D scan / fixed-delta4D second tune derivatives:')
print('x:', 2 * qx_coeff[2], fixed_delta.ddqx)
print('y:', 2 * qy_coeff[2], fixed_delta.ddqy)
# Second derivatives include the factorial2. Their estimates depend on the
# scan window, numerical optics precision and the different closure contracts.
# They are not guaranteed equal or a momentum-acceptance qualification.

########################################
# Differentiate the orbit using momentum at the same observation plane
########################################
for coordinate, column, first, second in [
    ('x', 6, 'dx', 'ddx'), ('px', 7, 'dpx', 'ddpx'),
]:
    coefficients = np.polynomial.polynomial.polyfit(
        rows[:, 5], rows[:, column], 2)
    print(coordinate, '6D scan / 4D first and second derivatives at',
          OBSERVATION)
    print(coefficients[1], fixed_delta[first, OBSERVATION],
          2 * coefficients[2], fixed_delta[second, OBSERVATION])
# Orbit derivatives use local delta; global tune fits above use mean delta.

########################################
# Measure first-order slip from both response problems
########################################
slip_slope  = np.polynomial.polynomial.polyfit(rows[:, 1], rows[:, 0], 1)[1]
print('Slippage scan / reference / magnetic slip factors:',
      -slip_slope / length, reference.slip_factor, fixed_delta.slip_factor)
print('Maximum requested-minus-tracked slippage [m]:',
      np.max(np.abs(rows[:, 0] - rows[:, 4])))

################################################################################
# Check the scan-window sensitivity of the second tune derivatives
################################################################################
half_window = []
for shift in np.linspace(-zeta_step / 2, zeta_step / 2, N_SAMPLES):
    tw          = line.twiss6d(zeta_shift = shift, chrom = False)
    mean_delta  = np.trapezoid(tw.delta, tw.s) / length
    half_window.append([mean_delta, tw.qx, tw.qy])
half_window = np.array(half_window)
for column, label, original in [(1, 'x', qx_coeff), (2, 'y', qy_coeff)]:
    narrowed = np.polynomial.polynomial.polyfit(
        half_window[:, 0], half_window[:, column], 2)
    print(label, 'second tune derivative: full / half slippage window:',
          2 * original[2], 2 * narrowed[2])
