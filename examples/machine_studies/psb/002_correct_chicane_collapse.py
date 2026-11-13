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
# Model from 001 and tabulated collapse
########################################
MODEL_FILE      = 'outputs/psb_chicane.json'
PROGRAMME_FILE  = 'test_data/psb_chicane/chicane_collapse.csv'
OUTPUT_FILE     = 'outputs/psb_corrected_collapse.json'

########################################
# Feed-forward sampling and optics constraints
########################################
END_TIME    = 5.5e-3                 # Collapse duration [s]
N_SAMPLES   = 17                     # Matched knots, including endpoints
MATCH_S     = 79.874                 # Match vertical optics here [m]
CORRECTORS  = [
    'kbrqfcorr', 'kbrqdcorr', 'kbrqd3corr', 'kbrqd14corr',
]

################################################################################
# Load the chicane and bind its circuits to the laboratory clock
################################################################################

########################################
# Read the preceding model and the stored programme
########################################
script_dir  = Path(__file__).resolve().parent
repo_root   = script_dir.parents[2]
line        = xt.load(script_dir / MODEL_FILE)
programme   = np.genfromtxt(
    repo_root / PROGRAMME_FILE, delimiter = ',', names = True)
line.insert('collapse.match', xt.Marker(), at = MATCH_S)

########################################
# Apply the dipole and induced-sextupole programmes
########################################
for knob in ['bsw_k0l', 'bsw_k2l']:
    function_name = f'programme.{knob}'
    line.functions[function_name] = xt.FunctionPieceWiseLinear(
        x = programme['time'], y = programme[knob])
    line[knob] = line.functions[function_name](line.ref['t_turn_s'])
# The table specifies signed integrated k0 [rad] and k2 [m^-2]. It models
# the eddy-current field through a prescribed circuit, not an EM calculation.

################################################################################
# Match a feed-forward table through the collapse
################################################################################

########################################
# Retain the chicane-off optics at the end of the programme
########################################
line['t_turn_s']    = END_TIME
tw_ref              = line.twiss()
sample_times        = np.linspace(0, END_TIME, N_SAMPLES)
corrections         = {name: [] for name in CORRECTORS}
uncorrected         = []
corrected           = []

########################################
# Restore both tunes and the local vertical Twiss parameters
########################################
for time_s in sample_times:
    line['t_turn_s'] = time_s
    for name in CORRECTORS:
        line[name] = 0.0
    tw = line.twiss()
    uncorrected.append([tw.qx, tw.qy, tw['bety', 'collapse.match']])
    opt = line.match(
        solve   = False,
        vary    = xt.VaryList(CORRECTORS, step = 1e-4),
        targets = [
            xt.TargetSet(qx = tw_ref.qx, qy = tw_ref.qy, tol = 1e-6),
            xt.TargetSet(
                bety    = tw_ref['bety', 'collapse.match'],
                alfy    = tw_ref['alfy', 'collapse.match'],
                at      = 'collapse.match',
                tol     = 1e-5,
                weight  = 100,
            ),
        ],
    )
    opt.solve()
    for name in CORRECTORS:
        corrections[name].append(line[name])
    tw = line.twiss()
    corrected.append([tw.qx, tw.qy, tw['bety', 'collapse.match']])
    print(f't = {time_s * 1e3:.3f} ms: '
          f'Qx = {tw.qx:.7f}, Qy = {tw.qy:.7f}, '
          f'bety = {tw["bety", "collapse.match"]:.6f} m')

################################################################################
# Bind the quadrupole corrections and inspect interpolation between knots
################################################################################

########################################
# Connect all four correction functions to the same clock
########################################
for name in CORRECTORS:
    function_name = f'correction.{name}'
    line.functions[function_name] = xt.FunctionPieceWiseLinear(
        x = sample_times, y = corrections[name])
    line[name] = line.functions[function_name](line.ref['t_turn_s'])

########################################
# Evaluate independent midpoints, without rematching
########################################
midpoint_times  = (sample_times[:-1] + sample_times[1:]) / 2
interpolated    = []
for time_s in midpoint_times:
    line['t_turn_s']    = time_s
    tw                  = line.twiss()
    interpolated.append([tw.qx, tw.qy, tw['bety', 'collapse.match']])
interpolated    = np.array(interpolated)
uncorrected     = np.array(uncorrected)
corrected       = np.array(corrected)
print('Maximum midpoint tune residuals:',
      np.max(np.abs(interpolated[:, :2] - [tw_ref.qx, tw_ref.qy]), axis = 0))
print('Maximum midpoint relative vertical-beta residual:',
      np.max(np.abs(interpolated[:, 2] / tw_ref['bety', 'collapse.match'] - 1)))
# Matching the knots does not make the interpolated optics exact, nor restore
# every beta function around the ring. The midpoint check measures that limit.

################################################################################
# Plot the programmed field and its correction
################################################################################
fig, axes = plt.subplots(3, 1, figsize = (10, 9), sharex = True)
axes[0].plot(programme['time'] * 1e3, programme['bsw_k0l'])
axes[0].set_ylabel('Bumper1 integrated k0 [rad]')
for index, name in enumerate(['Qx', 'Qy']):
    reference = [tw_ref.qx, tw_ref.qy][index]
    axes[1].plot(sample_times * 1e3, uncorrected[:, index] - reference,
                 '--', label = name + ' uncorrected')
    axes[1].plot(midpoint_times * 1e3, interpolated[:, index] - reference,
                 '.-', label = name + ' interpolated correction')
axes[1].set_ylabel('Tune - reference')
axes[1].legend()
axes[2].plot(sample_times * 1e3,
             uncorrected[:, 2] / tw_ref['bety', 'collapse.match'] - 1,
             '--', label = 'Uncorrected')
axes[2].plot(midpoint_times * 1e3,
             interpolated[:, 2] / tw_ref['bety', 'collapse.match'] - 1,
             '.-', label = 'Interpolated correction')
axes[2].set_ylabel('Local vertical beta / reference - 1')
axes[2].set_xlabel('Time [ms]')
axes[2].legend()
for axis in axes:
    axis.grid(alpha = 0.3)
fig.tight_layout()

################################################################################
# Save the clock-driven model for injection tracking
################################################################################
line['t_turn_s']    = 0.0
save_path           = script_dir / OUTPUT_FILE
line.to_json(save_path)
print('Saved corrected collapse:', save_path)
plt.show()
