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
MODEL_FILE      = 'test_data/hllhc15_thick/hllhc15_collider_thick.json'
SHARED_CIRCUIT  = 'kqx.l8'
MAGNET          = 'mqxa.1l8'
STRENGTH_CHANGE = 0.001                  # Fractional circuit change

################################################################################
# Inspect the two native beam representations
################################################################################

########################################
# Load both proton beams in one environment
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / MODEL_FILE)
tw0         = env.twiss(method = '4d', chrom = False)
for name in ['lhcb1', 'lhcb2']:
    line = env[name]
    print(name, 'p0*c [TeV]:', line.particle_ref.p0c[0] * 1e-12)
    print('Twiss reference frame:', tw0[name].reference_frame)
    tw0[name].rows[['ip1', 'ip5', 'ip8']].cols[
        'name s betx bety'].show()
# Beam2 is stored in its own tracking direction. Its saved reverse=True Twiss
# default returns optics in the counter-rotating convention used by this model;
# it does not reverse the order of elements used by Line.track.

########################################
# Anchor the surveys at IP1 with opposite local tangents
########################################
survey_b1   = env.lhcb1.survey(element0 = 'ip1')
survey_b2   = env.lhcb2.survey(element0 = 'ip1', theta0 = np.pi)
# The pi rotation puts both local reference trajectories into one global frame.
# An equal default theta0 for both beams would mirror the second ring.
ips = ['ip1', 'ip2', 'ip5', 'ip8']
for ip in ips:
    difference = np.array([
        survey_b2[coordinate, ip] - survey_b1[coordinate, ip]
        for coordinate in ['X', 'Y', 'Z']])
    print(ip, 'beam2 - beam1 survey [m]:', difference)

################################################################################
# Change one physical triplet circuit and observe both beam elements
################################################################################

########################################
# Record the signed gradients in the two tracking frames
########################################
reference_strength  = env[SHARED_CIRCUIT]
before              = [env[name][f'{MAGNET}/{name}'].k1
          for name in ['lhcb1', 'lhcb2']]
env[SHARED_CIRCUIT] = reference_strength * (1 + STRENGTH_CHANGE)
after               = [env[name][f'{MAGNET}/{name}'].k1
         for name in ['lhcb1', 'lhcb2']]
print('Shared circuit:', SHARED_CIRCUIT)
print('Before k1 for beam1/beam2 [m^-2]:', before)
print('After  k1 for beam1/beam2 [m^-2]:', after)
# These are two beam-specific element representations controlled by one circuit,
# not the same Python element. Reversing direction changes the signed k1 here.

########################################
# Observe the coupled constraint on the two collision optics
########################################
tw_changed = env.twiss(method = '4d', chrom = False)
for name in ['lhcb1', 'lhcb2']:
    print(name, 'IP8 beta changes [m]:',
          tw_changed[name]['betx', 'ip8'] - tw0[name]['betx', 'ip8'],
          tw_changed[name]['bety', 'ip8'] - tw0[name]['bety', 'ip8'])
env[SHARED_CIRCUIT] = reference_strength

################################################################################
# Plot the common geometry and the two optics responses
################################################################################
fig, axes = plt.subplots(1, 2, figsize = (12, 5))
axes[0].plot(survey_b1.Z * 1e-3, survey_b1.X * 1e-3, label = 'Beam1')
axes[0].plot(survey_b2.Z * 1e-3, survey_b2.X * 1e-3, '--', label = 'Beam2')
for ip in ips:
    axes[0].plot(survey_b1['Z', ip] * 1e-3,
                 survey_b1['X', ip] * 1e-3, 'ko', markersize = 3)
    axes[0].annotate(ip, (survey_b1['Z', ip] * 1e-3,
                         survey_b1['X', ip] * 1e-3))
axes[0].set(xlabel = 'Global Z [km]', ylabel = 'Global X [km]')
axes[0].set_aspect('equal', adjustable = 'box')
axes[0].legend()
for name in ['lhcb1', 'lhcb2']:
    tw      = tw_changed[name]
    tw_ref  = tw0[name]
    region  = np.abs(tw.s - tw['s', 'ip8']) < 300
    axes[1].plot(
        tw.s[region] - tw['s', 'ip8'],
        100 * (tw.betx[region] / tw_ref.betx[region] - 1),
        label = name,
    )
axes[1].set(xlabel = 's - s(IP8) in each Twiss table [m]',
            ylabel = 'Horizontal beta change [%]')
axes[1].legend()
for axis in axes:
    axis.grid(alpha = 0.3)
fig.tight_layout()
plt.show()
