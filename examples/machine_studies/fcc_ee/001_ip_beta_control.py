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
# Stored CAS Z model; paths relative to the repository root
########################################
MODEL_DIR       = 'test_data/example_lattices/fcc_ee_z'
IP              = 'ipd'
BETA_FACTOR     = 1.1
KNOB_SAMPLES    = 3

########################################
# Quadrupole and sextupole circuits
########################################
QUAD_KNOBS  = [
    'kqd16l', 'kqf17l', 'kqd18l', 'kqf19l', 'kqd20l',
    'kqd8m', 'kqf9m', 'kqd10m',
    'kqf17r', 'kqd18r', 'kqf19r', 'kqd20r',
    'kqd8mr', 'kqf9mr', 'kqd10mr', 'kqf2', 'kqd1',
]
SEXT_KNOBS  = ['ksf1', 'ksd1']

################################################################################
# Load the radiation-free periodic reference
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
model_dir   = repo_root / MODEL_DIR
env         = xt.load(model_dir / 'lattice.json')
line        = env.fccee_p_ring
env.call(model_dir / 'strengths.py')
line.configure_radiation(model = None)
tw0         = line.twiss4d(delta_chrom = 1e-6)
# Stored reference: 45.6 GeV/c positrons. No beam-beam elements are installed.
print('Reference p0c [GeV]:', line.particle_ref.p0c[0] * 1e-9)

################################################################################
# Generate a periodic IP beta-control knob
################################################################################

########################################
# Match the quadrupole endpoint and preserve the tunes
########################################
# The circuits repeat around the ring. This is a periodic match, not a match
# with fixed insertion boundaries or independently controlled individual IPs.
opt = line.match_knob(
    knob_name           = 'ip_beta_scale',
    knob_value_start    = 1,
    knob_value_end      = BETA_FACTOR,
    run                 = False,
    method              = '4d',
    chrom               = False,
    vary                = xt.VaryList(QUAD_KNOBS, step = 1e-8),
    targets             = [
        xt.Target('betx', at = IP, value = BETA_FACTOR * tw0['betx', IP],
                  tol = 1e-8, weight = 10),
        xt.Target('bety', at = IP, value = BETA_FACTOR * tw0['bety', IP],
                  tol = 1e-10, weight = 1000),
        xt.TargetSet(alfx = 0, alfy = 0, at = IP, tol = 1e-7),
        xt.TargetSet(qx = tw0.qx, qy = tw0.qy, tol = 1e-7),
    ],
)
opt.solve()
opt.generate_knob()

########################################
# Add chromaticity compensation at the same endpoint
########################################
env['ip_beta_scale']    = BETA_FACTOR
opt_chroma              = line.match_knob(
    knob_name           = 'ip_beta_scale',
    knob_value_start    = 1,
    knob_value_end      = BETA_FACTOR,
    run                 = False,
    method              = '4d',
    delta_chrom         = 1e-6,
    vary                = xt.VaryList(SEXT_KNOBS, step = 1e-5),
    targets             = xt.TargetSet(
        dqx = tw0.dqx, dqy = tw0.dqy, tol = 1e-3),
)
opt_chroma.solve()
opt_chroma.generate_knob()

################################################################################
# Check the origin, interior and matched endpoint
################################################################################

########################################
# Linear circuit interpolation does not give exact intermediate optics
########################################
knob_values = np.linspace(1, BETA_FACTOR, KNOB_SAMPLES)
results     = []
for value in knob_values:
    env['ip_beta_scale']    = value
    tw                      = line.twiss4d(delta_chrom = 1e-6)
    results.append(tw)
    print('Knob:', value)
    tw.rows[['ipa', 'ipd', 'ipg', 'ipj']].cols[
        'name betx bety alfx alfy'].show()
    print('Tunes:', tw.qx, tw.qy, 'chromaticities:', tw.dqx, tw.dqy)
# The knob matches endpoints; it neither solves at every use nor establishes
# hardware limits, off-momentum stability or collision performance.

########################################
# Compare the actual beta response with the requested scale
########################################
fig, axes = plt.subplots(1, 3, figsize = (13, 4), layout = 'constrained')
for name, label in [('betx', 'Horizontal'), ('bety', 'Vertical')]:
    ratios = np.array([tw[name, IP] / tw0[name, IP] for tw in results])
    axes[0].plot(knob_values, ratios, 'o-', label = label)
    axes[1].plot(knob_values, 100 * (ratios / knob_values - 1),
                 'o-', label = label)
axes[0].plot(knob_values, knob_values, 'k:', label = 'Requested scale')
axes[0].set_ylabel(f'Beta / nominal at {IP.upper()}')
axes[1].set_ylabel('Beta interpolation error [%]')
axes[1].axhline(0, color = 'grey', linestyle = ':')
for name, label in [('qx', 'Horizontal'), ('qy', 'Vertical')]:
    axes[2].plot(knob_values, [tw[name] - tw0[name] for tw in results],
                 'o-', label = label)
axes[2].set_ylabel('Tune change')
axes[2].axhline(0, color = 'grey', linestyle = ':')
for axis in axes:
    axis.set_xlabel('IP beta-control knob')
    axis.grid()
    axis.legend()
plt.show()
