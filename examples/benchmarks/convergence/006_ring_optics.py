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
# One operating point and one fixed Hamiltonian approximation
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
N_SLICES        = [2, 4, 8, 16, 32]
REFERENCE_N     = 64                     # Then doubled to check the reference
DELTA_STEP      = 2e-5                   # Chromatic finite-difference step
INTEGRATOR      = 'yoshida-6'
KICKS_PER_SLICE = 7                     # Yoshida-6 kicks per composed slice

########################################
# Fixed observable scales for dimensionless error comparisons
########################################
TUNE_SCALE      = 1.0                    # Tune units
BETA_SCALE      = 1.0                    # Relative beta error
CHROMA_SCALE    = 1.0                    # dQ/d(delta) units

################################################################################
# Keep the shared ring strengths fixed while refining thick quadrupoles
################################################################################

########################################
# Load the native ring without rematching any configuration
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
# The shared reference is a 1 GeV/c electron; RF voltage and radiation are zero.
# Bends retain their exact pure-bend cores and fixed linear edges. Thin
# sextupoles remain thin; this scan refines only the thick quadrupole splitting.
line.configure_quadrupole_model(
    model       = 'drift-kick-drift-exact',
    integrator  = INTEGRATOR,
)

########################################
# Double the reference resolution at an unchanged finite-difference step
########################################
references  = []
for count in [REFERENCE_N, 2 * REFERENCE_N]:
    line.configure_quadrupole_model(
        num_multipole_kicks = KICKS_PER_SLICE * count,
    )
    references.append(line.twiss4d(delta_chrom = DELTA_STEP))
reference   = references[1]
ref_tune    = max(abs(reference.qx - references[0].qx),
               abs(reference.qy - references[0].qy))
ref_beta    = max(np.max(np.abs(reference.betx / references[0].betx - 1)),
               np.max(np.abs(reference.bety / references[0].bety - 1)))
ref_chroma  = max(abs(reference.dqx - references[0].dqx),
                 abs(reference.dqy - references[0].dqy))
print('Reference refinement: tune, relative beta, chromaticity:',
      ref_tune, ref_beta, ref_chroma)

################################################################################
# Separate integration sensitivity from chromatic finite-difference sensitivity
################################################################################

########################################
# Scan the same quadrupole model and compare identical observation locations
########################################
errors  = []
for count in N_SLICES:
    line.configure_quadrupole_model(
        num_multipole_kicks = KICKS_PER_SLICE * count,
    )
    tw              = line.twiss4d(delta_chrom = DELTA_STEP)
    tune_error      = max(abs(tw.qx - reference.qx), abs(tw.qy - reference.qy))
    beta_error      = max(np.max(np.abs(tw.betx / reference.betx - 1)),
                     np.max(np.abs(tw.bety / reference.bety - 1)))
    chroma_error    = max(abs(tw.dqx - reference.dqx),
                         abs(tw.dqy - reference.dqy))
    errors.append([tune_error / TUNE_SCALE, beta_error / BETA_SCALE,
                   chroma_error / CHROMA_SCALE])
    print(count, 'slices: tune, relative beta, chromaticity errors:',
          errors[-1])

########################################
# Check the derivative step at the refined integration setting
########################################
line.configure_quadrupole_model(
    num_multipole_kicks = 2 * REFERENCE_N * KICKS_PER_SLICE,
)
half_step           = line.twiss4d(delta_chrom = DELTA_STEP / 2)
chroma_step_change  = max(abs(reference.dqx - half_step.dqx),
                         abs(reference.dqy - half_step.dqy))
print('Chromaticity change when halving delta_chrom:', chroma_step_change)
# This is a sensitivity check, not an absolute chromaticity-error certificate.

################################################################################
# Plot convergence and the resolution of the reference comparison
################################################################################
fig, axes   = plt.subplots(1, 3, figsize = (13, 4), constrained_layout = True)
errors      = np.array(errors)
for index, (label, floor) in enumerate([
    ('Tune error [1]', ref_tune / TUNE_SCALE),
    ('Relative beta error [1]', ref_beta / BETA_SCALE),
    ('Fixed-step chromaticity difference [1]', ref_chroma / CHROMA_SCALE),
]):
    axes[index].loglog(N_SLICES, errors[:, index], 'o-')
    axes[index].axhline(floor, color = 'k', linestyle = ':',
                       label = 'Reference refinement')
    axes[index].set(xlabel = 'Yoshida-6 slices per quadrupole', ylabel = label)
    axes[index].set_xticks(
        N_SLICES, labels = [str(count) for count in N_SLICES])
    axes[index].tick_params(axis = 'x', which = 'minor', labelbottom = False)
    axes[index].grid(which = 'major', alpha = 0.3)
    axes[index].legend(fontsize = 8)
axes[2].axhline(chroma_step_change / CHROMA_SCALE,
                color = 'C1', linestyle = '--', label = 'Halve delta_chrom')
axes[2].legend(fontsize = 8)
# No strengths were fitted to absorb integration errors. At small errors,
# roundoff and finite-difference noise prevent a useful fitted order.
# The orange derivative-step change can exceed the integration error. Converged
# tracking integration alone does not certify the chromaticity derivative.
plt.show()
