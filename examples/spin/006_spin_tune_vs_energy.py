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
# Planar electron ring with RF and radiation off
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'

########################################
# Bounded reference-momentum scan
########################################
P0C_VALUES      = np.linspace(9.8e9, 10.2e9, 9)   # p0*c [eV]

################################################################################
# Compare the native fractional spin tune with the planar a*gamma relation
################################################################################

########################################
# Load one geometry and update only its reference particle at each setting
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.configure_spin('auto')

# Normalized magnet strengths stay fixed: the implied physical fields follow
# reference rigidity. This is a family of settings, not an acceleration ramp.

########################################
# Request the public native spin-tune result at each energy
########################################
samples = []
for p0c in P0C_VALUES:
    line.set_particle_ref('electron', p0c = p0c)
    # The public polarization analysis also supplies the spin tune.
    twiss       = line.twiss4d(polarization_analysis = True)
    gamma       = float(line.particle_ref.gamma0[0])
    anomaly     = line.particle_ref.anomalous_magnetic_moment[0]
    a_gamma     = anomaly * gamma
    folded      = abs((a_gamma + 0.5) % 1.0 - 0.5)
    samples.append((p0c, gamma, a_gamma, folded, twiss.spin_tune_fractional))
samples = np.array(samples)
xt.Table(
    data    = {
        'p0c_GeV':          samples[:, 0] / 1e9,
        'gamma':            samples[:, 1],
        'a_gamma':          samples[:, 2],
        'folded_a_gamma':   samples[:, 3],
        'native_spin_tune': samples[:, 4]},
    index   = 'p0c_GeV').show(maxwidth = 'full')

# The native conjugate-eigenphase result is folded into [0,1/2]. It does not
# identify an integer part or the sign of an unwrapped spin tune. The planar
# a*gamma comparison applies to this closed, flat magnetic orbit.

################################################################################
# Display the spin tune across the folding cusps
################################################################################

########################################
# Compare native results with the folded analytic reference
########################################
# A dense analytic curve resolves the folding cusps between scan points.
p0c_curve   = np.linspace(P0C_VALUES[0], P0C_VALUES[-1], 401)
gamma_curve = np.sqrt(1 + (p0c_curve / line.particle_ref.mass0)**2)
tune_curve  = abs((anomaly * gamma_curve + 0.5) % 1.0 - 0.5)
fig, axis   = plt.subplots(figsize = (8, 4))
axis.plot(
    p0c_curve / 1e9,
    tune_curve,
    '--',
    label   = 'Folded a gamma',
)
axis.plot(samples[:, 0] / 1e9, samples[:, 4], 'o', label = 'Native spin tune')
axis.set_ylabel('Fractional spin tune')
axis.set_xlabel('Reference p0*c [GeV]')
axis.grid()
axis.legend()
fig.tight_layout()
plt.show()

########################################
# Quantify agreement at the sampled settings
########################################
print('Maximum native minus folded-reference difference:',
      np.max(np.abs(samples[:, 4] - samples[:, 3])))
