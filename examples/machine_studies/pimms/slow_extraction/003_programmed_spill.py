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
# Model and extraction programme
########################################
LATTICE_FILE    = 'test_data/example_lattices/pimms_extraction/lattice.py'
KSE1_FINAL      = 4.897              # Extraction sextupole k2 [m^-3]
KSE2_FINAL      = 0.2315             # Extraction sextupole k2 [m^-3]
RAMP_START      = 200                # Reference turns
RAMP_END        = 1000               # Reference turns
SEPTUM_X        = 0.035              # Absorbing horizontal boundary [m]

########################################
# Initial coasting beam and acquisition
########################################
N_PARTICLES     = 512
N_TURNS         = 1600
NEMITT_X        = 6.0e-6             # Normalized rms emittance [m]
NEMITT_Y        = 1.0e-6             # Normalized rms emittance [m]
SIGMA_DELTA     = 5e-4
BEAM_INTENSITY  = 1e10               # Represented protons
SPILL_BIN_TURNS = 100
SEED            = 2026

################################################################################
# Prepare the reference optics and a coasting beam
################################################################################

########################################
# Load the same 200 MeV proton extraction setting
########################################
repo_root       = Path(__file__).resolve().parents[4]
env             = xt.load(repo_root / LATTICE_FILE)
line            = env['pimms']
tw              = line.twiss4d()
circumference   = line.get_length()
rng             = np.random.default_rng(SEED)
particles       = line.build_particles(
    x_norm      = rng.normal(size = N_PARTICLES),
    px_norm     = rng.normal(size = N_PARTICLES),
    y_norm      = rng.normal(size = N_PARTICLES),
    py_norm     = rng.normal(size = N_PARTICLES),
    nemitt_x    = NEMITT_X,
    nemitt_y    = NEMITT_Y,
    delta       = rng.normal(scale = SIGMA_DELTA, size = N_PARTICLES),
    zeta        = rng.uniform(-circumference / 2, circumference / 2,
                              N_PARTICLES),
    weight      = BEAM_INTENSITY / N_PARTICLES,
    method      = '4d',
)
# No RF: the coasting longitudinal distribution is prescribed, while the
# transverse coordinates use native linear matching with dispersion included.
# The chosen horizontal emittance populates the shrinking resonance boundary.

########################################
# Install an ideal absorbing boundary at the observation septum
########################################
line.insert(
    'septum_aperture',
    xt.LimitRect(min_x = -0.1, max_x = SEPTUM_X,
                 min_y = -0.1, max_y = 0.1),
    at = 'extr_septum',
)
septum_index = line.element_names.index('septum_aperture')

################################################################################
# Ramp the extraction sextupoles and track the resulting spill
################################################################################

########################################
# Bind both sextupoles to one laboratory-time programme
########################################
line.functions['extraction_ramp']   = xt.FunctionPieceWiseLinear(
    x   = [0, RAMP_START * tw.t_rev0, RAMP_END * tw.t_rev0],
    y   = [0, 0, 1],
)
env['extraction_scale']             = 'extraction_ramp(t_turn_s)'
env['kse1']                         = f'{KSE1_FINAL} * extraction_scale'
env['kse2']                         = f'{KSE2_FINAL} * extraction_scale'
# The programme is sampled once per reference turn. This coasting-beam
# illustration does not resolve a kicker waveform within a revolution.
line.enable_time_dependent_vars = True
line.track(particles, num_turns = N_TURNS)

########################################
# Attribute boundary crossings separately from other losses
########################################
lost            = particles.state <= 0
at_septum       = lost & (particles.at_element == septum_index)
other_losses    = lost & ~at_septum
loss_turns      = particles.at_turn[at_septum]
loss_weights    = particles.weight[at_septum]
print('Tracked protons:', N_PARTICLES)
print('Surviving / septum-boundary / other:',
      np.count_nonzero(~lost), np.count_nonzero(at_septum),
      np.count_nonzero(other_losses))
print('Reference revolution period [us]:', tw.t_rev0 * 1e6)
# A hard boundary removes particles; it does not model septum wires, material,
# a deflecting field or the extracted transfer line. These crossings are not
# an extraction-efficiency prediction.

################################################################################
# Relate the spill to the programmed shrinking resonance region
################################################################################

########################################
# Bin the weighted boundary crossings in reference-turn intervals
########################################
edges           = np.arange(0, N_TURNS + SPILL_BIN_TURNS, SPILL_BIN_TURNS)
counts, edges   = np.histogram(loss_turns, bins = edges, weights = loss_weights)
bin_duration    = np.diff(edges) * tw.t_rev0
rate            = counts / bin_duration
times           = np.arange(N_TURNS) * tw.t_rev0
scale           = np.interp(np.arange(N_TURNS),
                            [0, RAMP_START, RAMP_END], [0, 0, 1])

########################################
# Compare programme, spill rate and phase-space intercepts
########################################
fig, axes   = plt.subplots(3, 1, figsize = (9, 9))
axes[0].plot(times * 1e3, scale)
axes[0].set(xlabel = 'Reference time [ms]', ylabel = 'Extraction scale')
axes[1].stairs(rate, edges * tw.t_rev0 * 1e3)
axes[1].set(xlabel = 'Reference time [ms]',
            ylabel = 'Boundary rate [protons/s]')
axes[2].plot(particles.x[~lost] * 1e3, particles.px[~lost] * 1e3,
             '.', markersize = 2, label = 'Surviving, final turn')
axes[2].plot(particles.x[at_septum] * 1e3, particles.px[at_septum] * 1e3,
             '.', markersize = 2, label = 'Boundary crossings, different turns')
axes[2].axvline(SEPTUM_X * 1e3, color = 'k', linestyle = '--')
axes[2].set(xlabel = 'x [mm]', ylabel = r'$p_x$ [$10^{-3}$]')
axes[2].legend(fontsize = 'small')
for axis in axes:
    axis.grid(alpha = 0.3)
fig.tight_layout()
plt.show()
