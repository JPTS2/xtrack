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
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Native magnetic ring with one physical octupole kick
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
K3LF            = 3.0e4                 # Existing oct.f integrated k3 [m^-3]
K3LD            = 0.0                   # Existing oct.d integrated k3 [m^-3]
NEMITT_X        = 1.0e-6                # Normalized launch emittance scale [m]
NEMITT_Y        = 1.0e-6                # [m]
DELTA_OFF       = 1.0e-4                # Fixed off-momentum (p-p0)/p0

########################################
# Polar launch grid and finite FFT record
########################################
R_RANGE         = (0.25, 2.0)           # Normalized transverse radius
THETA_RANGE     = (0.15, np.pi / 2 - 0.15)  # Polar grid angles [rad]
N_R             = 4
N_THETA         = 4
N_TURNS         = 512
N_FFT           = 2**14

########################################
# Local octupole-strength extrapolation
########################################
K3LF_REFERENCE  = 2.0e4                 # First tracked strength [m^-3]
K3LF_STEP       = 5.0e3                 # Tracked strength increment [m^-3]

################################################################################
# Add a nonlinear field to an independent native ring
################################################################################

########################################
# Power existing octupole circuits and inspect the linear reference
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
env['k3lf'] = K3LF
env['k3ld'] = K3LD

# knl[3] gives Delta px=-knl[3]*(x^3-3*x*y^2)/6 at each installed octupole.
# Geometry stays 1.5 km, with a 1 GeV/c electron reference and RF voltage zero.
# Radiation is off; the other nonlinear circuits stay unpowered.
reference = line.twiss4d(delta0 = 0.0)

########################################
# Prepare the CPU kernel needed for longitudinal freezing
########################################
# The footprint helper uses 4D launch normalization when freezing is requested.
# Only this context permits JIT compilation; a local C compiler may be needed.
line.discard_tracker()
context = xo.ContextCpu()
context.allow_kernel_compilation = True
line.build_tracker(_context = context)

################################################################################
# Measure a footprint from actual turn-by-turn tracking
################################################################################

########################################
# Track normalized polar launches at fixed momentum
########################################
# x_norm=r*cos(theta), y_norm=r*sin(theta), px_norm=py_norm=0. These are
# deterministic initial betatron phases, not a sampled beam distribution.
# Freeze zeta and delta numerically; this is not RF confinement.
footprint_settings = dict(
    nemitt_x            = NEMITT_X,
    nemitt_y            = NEMITT_Y,
    mode                = 'polar',
    r_range             = R_RANGE,
    theta_range         = THETA_RANGE,
    n_r                 = N_R,
    n_theta             = N_THETA,
    n_turns             = N_TURNS,
    n_fft               = N_FFT,
    zeta0               = 0.0,
    freeze_longitudinal = True,
    keep_fft            = False,
)
footprint = line.get_footprint(delta0 = 0.0, **footprint_settings)
# The public helper raises if a launch is lost: no lost-launch tune is plotted.

########################################
# Interpret geometric actions and the measured FFT frequencies
########################################
beta_gamma  = float(line.particle_ref.beta0[0] * line.particle_ref.gamma0[0])
jx          = (NEMITT_X / beta_gamma) * footprint.x_norm_2d**2 / 2
jy          = (NEMITT_Y / beta_gamma) * footprint.y_norm_2d**2 / 2

# Geometric J=(gamma*x^2+2*alpha*x*px+beta*px^2)/2 is in metres; no 2*pi factor.
# These are initial linear-optics actions, not invariants of the octupole map.
alias_x     = min(reference.qx % 1, 1 - reference.qx % 1)
alias_y     = min(reference.qy % 1, 1 - reference.qy % 1)
print('Full small-amplitude reference tunes:', reference.qx, reference.qy)
print('Their real-signal FFT aliases:', alias_x, alias_y)
print('Measured FFT-frequency ranges:',
      (np.min(footprint.qx), np.max(footprint.qx)),
      (np.min(footprint.qy), np.max(footprint.qy)))
print('FFT peak-grid spacing:', 1 / N_FFT,
      '; inverse record length:', 1 / N_TURNS)

# Real-signal FFT peaks lie in [0,0.5]; full tune branches are not recovered.
# Zero padding refines the peak grid, not the independent spectral resolution.

################################################################################
# Compare the launch grid with its tracked tune footprint
################################################################################

########################################
# Inspect the initial amplitude grid
########################################
fig, axes = plt.subplots(1, 2, figsize = (11, 4.5))
color = (jx + jy).ravel() * 1e9
axes[0].scatter(footprint.x_norm_2d, footprint.y_norm_2d, c = color)
axes[0].set_xlabel('Initial x_norm')
axes[0].set_ylabel('Initial y_norm')
axes[0].set_title('Normalized launch grid')
axes[0].set_aspect('equal')

########################################
# Inspect the measured tune footprint
########################################
for qx, qy in zip(footprint.qx, footprint.qy):
    axes[1].plot(qx, qy, '-', color = '0.7')
points = axes[1].scatter(footprint.qx, footprint.qy, c = color)
axes[1].plot(alias_x, alias_y, 'k*', label = 'Reference FFT alias')
axes[1].set_xlabel('Horizontal FFT frequency [oscillations/turn]')
axes[1].set_ylabel('Vertical FFT frequency [oscillations/turn]')
axes[1].legend()
fig.subplots_adjust(left = 0.07, right = 0.86, bottom = 0.16, wspace = 0.35)
color_axis = fig.add_axes([0.90, 0.20, 0.02, 0.65])
fig.colorbar(points, cax = color_axis, label = 'Initial geometric Jx + Jy [nm]')

################################################################################
# Separate momentum dependence from strength extrapolation
################################################################################

########################################
# Track the same amplitudes at a fixed momentum offset
########################################
off_momentum = line.get_footprint(
    delta0              = DELTA_OFF,
    keep_tracking_data  = True,
    **footprint_settings,
)
off_reference = line.twiss4d(delta0 = DELTA_OFF, chrom = False)
off_alias_x   = min(off_reference.qx % 1, 1 - off_reference.qx % 1)
off_alias_y   = min(off_reference.qy % 1, 1 - off_reference.qy % 1)
print('Recorded off-momentum delta range:',
      np.min(off_momentum.tracking_data.delta),
      np.max(off_momentum.tracking_data.delta))
print('Small-amplitude full-tune shifts:',
      off_reference.qx - reference.qx, off_reference.qy - reference.qy)
print('Linear chromatic predictions dQ/d(delta) * delta:',
      reference.dqx * DELTA_OFF, reference.dqy * DELTA_OFF)
print('FFT-alias shifts at the smallest launch radius:',
      np.mean(off_momentum.qx[:, 0] - footprint.qx[:, 0]),
      np.mean(off_momentum.qy[:, 0] - footprint.qy[:, 0]))

# Here Qx has fractional part > 0.5, so its FFT alias moves with OPPOSITE
# sign to the full tune; Qy does not. Finite-amplitude shifts also contain
# momentum-dependent detuning. They are not a chromaticity measurement.
# The finite-offset reference shift also includes higher-order chromaticity;
# its vertical value here differs visibly from the linear prediction.

########################################
# Compare native LinearRescale with actual tracking at the requested strength
########################################
# The helper tracks at reference and reference + step, then extrapolates
# frequencies to the current k3lf. It does not track at that requested value.
# Actual tracking at that strength checks this modest extrapolation.
rescaled = line.get_footprint(
    delta0                  = 0.0,
    linear_rescale_on_knobs  = xt.LinearRescale(
        knob_name   = 'k3lf',
        v0          = K3LF_REFERENCE,
        dv          = K3LF_STEP,
    ),
    **footprint_settings,
)
print('Requested k3lf after rescaling [m^-3]:', env['k3lf'])
print('Maximum rescaled minus actually tracked FFT frequency:',
      np.max(np.abs(rescaled.qx - footprint.qx)),
      np.max(np.abs(rescaled.qy - footprint.qy)))

# Agreement applies to this surviving grid and strength interval. Extrapolated
# frequencies do not establish survival or resonance behaviour elsewhere.

################################################################################
# Inspect the momentum shift and the rescaling error
################################################################################

########################################
# Compare actually tracked on- and off-momentum footprints
########################################
fig, axes = plt.subplots(1, 2, figsize = (11, 4.5))
axes[0].plot(footprint.qx.ravel(), footprint.qy.ravel(),
             'o', label = 'Tracked delta = 0')
axes[0].plot(off_momentum.qx.ravel(), off_momentum.qy.ravel(),
             'x', label = f'Tracked delta = {DELTA_OFF:g}')
axes[0].plot([alias_x, off_alias_x], [alias_y, off_alias_y],
             'k*', label = 'Small-amplitude aliases')
axes[0].set_xlabel('Horizontal FFT frequency [oscillations/turn]')
axes[0].set_ylabel('Vertical FFT frequency [oscillations/turn]')
axes[0].legend()

########################################
# Compare rescaling residuals with the FFT peak grid
########################################
axes[1].scatter(color, (rescaled.qx - footprint.qx).ravel(), label = 'x')
axes[1].scatter(color, (rescaled.qy - footprint.qy).ravel(), label = 'y')
axes[1].axhline(1 / N_FFT, color = '0.5', linestyle = '--',
                label = 'One FFT grid step')
axes[1].axhline(-1 / N_FFT, color = '0.5', linestyle = '--')
axes[1].set_xlabel('Initial geometric Jx + Jy [nm]')
axes[1].set_ylabel('Rescaled minus tracked frequency [1/turn]')
axes[1].legend()
fig.tight_layout()
plt.show()

# This finite surviving grid does not establish dynamic aperture, resonance
# stability, tune diffusion or a transmission probability.
