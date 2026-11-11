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
# A chamber around one arc quadrupole
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
ELEMENT         = 'arc.s1.c1.qd'
HALF_X          = 0.0015                 # Chamber horizontal half-width [m]
HALF_Y          = 0.002                 # Chamber vertical half-height [m]
PIPE_SHIFT_X    = 0.0003                # Chamber offset from reference [m]
N_SAMPLES       = 9                     # Longitudinal sampling points

########################################
# Native beam-envelope and directional sampling settings
########################################
NEMITT_X        = 2e-6                  # Normalized emittance [m rad]
NEMITT_Y        = 1e-6                  # Normalized emittance [m rad]
DELTA_RMS       = 1e-4                  # Relative momentum excursion
N_SIGMAS        = 6.0                   # Requested betatron envelope scale
N_RAYS          = 64                    # Directions for the clearance estimate

################################################################################
# Install the chamber geometry without changing the magnetic lattice
################################################################################

########################################
# Load the shared electron ring and place the pipe
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.twiss_default['method']    = '4d'
length      = line[ELEMENT].length
profile     = env.aperture.new_profile(
    'chamber', 'Rectangle',
    max_x   = HALF_X,
    max_y   = HALF_Y,
)
pipe        = env.aperture.new_pipe('quad.chamber', positions = [
    profile.place(at = 0),
    profile.place(at = length),
])
line.aperture.place_pipe(
    'installed.chamber', pipe,
    at      = ELEMENT,
    shift_x = PIPE_SHIFT_X,
)
line.aperture.build_model()
# RF voltage and radiation are zero. This is a local geometric-clearance
# calculation, not tracking acceptance or a collimation/scattering model.

########################################
# Sample the actual optics inside the quadrupole
########################################
start_s     = line.get_table()['s', ELEMENT]
sample_s    = start_s + np.linspace(0, length, N_SAMPLES)
line.aperture.halo_params.update({
    'nemitt_x':  NEMITT_X,
    'nemitt_y':  NEMITT_Y,
    'delta_rms': 0.0,
})
# The default halo ratios give an ellipse in normalized betatron coordinates.
# Closed-orbit and mechanical tolerances remain zero.

################################################################################
# Compare monoenergetic and dispersive beam envelopes
################################################################################

########################################
# Compute the limiting envelope scale from the native aperture model
########################################
monoenergetic, sampled      = line.aperture.get_aperture_sigmas_at_s(
    sample_s,
    method      = 'rays',
    num_rays    = N_RAYS,
)
line.aperture.halo_params['delta_rms']  = DELTA_RMS
clearance       = line.aperture.get_aperture_sigmas_for_twiss(
    sampled,
    method                  = 'rays',
    num_rays                = N_RAYS,
    output_cross_sections   = True,
    output_contact_points   = True,
)
# x and y below are limiting wall-contact coordinates, not the closed orbit.
clearance.cols['s n1 x y'].show(maxwidth = 'full')
# n1 is the largest betatron envelope scale that fits the selected halo model.
# The native envelope includes orbit and +/-D*DELTA_RMS excursions. It is not
# the Gaussian RMS size obtained by adding dispersion in quadrature.

########################################
# Inspect directional margins at the smallest sampled clearance
########################################
worst       = int(np.argmin(clearance.n1))
print('Smallest sampled clearance n1:', clearance.n1[worst])
print('Location in quadrupole [m]:', clearance.s[worst] - start_s)
directional, _  = line.aperture.get_hvd_aperture_sigmas_at_s(
    [clearance.s[worst]])
# These are selected direction families; n1 above scans all N_RAYS directions.
directional.cols['s n1_horizontal n1_vertical n1_diagonal'].show(
    maxwidth    = 'full')
# The minimum is sampled in s and ray direction; refine both resolutions when
# locating a limiting aperture. No dynamic or nonlinear acceptance is inferred.

################################################################################
# Show the limiting scale and the requested physical envelope
################################################################################
fig, axes   = plt.subplots(1, 2, figsize = (11, 4))
axes[0].plot(
    sample_s - start_s, monoenergetic.n1, 'o-',
    label   = 'Delta = 0',
)
axes[0].plot(
    sample_s - start_s, clearance.n1, 's-',
    label   = 'With momentum band',
)
axes[0].axhline(
    N_SIGMAS,
    color       = 'black',
    linestyle   = '--',
    label       = 'Requested',
)
axes[0].set_xlabel('Distance from quadrupole entrance [m]')
axes[0].set_ylabel('Available betatron envelope scale n1')
axes[0].legend()
line.aperture.plot_transverse(
    s_positions = [clearance.s[worst]],
    sigmas      = N_SIGMAS,
    middle      = 'beam',
    ax          = axes[1],
)
axes[1].set_xlabel('Local reference x [m]')
axes[1].set_ylabel('Local reference y [m]')
axes[1].set_title('Chamber and requested envelope at the limiting sample')
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
