# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
import xfields as xf
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Native electron ring and Gaussian source
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
BUNCH_INTENSITY = 1.0e11                # Electrons represented by the weights
NEMITT          = 4.0e-8                # Normalized transverse rms scale [m]
SIGMA_Z         = 0.02                  # Prescribed longitudinal rms size [m]
SC_LENGTH       = 1.0                   # Effective thin interaction length [m]

########################################
# Identical seeded macroparticle launches
########################################
N_MACRO         = 512                   # Even: mirrored Gaussian sample
SEED            = 1729
OFFSET_X        = 5.0e-6                # Centroid displacement [m]
OFFSET_Y        = -3.0e-6               # [m]
N_TURNS         = 4

################################################################################
# Prepare the common initial bunch and inspect the prescribed force scale
################################################################################

########################################
# Sample the external-field matched covariance, then displace the centroid
########################################
repo_root   = Path(__file__).resolve().parents[3]
base_env    = xt.load(repo_root / LATTICE_FILE)
base        = base_env['demo']
reference   = base.twiss4d(chrom = False)
rng         = np.random.default_rng(SEED)
half        = rng.normal(size = (N_MACRO // 2, 5))
normal      = np.vstack([half, -half])
initial = base.build_particles(
    method      = '4d',
    x_norm      = normal[:, 0],
    px_norm     = normal[:, 1],
    y_norm      = normal[:, 2],
    py_norm     = normal[:, 3],
    zeta        = SIGMA_Z * normal[:, 4],
    delta       = 0.0,
    nemitt_x    = NEMITT,
    nemitt_y    = NEMITT,
    weight      = BUNCH_INTENSITY / N_MACRO,
)
initial.x += OFFSET_X
initial.y += OFFSET_Y

# Mirrored draws center the sample before displacement. Its covariance follows
# external optics; this is not a space-charge-matched equilibrium.

########################################
# Estimate the central defocusing of the prescribed Gaussian field
########################################
particle_ref    = base.particle_ref
beta0           = float(particle_ref.beta0[0])
gamma0          = float(particle_ref.gamma0[0])
gemitt          = NEMITT / (beta0 * gamma0)
sigma_x         = np.sqrt(reference.betx[0] * gemitt)
sigma_y         = np.sqrt(reference.bety[0] * gemitt)
line_density    = BUNCH_INTENSITY / (np.sqrt(2 * np.pi) * SIGMA_Z)
factor          = (2 * particle_ref.get_classical_particle_radius0()
                   * line_density * SC_LENGTH / (beta0**2 * gamma0**3))
kappa_x         = factor / (sigma_x * (sigma_x + sigma_y))
kappa_y         = factor / (sigma_y * (sigma_x + sigma_y))
print('Reference electron p0*c [eV], gamma:', particle_ref.p0c[0], gamma0)
print('Prescribed sigma_x, sigma_y [m]:', sigma_x, sigma_y)
print('Peak line density [electrons/m]:', line_density)
print('Central kick gradients Delta px/x, Delta py/y [m^-1]:', kappa_x, kappa_y)

################################################################################
# Track frozen and genuinely updating analytic Gaussian fields
################################################################################

########################################
# Give each configuration its own source and native ring
########################################
lines   = {}
fields  = {}
for label, updating in [('Frozen', False), ('Updating', True)]:
    env     = xt.load(repo_root / LATTICE_FILE)
    line    = env['demo']
    profile = xf.LongitudinalProfileQGaussian(
        number_of_particles = BUNCH_INTENSITY,
        sigma_z             = SIGMA_Z,
        q_parameter         = 1.0,
    )
    field   = xf.SpaceChargeBiGaussian(
        longitudinal_profile        = profile,
        update_on_track             = updating,
        length                      = SC_LENGTH,
        mean_x                      = reference.x[0],
        mean_y                      = reference.y[0],
        sigma_x                     = sigma_x,
        sigma_y                     = sigma_y,
        z_kick_num_integ_per_sigma   = 0,
    )
    line.insert('spacecharge', field, at = 0.0)
    lines[label] = line
    fields[label] = field

# Both fields act. The frozen source stays centered on the clean reference.
# Updating fits weighted moments of the full bunch and is classified collective.
# This models one local 1 m interaction, not distributed ring space charge.

########################################
# Record the bunch moments before each interaction and its applied source
########################################
histories   = {}
outputs     = {}
for label, line in lines.items():
    particles   = initial.copy()
    history     = []
    for turn in range(N_TURNS):
        alive           = particles.state > 0
        weights         = particles.weight[alive]
        input_moments   = []
        for coordinate in ['x', 'y']:
            values  = getattr(particles, coordinate)[alive]
            mean    = np.average(values, weights = weights)
            sigma   = np.sqrt(np.average(
                (values - mean)**2,
                weights = weights,
            ))
            input_moments.extend([mean, sigma])
        line.track(particles, num_turns = 1)
        field = fields[label]
        source_moments = [
            field.mean_x, field.sigma_x, field.mean_y, field.sigma_y,
        ]
        history.append((np.array(input_moments), np.array(source_moments)))
    histories[label] = history
    outputs[label] = particles
    print(label, 'line.iscollective:', line.iscollective,
          '; surviving macroparticles:', np.count_nonzero(particles.state > 0))

################################################################################
# Inspect the collective contract and the resulting physical differences
################################################################################

########################################
# Compare input moments with the source used at each interaction
########################################
print('All moments below are in metres, at the interaction before each turn:')
print('Case/turn       input mean_x      source mean_x      input sigma_x'
      '     source sigma_x')
for label, history in histories.items():
    for turn, (beam, source) in enumerate(history):
        print(f'{label:8s} {turn:2d} {beam[0]:17.9e} {source[0]:17.9e}'
              f' {beam[1]:17.9e} {source[1]:17.9e}')

# The last applied source describes the final turn's entrance, not its exit.

########################################
# Compare final trajectories by particle ID
########################################
frozen          = outputs['Frozen']
updating        = outputs['Updating']
order_frozen    = np.argsort(frozen.particle_id)
order_updating  = np.argsort(updating.particle_id)
for coordinate, scale in [('x', sigma_x), ('y', sigma_y)]:
    difference      = (getattr(updating, coordinate)[order_updating]
                       - getattr(frozen, coordinate)[order_frozen])
    rms_difference  = np.sqrt(np.mean(difference**2))
    print(coordinate, 'final trajectory RMS difference [m]:', rms_difference,
          '; relative to initial prescribed sigma:', rms_difference / scale)

# This is Gaussian moment closure, not a PIC solve: the longitudinal profile
# stays prescribed, and there is no longitudinal SC kick, RF or radiation.

# Tracking an updating field on separate chunks would fit different sources;
# the full bunch is the collective input. No long-term stability is inferred.
