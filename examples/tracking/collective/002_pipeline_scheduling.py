# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import xfields as xf
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Two identical proton bunches and one abstract interaction point
########################################
P0C                 = 7e12              # Proton reference p0*c [eV]
BUNCH_POPULATION    = 1e11              # Real protons per bunch
N_MACROPARTICLES    = 256               # Multiple of four for reflection pairs
SIGMA_X             = 1e-4              # Gaussian population scale [m]
SIGMA_Y             = 0.8e-4            # [m]
LOCAL_OFFSET_X      = 0.5e-4            # Both local x means; opposite lab x [m]
SEED                = 742

########################################
# Linear one-turn maps and scheduling duration
########################################
QX                  = 0.31
QY                  = 0.32
BETA                = 10.0             # Both map beta functions [m]
TURN_LENGTH         = 100.0            # Abstract reference-path length [m]
N_TURNS             = 8

################################################################################
# Register the two bunch identities with a single-process pipeline manager
################################################################################
# Each branch is one head-on collision followed by a linear ring map, not a
# geometric collider model. The 2D soft-Gaussian interaction fits the partner's
# transverse moments every turn; no slicing, crossing angle, RF or radiation.
manager = xt.PipelineManager()
for name in ['b1', 'b2']:
    manager.add_particles(name, 0)       # Single-process rank zero.
manager.add_element('ip')

################################################################################
# Prepare finite bunch samples and genuine moment-exchanging beam elements
################################################################################

########################################
# Draw Gaussian samples with exact reflection symmetry
########################################
rng                 = np.random.default_rng(SEED)
seed_coordinates    = rng.normal(size = (4, N_MACROPARTICLES // 4))
# Independent x/px and y/py reflections remove centroid and xy-covariance noise.
coordinates         = np.concatenate([
    seed_coordinates * np.array([sx, sx, sy, sy])[:, None]
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
], axis = 1)
bunches     = {}
lines       = {}
branches    = []

########################################
# Connect each collision element to the actual partner bunch
########################################
for name, partner in [('b1', 'b2'), ('b2', 'b1')]:
    particles = xt.Particles(
        p0c     = P0C,
        mass0   = xt.PROTON_MASS_EV,
        q0      = 1,
        weight  = BUNCH_POPULATION / N_MACROPARTICLES,
        x       = LOCAL_OFFSET_X + SIGMA_X * coordinates[0],
        px      = SIGMA_X / BETA * coordinates[1],
        y       = SIGMA_Y * coordinates[2],
        py      = SIGMA_Y / BETA * coordinates[3],
    )
    particles.init_pipeline(name)
    config = xf.ConfigForUpdateBeamBeamBiGaussian2D(
        pipeline_manager        = manager,
        element_name            = 'ip',
        partner_particles_name  = partner,
        update_every            = 1,
    )
    interaction = xf.BeamBeamBiGaussian2D(
        other_beam_q0       = 1,
        other_beam_beta0    = float(particles.beta0[0]),
        config_for_update   = config,
    )
    line = xt.Line(
        elements        = [
            interaction,
            xt.LineSegmentMap(
                length  = TURN_LENGTH,
                qx      = QX,
                qy      = QY,
                betx    = BETA,
                bety    = BETA,
            ),
        ],
        element_names   = ['ip', 'turn_map'],
    )
    line.set_particle_ref('proton', p0c = P0C)
    line.build_tracker(enable_pipeline_hold = True)
    bunches[name] = particles
    lines[name] = line
    branches.append(xt.PipelineBranch(line = line, particles = particles))

################################################################################
# Let the scheduler hold and resume branches waiting for partner moments
################################################################################
# Both bunches must send their moments at a collision before either can finish
# that collision. Sequentially tracking one whole line cannot provide this.
# The multitracker resumes real on-hold interactions, not fixed counter pauses.
multitracker = xt.PipelineMultiTracker(
    branches            = branches,
    enable_debug_log    = True,
)
multitracker.track(num_turns = N_TURNS, turn_by_turn_monitor = True)

################################################################################
# Inspect completed turns, actual holds and the received physical moments
################################################################################

########################################
# Check that both bunches progressed through the requested turns
########################################
for name in ['b1', 'b2']:
    particles = bunches[name]
    print(name, 'turns / states:', np.unique(particles.at_turn),
          np.unique(particles.state))
print('Actual scheduler holds:', len(multitracker.debug_log))
for event in multitracker.debug_log[:4]:
    print('branch / tracking-session turn / held element:',
          event['branch'], event['track_session_turn'],
          event['held_by_element'])

########################################
# Compare received moments with the partner at the last collision
########################################
# The received means use x_other=-x_partner and y_other=y_partner. Same
# positive charges repel. Only uncorrelated reflection-symmetric bunches are
# qualified here; general tilted Gaussian profiles are not qualified.

# Fields retain the partner's PRE-COLLISION moments at the last IP passage,
# not the coordinates after the final one-turn map.
for name, partner in [('b1', 'b2'), ('b2', 'b1')]:
    interaction = lines[name]['ip']
    record      = lines[partner].record_last_track
    weights     = bunches[partner].weight
    partner_x   = np.average(record.x[:, -1], weights = weights)
    partner_y   = np.average(record.y[:, -1], weights = weights)
    print(name, 'received partner population:',
          interaction.other_beam_num_particles)
    print('Received / independently recorded opposing x [m]:',
          interaction.other_beam_shift_x, -partner_x)
    print('Received / independently recorded opposing y [m]:',
          interaction.other_beam_shift_y, partner_y)

# This single-process workflow qualifies scheduling and moment updates. It does
# not demonstrate MPI scaling or the accuracy of a Gaussian beam approximation.
