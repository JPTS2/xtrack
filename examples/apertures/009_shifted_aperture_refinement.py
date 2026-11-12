# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Thick magnetic transport between two transformed checks
########################################
P0C             = 1e9                  # Electron reference p0*c [eV]
QUAD_LENGTH     = 2.0                  # [m]
QUAD_K1         = 0.25                 # Nonzero normal gradient [m^-2]
START_AXES      = (0.020, 0.015)        # Ellipse half axes [m]
END_HALF_X      = 0.010                # Rectangle horizontal half-width [m]
END_HALF_Y      = 0.012                # Vertical half-width [m]
START_SHIFT     = (0.003, -0.002)       # Line-frame aperture centre x/y [m]
END_SHIFT       = (-0.002, 0.003)       # [m]
START_ROLL      = 0.15                 # Aperture-frame roll [rad]
END_ROLL        = -0.10                # [rad]

########################################
# Deterministic entrance probes
########################################
X_START         = np.array([0, 0, 0, 0, 0, 0, 0, 0.04])  # [m]
PX_START        = np.array([-.015, -.010, -.006, 0, .006, .010, .015, 0])
PY_START        = np.array([0, .002, -.002, 0, .002, -.002, 0, 0])
# px and py are canonical momenta normalized to reference p0, not slopes.
# The last probe starts outside the first aperture; it has no upstream bracket.

########################################
# Numerical characterization and longitudinal resolution
########################################
N_THETA         = 96
R_MAX           = 0.06                 # Beyond both transformed openings [m]
DR              = 1e-6                 # Radial boundary sampling [m]
DS              = 0.02                 # Inserted aperture spacing [m]

################################################################################
# Track the installed shifted and rolled apertures around a real quadrupole
################################################################################

########################################
# Keep aperture alignment separate from magnetic alignment
########################################
line = xt.Line(elements = {
    'aper.start': xt.LimitEllipse(
        a           = START_AXES[0],
        b           = START_AXES[1],
        shift_x     = START_SHIFT[0],
        shift_y     = START_SHIFT[1],
        rot_s_rad   = START_ROLL,
    ),
    'quad': xt.Quadrupole(length = QUAD_LENGTH, k1 = QUAD_K1),
    'aper.end': xt.LimitRect(
        min_x       = -END_HALF_X,
        max_x       = END_HALF_X,
        min_y       = -END_HALF_Y,
        max_y       = END_HALF_Y,
        shift_x     = END_SHIFT[0],
        shift_y     = END_SHIFT[1],
        rot_s_rad   = END_ROLL,
    ),
})
line.set_particle_ref('electron', p0c = P0C)
line.skip_end_turn_actions   = True
# This is open transport: skip ring boundary bookkeeping. RF and radiation
# are absent. The magnet is aligned; only the two aperture frames differ.

########################################
# Record the first installed check that rejects each probe
########################################
initial = line.build_particles(x = X_START, px = PX_START, py = PY_START)
coarse  = initial.copy()
line.track(coarse)

################################################################################
# Refine only losses with a known upstream and downstream aperture
################################################################################

########################################
# Characterize the transformed endpoints and interpolate convex polygons
########################################
refined     = coarse.copy()
refinement  = xt.LossLocationRefinement(
    line,
    n_theta             = N_THETA,
    r_max               = R_MAX,
    dr                  = DR,
    ds                  = DS,
    save_refine_lines   = True,
)
end_index   = line.get_table().rows.indices['aper.end'][0]
refinement.refine_loss_location(
    refined, i_apertures = [end_index], with_progress = False)
# Native refinement backtracks through the supported thick Quadrupole, then
# retracks through that magnetic map with extra aperture checks. Its curved
# transverse trajectories are not the straight rays used in008.
# The endpoint profiles are convex and contain the line origin. Interpolating
# their characterized polygons assumes a chamber wall between installed checks.

########################################
# Preserve injection losses without inventing a preceding chamber segment
########################################
coarse.sort(interleave_lost_particles = True)
refined.sort(interleave_lost_particles = True)
checks  = np.array([
    line.element_names[int(index)] if state == 0 else 'survived'
    for index, state in zip(coarse.at_element, coarse.state)
])
xt.Table(
    data    = {
        'particle_id':  refined.particle_id,
        'coarse_check': checks,
        'state':        refined.state,
        'coarse_s':     coarse.s,
        'refined_s':    refined.s,
    },
    index   = 'particle_id',
).show(maxwidth = 'full')
print('Longitudinal sampling DS [m]:', DS)
# The first-check loss remains at s=0: no upstream bracket was supplied.
# at_element continues to identify the original check, while s is refined.

################################################################################
# Inspect the transformed opening model and its loss estimates
################################################################################

########################################
# Obtain the sampled endpoint polygons through public line access
########################################
interpolated    = refinement.refine_lines[end_index]
interp_table    = interpolated.get_table()
polygon_names   = interp_table.rows[
    interp_table.element_type == 'LimitPolygon'].name
start_polygon   = interpolated.get(polygon_names[0])
end_polygon     = interpolated.get(polygon_names[-1])
fig, axes       = plt.subplots(1, 2, figsize = (11, 4.5))
for polygon, label in [(start_polygon, 'Sampled entrance opening'),
                        (end_polygon, 'Sampled exit opening')]:
    axes[0].plot(polygon.x_closed * 1e3, polygon.y_closed * 1e3, label = label)
losses          = (coarse.state == 0) & (coarse.at_element == end_index)
axes[0].plot(refined.x[losses] * 1e3, refined.y[losses] * 1e3,
              'o', label = 'Interpolated-wall losses')
axes[0].set_title('Openings and impacts projected onto line x/y')
axes[0].set_xlabel('Line x [mm]')
axes[0].set_ylabel('Line y [mm]')
axes[0].set_aspect('equal')

########################################
# Separate refined downstream losses from the unbracketed entrance loss
########################################
for particles, label, marker in [(coarse, 'Installed checks', 'x'),
                                 (refined, 'Refined positions', 'o')]:
    lost    = particles.state == 0
    axes[1].plot(particles.particle_id[lost], particles.s[lost],
                  marker, label = label)
axes[1].set_xlabel('Particle ID')
axes[1].set_ylabel('Loss s [m]')
for axis in axes:
    axis.grid()
    axis.legend(fontsize = 8)
fig.tight_layout()
plt.show()
