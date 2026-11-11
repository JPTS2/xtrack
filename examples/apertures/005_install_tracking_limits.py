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
# Straight drift and tapered circular chamber
########################################
LENGTH          = 2.0                   # [m]
P0C             = 1e9                   # Electron reference momentum [eV/c]
RADIUS_START    = 0.020                 # Entrance chamber radius [m]
RADIUS_END      = 0.010                 # Exit chamber radius [m]

########################################
# Tracking stations and initial coordinates
########################################
CHECK_S         = [0.0, 1.0, 2.0]      # Installed aperture stations [m]
X_INITIAL       = np.array([0, 0.006, 0.012, 0.018, 0.024])  # [m]

################################################################################
# Define chamber geometry separately from particle tracking limits
################################################################################

########################################
# Build a straight line and its pipe model
########################################
env     = xt.Environment()
line    = env.new_line(components = [
    env.new('start', xt.Marker),
    env.new('drift', xt.Drift, length = LENGTH, model = 'exact'),
    env.new('end', xt.Marker),
])
line.set_particle_ref('electron', p0c = P0C)
entrance    = env.aperture.new_profile(
    'entrance', 'Circle', radius = RADIUS_START)
exit_shape  = env.aperture.new_profile(
    'exit', 'Circle', radius = RADIUS_END)
pipe        = env.aperture.new_pipe(
    'taper',
    positions   = [entrance.place(at = 0), exit_shape.place(at = LENGTH)],
)
line.aperture.place_pipe('chamber', pipe, at = 'start')
line.aperture.build_model()

########################################
# Show that a pipe model alone does not remove particles
########################################
initial         = line.build_particles(x = X_INITIAL)
without_limits  = initial.copy()
line.track(without_limits)
print('Alive with geometry only:', np.count_nonzero(without_limits.state > 0))

################################################################################
# Sample the chamber into native tracking elements
################################################################################

########################################
# Convert local cross-sections into LimitPolygon snapshots
########################################
limits  = line.aperture.get_limit_elements_at_s(CHECK_S)
# This returns ordinary LimitPolygon elements in local survey coordinates.
# The geometry model itself is not a continuously active tracking aperture.
line.discard_tracker()
for index, (s, limit) in enumerate(limits.items()):
    line.insert(f'aper.{index}', limit, at = s)
line.get_table().cols['name s element_type'].show(maxwidth = 'full')
# Snapshots must be regenerated after changing the chamber model. The chosen
# stations set the longitudinal sampling; no material interaction is included.

########################################
# Track the same particles through the installed checks
########################################
particles   = initial.copy()
line.track(particles)
order       = np.argsort(particles.particle_id)
state       = np.asarray(particles.state)[order]
loss_s      = np.where(state > 0, np.nan, np.asarray(particles.s)[order])
xt.Table(
    data    = {
        'particle_id':  np.asarray(particles.particle_id)[order],
        'x_initial':    X_INITIAL,
        'state':        state,
        'loss_s':       loss_s},
    index   = 'particle_id').show(maxwidth = 'full')
# Surviving particles have no loss location. A recorded loss is at an installed
# station, not necessarily where the trajectory first met the assumed wall.

################################################################################
# Distinguish the chamber wall from the sampled tracking checks
################################################################################
s_plot      = np.linspace(0, LENGTH, 100)
radius      = RADIUS_START + (RADIUS_END - RADIUS_START) * s_plot / LENGTH
fig, ax     = plt.subplots(figsize = (9, 4))
ax.plot(s_plot, radius * 1e3, 'k-', label = 'Assumed chamber wall')
ax.plot(s_plot, -radius * 1e3, 'k-')
for s in CHECK_S:
    ax.axvline(s, color = 'grey', linestyle = ':')
for x, s_loss in zip(X_INITIAL, loss_s):
    stop    = LENGTH if np.isnan(s_loss) else s_loss
    ax.plot([0, stop], [1e3 * x, 1e3 * x], color = 'C0', alpha = 0.6)
lost    = state <= 0
ax.plot(loss_s[lost], X_INITIAL[lost] * 1e3, 'rx', label = 'Recorded loss')
ax.set_xlabel('s [m]; dotted lines: installed checks')
ax.set_ylabel('x [mm]')
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()
