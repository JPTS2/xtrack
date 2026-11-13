# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Proton transport and offset physical opening
########################################
P0C             = 10.0e9                # Proton reference p0*c [eV]
BUMP_X          = -1e-3                 # Orbit through the aperture [m]
APERTURE_X      = (-1.5e-3, -0.5e-3)    # Allowed horizontal interval [m]
BETX_START      = 1.0                   # [m]
BETY_START      = 1.0                   # [m]

########################################
# Matching precision
########################################
X_TOL           = 1.0e-8                # Position target tolerance [m]
PX_TOL          = 1.0e-8                # Canonical px tolerance [1]
KICK_STEP       = 1e-6                  # Integrated dipole-strength step [rad]

################################################################################
# Build the tracking line
################################################################################
env = xt.Environment()
for index in range(1, 5):
    env[f'kick.{index}'] = 0.0

line = env.new_line(components = [
    env.new('c1', xt.Multipole, knl = ['kick.1'], at = 1.0),
    env.new('c2', xt.Multipole, knl = ['kick.2'], at = 2.0),
    env.new('aperture', xt.LimitRect,
            min_x = APERTURE_X[0], max_x = APERTURE_X[1], at = 2.5),
    env.new('c3', xt.Multipole, knl = ['kick.3'], at = 3.0),
    env.new('c4', xt.Multipole, knl = ['kick.4'], at = 4.0),
    env.new('end', xt.Marker, at = 5.0),
])
line.set_particle_ref('proton', p0c = P0C)

# The unsteered reference trajectory misses the displaced aperture.
# The native dipole kicks have zero length. Radiation is off.
particles_before = line.particle_ref.copy()
line.track(particles_before)
print('Before matching:')
particles_before.get_table().cols['state at_element'].show()

################################################################################
# Create an aperture-free twin
################################################################################
# select shares the magnets and variables but gives a separate sequence.
# Only the twin replaces the aperture; tracking keeps the actual aperture.
line_optics = line.select()
env.new('observation', xt.Marker)
line_optics.replace('aperture', 'observation')

################################################################################
# Match a local orbit bump
################################################################################
# Open 4D optics use zero entrance orbit and dispersion by default.
# knl[0] is a normal integrated dipole strength; its horizontal kick has the
# opposite sign for this reference species. Match x/px rather than guessing it.
opt = line_optics.match(
    method      = '4d',
    solve       = False,
    betx        = BETX_START,
    bety        = BETY_START,
    vary        = xt.VaryList(
        [f'kick.{i}' for i in range(1, 5)], step = KICK_STEP),
    targets     = [
        xt.TargetSet(x = BUMP_X, at = 'observation', tol = X_TOL),
        xt.TargetSet(px = 0.0, at = 'observation', tol = PX_TOL),
        xt.TargetSet(x = 0.0, at = xt.END, tol = X_TOL),
        xt.TargetSet(px = 0.0, at = xt.END, tol = PX_TOL),
    ],
)
opt.solve()
opt.target_status()
tw = line_optics.twiss(
    method      = '4d',
    betx        = BETX_START,
    bety        = BETY_START,
)
tw.cols['name s x px'].show()

################################################################################
# Track through the original aperture
################################################################################
# The shared kick variables apply the matched bump in both sequences.
particles_after = line.particle_ref.copy()
line.track(particles_after)
print('After matching:')
particles_after.get_table().cols['state at_element'].show()

# The input reference misses the opening; the matched one passes it.
print('The two sequences share c1:', line.get('c1') is line_optics.get('c1'))

################################################################################
# Inspect the aperture-free optics against the physical opening
################################################################################
fig, axis = plt.subplots(figsize = (8, 4))
axis.plot(tw.s, tw.x * 1e3, '.-', label = 'Matched twin orbit')
aperture_s = line.get_table()['s', 'aperture']
axis.plot([aperture_s, aperture_s], [bound * 1e3 for bound in APERTURE_X],
          color = 'grey', linewidth = 8,
          label = 'Physical opening')
axis.set(xlabel = 's [m]', ylabel = 'Horizontal orbit [mm]')
axis.grid()
axis.legend()
fig.tight_layout()
plt.show()
