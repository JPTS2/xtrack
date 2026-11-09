# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Further examples: examples/particles/011_capacity_and_insertion.py

################################################################################
# Required packages
################################################################################
import numpy as np
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Unit-weight electron population
########################################
P0C         = 1e9                       # Electron reference p0*c [eV]
X           = np.array([-0.3, -0.1, 0.1, 0.3]) * 1e-3  # [m]

########################################
# Deterministic illustrative process
########################################
X_CUT       = 0.0                       # Select live parents with x>X_CUT [m]
PX_KICK     = 1e-5                      # Daughter's added normalized px

################################################################################
# Implement the supported Python interaction-process contract
################################################################################

########################################
# Return product coordinates and modify the selected parent states
########################################
class ReplaceSelectedParticles:

    def __init__(self, x_cut, px_kick):
        self.x_cut      = x_cut
        self.px_kick    = px_kick

    def interact(self, particles):
        selected    = (particles.state > 0) & (particles.x > self.x_cut)
        if not np.any(selected):
            return None
        daughters   = particles.filter(selected)
        daughters.px += self.px_kick
        products    = daughters.to_dict()
        products['mass_ratio'] = daughters.mass_ratio
        products['parent_particle_id'] = daughters.particle_id.copy()
        particles.state[selected] = 0
        return products

# filter/to_dict copy coordinates and location fields through native APIs.
# mass_ratio is derived, so the product payload supplies it explicitly.
# parent_particle_id identifies the original parent, not the new product ID.

########################################
# Wrap the process with native BeamInteraction
########################################
interaction = xt.BeamInteraction(
    interaction_process = ReplaceSelectedParticles(X_CUT, PX_KICK),
)

# This toy replaces each selected electron by one same-species daughter at the
# same position with a prescribed kick; it is not a scattering or shower model.
# No interaction cross section or physical energy balance is modeled.

# Inputs have unit weights; this wrapper does not transfer product weights.

################################################################################
# Invoke the CPU interaction once and follow the resulting IDs
################################################################################

########################################
# Reserve product slots and show the original live IDs
########################################
# Capacity management is covered in particles/011_capacity_and_insertion.py.
particles   = xt.Particles(
    _context    = xo.ContextCpu(),
    _capacity   = 8,
    pdg_id_0    = 'electron',
    p0c         = P0C,
    x           = X,
)
print('Original live IDs:', particles.particle_id[particles.state > 0])

########################################
# Inspect native product IDs and the retained parent histories
########################################
# BeamInteraction calls interact, appends products and assigns their new IDs.
interaction.track(particles)
occupied    = particles.remove_unused_space()
occupied.get_table().cols[
    'particle_id parent_particle_id x px state'].show(maxwidth = 'full')

# With these defaults, daughters 4/5 retain the lost parents' IDs 2/3.
# Positions are in metres; px is canonical momentum normalized to reference p0.
# Use IDs rather than storage order after the native reorganization.
