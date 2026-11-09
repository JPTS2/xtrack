# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import json
from pathlib import Path

import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################
P0C             = 1e9                     # Proton reference p0*c [eV]
PARTICLE_IDS    = np.array([10, 20, 30])
X               = np.array([-1e-3, 0, 1e-3])  # [m]
DELTA           = np.array([-1e-3, 0, 1e-3])
WEIGHTS         = np.array([1e6, 2e6, 1e6])  # Real particles per macroparticle
JSON_FILE       = 'outputs/particles.json'  # Relative to this script

################################################################################
# Construct a particle snapshot
################################################################################
particles = xt.Particles(
    pdg_id_0    = 'proton',
    p0c         = P0C,
    particle_id = PARTICLE_IDS,
    x           = X,
    delta       = DELTA,
    weight      = WEIGHTS)

################################################################################
# Save and restore the particle definition
################################################################################

########################################
# Round-trip through a dictionary
########################################
snapshot  = particles.to_dict()
from_dict = xt.Particles.from_dict(snapshot)

########################################
# Write JSON and reconstruct from its contents
########################################
# Default serialization includes stored RNG state. compact=True would omit
# that state by default; keep the full snapshot for stochastic continuation.
json_path = Path(__file__).resolve().parent / JSON_FILE
json_path.parent.mkdir(parents = True, exist_ok = True)
particles.to_json(json_path)
with json_path.open() as file:
    restored = xt.Particles.from_dict(json.load(file))
print('Saved particles to:', json_path)

################################################################################
# Inspect identifiers, coordinates, weights and reference momentum
################################################################################
for label, batch in [
    ('Original', particles), ('Dictionary', from_dict), ('JSON', restored),
]:
    print(label)
    batch.get_table().cols[
        'particle_id x delta weight p0c'].show(maxwidth = 'full')
# This snapshot contains particles, not the line or its settings. Save the
# associated model separately when preparing a restart of a tracking study.
