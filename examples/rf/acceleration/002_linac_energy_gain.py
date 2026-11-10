# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Ideal proton accelerating gap
########################################
KINETIC_START   = 100e6                 # Reference kinetic energy [eV]
VOLTAGE         = 5e6                   # [V]; reference proton charge +e
FREQUENCY       = 100e6                 # [Hz]
PHASE           = np.pi / 2             # Crest for zeta=0, in radians

########################################
# Two particles and a downstream drift
########################################
DELTA_START     = [0.0, 1e-3]            # (p-p0)/p0
PX_START        = [1e-4, -1e-4]          # Canonical Px/p0
PY_START        = [2e-4, 0.0]            # Canonical Py/p0
DRIFT_LENGTH    = 2.0                   # [m]

################################################################################
# Define physical acceleration and reference rebasing as separate elements
################################################################################

########################################
# Let the native particle API calculate the new design reference momentum
########################################
new_reference   = xt.Particles(
    'proton',
    kinetic_energy0 = KINETIC_START + VOLTAGE * np.sin(PHASE),
)

########################################
# Build a thin gap, a reference change and a drift in their physical order
########################################
line    = xt.Line(elements = {
    'entrance': xt.Marker(),
    'gap': xt.Cavity(voltage = VOLTAGE, frequency = FREQUENCY, phase = PHASE),
    'after.gap': xt.Marker(),
    'rebase': xt.ReferenceEnergyChange(p0c = new_reference.p0c[0]),
    'after.rebase': xt.Marker(),
    'flight': xt.Drift(length = DRIFT_LENGTH),
    'exit': xt.Marker(),
})
line.set_particle_ref('proton', kinetic_energy0 = KINETIC_START)
particles       = line.build_particles(
    zeta    = 0.0,
    delta   = DELTA_START,
    px      = PX_START,
    py      = PY_START,
)
initial         = particles.copy()

# This thin longitudinal kick omits RF-gap transverse focusing and field maps.
# No radiation or collective force is included.

################################################################################
# Observe the RF kick and the subsequent coordinate rebasing independently
################################################################################

########################################
# Stop before and after the native reference-energy change
########################################
line.track(particles, ele_stop = 'after.gap')
after_gap       = particles.copy()
line.track(particles, ele_start = 'after.gap', ele_stop = 'after.rebase')
after_rebase    = particles.copy()

########################################
# Apply only the reference change to an unaccelerated control
########################################
reference_only  = initial.copy()
line['rebase'].track(reference_only)
print('Physical RF energy gain [eV]:', after_gap.energy - initial.energy)
print('Energy change on rebasing [eV]:', after_rebase.energy - after_gap.energy)
print('Reference-only control energy change [eV]:',
      reference_only.energy - initial.energy)

################################################################################
# Inspect physical energy and momenta alongside their reference coordinates
################################################################################
for label, sample in [('Initial', initial), ('After gap', after_gap),
                      ('After rebase', after_rebase),
                      ('Rebase without RF', reference_only)]:
    print(label)
    xt.Table(
        data    = {
            'particle_id':      sample.particle_id,
            'kinetic_MeV':      (sample.energy - sample.mass0) / 1e6,
            'p0c_MeV':          sample.p0c / 1e6,
            'delta':            sample.delta,
            'px':               sample.px,
            'Px_eV_c':          sample.px * sample.p0c,
            'py':               sample.py,
            'Py_eV_c':          sample.py * sample.p0c,
        },
        index   = 'particle_id',
    ).show(maxwidth = 'full')

# RF changes the physical energy. Rebasing preserves it and the physical
# transverse momenta, while changing delta, px and py through the new p0.

line.track(particles, ele_start = 'after.rebase', ele_stop = 'exit')
print('Drift exit x/y [m] and particle states:',
      particles.x, particles.y, particles.state)
