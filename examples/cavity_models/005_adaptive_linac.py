# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""What `adaptive` picks along a dummy electron linac, and why.

Electrons are accelerated from 3 MeV/c through standing-wave capture cells and
travelling-wave cells, the reference following the energy gain, and then pass
weak trim cells. `adaptive` compares each particle's own reference with the
cell, so the map changes along the linac: phase-resolved while the RF is strong
against the momentum, paraxial for a weaker standing wave, phase-averaged for a
travelling wave, and longitudinal where the gain is negligible.

Each cell's choice is found by tracking the same particle with each explicit
model: the model that gives the same result is the one picked.
"""

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################
P0C_START_EV = 3e6
FREQUENCY_HZ = 1.3e9
CELL_LENGTH_M = 1.
CELLS = (
    ('capture', 20e6, .25), ('capture', 20e6, .25), ('capture', 20e6, .25),
    ('travelling', 20e6, 0.), ('travelling', 20e6, 0.), ('travelling', 20e6, 0.),
    ('trim', 1e3, 0.), ('trim', 1e3, 0.),
)

MODELS = ('longitudinal-only', 'phase-averaged', 'phase-resolved-paraxial', 'phase-resolved')

################################################################################
# One line per model
################################################################################
lines = {}
for model in ('adaptive',) + MODELS:
    lines[model] = xt.Line(elements={'rf': xt.Cavity(model=model, length=CELL_LENGTH_M,
                                                     frequency=FREQUENCY_HZ, phase=np.pi/2)})
    lines[model].particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, p0c=P0C_START_EV)
    lines[model].build_tracker()


def track(model, p0c, voltage, eta):
    line = lines[model]
    line['rf'].voltage = voltage
    line['rf'].eta = eta
    particles = xt.Particles(mass0=xt.ELECTRON_MASS_EV, p0c=p0c, x=1e-3, px=1e-4, zeta=1e-4)
    line.track(particles)
    assert particles.state[0] == 1
    return particles


def picked_model(p0c, voltage, eta):
    adaptive = track('adaptive', p0c, voltage, eta)
    for model in MODELS:
        fixed = track(model, p0c, voltage, eta)
        if all(getattr(fixed, name)[0] == getattr(adaptive, name)[0]
               for name in ('x', 'px', 'zeta', 'delta')):
            return model, adaptive
    raise RuntimeError('adaptive matched no explicit model')


################################################################################
# Along the linac
################################################################################
p0c = P0C_START_EV
momenta, picks = [], []
wave_number = 2*np.pi*FREQUENCY_HZ/299792458.
print(f'{"cell":>4} {"section":>10} {"p0c [MeV/c]":>12} {"eta":>5} {"g/p0":>9}  model')
for ii, (section, voltage, eta) in enumerate(CELLS):
    model, particles = picked_model(p0c, voltage, eta)
    strength = voltage/(CELL_LENGTH_M*wave_number*p0c)
    print(f'{ii:4d} {section:>10} {p0c/1e6:12.2f} {eta:5.2f} {strength:9.2e}  {model}')
    momenta.append(p0c)
    picks.append(model)
    p0c = p0c*(1 + particles.delta[0])

assert set(picks) == set(MODELS)

################################################################################
# Plot
################################################################################
plt.figure(figsize=(7, 3))
plt.semilogy(range(len(CELLS)), momenta, 'k-', lw=1)
for model, color in zip(MODELS, ('C0', 'C1', 'C2', 'C3')):
    chosen = [ii for ii, pick in enumerate(picks) if pick == model]
    plt.semilogy(chosen, [momenta[ii] for ii in chosen], 'o', color=color, label=model)
plt.xlabel('Cell')
plt.ylabel('Reference momentum [eV/c]')
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
