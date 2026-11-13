# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Native ring and selected local reference conditions
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
OBSERVATION     = 'ip.1'                # Waist marker at s=205 m
QUAD_NAMES      = ['arc.s1.c1.qf.in', 'arc.s1.c1.qd']
KNOB_NAMES      = ['local_kqf', 'local_kqd']

########################################
# Explicit perturbation and matching precision
########################################
STRENGTH_FACTOR = [1.0002, 0.9998]      # Applied to these two magnets only
K1_STEP         = 1e-5                  # Finite-difference strength step [m^-2]
BETA_TOL        = 1e-8                  # [m]
ALPHA_TOL       = 1e-9                  # Dimensionless

################################################################################
# Capture a reference before perturbing two individual quadrupoles
################################################################################

########################################
# Give the selected magnets individual live strength knobs
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
# The 1500 m ring supplies a 1 GeV/c electron reference, with RF voltage
# zero and radiation off.
# Replacing these two strength expressions leaves their initial values intact;
# the remaining quadrupoles continue to follow the native family variables.
for name, knob in zip(QUAD_NAMES, KNOB_NAMES):
    env[knob] = line[name].k1
    env.set(name, k1 = knob)
reference       = line.twiss4d()
reference_k1    = [env[knob] for knob in KNOB_NAMES]

########################################
# Apply a stated perturbation after saving the reference
########################################
for knob, factor in zip(KNOB_NAMES, STRENGTH_FACTOR):
    env[knob] *= factor
perturbed   = line.twiss4d()

################################################################################
# Combine Twiss, variable and element-field targets
################################################################################

########################################
# Select values from a Twiss table at a named location
########################################
# These targets reevaluate Twiss, but retain the saved reference values.
twiss_targets = [
    reference.target('betx', at = OBSERVATION, tol = BETA_TOL),
    reference.target('alfx', at = OBSERVATION, tol = ALPHA_TOL)]

########################################
# Bound the variable and the field controlled by the same varied knobs
########################################
# Each bound allows one quarter of the applied perturbation. The restored
# reference satisfies both; the perturbed strengths violate both.
upper_k1        = [value + 0.25 * abs(value * (factor - 1))
            for value, factor in zip(reference_k1, STRENGTH_FACTOR)]
variable_target = env.vars.target(
    KNOB_NAMES[0],
    value   = xt.LessThan(upper_k1[0]),
    tol     = 1e-10)
field_target    = line.target(
    lambda ll: ll[QUAD_NAMES[1]].k1,
    value   = xt.LessThan(upper_k1[1]),
    tol     = 1e-10)

########################################
# Inspect what each action evaluates and solve the combined problem
########################################
# The variable and field actions read the current model without calling Twiss.
# All targets depend on the same two independent magnet-strength variables.
opt = line.match(
    method  = '4d',
    solve   = False,
    vary    = xt.VaryList(KNOB_NAMES, step = K1_STEP),
    targets = twiss_targets + [variable_target, field_target])
for target in opt.targets:
    print('Target action:', type(target.action).__name__)
print('Before matching:')
opt.target_status()
opt.solve()
print('After matching:')
opt.target_status()
restored    = line.twiss4d()

################################################################################
# Inspect the reference, perturbed and rematched values
################################################################################

########################################
# Compare the selected beta and alpha at the same marker
########################################
print('Observation:', OBSERVATION, 'at s [m]:', reference['s', OBSERVATION])
print(f'{"Quantity":<10} {"Reference":>14} {"Perturbed":>14} {"Restored":>14}')
for quantity in ['betx', 'alfx']:
    print(f'{quantity:<10} {reference[quantity, OBSERVATION]:14.9f}'
          f' {perturbed[quantity, OBSERVATION]:14.9f}'
          f' {restored[quantity, OBSERVATION]:14.9f}')
print('betx has units m; alfx is dimensionless.')

########################################
# Inspect which individual strength knobs were changed
########################################
for name, knob in zip(QUAD_NAMES, KNOB_NAMES):
    print(name, knob, 'restored k1 [m^-2]:', env[knob])

print('Variable upper bound [m^-2]:', upper_k1[0])
print('Field upper bound [m^-2]:', upper_k1[1])
# Restoring these local observables does not target the entire Twiss table.
