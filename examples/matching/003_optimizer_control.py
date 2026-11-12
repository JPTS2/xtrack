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
# Model and targets
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
QX_TARGET       = 8.760                 # Full tunes, including integer part
QY_TARGET       = 10.020
PHASE_START     = 'bpm.s1.f'
PHASE_END       = 'bpm.s2.f'
PHASE_INCREMENT = 1e-4                   # Relative phase increment [turns]

########################################
# Solver settings
########################################
TUNE_TOL        = 1.0e-8
K1_STEP         = 1.0e-5                # Finite-difference step [m^-2]
K1_MIN          = -0.10                 # Lower family-strength bound [m^-2]
K1_MAX          = 0.10                  # Upper family-strength bound [m^-2]
MANUAL_STEPS    = 1
QX_RETARGET     = 0.002                 # Temporary horizontal tune increment

################################################################################
# Load the ring and define the bounded matching problem
################################################################################

########################################
# Load the model
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
k1_before   = {name: env[name] for name in ['kqf', 'kqd']}

########################################
# Define variables and targets
########################################
# The ring supplies an electron reference at p0*c = 1 GeV, with RF voltage
# zero and radiation off. Periodic 4D optics uses delta=0 at fixed p0.
# Family variables drive all their quadrupoles through lattice expressions.

# The numerical bounds illustrate solver control, not hardware limits, and do
# not guarantee stable intermediate optics.
opt = line.match(
    method  = '4d',
    solve   = False,
    vary    = [
        xt.Vary('kqf', step = K1_STEP, limits = [K1_MIN, K1_MAX], tag = 'qf'),
        xt.Vary('kqd', step = K1_STEP, limits = [K1_MIN, K1_MAX], tag = 'qd'),
    ],
    targets = [
        xt.Target('qx', value = QX_TARGET, tol = TUNE_TOL, tag = 'horizontal'),
        xt.Target('qy', value = QY_TARGET, tol = TUNE_TOL, tag = 'vertical'),
    ],
)

################################################################################
# Control and inspect the optimization
################################################################################

########################################
# Inspect the initial state
########################################
# solve = False creates the optimizer without changing the family strengths.
print('Strengths before/after optimizer construction [m^-2]:')
for name, strength in k1_before.items():
    print(f'{name}: {strength:+.9f} -> {env[name]:+.9f}')
print('Before taking any optimizer steps:')
opt.target_status()
opt.vary_status()

########################################
# Take a fixed number of steps
########################################
# step(n) advances the optimizer; it does not require target convergence.
opt.step(MANUAL_STEPS)
print('After the manually requested steps:')
opt.target_status()
opt.vary_status()
print('Original strengths from iteration 0:',
      opt.get_knob_values(iteration = 0))
print('Current strengths:', opt.get_knob_values())
print('Iteration history:')
opt.log().show(maxwidth = 'full')

########################################
# Continue until the target tolerances are met
########################################
# solve() continues from the current state rather than resetting the strengths.
opt.solve()
opt.target_status()
opt.vary_status()

################################################################################
# Retarget with one active control, then restore a tagged solution
################################################################################

########################################
# Save the matched strengths and change the active problem
########################################
opt.tag(tag = 'two_tunes_matched')
kqd_matched = env['kqd']
opt.disable(target = 'vertical')
opt.disable(vary = 'qd')
opt.targets[0].value = QX_TARGET + QX_RETARGET
opt.solve()
temporary   = line.twiss4d()
print('One-control result Qx/Qy:', temporary.qx, temporary.qy)
print('Requested Qx:', QX_TARGET + QX_RETARGET)
print('Qy is temporarily unconstrained.')
print('Fixed kqd before/after [m^-2]:', kqd_matched, env['kqd'])
opt.target_status()
opt.vary_status()

########################################
# Restore the problem definition and reload stored strengths
########################################
# reload restores variable values and logged enabled states, not target values.
opt.enable(target = True)
opt.enable(vary = True)
opt.targets[0].value = QX_TARGET
opt.reload(iteration = 0)
initial_again   = line.twiss4d()
print('Reloaded initial Qx/Qy:', initial_again.qx, initial_again.qy)
opt.reload(tag = 'two_tunes_matched')
opt.target_status()
opt.log().show(maxwidth = 'full')

################################################################################
# Check the resulting tunes and family strengths
################################################################################
tw = line.twiss(method = '4d')
print(f'Final Qx = {tw.qx:.10f}; residual = {tw.qx - QX_TARGET:+.3e}')
print(f'Final Qy = {tw.qy:.10f}; residual = {tw.qy - QY_TARGET:+.3e}')
print(f'Allowed family strengths: [{K1_MIN}, {K1_MAX}] m^-2')
for name in ['kqf', 'kqd']:
    strength = env[name]
    print(f'{name} = {strength:.10f} m^-2; '
          f'within bounds: {K1_MIN <= strength <= K1_MAX}')

################################################################################
# Clone the optimizer and add a local phase constraint
################################################################################

########################################
# Free one tune to bound a relative phase
########################################
# The clone operates on the same ring. It has its own target definitions,
# but changing its strengths also changes the original optimizer's model.
phase_ref   = tw['mux', PHASE_END] - tw['mux', PHASE_START]
phase_goal  = phase_ref + PHASE_INCREMENT
phase_opt   = opt.clone(add_targets = xt.TargetRelPhaseAdvance(
    'mux',
    value   = xt.GreaterThan(phase_goal),
    start   = PHASE_START,
    end     = PHASE_END,
    tol     = TUNE_TOL,
))
# Use both families to satisfy the phase bound while retaining Qy; release Qx.
phase_opt.disable(target = 'horizontal')
phase_opt.solve()
phase_twiss = line.twiss4d()
print('Lower bound / achieved phase advance [turns]:', phase_goal,
      phase_twiss['mux', PHASE_END] - phase_twiss['mux', PHASE_START])
print('Retained Qy:', phase_twiss.qy)
print('Unconstrained Qx:', phase_twiss.qx)
phase_opt.target_status()

########################################
# Restore the original two-tune solution
########################################
opt.reload(tag = 'two_tunes_matched')
restored = line.twiss4d()
print('Restored Qx/Qy:', restored.qx, restored.qy)
