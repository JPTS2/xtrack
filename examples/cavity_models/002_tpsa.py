# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""TPSA tracking and physical-parameter derivatives of a native cavity.

Run this example cell by cell, editing the user parameters below.
Parameter derivatives include the fixed-beta voltage calibration and RF clock.
Eta is the squared counter/forward amplitude ratio; its zero endpoint has no
ordinary eta Taylor chart. The reference momentum stays fixed throughout.
All results use native canonical coordinates. Radiation is off; this example
is not a physical-source, numerical-error or backend qualification.
"""

################################################################################
# Required packages
################################################################################
from pathlib import Path
import json

import numpy as np
import madng_tpsa
import xobjects as xo
import xtrack as xt
import xtrack.tpsa as xtpsa

################################################################################
# User parameters
################################################################################

########################################
# Physical model and RF settings
########################################
MODEL               = "phase-resolved"
LENGTH_M            = 0.05
VOLTAGE_V           = 20E3
FREQUENCY_HZ        = 400E6
ETA                 = 0.25
PHASE_RAD           = 0.4
COUNTER_PHASE_RAD   = 0.3
CALIBRATION_BETA    = 0.9

########################################
# Reference, TPSA order and resolution
########################################
GAMMA0              = 2.4
TPSA_ORDER         = 2
NUM_GAUSS_CELLS     = 8
OUTPUT_JSON         = None  # For example, Path("tpsa_results.json")
CONTEXT             = xo.ContextCpu(omp_num_threads = 0)

assert TPSA_ORDER >= 2, "The displayed mixed coefficient requires order two"
assert np.isfinite(ETA) and ETA > 0., "This eta-seeded example requires finite eta > 0"

################################################################################
# Helpers
################################################################################
COORDINATES = ("x", "px", "y", "py", "zeta", "delta")


def particle_coordinates(particles):
    return np.array([float(getattr(particles, name)[0])
                     for name in COORDINATES])

################################################################################
# Line and reference particle
################################################################################
mass0 = xt.ELECTRON_MASS_EV
momentum_over_mc = np.sqrt(GAMMA0**2 - 1)
launch = dict(
    mass0 = mass0, q0 = 1., p0c = mass0 * momentum_over_mc,
    x = 4E-4, px = 0.005 / momentum_over_mc,
    y = -3E-4, py = -0.004 / momentum_over_mc,
    zeta = 3E-4, delta = 0.002)

cavity = xt.Cavity(
    _context = CONTEXT, model = MODEL, length = LENGTH_M,
    voltage = VOLTAGE_V, frequency = FREQUENCY_HZ,
    counter_phase = COUNTER_PHASE_RAD, calibration_beta = CALIBRATION_BETA,
    num_kicks = NUM_GAUSS_CELLS)
line = xt.Line(elements = {"rf": cavity})
line.particle_ref = xt.Particles(
    _context = CONTEXT, mass0 = mass0, q0 = 1.,
    p0c = mass0 * momentum_over_mc)
line.reset_s_at_end_turn = False

########################################
# Ordinary variables own the RF controls
########################################
line.vars["rf_eta"] = ETA
line.vars["rf_phase"] = PHASE_RAD
line.element_refs["rf"].eta = line.vars["rf_eta"]
line.element_refs["rf"].phase = line.vars["rf_phase"]
line.build_tracker(_context = CONTEXT)

################################################################################
# Ordinary tracking
################################################################################
scalar = xt.Particles(_context = CONTEXT, **launch)
line.track(scalar)
assert scalar.state[0] == 1

################################################################################
# TPSA tracking and live knob refresh
################################################################################
descriptor = madng_tpsa.Descriptor(
    6, TPSA_ORDER, num_params = 2, param_order = 1)
knobs = xtpsa.KnobParameters(line, ["rf_eta", "rf_phase"], descriptor)
knobs.apply()
try:
    series = xtpsa.ParticlesTpsa(
        order = TPSA_ORDER, descriptor = descriptor, **launch)
    line.track(series)
    initial_map = dict(
        constant_part = series.const_part.tolist(),
        state_jacobian = series.jacobian().tolist(),
        parameter_names = ["eta", "phase_rad"],
        parameter_jacobian = series.param_jacobian().tolist(),
        d2x_dpx_deta = float(series.coefficient("x", (0, 1, 0, 0, 0, 0, 1, 0))),
        maximum_scalar_constant_difference = float(np.max(
            abs(series.const_part - particle_coordinates(scalar)))))

    knobs.refresh([0.36, 0.55])
    refreshed = xtpsa.ParticlesTpsa(
        order = TPSA_ORDER, descriptor = descriptor, **launch)
    line.track(refreshed)
    refreshed_map = dict(
        eta = 0.36, phase_rad = 0.55,
        constant_part = refreshed.const_part.tolist(),
        parameter_jacobian = refreshed.param_jacobian().tolist())
finally:
    knobs.teardown()

################################################################################
# Ordinary set, copy and serialization after TPSA tracking
################################################################################
line.vars["rf_phase"] = 0.65
line.set("rf", counter_phase = 0.35)
copied = line.copy()
restored = xt.Line.from_dict(copied.to_dict())
restored.build_tracker(_context = CONTEXT)
original_particle = xt.Particles(_context = CONTEXT, **launch)
restored_particle = xt.Particles(_context = CONTEXT, **launch)
line.track(original_particle)
restored.track(restored_particle)

################################################################################
# Results
################################################################################
results = dict(
    model = MODEL, order = TPSA_ORDER, num_kicks = NUM_GAUSS_CELLS,
    reference_p0c_eV = launch["p0c"], coordinates = list(COORDINATES),
    initial = initial_map, refreshed = refreshed_map,
    scalar_storage_restored = not bool(line["rf"]._tpsa_enabled),
    derived_carrier_preserved = line["rf"].wave_number is None,
    copy_serialization_maximum_difference = float(np.max(abs(
        particle_coordinates(original_particle) - particle_coordinates(restored_particle)))))
print(json.dumps(results, indent = 2))
if OUTPUT_JSON is not None:
    OUTPUT_JSON = Path(OUTPUT_JSON).resolve()
    OUTPUT_JSON.parent.mkdir(parents = True, exist_ok = True)
    OUTPUT_JSON.write_text(json.dumps(results, indent = 2) + "\n")
