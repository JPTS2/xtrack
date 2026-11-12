# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path
from time import perf_counter
import platform

import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
RF_VOLTAGE      = 1e5                   # [V], stable longitudinal focusing
N_REPEATS       = 7
OPEN_START      = 'arc.s1.c1.qf.in'
OPEN_END        = 'straight.s1.start'

################################################################################
# Set a common RF-on operating point and retain entrance optics
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
env['voltage_rf']       = RF_VOLTAGE
reference   = line.twiss4d(chrom = False)
initial     = reference.get_twiss_init(OPEN_START)
print('Python / Xtrack:', platform.python_version(), xt.__version__)
print('Platform:', platform.platform())
print('Ring elements / length [m]:',
      len(line.element_names), line.get_length())

################################################################################
# Compare public optics operations after an excluded warmup
################################################################################

########################################
# Specify the physics and the propagated range for each operation
########################################
cases   = [
    ('periodic_4d', dict(method = '4d', chrom = False)),
    ('periodic_4d_chromatic', dict(method = '4d', chrom = True)),
    ('periodic_6d', dict(method = '6d', chrom = False)),
    ('open_arc_4d', dict(method = '4d', chrom = False,
                         start = OPEN_START, end = OPEN_END, init = initial)),
]
# 4D freezes longitudinal motion; 6D solves the synchronous orbit and third
# mode. The open arc reuses supplied entrance optics and has a shorter range.
# These different operations are not equal-physics algorithm speed rankings.

########################################
# Record repeated wall times and inspect the resulting tables
########################################
rows    = []
for label, options in cases:
    line.twiss(**options)
    samples = []
    for repeat in range(N_REPEATS):
        started = perf_counter()
        tw      = line.twiss(**options)
        samples.append(perf_counter() - started)
    rows.append((label, len(tw.name), tw.s[-1] - tw.s[0],
                 *np.percentile(samples, [50, 25, 75])))
    print(label, 'entrance/exit beta_x [m]:', tw.betx[0], tw.betx[-1])

################################################################################
# Report the complete calculation cost
################################################################################
columns = ['operation', 'rows', 'length_m', 'median_s', 'q25_s', 'q75_s']
xt.Table(
    data    = {name: np.array(values)
               for name, values in zip(columns, zip(*rows))},
    index   = 'operation',
).show(maxwidth = 'full')
# Imports, model loading and initial tracker construction are excluded.
# The timer includes each requested closed-orbit, map and table calculation.
