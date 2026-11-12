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

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Anchored assembly workload
########################################
CELL_COUNTS     = [16, 64, 256]
CELL_LENGTH     = 50.0                  # [m]

########################################
# Individual appends and repeated measurements
########################################
APPEND_COUNTS   = [100, 500, 1000]
INITIAL_COUNTS  = [1, 1000, 10000]
LATTICE_FILE    = 'test_data/hllhc15_thick/lhc_thick_with_knobs.json'
MACHINE_APPENDS = 1000
N_REPEATS       = 5
# One fresh-model warmup precedes the measured repeats at each size.

################################################################################
# Report the construction environment
################################################################################
print('Python / Xtrack:', platform.python_version(), xt.__version__)
print('Platform:', platform.platform())
print('Fresh models; imports and component preparation excluded from assembly')
repo_root   = Path(__file__).resolve().parents[3]

################################################################################
# Measure construction operations on independent environments
################################################################################

########################################
# Resolve anchored placements and assemble a line
########################################
assembly_times  = []
for n_cells in CELL_COUNTS:
    samples = []
    for repeat in range(N_REPEATS + 1):
        env         = xt.Environment()
        components  = []
        env.new('quad', xt.Quadrupole, length = 2.0)
        for cell in range(n_cells):
            prefix  = f'c{cell}'
            origin  = CELL_LENGTH * cell
            components.extend([
                env.new(prefix + '.q1', 'quad',
                    anchor  = 'start',
                    at      = origin + 1.0),
                env.new(prefix + '.q2', 'quad',
                    anchor  = 'start',
                    at      = 10.0,
                    from_   = prefix + '.q1@end'),
                env.new(prefix + '.sext', xt.Sextupole,
                    length  = 0.1,
                    anchor  = 'end',
                    at      = -1.0,
                    from_   = prefix + '.q2@start'),
                env.new(prefix + '.q3', 'quad', at = origin + 20.0),
                env.new(prefix + '.q4', 'quad',
                    anchor  = 'start',
                    at      = prefix + '.q3@end'),
                env.new(prefix + '.bpm', xt.Marker,
                    at  = prefix + '.q2@start'),
                env.new(prefix + '.marker', xt.Marker,
                    at  = prefix + '.bpm@end'),
            ])
        started = perf_counter()
        line    = env.new_line(
            length      = n_cells * CELL_LENGTH,
            components  = components,
        )
        elapsed = perf_counter() - started
        if repeat > 0:
            samples.append(elapsed)
    assembly_times.append(samples)
    print('Cells / placed components / length [m]:',
          n_cells, len(components), line.get_length())
# These unpowered cells define a placement workload, not a matched ring.

########################################
# Vary initial line size and include the stored HL-LHC workload
########################################
append_times    = []
append_results  = []
cases           = [(f'{count} drifts', count) for count in INITIAL_COUNTS]
cases.append(('HL-LHC', None))
for label, initial_count in cases:
    counts  = [MACHINE_APPENDS] if initial_count is None else APPEND_COUNTS
    for n_markers in counts:
        samples = []
        for repeat in range(N_REPEATS + 1):
            if initial_count is None:
                line    = xt.load(repo_root / LATTICE_FILE)
            else:
                line    = xt.Line(elements = {
                    f'base.{index}': xt.Drift(length = 1.0)
                    for index in range(initial_count)})
            initial_elements    = len(line.element_names)
            length_before       = line.get_length()
            started = perf_counter()
            for index in range(n_markers):
                line.append(f'benchmark.marker.{index}', xt.Marker())
            elapsed = perf_counter() - started
            if repeat > 0:
                samples.append(elapsed)
        append_times.append(samples)
        append_results.append((label, initial_elements, n_markers, samples))
        print('Case / initial elements / appended / final elements / length:')
        print(label, initial_elements, n_markers,
              len(line.element_names), line.get_length())
        print('Circumference change [m]:', line.get_length() - length_before)
# Fresh construction or machine loading is outside the timer. The timed work
# includes marker creation and individual append calls. Drift-only lines isolate
# initial-size scaling; the HL-LHC case also retains its strength expressions.

################################################################################
# Show timing variation and workload scaling
################################################################################
fig, axes   = plt.subplots(1, 2, figsize = (10, 4))
timings = np.array(assembly_times)
median  = np.median(timings, axis = 1)
bounds  = np.percentile(timings, [25, 75], axis = 1)
print('Cells assembled: median [q25, q75] seconds')
for size, centre, low, high in zip(CELL_COUNTS, median, *bounds):
    print(f'{size:6d}: {centre:.6f} [{low:.6f}, {high:.6f}]')
axes[0].plot(CELL_COUNTS, median, 'o-', label = 'Median')
axes[0].fill_between(CELL_COUNTS, *bounds, alpha = 0.2,
                     label = 'Interquartile range')
axes[0].set(xlabel = 'Cells assembled', ylabel = 'Wall time [s]')

print('Appends: case / initial elements / appended / median [q25, q75] s')
for label, initial_count in cases:
    selected    = [row for row in append_results if row[0] == label]
    sizes       = [row[2] for row in selected]
    median      = np.array([np.median(row[3]) for row in selected])
    bounds      = np.array([np.percentile(row[3], [25, 75])
                            for row in selected]).T
    for row, centre, low, high in zip(selected, median, *bounds):
        print(f'{label}: {row[1]} / {row[2]} / '
              f'{centre:.6f} [{low:.6f}, {high:.6f}]')
    axes[1].errorbar(sizes, median,
                     yerr   = [median - bounds[0], bounds[1] - median],
                     fmt    = 's' if initial_count is None else 'o-',
                     capsize = 4, label = label)
axes[1].set(xlabel = 'Markers appended', ylabel = 'Wall time [s]')
for axis in axes:
    axis.legend()
fig.tight_layout()
plt.show()
