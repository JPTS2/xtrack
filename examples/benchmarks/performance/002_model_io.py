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
N_REPEATS       = 5
OUTPUT_FOLDER   = 'outputs/model_io'

################################################################################
# Prepare the native environment and generated output paths
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
source      = repo_root / LATTICE_FILE
env         = xt.load(source)
line        = env['demo']
baseline    = line.twiss4d(chrom = False)
output      = Path(__file__).resolve().parent / OUTPUT_FOLDER
output.mkdir(parents = True, exist_ok = True)
print('Python / Xtrack:', platform.python_version(), xt.__version__)
print('Platform:', platform.platform())
print('Elements / circumference [m]:',
      len(line.element_names), line.get_length())

################################################################################
# Time environment serialization and loading
################################################################################

########################################
# Measure plain JSON and gzip JSON on the same model
########################################
rows    = []
for filename in ['demo.json', 'demo.json.gz']:
    path        = output / filename
    save_times  = []
    load_times  = []
    for repeat in range(N_REPEATS + 1):
        started = perf_counter()
        env.to_json(path)
        saved   = perf_counter() - started

        started = perf_counter()
        restored    = xt.load(path)
        loaded  = perf_counter() - started
        if repeat > 0:
            save_times.append(saved)
            load_times.append(loaded)
    for label, samples in [('save', save_times), ('load', load_times)]:
        rows.append((filename, label, *np.percentile(samples, [50, 25, 75]),
                     path.stat().st_size))

    ########################################
    # Verify the restored operating point outside the timer
    ########################################
    restored_line   = restored['demo']
    tw              = restored_line.twiss4d(chrom = False)
    print(filename, 'length / p0c:',
          restored_line.get_length(), restored_line.particle_ref.p0c[0])
    print('Restored - source tunes:', tw.qx - baseline.qx, tw.qy - baseline.qy)
    print('Restored - source kqf:', restored['kqf'] - env['kqf'])

################################################################################
# Compare loading the native construction script
################################################################################
native_times    = []
for repeat in range(N_REPEATS + 1):
    started = perf_counter()
    rebuilt = xt.load(source)
    elapsed = perf_counter() - started
    if repeat > 0:
        native_times.append(elapsed)
rows.append(('lattice.py', 'build',
             *np.percentile(native_times, [50, 25, 75]), source.stat().st_size))
# Native loading executes geometry and strength construction; JSON restores
# serialized objects. The .py byte count excludes its referenced source files.

################################################################################
# Inspect costs with explicit timing boundaries
################################################################################
columns = ['format', 'operation', 'median_s', 'q25_s', 'q75_s', 'bytes']
xt.Table(
    data    = {name: np.array(values)
               for name, values in zip(columns, zip(*rows))},
    index   = 'format',
).show(maxwidth = 'full')
print('Generated files:', output)
# Each case has one excluded warmup. File reads use the operating-system cache;
# writes include serialization and buffered I/O, without forcing data to disk.
# Tracker construction, compilation, optics and imports are outside the timers.
