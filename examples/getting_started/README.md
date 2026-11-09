<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Getting started

Work through the numbered examples to construct and inspect a line, calculate
open and periodic optics, match tunes, track particles, compare measurements at
two locations, and save a model. Each script runs independently; there is no
need to run the preceding files first. Follow the topic links for more detail,
or use the [examples index](../README.md) to choose another subject.

| Example | Topic | Continue with |
| --- | --- | --- |
| [01_build_line.py](01_build_line.py) | Elements, line assembly and strength expressions | [line construction](../line_construction/README.md) |
| [02_load_and_inspect.py](02_load_and_inspect.py) | Loading a model and exploring its element table | [line construction](../line_construction/README.md) |
| [03_open_optics.py](03_open_optics.py) | Optics from specified entrance conditions | [transverse optics](../optics/transverse/README.md) |
| [04_periodic_optics.py](04_periodic_optics.py) | Periodic optics, tunes and chromaticities | [transverse optics](../optics/transverse/README.md) |
| [05_match_tunes.py](05_match_tunes.py) | Tune matching with quadrupole families | [matching](../matching/README.md) |
| [06_track_particles.py](06_track_particles.py) | Normalized launches and turn-by-turn tracking | [particle preparation](../particles/README.md), [tracking](../tracking/), [particle records](../monitors/) |
| [07_monitors_at_elements.py](07_monitors_at_elements.py) | Particle monitors at different locations | [monitors](../monitors/) |
| [08_save_and_load.py](08_save_and_load.py) | Model persistence and expressions | [environment persistence](../line_construction/015_save_and_load_environment.py) |

Run the files in your Xsuite environment. Parameters to change are collected
near the top of each file.

`03` retains the small native transport line for specified entrance conditions.
The other model-loading lessons use the shared
[demo ring](../../test_data/example_lattices/demo_ring/README.md): a 1500 m ring
with a 1 GeV/c electron reference and a low-beta insertion. RF voltage is zero
and radiation is off. Ring optics use a 4D periodic solution. During tracking,
momentum offset is constant and the longitudinal coordinate can evolve.

`05` changes the two arc-family trims to request tune shifts near +0.002 in
each plane. It does not preserve all insertion optics. `07` installs actual
particle monitors at the ideal BPM planes `bpm.s1.f` and `bpm.s4.f`, at 1 m
and 751 m. The model's passive BPM devices do not themselves record particles.

`08_save_and_load.py` writes `outputs/getting_started_demo.json` beside the
script. It never overwrites the input lattice.
