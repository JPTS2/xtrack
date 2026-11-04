<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Getting started

A short introduction to constructing a line, exploring a model, calculating
optics, matching and tracking. Each example runs independently. The topic folders
provide more detailed examples.

| Example | Topic | Continue with |
| --- | --- | --- |
| [01_build_line.py](01_build_line.py) | Elements, line assembly and strength expressions | [line construction](../line_construction/README.md) |
| [02_load_and_inspect.py](02_load_and_inspect.py) | Loading a model and exploring its element table | [line construction](../line_construction/README.md) |
| [03_open_optics.py](03_open_optics.py) | Optics from specified entrance conditions | [Twiss examples](../twiss/) |
| [04_periodic_optics.py](04_periodic_optics.py) | Periodic optics, tunes and chromaticities | [Twiss examples](../twiss/) |
| [05_match_tunes.py](05_match_tunes.py) | Tune matching with quadrupole families | [matching](../match/) |
| [06_track_particles.py](06_track_particles.py) | Normalized launches and turn-by-turn tracking | [tracking](../tracker/), [monitors](../monitor/) |
| [07_monitors_at_elements.py](07_monitors_at_elements.py) | Particle monitors at different locations | [monitors](../monitor/) |
| [08_save_and_load.py](08_save_and_load.py) | Model persistence and expressions | [JSON persistence](../to_json/) |

Run the files in your Xsuite environment. Parameters to change are collected
near the top of each file.

The [native models](../../test_data/example_lattices/README.md) are a transport
line and a magnetic ring, both with a 1 GeV/c electron reference. There is no RF
or radiation. Open optics use specified entrance conditions; ring optics use a
4D periodic solution. During tracking, momentum offset is constant and the
longitudinal coordinate can evolve.

`08_save_and_load.py` writes `outputs/getting_started_ring.json` beside the
script. It never overwrites the input lattice.
