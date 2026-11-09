<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Tables

Learn the table operations used by line descriptions, surveys, Twiss results
and measurement records. These examples use small synthetic BPM readings, with
positions and offsets in metres; they do not require a lattice or calculate
optics. Each script runs independently.

| Example | Lesson |
| --- | --- |
| [001_create_and_access_tables.py](001_create_and_access_tables.py) | Create named rows, read columns and cells, exchange columns with pandas |
| [002_select_and_calculate.py](002_select_and_calculate.py) | Select names, ranges, labels and conditions; calculate columns and edit a copy |
| [003_save_and_load_tables.py](003_save_and_load_tables.py) | Preserve columns and scalar metadata through dictionaries, JSON and Xtrack CSV |
| [004_hdf5_and_tfs_tables.py](004_hdf5_and_tfs_tables.py) | Store HDF5 and choose a TFS text layout with explicit fields and precision |

The first two examples use NumPy and Xtrack, whose table implementation also
uses pandas. HDF5 requires `h5py`; reading TFS requires `tfs-pandas` (imported
internally as `tfs`). TFS is a file format here; no MAD-X installation or live
external-code interface is involved.

The persistence examples write to `outputs/` beside the scripts and display
the restored metadata and numerical differences. Their data is deliberately
simple: arbitrary Python objects or all fields of a specialised table need not
be representable by every format. Conversion to pandas exchanges columns,
rather than preserving scalar table attributes.

For a model's element table, start with
[load and inspect](../getting_started/02_load_and_inspect.py). For model
persistence, see [save and load](../getting_started/08_save_and_load.py).
Twiss calculations and their specialised table methods belong with
[optics](../twiss/). See the [examples overview](../README.md) for other subjects.
