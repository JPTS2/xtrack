<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Tables

Learn the table operations used by line descriptions, surveys, Twiss results
and measurement records. These examples use small synthetic BPM readings, with
positions and offsets in metres. The TFS example also constructs a short line
to export element types and reference quantities; none calculates optics.
Each script runs independently.

| Example | Lesson |
| --- | --- |
| [001_create_and_access_tables.py](001_create_and_access_tables.py) | Create named rows, read columns and cells, exchange columns with pandas |
| [002_select_and_calculate.py](002_select_and_calculate.py) | Select names, ranges, labels and conditions; calculate columns and edit a copy |
| [003_save_and_load_tables.py](003_save_and_load_tables.py) | Preserve columns and scalar metadata through dictionaries, JSON and Xtrack CSV |
| [004_tfs_tables.py](004_tfs_tables.py) | Tailor TFS presets, use JSON/TOML layouts and export physical-reference headers |
| [005_hdf5_tables.py](005_hdf5_tables.py) | Store columns and scalar metadata in a named HDF5 group |

The first two examples use NumPy and Xtrack, whose table implementation also
uses pandas. HDF5 requires `h5py`; reading TFS requires `tfs-pandas` (imported
internally as `tfs`). TFS is a file format here; no MAD-X installation or live
external-code interface is involved.

The [TOML layout](bpm_format.toml) selects the same output fields and precision
as the dictionary and generated JSON in `004`. Output names can differ from
table attributes through `source_name`; the table itself is unchanged. Units
must be converted explicitly before writing. A TFS layout alone does not
convert coordinate or momentum-derivative conventions.

The persistence examples write to `outputs/` beside the scripts and display
the restored metadata and numerical differences. Their data is deliberately
simple: arbitrary Python objects or all fields of a specialised table need not
be representable by every format. Conversion to pandas exchanges columns,
rather than preserving scalar table attributes.

For a model's element table, start with
[load and inspect](../getting_started/02_load_and_inspect.py). For model
persistence, see [save and load](../getting_started/08_save_and_load.py).
Twiss calculations and their specialised table methods belong with
[optics](../optics/transverse/README.md). See the
[examples overview](../README.md) for other subjects.
