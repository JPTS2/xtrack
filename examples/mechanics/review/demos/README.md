# Local assembly review

This heavier visual demonstration is kept outside the fourteen short lessons.
It uses the native mechanics model and table/history interface; the adjacent
helper contains presentation code only. Publishing or promoting it into the
main example set remains a later choice.

```bash
PYTHONPATH=. MPLBACKEND=Agg python examples/mechanics/review/demos/015_girder_assembly.py
```

The exact line sequence is 0.5 m quadrupole, 0.5 m drift, 1.5 m bend, 0.5 m
drift and 0.5 m quadrupole: **3.5 m in total**. The reference bend angle is
60 mrad. The straight rigid girder is 3.7 m long, including end overhang;
the bend is a separate rigid body mounted at its surveyed chord centre.
Neither the bend nor the girder changes shape during motion.

The girder has all six prescribed coordinates. Each magnet inherits its motion
and has additional relative translation/rotation inputs. These are illustrative
harmonic inputs, with explicit installation placements and retained deliberate
quadrupole offsets/roll. No structural force law is inferred from the mounts.
The model checks its element poses against actual Xtrack alignment lowering and
native element-frame survey before rendering.

The animation magnifies the girder's transform, then each magnet's transform
relative to the girder, and finally composes them. This keeps rigid mounts
attached even when translation/rotation gains differ. The local checks cover
that invariant and preservation of cuboid dimensions. The lower panel reports
relative magnet motion in physical nm, without display magnification.

Outputs are in [results/015_girder_assembly](../results/015_girder_assembly):

- [Animated assembly](../results/015_girder_assembly/girder_assembly.gif).
- [Static assembly view](../results/015_girder_assembly/assembly.svg).
- [Terminal output and separate timing costs](../results/015_girder_assembly/terminal.txt).
- [Run metadata](../results/015_girder_assembly/result.json).

The default is 100 frames over four physical seconds, played at 25 fps. GIF
rendering/encoding is measured separately from native motion/history work.
Pillow retains encoding frames on the host; this local export can use much more
memory than the four-body pose data. It creates no tracking or GPU history cache.
`N_FRAMES`, `DISPLAY_SCALES` and all prescribed inputs are editable parameters.

The figure uses a subplot grid: the complete assembly occupies the upper panel,
with a compact history/status row below. One fixed fit contains the swept
geometry for every frame; it does not crop or change the camera during playback.

The captured run prepared the selected histories in 0.695 s and rendered/encoded
the GIF in 8.15 s. The GIF is 4.4 MiB. Peak process RSS was about 798 MiB including
scientific Python, rendering and Pillow frames; the small native pose model and
explicit histories are separate from this export cost. Source and helper hashes
are recorded with the result.
