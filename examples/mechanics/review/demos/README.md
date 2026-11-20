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

The girder has all six prescribed coordinates at 0.25 Hz: one slow cycle in the
four-second clip, with 180 nm peak vertical input. Each magnet inherits that
motion and has smaller relative inputs at 2–3 Hz (8–12 cycles per clip), with
14–20 nm peak vertical displacement. These are illustrative
harmonic inputs, with explicit installation placements and retained deliberate
quadrupole offsets/roll. No structural force law is inferred from the mounts.
The model checks its element poses against actual Xtrack alignment lowering and
native element-frame survey before rendering.

The animation magnifies the girder's transform, then each magnet's transform
relative to the girder, and finally composes them. The current display uses the
same gain of one million for every translation and rotation, preserving the
common/relative amplitude hierarchy. Wire outlines move rigidly with the girder:
they show where each magnet would be without its additional relative input.
The coloured magnets make the small fast residual visible against those carried
outlines. The local checks cover rigid inheritance even with unequal gains and
preservation of cuboid dimensions.

The lower row separately plots slow girder displacement and fast magnet/girder
motion in physical nm, without display magnification. Their axes show the
different amplitude scales. Peak common displacement divided by peak relative
displacement at the displayed magnet anchors is 8.28 for Q1, 9.12 for the bend
and 16.83 for Q2; these are ratios of maxima over the clip. The example checks
that each exceeds four.

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
with a compact pair of histories and status below. One fixed fit contains the swept
geometry for every frame; it does not crop or change the camera during playback.

The captured run prepared the selected histories in 0.749 s and rendered/encoded
the GIF in 12.04 s. The GIF is 3.8 MiB. Peak process RSS was about 770 MiB including
scientific Python, rendering and Pillow frames; the small native pose model and
explicit histories are separate from this export cost. Source and helper hashes
are recorded with the result.
