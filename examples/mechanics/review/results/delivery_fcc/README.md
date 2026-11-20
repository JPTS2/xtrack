# FCC delivery captures

The [walkthrough](../../../FCC_DEMO.md) explains the runnable lesson and figures.

| Capture | Meaning |
| --- | --- |
| `016_fcc_ground_motion/` | Default 512-sample, blocked-fit, one-particle lesson; 23.57 s and four figures |
| `throughput_variant/` | Earlier full 1/32-particle comparison; 102.11 s, whole-record contact fit; exact lesson source snapshot retained |
| `plane_wave_variant/` | Eight-turn functional check with explicit parameter overrides; not a throughput record |
| `installation_probe.json` | Uncorrected installation-amplitude survival check; no dynamic sources |

The throughput variant's `initial_5um_installation_failure.txt` is the preceding
failed 5 µm installation launch, not a failure of the subsequent 0.5 µm run.
No tracking guard was weakened. Source hashes, all timing repetitions and peak
RSS are retained in the result records. PNG previews are generated locally;
the SVG figures are retained here, with two selected figures also in the draft
walkthrough.

The separate `delivery_slice/clean_examples/` records a replay from the selected
draft patch, including the pinned FCC input hash. Those are verification runs;
the walkthrough retains the timing basis above.
