# FCC input for the mechanics demonstrations

Lesson [016](../016_fcc_ground_motion.py) runs the positron Z line from the
requested [FCC development model](https://gitlab.cern.ch/acc-models/fcc/acc-models-fcc-ee/-/tree/dev/108.1.0).
It uses the actual thick input, without replacing it by a maintained example or
flattening its geometry expressions. The model is an external input and is not
redistributed in this draft.

The recorded input is:

| Quantity | Value |
| --- | --- |
| Repository | `https://gitlab.cern.ch/acc-models/fcc/acc-models-fcc-ee.git` |
| Requested branch | `dev/108.1.0` |
| Pinned commit | `8805a5bb8bd568f5de2e1558037371905ecfc985` |
| Model's own VERSION | `dev/LCC_108.3.0` |
| File | `lattices/z/fccee_z_beam_1_positrons.json` |
| Line | `fccee_p_ring` |
| JSON SHA-256 | `dd7c7749d1bedfa5faf41bbfd885312bbc38bec8a817460bc0627f3c4b6094fe` |

The branch label and internal VERSION differ; both are retained above.
For a separate local input checkout:

```bash
git clone --branch dev/108.1.0 --single-branch \
    https://gitlab.cern.ch/acc-models/fcc/acc-models-fcc-ee.git /path/to/fcc-model
git -C /path/to/fcc-model checkout 8805a5bb8bd568f5de2e1558037371905ecfc985
```

Run from the Xtrack checkout, selecting that file explicitly:

```bash
XTRACK_MECHANICS_FCC_INPUT=/path/to/fcc-model/lattices/z/fccee_z_beam_1_positrons.json \
    PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python \
    examples/mechanics/016_fcc_ground_motion.py
```

The default path already resolves to the isolated `../inputs/` checkout in this
workspace. Ordinary editable **User parameters** select the line, context,
ground PSD/seed, correlations, waves/pump, response classes, installation,
record bounds and tracking duration. The example makes no network requests and
does not modify the model checkout. Lesson 015 can use the same FCC file for
independent bunch phases; its line/quad/frequency parameters state that variant.

One support per quadrupole and the cyclic response classes in 016 are explicit
demonstration assumptions. The
[magnet catalogue](https://github.com/skostogl/fcc-ee-magnet-catalogue)
does not establish engineering girder assignments for this input. Replace these
rules with verified support data before making an FCC mechanical prediction.
The ground spectra and conditional transfer functions are illustrative too.
