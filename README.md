# dmipy-replay

The reader of Monte-Carlo diffusion-MRI replay packs: **no walk, no GPU, numpy at its core.**

A *replay pack* (`.rpk`, [replay-pack-spec](https://github.com/dmrai-lab/replay-pack-spec) RPK.md) stores what a
Monte-Carlo walk leaves of every walker that no acquisition changes: its path as sine bands, the pool it is in, its
wall contact and the susceptibility field it read along the way. Every acquisition is then an inner product against
the pack, including any gradient waveform, RF schedule and B1 scale, T2/T1 per pool, surface relaxivity, field
strength and orientation. Packs live on the Hugging Face Hub under
[SubstrateCommons](https://huggingface.co/SubstrateCommons). This package reads them: the Hub client, the pack
container, the acquisition object and its readers (Pulseq, b-tables), the replay, the pose replay over orientation
distributions, the replay phantoms, and the layouts a whole grid of voxels is stored in.

```python
import dmipy_replay as dr

pack = dr.ReplayPack.load("hf://SubstrateCommons/canonical-pores/packs/cylinder/d04.00um.rpk")
seq = dr.pgse([[1, 0, 0], [0, 0, 1]], 0.01, 0.03, bvalues=[3e9, 3e9])        # SI: s, s/m^2
signal = pack.replay(seq)                                                    # the bare diffusion signal
signal = pack.replay(seq, tissue=pack.nominal, scanner=pack.nominal_field_T) # the substrate's nominal tissue
```

`dr.hub.packs("SubstrateCommons/canonical-pores")` lists a dataset's packs with their sizes, bands, floors and
licences.

## The knobs

| knob | how | tier |
|---|---|---|
| any gradient waveform (PGSE, OGSE, PGSTE, b-tensor, arbitrary) | `pack.replay(seq)` | C0 |
| T2 / T1 per pool | `tissue=dr.Tissue(T2={"intra": 0.08, "extra": 0.06})` | C1 |
| surface relaxivity (transverse, longitudinal) | `tissue=dr.Tissue(rho2=, rho1=)` | C2 |
| susceptibility field: B0, chi, its direction | `tissue=dr.Tissue(chi_iso=, chi_aniso=)`, `scanner=7.0` | C3 |
| the substrate's pose | `orientation=` a 3x3 rotation or the lab direction of its axis | all |
| a distribution of poses (Watson, Bingham, an FOD, every pose) | `orientation=dr.Distribution.watson(8.0)`, `dr.FOD.from_sh(c, basis=)`; `pack.pose_response(seq)` | all |
| B1+ scale, finite / shaped pulses, pulse trains, off-resonance | `pack.replay_bloch(seq, b1_scale=0.9, off_resonance_T=)` | all |
| a faster diffusivity, a smaller or larger substrate | `tissue=dr.Tissue(D=)`, `pack.at_length_scale(s)` | views |
| one pool, every walker | `compartment=1`, `pack.walker_signals(seq)` | |

A pack replays only the tiers it carries and **refuses** a knob outside them by name. It never returns a
plausible number in place of one it cannot compute. Each pack states the envelope it is certified over,
`pack.envelope` (a `dr.Envelope`, in physical terms: the temporal band `band_hz`, the per-axis amplitude `G_max`, the
field strengths, susceptibility scale, refocusing depth and surface relaxivity; `dr.envelope_of("prisma")` is a
scanner class's). A waveform that needs more band than `band_hz` (`pack.waveform_band(seq)`, at the pack's stated
numerical cap less its other certified terms, `pack.truncation_budget`) or plays more than `G_max` on an axis is
refused by name, and so is a field strength or susceptibility beyond the certified field scale (`pack.field_envelope`)
or a relaxivity beyond the contact envelope.

## Plan before read

A replay decodes only the position bands and field modes the acquisition needs: the fewest whose dropped tail costs
at most `tol` (0.25 by default) times the pack's Monte-Carlo floor. `pack.plan(seq)` states that read before any
band is decoded: the bands and modes, the error each truncation costs, where the variances it was weighed with come
from, and the bytes read against the bytes stored. The array a replay returns carries its plan (`S.plan`); `K=` and
`modes=` read a stated depth, `tol=None` reads every stored band and mode. A prefix read is bit for bit the full
decode cut at the same depth, and each tensor selection is decoded once per pack and held
(`pack.decode_budget`, `pack.release_decoded()`).

A pack on the Hub need not be downloaded to be replayed:

```python
pack = dr.ReplayPack.open("hf://SubstrateCommons/canonical-pores/packs/cylinder/d04.00um.rpk")
plan = pack.plan(seq)            # from the file's header alone
S = pack.replay(seq)             # fetches by HTTP range only the tensors the plan names
whole = pack.load()              # the whole file, checked against the dataset's manifest
```

## The Hub

Every Hub read pins the dataset's revision once (a branch or tag is resolved to its commit sha when the pack or
layout is opened), so the manifest, the pack and every byte range come from one commit. A whole pack is downloaded
once into the cache (`$DMIPY_REPLAY_CACHE`, else `~/.cache/dmipy-replay`, keyed by that commit) and its sha256 is
checked against the manifest's row at the same commit; a mismatch is refused. `ReplayPack.load(uri, windows=range(k))`
reads only a pack's leading storage windows. A token is read from `HF_TOKEN` or the Hub client's token file when
present; public datasets need none.

## Orientation distributions and phantoms

A pack answers for one microstructure at any pose; a voxel holds a distribution of poses. `pack.pose_response(seq)`
expands the pack's response over every pose of its substrate in the real Wigner basis (`dmipy_replay.replay.so3`), by
the closed form (the per-walker Rayleigh expansion) or a quadrature, with its band and a certified misfit; a
`Distribution` (one pose, an axis, a Watson, a Bingham, uniform) or an `FOD` composes against it by an inner product,
which is what `replay(orientation=...)` does. `pack.pose_series(classes)` takes every encoding class of a machine in
one pass over the walkers; `dmipy_replay.replay.pathways.train_response` an RF train as a sum of gates;
`pack.study(Study(...))` a protocol on several tissues and scanners at once.

A **replay phantom** (`.rph`, RPH.md) is a voxel grid that cites packs: `dmipy_replay.phantom.Phantom.compose(grid,
fractions=, orientation=)` places packs and closed forms (`PackSubstrate`, `FreeWater`, `Inert`) in voxels with a pose
per voxel (`Peaks`, `ODF`, `Watson`, `Frames`, `Fan`), `Phantom.partition(pack, grid)` cuts one walk into the voxels
its walkers started in; `replay(seq, scanner=)`, `replay_train` and `replay_bloch` image it, with a machine's own
field, gradient non-linearity and concomitant fields per voxel (`ScannerLimits`); `write` / `read` the file.

## Layouts of a whole grid

- **The columnar layout** (`dr.ColumnarPack`, `ReplayPack.open("hf://owner/name/prefix")`): the walkers of a grid of
  voxels in columns, read by reference over HTTP range requests. `plan(seq)` and `plan_for_envelope(envelope_of("prisma"))` say
  which bands, field modes, tiers, rows and bytes an acquisition (or every waveform inside an envelope) needs before
  any transfer; `view(K=, voxels=)` is an ordinary `ReplayPack` of those rows; `image(seq)` replays the whole grid
  in one pass over the rows.
- **The time-segment layout** (`dmipy_replay.replay.time_segments.TimeSegmentLayout`): a walk contracted once
  against the terms of stored partitions of its time (constant segments, trapezoids, cosines, tensor encodings), so
  that every acquisition built on a stored partition, at any amplitude and direction on each term, is a few products
  per walker. `fit(seq)` names the partition a sequence fits and refuses one that fits none; `image(seq, tissue=,
  scanner=, b0_direction=)` forms every voxel's signal and split-half floor from the packed words.

## Acquisitions

Every acquisition is a `ScannerSequence` (replay-pack-spec ACQUISITION.md): the physical gradient, the RF schedule,
the readout. Builders cover the families (`pgse`, `pgste`, `ogse`, `cpmg`, `gre`, `ste`, `pte`, `splice`), and
**Pulseq files are read natively**:

```python
from dmipy_replay.sequences.pulseq import from_pulseq
seq = from_pulseq("my_diffusion.seq")           # gradients exactly, every RF pulse with the B1(t) it plays
pack.replay_bloch(seq, b1_scale=0.92)           # the real pulses at 92 % transmit
```

The reader is the file's own text and numpy (formats 1.4 and 1.5). Each RF pulse carries its played B1(t) as its
envelope, including shape, phase and carrier offset, so the vector-Bloch route plays the actual pulse. A B1+ *map*
is not part of a Pulseq file; it is the `b1_scale` knob (per pack, or per walker).

A b-value table (s/mm², one direction per volume) is `from_btable(bvals, bvecs, delta=, Delta=, TE=)`. A table
carries no timing, so `delta` and `Delta` are required; nothing fills them in from a default.

## Dependencies and licence

`numpy`, nothing else, for everything above but the pose expansion: `import dmipy_replay` loads numpy alone
(`tests/test_light.py`). The `.rpk` and `.rph` containers are read and written without the safetensors wheel
(`dmipy_replay.replay.container`, the file mapped copy-on-write) and the Hub over plain HTTPS with the standard library.
The extras: `pose` (scipy, threadpoolctl: the pose expansion's special functions and rotations, and the CPU thread
cap, `dr.cap_cpu_threads` / `DMIPY_REPLAY_CPU_THREADS`), `jax` and `torch` (the same arithmetic on another array
library and device), `all`. Every route runs on numpy unless `backend="jax"` or `backend="torch"` is stated; a route
whose extra is not installed is refused with its `pip install 'dmipy-replay[...]'`.

The **code** is Apache-2.0. The **packs** carry their own terms in their headers (`pack_license`): the phantoms on
SubstrateCommons are CC BY-NC-SA 4.0 or a commercial licence, and the canonical pores are CC BY 4.0.

## How this relates to dmipy-sim

[dmipy-sim](https://github.com/dmrai-lab/dmipy-sim) walks the substrates and writes the packs and layouts: the walk,
the encoding, the certificate, publishing, and the operations that re-encode a pack (`truncate_pack`, `rewindow_pack`,
`prefix_pack`, `extend_pack`). It imports this package for everything a reader does and re-exports none of it, so
there is one `ReplayPack` and one implementation of every reading. `tests/test_parity.py`, `tests/test_columnar.py`
and `tests/test_time_segments.py` hold this package's reads to dmipy-sim's own reads of the same packs and layouts
(records taken at dmipy-sim `8d7b97f`, frozen); `tests/parity_move/` holds every route that came from dmipy-sim
bit for bit to its outputs there.
