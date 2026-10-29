# dmipy-replay

The reader of Monte-Carlo diffusion-MRI replay packs: **numpy only, no walk, no GPU.**

A *replay pack* (`.rpk`, [replay-pack-spec](https://github.com/dmrai-lab/replay-pack-spec) RPK.md) stores what a
Monte-Carlo walk leaves of every walker that no acquisition changes: its path as sine bands, the pool it is in, its
wall contact and the susceptibility field it read along the way. Every acquisition is then an inner product against
the pack, including any gradient waveform, RF schedule and B1 scale, T2/T1 per pool, surface relaxivity, field
strength and orientation. Packs live on the Hugging Face Hub under
[SubstrateCommons](https://huggingface.co/SubstrateCommons). This package reads them: the Hub client, the pack
container, the acquisition object and its readers (Pulseq, b-tables), the replay, and the layouts a whole grid of
voxels is stored in.

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
| B1+ scale, finite / shaped pulses, pulse trains, off-resonance | `pack.replay_bloch(seq, b1_scale=0.9, off_resonance_T=)` | all |
| a faster diffusivity, a smaller or larger substrate | `tissue=dr.Tissue(D=)`, `pack.at_length_scale(s)` | views |
| one pool, every walker | `compartment=1`, `pack.walker_signals(seq)` | |

A pack replays only the tiers it carries and **refuses** a knob outside them by name. It never returns a
plausible number in place of one it cannot compute. Each pack states the envelope it is certified over: a waveform
whose band truncation would cost more than the pack's stated numerical cap less its other certified terms
(`pack.truncation_budget`) is refused, naming the band it needs (`pack.waveform_band(seq)`), and so is a field
strength or susceptibility beyond the certified field scale (`pack.field_envelope`) or a relaxivity beyond the contact
envelope.

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

## Layouts of a whole grid

- **The columnar layout** (`dr.ColumnarPack`, `ReplayPack.open("hf://owner/name/prefix")`): the walkers of a grid of
  voxels in columns, read by reference over HTTP range requests. `plan(seq)` and `plan_for_class("prisma")` say which
  bands, field modes, tiers, rows and bytes an acquisition (or every waveform a scanner class can play) needs before
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

`numpy`, nothing else. The `.rpk` container is read without the safetensors wheel (`dmipy_replay.replay.container`,
the file mapped copy-on-write) and the Hub over plain HTTPS with the standard library.

The **code** is Apache-2.0. The **packs** carry their own terms in their headers (`pack_license`): the phantoms on
SubstrateCommons are CC BY-NC-SA 4.0 or a commercial licence, and the canonical pores are CC BY 4.0.

## How this relates to dmipy-sim

[dmipy-sim](https://github.com/dmrai-lab/dmipy-sim) walks the substrates and writes the packs and layouts; it
imports this package for everything a reader does, so the acquisition object, the builders, the replay contraction,
the plan and the layouts' readers are one implementation. dmipy-sim adds the producer's routes and the JAX and torch
backends of the same arithmetic, and composes orientation distributions (FODs, Watson / Bingham dispersion) through
its pose expansion. `tests/test_parity.py`, `tests/test_columnar.py` and `tests/test_time_segments.py` hold this
package's reads to dmipy-sim's own reads of the same packs and layouts; `tools/make_fixtures.py`,
`tools/make_columnar_fixtures.py` and `tools/make_segment_fixtures.py` (dev-only, need dmipy-sim) record them.
