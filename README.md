# dmipy-replay

Replay Monte-Carlo diffusion-MRI walks with any acquisition you like: **numpy only, no walk, no GPU.**

A *replay pack* (`.rpk`, [replay-pack-spec](https://github.com/dmrai-lab/replay-pack-spec) RPK.md) stores what a
Monte-Carlo walk leaves of every walker that no acquisition changes: its path as sine bands, the pool it is in, its
wall contact and the susceptibility field it read along the way. Every acquisition is then an inner product against
the pack, including any gradient waveform, RF schedule and B1 scale, T2/T1 per pool, surface relaxivity, field
strength and orientation. Packs live on the Hugging Face Hub under
[SubstrateCommons](https://huggingface.co/SubstrateCommons). This package pulls a pack and replays it.

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
plausible number in place of one it cannot compute. A pack also states the band it was walked at and refuses a
waveform beyond that band.

## Acquisitions

Every acquisition is a `ScannerSequence` (replay-pack-spec ACQUISITION.md): the physical gradient, the RF schedule,
the readout. Builders cover the families (`pgse`, `pgste`, `ogse`, `cpmg`, `gre`, `ste`, `pte`, `splice`), and
**Pulseq files are read natively**:

```python
from dmipy_replay.sequences.pulseq import from_pulseq
seq = from_pulseq("my_diffusion.seq")           # gradients exactly, every RF pulse with the B1(t) it plays
pack.replay_bloch(seq, b1_scale=0.92)           # the real pulses at 92 % transmit
```

The reader is the file's own text and numpy (formats 1.4 and 1.5; no pypulseq). Each RF pulse carries its played
B1(t) as its envelope, including shape, phase and carrier offset, so the vector-Bloch route plays the actual pulse.
A B1+ *map* is not part of a Pulseq file; it is the `b1_scale` knob (per pack, or per walker).

A b-value table (s/mm², one direction per volume) is `from_btable(bvals, bvecs, delta=, Delta=, TE=)`. A table
carries no timing, so `delta` and `Delta` are required; nothing fills them in from a default.

## From DIPY

`dmipy_replay.dipy.multi_pack` is DIPY's `multi_tensor` with a pack where the tensor was: the same gradient table,
`angles` (degrees, or unit vectors), `fractions` (percent), `S0`, `snr` (Rician), and the same return value,
`(S, sticks)`. What it adds are the knobs:

```python
from dipy.core.gradients import gradient_table
from dmipy_replay.dipy import multi_pack, Tissue

gtab = gradient_table(bvals, bvecs=bvecs, big_delta=0.03, small_delta=0.012)   # seconds
S, sticks = multi_pack(gtab, [bundle, bundle], angles=[(0, 0), (60, 0)], fractions=[50, 50],
                       tissue=Tissue(T2={"intra": 0.07, "extra": 0.06, "myelin": 0.015}, chi_iso=-1e-7),
                       TE=0.08, scanner=3.0, S0=100, snr=30)
```

It refuses a table without `big_delta`/`small_delta`, and a `tissue` without a `TE`. DIPY is never imported: the
table is read by its attributes. `pack_licenses(packs)` gives the terms each pack is published under.

## Dependencies and licence

`numpy`, nothing else. The `.rpk` container is read without the safetensors wheel (`dmipy_replay.replay.container`)
and the Hub over plain HTTPS with the standard library (`HF_TOKEN` or the Hub client's token file for a private
dataset; public datasets need none).

The **code** is Apache-2.0. The **packs** carry their own terms in their headers (`pack_license`): the phantoms on
SubstrateCommons are CC BY-NC-SA 4.0 or a commercial licence, and the canonical pores (the ruler) are CC BY 4.0.

## How this relates to dmipy-sim

[dmipy-sim](https://github.com/dmrai-lab/dmipy-sim) walks the substrates and writes the packs. This package is the
reader side of the same code: the acquisition object, the builders and the replay contraction are one
implementation, and `tests/test_parity.py` holds every route to dmipy-sim's own replay of the same pack, bit for
bit. `tools/make_fixtures.py` (dev-only, needs dmipy-sim) records those numbers.

Not here yet: orientation *distributions* (FODs, Watson / Bingham dispersion), which compose through the pose
expansion; replay phantoms (`.rph`); and the time-segment page layouts.
