# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project follows [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-02

dmipy-replay turns a replay pack into a signal or an image: the signal side of the dmipy ecosystem. NumPy at its core,
no walk, no GPU. [dmipy-sim](https://github.com/dmrai-lab/dmipy-sim) walks the substrates and produces the packs; this
package reads them.

- **Packs.** The Hub client (a dataset revision pinned once, sha256 checked against the manifest, ranged HTTP reads), the
  `.rpk` container and one `ReplayPack` class (`ReplayPack.load` / `.open`, `read_rpk`, `RemotePack` for a pack read by
  range), and `plan` / `truncation_budget` to state a read before any band is decoded.
- **Replay.** Any gradient waveform, with tissue (T2, T1, surface relaxivity, susceptibility), field strength and
  orientation as knobs; `replay_bloch` for RF schedules and B1 scale; RF trains as sums of coherence pathways.
- **Orientation distributions.** `pose_response` and `pose_series` expand a pack's response over every pose in the real
  Wigner basis (`so3`); a `Distribution` (axis, Watson, Bingham, uniform) or an `FOD` composes against it. A `Study`
  replays a protocol on several tissues and scanners from one read.
- **Images.** `dmipy_replay.phantom` composes packs into a voxel grid, or cuts one walk into it (`Phantom`, the `.rph`
  file), with a machine's own field, gradient non-linearity and concomitant fields per voxel. The columnar and
  time-segment layouts read a whole grid of voxels by reference. JAX and PyTorch run the image routes where stated.
- **Acquisition.** `ScannerSequence` with the sequence builders (`pgse`, `pgste`, `ogse`, `cpmg`, `splice`, `gre`, `ste`,
  `pte`), the Pulseq reader (formats 1.4 and 1.5), `from_btable`, the cited scanner catalogue (`ScannerLimits`) and the
  substrate and tissue specification.
- **The guarantee is one `Envelope`.** A pack states what it is certified for in physical terms: the temporal band, the
  per-axis `G_max`, the field strengths, the susceptibility scale, the refocusing depth and the surface relaxivity.
  `envelope_of(scanner)` gives a scanner class's envelope. Every replay route refuses a request outside the envelope by
  name, and a pack that states none.
- **Refused by name.** A pack written by an older producer (re-walk it: a pack is never converted) and a tier the pack
  does not carry.
- **Names.** A sequence's b-value and b-tensor are `seq.b()` and `seq.btensor()`; `calc_b` and `calc_btensor` are refused
  by name. The substrate spec's pool is `PoolSpec`, and the source of a columnar layout's bytes is
  `columnar.ColumnSource`; `Pool` and `Source` are refused by name with the new spelling.
- **Pairs with** dmipy-sim 3.0 (the producer of packs). It depends on no other dmipy package. Python >= 3.10, NumPy >= 2.0;
  the extras `pose` (SciPy, threadpoolctl), `jax`, `torch` and `all` open the routes that need them, and a route whose
  extra is missing is refused with its `pip install`.
- **Licence** Apache-2.0 for the code; each pack states its own terms in its header.
