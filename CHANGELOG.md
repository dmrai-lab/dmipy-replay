# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### 0.2.0

dmipy-replay reads replay packs: NumPy only, no walk, no GPU.

- **Packs.** The Hub client (a dataset revision pinned once, sha256 checked against the manifest, ranged HTTP reads),
  the `.rpk` container, `ReplayPack.replay` for any gradient waveform, tissue (T2, T1, surface relaxivity,
  susceptibility), field strength and orientation, `replay_bloch` for RF schedules and B1 scale, and `plan` /
  `truncation_budget` to state a read before any band is decoded.
- **Grids.** The columnar layout and the time-segment layout read a whole grid of voxels by reference.
- **Acquisition.** `ScannerSequence`, the sequence builders, the Pulseq reader, the scanner catalogue
  (`ScannerLimits`) and the substrate and tissue specification.
- **The guarantee.** A pack states the scanner class it is certified for. Every replay route refuses a request outside
  it by name: a gradient above `G_max`, a slew above `slew_max`, more refocusing pulses than the class allows, a field
  strength or relaxivity beyond the certified scale, a waveform beyond the stated numerical cap. A pack that states no
  class is refused.
- **Pairs with** [dmipy-sim](https://github.com/dmrai-lab/dmipy-sim) 3.0 (the producer of packs). Python >= 3.10,
  NumPy >= 2.0; `safetensors` only in the `test` extra.
- **Licence** Apache-2.0.

> **Pending, not yet landed (this entry is completed before the tag).**
> The pose replay (`pose_response(s)`, `pose_series`, the device kernels, `so3`), `pathways` and `ReplayPhantom`
> (with `fod` and `study`) move here from dmipy-sim, with outputs checked bit-identical. The physical `Envelope`
> (`Envelope(band_hz, G_max, field and tissue bounds)`, `envelope_of(scanner)` presets; the reader checks band and
> amplitude) lands here.
