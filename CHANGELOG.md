# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

- **A wall that binds states its bound pool's mean dwell** (`Wall.mt_dwell`, a `Dwell` of seconds per side;
  dmrai-lab/dmrai#72). Stated exactly when a side's `mt_reactivity` is positive, positive on each side that binds and
  0 on each side that does not, refused by name otherwise; a time, so `SubstrateSpec.scaled` keeps it. A wall that
  binds nothing writes none: every existing spec reads and writes as it did.
- **A time view at any diffusivity** (`ReplayPack.at_diffusivity`, `Tissue(D=)`, `Tissue(kappa=)`): a view at
  `a = D / D_walk` is the walk at the mapped request -- `G(t)` on the view is `G(t' / a) / a` on the walk, `B0 chi` is
  `B0 chi / a`, `rho2` is `rho2 / a` -- so its envelope is the walk's moved by `a` (`band_hz` and `G_max` times `a`, and a
  slower view's `chi_max` times `a`) and every request inside it is certified by the walk's own terms. A slower view
  (`a < 1`) is served: its band is `a` times the walked band, and a waveform beyond it, a gradient beyond `a` times
  `G_max` or a field scale beyond the moved one is refused by name, the refusal naming the view.
  `provenance.diffusivity_scaled` records `a` cumulative over repeated views.
- **Filter-exchange imaging** (`sequences.fexi`, `fexi_protocol`, `fexi_bvalues`): FEXI as one `ScannerSequence` -- a
  bipolar filter, a crusher, the store, the mixing time `t_m`, the recall, a crusher, a bipolar detection; three 90s and
  no 180, each block refocused on its own, the crushers declared before the store and after the recall to select the
  stimulated echo. `b_f` (the filter's b) and `bvalues` (the detection's) per row, realised exactly; `fexi_protocol` is
  the filter off and on at every mixing time and detection b-value (one sequence per mixing time); `fexi_bvalues` reads
  each row's two b-values back off the object and refuses a sequence whose encoding runs across the storage (a PGSTE).
- **A wall's permeability per instance and over time** (`spec.substrate.PermeabilityTable`, `Wall.permeability_table`):
  piecewise constant in time, one value per instance of the wall's surface in each row and each crossing direction.
  The validator holds it to its shape (positive, strictly increasing breakpoints; one row per interval; one value per
  instance) and the wall's `permeability` to its largest value per direction; `SubstrateSpec.scaled` multiplies its
  values as rates and keeps its breakpoints. The schema carries it.

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
- **The forms are dmipy-analytic's.** The physical constants (`GAMMA`, `GAMMA_BAR`), the real spherical-harmonic basis, the
  Watson and Bingham forms and the extended phase graph are [dmipy-analytic](https://github.com/dmrai-lab/dmipy-analytic)'s,
  and dmipy-replay imports them. The moved names are refused by name with their new spelling (`dmipy_replay.GAMMA` says
  `from dmipy_analytic.constants import GAMMA`, and the moved names of `constants`, `replay.so3` and `acquisition.epg`
  each say theirs), and `math.sh_analytical` and `acquisition.epg_state`, which moved whole, are gone.
- **Refused by name.** A pack written by an older producer (re-walk it: a pack is never converted) and a tier the pack
  does not carry.
- **Names.** A sequence's b-value and b-tensor are `seq.b()` and `seq.btensor()`; `calc_b` and `calc_btensor` are refused
  by name. The substrate spec's pool is `PoolSpec`, and the source of a columnar layout's bytes is
  `columnar.ColumnSource`; `Pool` and `Source` are refused by name with the new spelling.
- **Pairs with** dmipy-sim 3.0 (the producer of packs). It depends on dmipy-analytic `>=3,<4`, which imports with NumPy
  alone. Python >= 3.10, NumPy >= 2.0; the extras `pose` (SciPy, threadpoolctl, `dmipy-analytic[special]`), `jax`, `torch`
  and `all` open the routes that need them, and a route whose extra is missing is refused with its `pip install`.
- **Licence** Apache-2.0 for the code; each pack states its own terms in its header.
