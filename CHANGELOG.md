# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

- **An envelope states one maximum field strength.** `Envelope` holds `B0_max` (a number in tesla, or `None`) in the
  place of `B0_list`, and `envelope_of(scanner, *, B0_max=None, ...)` takes it in the place of `B0_list=()`; the
  property `B0_max` is now the field. A sequence, a string or a bool is refused by name. `to_dict()` gives `B0_max`.
  A header keeps its key `replay_envelope.acquisition.B0_list`, written with one entry (`[B0_max]`, empty for
  `None`); a header's list, and a stored dictionary's `B0_list` read by `Envelope.from_dict`, are read at their
  largest entry by one function that `Envelope.from_header` and `field_envelope_of` share. No stored file changes.
- **The Bloch route states its longitudinal model.** `replay_bloch` (numpy and JAX) multiplies each walker's `M_z` by
  `exp(-dt / T1)` per piece, the MT bound pool's by `exp(-dt / T1_bound)`, and adds no recovery toward equilibrium;
  a walker starts at equilibrium. Its docstrings say so and state the size of the omitted recovery on the reads it
  was measured on: none on an ideal 90-180 spin echo; about 1e-2 on an unencoded stimulated echo and from about 2e-4
  to 7e-3 on an encoded one, depending on how far the walkers are spread; the whole signal on a repeated excitation
  train.
- **A built pulsed pair states the timings it plays.** `Encoding.delta_built` and `Encoding.Delta_built` (seconds per
  measurement) of a `pgse` or `pgste` sequence are its pair as built, read off the stored samples of the effective
  gradient (`sequences.assemble.pair_timings`): the mean half-amplitude width of the first two lobes and the separation
  of their centres, by linear interpolation at half each lobe's peak; NaN on a row with no pair. `Encoding.delta` and
  `Encoding.Delta` are the nominal timings the builder was asked for, unchanged (`pgste`: `Delta = delta + TM`, edge to
  edge). Every other family, and `from_waveform`, states None for the built pair.
- **The time-segment read path refuses a waveform the stated slew cannot play.** `TimeSegmentLayout.fit`, `image` and
  `delivery` refuse a sequence whose played gradient steps faster than the slew limit they are given, by
  `ScannerSequence.check_slew` (the measurement, the axis, the step and the limit named). The limit is the call's
  `slew_max=` (T/m/s, one value or one per axis), else, for `delivery`, its scanner's `ScannerLimits.slew_max`, else the
  layout manifest's `source.envelope["slew_max"]`; with none stated no slew is checked. A delivery given a scanner now
  refuses a waveform that scanner's slew cannot play (a square pulse, for example); an idealised waveform is delivered
  by stating limits without a slew, `dataclasses.replace(limits, slew_max=None)`.
- **Every builder plays within the slew it is given.** At a finite `slew_rate` no step of a built sequence's played
  gradient, on any axis, exceeds the slew (`ScannerSequence.check_slew` refuses one that does, `ScannerSequence.peak_slew`
  reports the steps; every builder applies the check), for every ramp length, a ramp shorter than one step included:
  a ramped lobe keeps every sample whose mid-step lies inside its span, a train of lobes is one sampled waveform, and a
  self-refocusing pair is a lobe and its exact negative starting on the first step after the first lobe's ideal end
  (an axis pair is less than one step longer than its ideal). A ramped lobe's area is its ideal area to within
  `min(1, dt / (4 eps)) g dt`. An automatic `TE` at a finite slew is `minimum_te` (the continuous floor) extended by the
  fewest whole steps at which every block fits whole; an explicit `TE` that leaves a gradient-echo block less room is
  refused by name. `Encoding.qvalues` of a built sequence is the largest `|q(t)|` its effective gradient reaches along
  the measurement's direction, from the built samples (`assemble(state_q=)` in place of `q_width=`). The stated q
  changes: a ramped pulsed pair by the built area and amplitude (`pgse` 9 / 29 ms, b 7.5e9 at 600 T/m/s: +2.1e-5; the
  old nominal value exceeded the old built lobe by 1.7e-4); a square pulsed pair by its whole-step lobe (up to one
  step of the lobe, -2.6e-3 for `pgse` 9 / 29 ms); `cpmg` and `splice` by their ramps (-8e-3, -3e-3); the trapezoidal
  OGSE from `g sigma` to one lobe's q (about `1 / (2 f sigma)` of it: 5.11e4 -> 6.22e3 /m at 100 Hz, 40 ms); the cosine
  OGSE from `g sigma` to about a quarter period's (4.26e4 -> 7.04e3 /m at 50 Hz, 20 ms). The one reader of
  `qvalues` in the package, `set_b`, scales whatever is stated and is otherwise unchanged.
- **The `bound` column is read** (RPK.md 5.3, 6.5, 8.8). A pack that carries the C1 `bound` occupancy column is replayed with
  the joint occupancy `f_bound = b`, `f_g = (1 - b) [comp = g]` and the rate `R(t) = sum_c f_c(t) R_c`, exact over the runs of
  both columns, by `ReplayPack.replay`, `walker_signals`, `walker_phases`, the study route (`walker_primitives` holds one
  extra exposure column for the bound pool; `Primitives.signals` applies its rates) and `replay_bloch`. The occupancy columns
  the client reads are `comp` and `bound`. The bound pool's parameters are explicit keyword arguments: `T2_b` and `T1_b`
  (seconds) of `replay`, `walker_signals`, `walker_phases`, `replay_bloch` and `Primitives.signals`, and `delta_omega_b`
  (rad/s) of `replay_bloch` only, each required and refused by name on a pack that carries the column and refused on a
  pack that does not; there is no default. A pack without the column reads exactly as before on the scalar routes.
- **The Bloch route reads the exact occupancy runs of each piece** (RPK.md 9.5.7, 6.5). Each piece of the walk (the block
  grid cut at every RF instant) takes the occupancy of its own interval, the sum over the runs of the overlap of the run
  with the piece times the run's occupancy: a walker that changes pool inside a block spends each side of the change in its
  pool, and a block that a pulse splits is split at the pulse. Before, the label at each block's end was applied to the
  whole block. The propagators (`replay_bloch`, `replay_bloch_jax`) take `comp_traj` `(n_walkers, n_pieces, n_pools)` and
  `bound_frac` `(n_walkers, n_pieces)`, each the average over the piece of its pool's occupancy
  (`compression.occupancy_pieces`); the per-block-end spelling is removed. A walker with no occupancy event in the read
  window `(0, TE]` reads as before, bit for bit; a walker with one changes. The longitudinal relaxation of the propagator is
  unchanged.
- **A wall that binds states its bound pool's mean dwell** (`Wall.mt_dwell`, a `Dwell` of seconds per side).
  Stated exactly when a side's `mt_reactivity` is positive, positive on each side that binds and
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
