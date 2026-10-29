# Dedup ledger: dmipy-sim -> dmipy-replay

dmipy-replay holds the reader side of dmipy-sim at sim `main` cd105e1 (2026-10-29). dmipy-sim imports it for every
definition below and keeps none of them (`tools/dedup_sim.py`, `dedup_attrs.py`, `dedup_classes.py`,
`dedup_fix_names.py` rewrite a sim checkout to that state). This file lists exactly what lives here and every place
the text is not sim's, so that a later sim change to a reader is brought here by `tools/dedup_sync.py`.

## Whole modules (imports relative, `dmipy_sim` -> `dmipy_replay` in docstrings)

| dmipy-sim | dmipy-replay |
|---|---|
| `constants.py` | `constants.py` |
| `acquisition/{scanner_sequence, rf, timing, waveforms, prescription, epg, epg_state, scanners, scanner_constants, maxwell, solid_harmonics}.py`, `scanner_constants.json`, `scanner_catalogue.schema.json` | `acquisition/` |
| `sequences/{builders, assemble, _helpers}.py` | `sequences/` |
| `math/gradient_conversions.py` | `math/` |
| `spec/{substrate, tissue}.py`, `substrate.schema.json` | `spec/` |
| `replay/{_replay_kernel, decoded, planning}.py` | `replay/` |
| `phantom/grid.py` | `phantom/grid.py` |

## Subsets (the source text of each definition sim's, its imports pointed at the replay module that defines the name)

| dmipy-sim | dmipy-replay | what lives here |
|---|---|---|
| `replay/compression.py` | `replay/compression.py` | the reader: band projections (`segment_projection` with shapes), container decoding (`unpack_bits`, `_load_range`, `_unpack_from`, `_range_values`), the decoded-once and prefix reads (`_decoded_once`, `_decoded_prefix`, `read_position_coeffs(bands=)`, `_read_position_coeffs`), the stored shapes (`position_shape`, `position_starts`, `_position_tensors`, `_no_position_axes`), `walker_rows`, `_per_walker`, `positions_at` / `local_time_at`, occupancy readers (`occupancy_events`, `_occupancy_events`, `_occupancy_runs`), `bridge_bands`, `CumulativeGate`, `relaxation_logweight_runs`, `surface_logweight_bridge`, `rescale_bands`, the predicates |
| `replay/replay.py` | `replay/replay.py` | `read_rpk(cls=)`, the refusals, the field rule (`field_term(envelope=)`, `scanner_field`, `declared_susceptibility_field`), `truncation_budget`, `NUMERICAL_TERMS`, `_measured_mode_variance`, `ReplayPack`'s read, plan and replay methods (`open`, `arrays` / `decode_budget` / `release_decoded`, `plan`, `_read_plan`, `_full_plan`, `_mode_variance`, `field_envelope`, `truncation_budget`, `waveform_band`, `_check_band`, `walkers`, `replay(tol=, K=, modes=)` and the per-walker routes), `compile_scheme`, `replay_signal`, `replay_coefficients`, `surface_logweight` |
| `replay/bank.py` | `replay/field.py` | `frame_from_axis`, `has_susc_path`, `susc_path_coeffs(modes=)`, `_susc_path_coeffs`, `_susc_path_names`, `_susc_path_expanded_names`, `path_mode_count`, `susc_path_field`, `susc_path_decode`, `path_field_integral(modes=)`, `_refuse_trace_component`, `_mode_sample_matrix` |
| `fields/hollow_cylinder.py` | `replay/field.py` | `CHANNEL_NAMES`, `ISO_CHANNEL_NAMES`, `q_of_H`, `iso_P6`, `iso_weights`, `field_terms`, `contract` |
| `replay/bank.py` | `replay/storage.py` | `STORAGE_BAND_MARGIN`, `storage_band_hz`, `mode_count` |
| `replay/trajectories.py` | `replay/bloch.py` | `replay_bloch` (numpy) and its terms: `_rf_increment`, `finite_180_longitudinal_dwell`, `_BlochTerms`, `_bloch_timeline`, `_bloch_replay_terms`, `_bloch_replay_output` |
| `engine/bloch.py` | `replay/bloch.py` | `_build_crusher` |
| `coefficient_walk.py` | `replay/events.py` | `EventList`, `events_label_at`, `events_at_instants`, `events_within` |
| `replay/so3.py` | `replay/so3.py` | `rotation_of` |
| `replay/study.py` | `replay/primitives.py` | `Primitives` (the one reduction a (tissue, scanner) pair is applied by) |
| `replay/columnar.py` | `replay/columnar.py` | `Source` (on `urllib`), `ColumnarPack` (plan, `truncation_budget`, views, image), `open_columnar` |
| `replay/remote.py` | `replay/remote.py` | `RemotePack` (on `urllib`), `_Skeleton`, `_Stub` |
| `replay/time_segments.py` | `replay/time_segments.py` | the format (`MANIFEST`, `FORMAT`, `TILE`, `WORD_BITS`, `read_manifest`), the words (`_decode`, `unpack_words`, `_layout_of`), `Shape`, `Partition`, `_lobes`, `_group_key`, `_field_of_tiers`, `Delivered`, the row arithmetic (`_field_terms_of_words`, `_field_offset`, `_phase`, `_row_terms`), `TimeSegmentLayout` without `delivery`, and a numpy backend |
| `replay/publish.py` | `replay/hub.py` | `parse_uri`, `is_hub_uri`, `fetch`, `fetch_windows`, `window_prefix`, `_coalesce`, `_window_tensor_names`, on `urllib`; `fetch_tree` |
| `sequences/pulseq.py` | `sequences/pulseq.py` | `from_pulseq`, `pulseq_timing`: a native reader |

## Where the text is not sim's

1. `compression._band_ranges` reads declared ranges only. The `BandCurve` / `DerivedContainer` branches are
   producer-side; sim resolves them before calling (`resolve_container`).
2. `bloch.replay_bloch` / `_bloch_replay_terms` take no `susceptibility=` provider (a JAX callable sampled along raw
   paths). The field reaches them as `extra_phase_per_step`, as a pack's replay passes it. Sim's
   `trajectories.replay_bloch` samples its provider into `extra_phase_per_step` and calls this one;
   `replay_bloch_jax` is sim's.
3. `ReplayPack.replay_bloch(backend=)` runs a propagator of `BLOCH_BACKENDS` (`"numpy"` here); sim's subclass adds
   `"jax"`.
4. `ReplayPack._as_distribution` refuses a distribution of poses (`NotImplementedError`): the pose expansion
   (`PoseResponse`, `pose_response(s)`, `pose_series`, `so3.Distribution`, `fod`, `pose_device`) is sim's, and sim's
   subclass reads one. `ReplayPack.window` / `walkers` build `type(self)`.
5. `read_rpk` reads the container with `replay/container.py` (numpy; the file mapped copy-on-write) and builds `cls`.
   `write_rpk` is sim's.
6. `ReplayPack.load("hf://...")` goes through `hub.fetch` (cache `$DMIPY_REPLAY_CACHE`, else `~/.cache/dmipy-replay`,
   keyed by the dataset's commit; sha256 against the manifest at the same commit) instead of `hf_hub_download`.
   `ReplayPack.open` routes `hf://.../x.rpk` to `RemotePack` and anything else to `open_columnar`; sim's subclass
   routes to its own subclasses of both.
7. `sequences/pulseq.py` is a native reader (formats 1.4 / 1.5). Sim's reader went through pypulseq and named every
   foreign pulse 90 / 180 by its use; this one reads each pulse's played B1(t) into its envelope, so `flip_deg` is
   `gamma int |B1| dt` of the file (the `RFEvent` convention: a sinc 90's label reads ~145 deg, its net rotation is
   played by the Bloch route). Gradient, grid, readout and RF instants equal sim's on every fixture
   (`tests/test_pulseq.py`). `to_pulseq` (export, pypulseq) is sim's.
8. `columnar.Source` reads one pinned commit on `urllib` and refuses a range request answered with anything but 206
   (retried, never sliced). `ColumnarPack.image` forms the voxel sums through `IMAGE_BACKENDS` (`"numpy"` here; sim
   adds `"jax"`, its default), takes `voxels=`, and `view` builds `pack_class` (sim sets its `ReplayPack`). Added
   here: `plan_for_class` / `bands_for_class` and `pack_license`. `image_study` (a `Study`) is sim's.
9. `remote.RemotePack` reads ranges with `hub._get` at a commit resolved once (`revision=`, `main` by default; no
   `session=`), builds `pack_class` (sim sets its `ReplayPack`), and `load()` is `hub.fetch` at that commit.
10. `time_segments.TimeSegmentLayout.image` / `tier_maps` run on a backend of `BACKENDS` (an object: `device`, `put`,
    `context`, `field_spec`, `sums`, `voxel_sums`), `"numpy"` here and `DEFAULT_BACKEND = "numpy"`; the numpy
    backend is the same float32 phase / complex64 exponential / complex128 sum arithmetic as sim's JAX kernel, in
    another summation order. Sim's subclass adds `"jax"` (its default) and `"torch"` and `delivery` (the bore).
    `open` is a classmethod on `hub.fetch_tree`. The row helpers take numpy or JAX arrays through their own array
    namespace (`w.__array_namespace__()`); torch tensors keep their branch.
11. Module docstrings of `replay.py`, `bloch.py`, `events.py`, `field.py`, `so3.py`, `remote.py`, `time_segments.py`,
    `primitives.py` are written for the reader; `replay.__all__` lists what lives here.
12. `acquisition/scanner_constants.json` names `dmipy_replay.replay.storage.storage_band_hz` where sim names
    `dmipy_sim.replay.bank.storage_band_hz` (the storage rule lives in `replay/storage.py`).

## Sim defects found on the way

- `coefficient_walk.EventList` defines `__iter__` twice (the second wins; same behaviour).
- `acquisition/rf.py:522`: `` `RFEvent`\ s `` is an invalid escape (SyntaxWarning on import).
- replay-pack-spec ACQUISITION.md 10.1 names dmipy-sim as Apache-2.0; it is AGPL-3.0 OR commercial. The reference
  implementation of ACQUISITION.md is this package.
- A Pulseq grid instant exactly on an arbitrary gradient's edge reads the event's own end value (sim's rule, kept).
  The format leaves that one instant to the reader: the sample-and-hold convention would read the next event.

## Bringing a later sim `main` here, and sim to this package

From this repository: `python tools/dedup_sync.py SIM_CHECKOUT cd105e1 [--apply] [--add module:NAME ...]` lists every
definition sim changed since this state (`--apply` takes sim's text, re-pointing its imports), every one sim added,
and every one that diverged here; then the fixtures are recorded again from a pristine sim checkout
(`tools/make_fixtures.py`, `make_columnar_fixtures.py`, `make_segment_fixtures.py`) and the suite run.

From a sim checkout on a fresh `main`: `tools/dedup_sim.py --apply` (imports, module-name strings, the
`dmipy_sim.replay` package's re-exports, deletions), `tools/dedup_attrs.py --apply`, `tools/dedup_classes.py
--apply` (sim's `ReplayPack`, `ColumnarPack`, `TimeSegmentLayout` become subclasses), `tools/dedup_fix_names.py`
until it reports nothing, `tools/dedup_check_names.py` (every sim-internal import resolves), then the hand-written
parts: `read_rpk` (`cls=ReplayPack`), `ReplayPack.BLOCH_BACKENDS` / `_as_distribution`, `columnar.py` (the subclass,
the JAX voxel sums), `remote.py` (the subclass), the time-segment backends, `trajectories.replay_bloch` (the wrapper),
the `replay_bloch(jax=)` callers (`backend=`), and the suite in groups.
