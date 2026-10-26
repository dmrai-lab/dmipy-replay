# Dedup ledger: dmipy-sim -> dmipy-replay

dmipy-replay is cut from dmipy-sim `main` 194dad6 (2026-10-21). Until the dedup branch on dmipy-sim merges, the code
below exists twice. Afterwards it exists only here, and sim imports it (an Apache-2.0 dependency of an AGPL package).
This file lists exactly what moved and every place the copy is not verbatim, so that the sim branch knows what to
delete and what to adapt.

## Moved verbatim (imports relative, `dmipy_sim` -> `dmipy_replay` in docstrings)

| dmipy-sim | dmipy-replay |
|---|---|
| `constants.py` | `constants.py` |
| `acquisition/{scanner_sequence, rf, timing, waveforms, prescription, epg, epg_state, scanners, scanner_constants, maxwell, solid_harmonics}.py`, `scanner_constants.json`, `scanner_catalogue.schema.json` | `acquisition/` |
| `sequences/{builders, assemble, _helpers}.py` | `sequences/` |
| `math/gradient_conversions.py` | `math/` |
| `spec/{substrate, tissue}.py`, `substrate.schema.json` | `spec/` |
| `replay/_replay_kernel.py` | `replay/_replay_kernel.py` |

## Moved as a subset (source text of each function exact)

| dmipy-sim | dmipy-replay | what moved |
|---|---|---|
| `replay/compression.py` | `replay/compression.py` | the reader: band projections, container decoding, `positions_at` / `local_time_at`, occupancy readers, `CumulativeGate`, `relaxation_logweight_runs`, `surface_logweight_bridge`, `rescale_bands`, the predicates |
| `replay/replay.py` | `replay/replay.py` | `read_rpk`, the refusals, the field rule (`field_term`, `scanner_field`, `declared_susceptibility_field`), `ReplayPack`'s read and replay methods, `compile_scheme`, `replay_signal`, `replay_coefficients`, `surface_logweight` |
| `replay/bank.py` | `replay/field.py` | `frame_from_axis`, `has_susc_path`, `susc_path_coeffs`, `susc_path_field`, `susc_path_decode`, `path_field_integral`, `_refuse_trace_component`, `_mode_sample_matrix` |
| `fields/hollow_cylinder.py` | `replay/field.py` | `CHANNEL_NAMES`, `ISO_CHANNEL_NAMES`, `q_of_H`, `iso_P6`, `iso_weights`, `field_terms`, `contract` |
| `replay/trajectories.py` | `replay/bloch.py` | `replay_bloch` (numpy) and its terms: `_rf_increment`, `finite_180_longitudinal_dwell`, `_BlochTerms`, `_bloch_timeline`, `_bloch_replay_terms`, `_bloch_replay_output` |
| `engine/bloch.py` | `replay/bloch.py` | `_build_crusher` |
| `coefficient_walk.py` | `replay/events.py` | `EventList`, `events_label_at`, `events_at_instants`, `events_within` |
| `replay/so3.py` | `replay/so3.py` | `rotation_of` |
| `replay/publish.py` | `replay/hub.py` | `parse_uri`, `is_hub_uri`, `fetch`, `fetch_windows`, `window_prefix`, `_coalesce`, `_window_tensor_names`, rewritten on `urllib` |

## Where the copy is not verbatim

1. `compression._band_ranges` reads declared ranges only. The `BandCurve` / `DerivedContainer` branches are
   producer-side; sim resolves them before calling (`resolve_container`). On dedup sim keeps both classes,
   `band_curve`, `derive_container`, `BAND_CONTAINER` and resolves first.
2. `bloch.replay_bloch` / `_bloch_replay_terms` take no `susceptibility=` provider (a JAX callable sampled along raw
   paths). The field reaches them as `extra_phase_per_step`, as a pack's replay already passes it. On dedup sim's
   `trajectories.replay_bloch` becomes a wrapper that samples its provider into `extra_phase_per_step` and calls
   this one; `replay_bloch_jax` stays in sim.
3. `ReplayPack.replay_bloch` takes no `jax=` (the JAX propagator is sim's). Sim's `ReplayPack` subclass adds it.
4. `replay._as_distribution` refuses a distribution of poses (`NotImplementedError`) until the pose expansion
   (`PoseResponse`, `pose_response(s)`, `pose_series`, `so3`, `fod`, `pose_device`) moves (phase 2).
5. `read_rpk` reads the container with `replay/container.py` (numpy; the safetensors wheel is a test oracle only).
   `write_rpk` is not here; sim keeps it, or calls `container.write_safetensors`.
6. `ReplayPack.load("hf://...")` goes through `hub.fetch` (cache `$DMIPY_REPLAY_CACHE`, else `~/.cache/dmipy-replay`,
   keyed by the dataset's commit; sha256 against the manifest at the same commit) instead of `hf_hub_download`.
7. `sequences/pulseq.py` is a native reader (formats 1.4 / 1.5). Sim's reader goes through pypulseq and names every
   foreign pulse 90 / 180 by its use; this one reads each pulse's played B1(t) into its envelope, so `flip_deg` is
   `gamma int |B1| dt` of the file (the `RFEvent` convention: a sinc 90's label reads ~145 deg, its net rotation is
   played by the Bloch route). Gradient, grid, readout and RF instants equal sim's on every fixture
   (`tests/test_pulseq.py`), including sim's half-raster widening of arbitrary-gradient spans. `to_pulseq`
   (export, pypulseq) stays in sim. `acquisition/timing.SequenceTiming.from_pulseq` now reads through the native
   reader.
8. Module docstrings of `replay.py`, `bloch.py`, `events.py`, `field.py`, `so3.py` are rewritten for the reader;
   `replay.__all__` lists what moved.

## Found on the way (sim defects, fix on the dedup branch)

- `coefficient_walk.EventList` defines `__iter__` twice (the second wins; same behaviour).
- `acquisition/rf.py:522`: `` `RFEvent`\ s `` is an invalid escape (SyntaxWarning on import).
- replay-pack-spec ACQUISITION.md 10.1 names dmipy-sim as Apache-2.0; it is AGPL-3.0 OR commercial. The reference
  implementation of ACQUISITION.md is now this package.
- A Pulseq grid instant exactly on an arbitrary gradient's edge reads the event's own end value (sim's rule, kept).
  The format leaves that one instant to the reader: the sample-and-hold convention would read the next event.

## The dedup branch on dmipy-sim (`dedup/replay`)

1. `pyproject`: `dmipy-replay` a dependency (git URL until it is on PyPI).
2. Delete the verbatim modules; `dmipy_sim.acquisition`, `.sequences` (builders), `.constants`, `.spec.tissue`,
   `.spec.substrate` re-export from `dmipy_replay` (one spelling: `dmipy_sim.pgse is dmipy_replay.pgse`).
3. `dmipy_sim.replay.replay.ReplayPack` subclasses `dmipy_replay.ReplayPack` and keeps the producer and device
   routes: `open` (columnar), `save`, `publish`, `truncate`, `end_state`, `extend`, `rewindow`,
   `restate_surface_envelope`, `prefix`, `pose_*`, `study`, `walker_primitives`; `BLOCH_BACKENDS` adds
   `"jax"` (`replay_bloch(backend="jax")`).
4. `compression`, `bank`, `trajectories`, `fields.hollow_cylinder`, `engine.bloch`, `coefficient_walk` import the
   moved functions instead of defining them.
5. `sequences/pulseq.py` keeps `to_pulseq` and imports `from_pulseq` / `pulseq_timing` from replay.
6. The sim suite green with dmipy-replay installed from this repository; `tests/test_api_surface.py` re-pointed.

Re-cut on a fresh sim `main`, from the sim checkout: `tools/dedup_sim.py --apply` (imports, module-name strings, the
`dmipy_sim.replay` package's re-exports, deletions), `tools/dedup_attrs.py --apply`, `tools/dedup_fix_names.py` until
it reports nothing, then `tools/dedup_check_names.py` (every sim-internal import resolves), pyflakes, the suite.
