"""The time-segment layout read here against dmipy-sim's own reads of the same layout (``data/segment_reference.json``, recorded at dmipy-sim ``8d7b97f`` and frozen):
the words and the tiers are the same code and the same bits, the fits the same to the least-squares solver's rounding; the image is the same float32 arithmetic on
numpy that sim's kernel runs on JAX, equal to the float32 rounding of the phases."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_replay.replay.time_segments import Partition, TimeSegmentLayout
from segment_scenarios import TISSUE_KW, acquisitions

DATA = Path(__file__).parent / "data"
REF = np.load(DATA / "segment_reference.npz")
INFO = json.loads((DATA / "segment_reference.json").read_text())
NS = SimpleNamespace(pgse=dr.pgse, pgste=dr.pgste, ogse=dr.ogse, ScannerSequence=dr.ScannerSequence, Partition=Partition,
                     Tissue=dr.Tissue)
ACQ = acquisitions(NS)
RTOL, ATOL = 2e-5, 2e-6                       # float32 phases and complex64 exponentials, summed in another order


@pytest.fixture(scope="module")
def lay():
    return TimeSegmentLayout(str(DATA / "time_segments"))


def test_the_words_decode_to_sims_vectors_and_tiers(lay):
    for name in lay.partitions:
        np.testing.assert_array_equal(lay.vectors(name), REF[f"vectors/{name}"])
    for g in lay.manifest["groups"]:
        for k, v in lay.tier_columns(g).items():
            np.testing.assert_array_equal(v, REF[f"tiers/{g}/{k}"])


@pytest.mark.parametrize("key", sorted(ACQ))
def test_a_sequence_fits_sims_partition_at_sims_levels(lay, key):
    name, L = lay.fit(ACQ[key][1])
    assert name == INFO[f"{key}/partition"] == ACQ[key][0]
    # shaped and overlapping terms take a least-squares fit, whose last bits are the LAPACK build's
    np.testing.assert_allclose(L, REF[f"{key}/levels"], rtol=1e-10, atol=1e-12 * float(np.abs(REF[f"{key}/levels"]).max()))


@pytest.mark.parametrize("key", sorted(ACQ))
def test_the_numpy_image_is_sims(lay, key):
    S, floor = lay.image(ACQ[key][1])
    np.testing.assert_allclose(S, REF[f"{key}/S"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(floor, REF[f"{key}/floor"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("key", sorted(ACQ))
def test_the_tiered_image_is_sims(lay, key):
    kw = dict(tissue=dr.Tissue(**TISSUE_KW), scanner=3.0, b0_direction=(0.6, 0.0, 0.8))
    if f"{key}/S_tiers" in INFO["failed"]:
        with pytest.raises(Exception) as e:
            lay.image(ACQ[key][1], **kw)
        assert type(e.value).__name__ == INFO["failed"][f"{key}/S_tiers"].split(":")[0]
        return
    S, _ = lay.image(ACQ[key][1], **kw)
    np.testing.assert_allclose(S, REF[f"{key}/S_tiers"], rtol=RTOL, atol=ATOL)


def test_a_chunked_read_is_the_resident_read(lay):
    seq = ACQ["box"][1]
    a, _ = lay.image(seq)
    b, _ = lay.image(seq, resident=False)
    np.testing.assert_array_equal(a, b)


def test_the_tier_maps_are_sims(lay):
    maps = lay.tier_maps("box", dr.Tissue(**TISSUE_KW), 3.0, b0_direction=(0.6, 0.0, 0.8))
    for k, v in maps.items():
        if isinstance(v, dict):
            for p, m in v.items():
                np.testing.assert_allclose(m, REF[f"maps/{k}/{p}"], rtol=1e-12, atol=1e-15)
        elif v is not None:
            np.testing.assert_allclose(v, REF[f"maps/{k}"], rtol=1e-5, atol=1e-7)


def test_a_backend_it_does_not_have_is_refused_by_name(lay):
    with pytest.raises(ValueError, match="backend is one of"):
        lay.image(ACQ["box"][1], backend="cupy")


def test_a_waveform_off_the_stored_shapes_is_refused(lay):
    seq = ACQ["trap"][1]
    G = np.asarray(seq.G, np.float32).copy()
    on = np.flatnonzero(np.abs(G).max(axis=(0, 2)) > 0)
    mid = on[len(on) // 4]
    G[:, mid:mid + 5] *= 1.3                                           # a bump on the first lobe's plateau
    with pytest.raises(ValueError, match="fits no stored partition"):
        lay.fit(dr.ScannerSequence(G=G, dt=seq.dt, rf=seq.rf, readout=seq.readout))


def test_a_hub_layout_opens_from_the_files_its_partitions_need(tmp_path, monkeypatch):
    """``open("hf://...")`` pins the revision once and fetches the manifest, then the shared columns, the tiers and the
    words of the partitions asked for, into the cache under that commit, and nothing else."""
    import shutil
    from dmipy_replay.replay import hub
    monkeypatch.setenv("DMIPY_REPLAY_CACHE", str(tmp_path))
    asked = []
    monkeypatch.setattr(hub, "revision_of", lambda repo, revision="main": "c0ffee")

    def download(url, dest):
        rel = url.split("/resolve/c0ffee/", 1)[1].split("/", 1)[1]
        asked.append(rel)
        shutil.copy(DATA / "time_segments" / rel, dest)
    monkeypatch.setattr(hub, "_download", download)
    lay = TimeSegmentLayout.open("hf://owner/name/time_segments", partitions=["box"])
    assert asked[0] == "manifest.json" and not any(a.startswith("p_") and a != "p_box.npy" for a in asked)
    S, _ = lay.image(ACQ["box"][1])
    np.testing.assert_allclose(S, REF["box/S"], rtol=RTOL, atol=ATOL)
