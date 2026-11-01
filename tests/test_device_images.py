"""The study and the device images on the committed fixtures: a study on a pack is the replay of each pair, a columnar
layout's study image is the per-pair image, and JAX and torch image a layout as numpy does, to the float32 rounding
of the phases. NumPy is the default everywhere; JAX and torch run where they are stated."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_replay.replay.study import Acquisition, Study, StudyProtocol

DATA = Path(__file__).parent / "data"


def _cols():
    from columnar_scenarios import seqs, tissue_of
    lay = dr.open_columnar(str(DATA / "columnar_quant"))
    ns = SimpleNamespace(pgse=dr.pgse, set_b=dr.set_b, Tissue=dr.Tissue)
    return lay, seqs(ns), tissue_of(ns, lay.meta)


def test_a_study_on_a_pack_is_the_replay_of_every_pair():
    pk = dr.read_rpk(DATA / "cylinder.rpk")
    T = pk.T_window
    a = dr.pgse([[0, 0, 1], [1, 0, 0]], 0.2 * T, 0.5 * T, bvalues=[1e9, 1e9], TE=0.95 * T, slew_rate=np.inf)
    b = dr.pgse([[0, 1, 0]], 0.2 * T, 0.5 * T, bvalues=[5e8], TE=0.95 * T, slew_rate=np.inf)
    tissue = dr.Tissue(T2={p["name"]: 0.07 for p in pk.meta["substrate"]["pools"]})
    S = pk.study(Study(StudyProtocol([Acquisition(a), Acquisition(b)]), tissues=(None, tissue)))
    for k, t in enumerate((None, tissue)):
        ref = np.concatenate([np.ravel(pk.replay(s, tissue=t)) for s in (a, b)])          # both at the plan's bands
        np.testing.assert_allclose(S[k], ref, rtol=0, atol=1e-12)


def test_the_study_image_is_the_per_pair_image():
    lay, S, tissue = _cols()
    st = Study(StudyProtocol([Acquisition(s) for s in S.values()]), tissues=(None, tissue))
    img, _floor, plan = lay.image(st)                                   # a Study goes to image_study
    for k, t in enumerate((None, tissue)):
        ref = lay.image(list(S.values()), tissue=t)[0]                      # one pass, the protocol's widest band
        np.testing.assert_allclose(img[k], ref, rtol=1e-10, atol=1e-12, equal_nan=True)


def test_jax_images_a_columnar_layout_as_numpy_does():
    pytest.importorskip("jax")
    lay, S, tissue = _cols()
    for s in S.values():
        a, b = lay.image(s, settings=[(None, None), (tissue, 3.0)])[0], lay.image(s, settings=[(None, None), (tissue, 3.0)], backend="jax")[0]
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-12, equal_nan=True)
    st = Study(StudyProtocol([Acquisition(s) for s in S.values()]), tissues=(None, tissue))
    np.testing.assert_allclose(lay.image_study(st, backend="jax")[0], lay.image_study(st)[0], rtol=1e-10, atol=1e-12,
                               equal_nan=True)


@pytest.mark.parametrize("backend,device", [("jax", None), ("torch", "cpu")])
def test_jax_and_torch_image_a_time_segment_layout_as_numpy_does(backend, device):
    pytest.importorskip(backend)
    from dmipy_replay.replay.time_segments import Partition, TimeSegmentLayout
    from segment_scenarios import acquisitions
    ns = SimpleNamespace(pgse=dr.pgse, pgste=dr.pgste, ogse=dr.ogse, ScannerSequence=dr.ScannerSequence,
                         Partition=Partition, Tissue=dr.Tissue)
    lay = TimeSegmentLayout(str(DATA / "time_segments"))
    for key, (_part, seq) in sorted(acquisitions(ns).items()):
        S, floor = lay.image(seq)
        S2, floor2 = lay.image(seq, backend=backend, device=device)
        np.testing.assert_allclose(S2, S, rtol=2e-5, atol=2e-6, equal_nan=True)   # float32 phases, summed in another order
        np.testing.assert_allclose(floor2, floor, rtol=2e-5, atol=2e-6, equal_nan=True)


def test_an_unknown_study_backend_is_refused_by_name():
    lay, S, _t = _cols()
    with pytest.raises(ValueError, match="'numpy' or 'jax'"):
        lay.image_study(Study(StudyProtocol([Acquisition(next(iter(S.values())))])), backend="cupy")
