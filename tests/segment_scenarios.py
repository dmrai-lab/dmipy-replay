"""The partitions and acquisitions of the time-segment fixtures, written once against a namespace (``ns`` carries the
builders, ``ScannerSequence``, ``Partition`` and ``Tissue`` of whichever package runs them), shared by the fixture tool
and the tests. One 0.1 ms grid over a 50 ms echo."""
import numpy as np

N_T, TE = 501, 0.05
TISSUE_KW = dict(T2={"intra": 0.07, "extra": 0.06, "myelin": 0.015}, T1={"intra": 1.0, "extra": 1.2, "myelin": 0.4},
                 rho2=2e-5, chi_iso=-1e-7, chi_aniso=-1e-8)
_I0, _I1 = 60, 210                                                   # the tensor encoding's interval (samples)


def templates(ns):
    """A square PGSE, a PGSE with 0.25 ms ramps, a cosine OGSE, a tensor encoding's three shapes (three whole sines on one
    interval before the 180, one per row along x), a stimulated echo, and a double diffusion encoding (two bipolar
    pairs about the spin echo's pulse)."""
    box = ns.pgse([[0, 0, 1]], 4e-3, 10e-3, gradient_strengths=0.1, TE=TE, n_t=N_T, slew_rate=np.inf)
    trap = ns.pgse([[0, 0, 1]], 8e-3, 20e-3, gradient_strengths=0.05, TE=TE, n_t=N_T, slew_rate=200.0)
    cos = ns.ogse([[0, 0, 1]], 50.0, 0.02, shape="cosine", gradient_strengths=0.05, TE=TE, n_t=N_T, slew_rate=200.0)
    k = np.arange(_I1 - _I0)
    G = np.zeros((3, N_T, 3), np.float32)
    for j in range(3):
        G[j, _I0:_I1, 0] = 0.05 * np.sin(2.0 * np.pi * (j + 1) * k / (_I1 - _I0))
    tensor = ns.ScannerSequence(G=G, dt=box.dt, rf=box.rf, readout=box.readout)
    ste = ns.pgste([[0, 0, 1]], 3e-3, 10e-3, gradient_strengths=0.1, n_t=N_T, slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))
    G = np.zeros((1, N_T, 3), np.float32)
    for i0, v in ((5, [0, 0, 0.1]), (35, [0, 0, -0.1]), (255, [0.1, 0, 0]), (285, [-0.1, 0, 0])):
        G[0, i0:i0 + 25, :] = v
    dde = ns.ScannerSequence(G=G, dt=box.dt, rf=box.rf, readout=box.readout)
    return dict(box=box, trap=trap, cos=cos, tensor=tensor, pgste=ste, dde=dde)


def partitions(ns):
    t = templates(ns)
    ste = t["pgste"]
    lobes = ns.Partition.of("lobes", ste).segments
    eg = np.asarray(ste.effective_gate)
    z = np.flatnonzero(eg == 0)
    z = z[(z * ste.dt > lobes[0][1]) & (z * ste.dt < lobes[1][0])]
    mixing = (z[0] * ste.dt, (z[-1] + 1) * ste.dt)
    i0, i1 = _I0 * t["tensor"].dt, _I1 * t["tensor"].dt
    return {"box": ns.Partition.of("box", t["box"]), "trap": ns.Partition.from_sequence("trap", t["trap"]),
            "cos": ns.Partition.from_sequence("cos", t["cos"]),
            "tensor": ns.Partition.from_sequence("tensor", t["tensor"], terms=[(i0, i1)] * 3, rows=[0, 1, 2]),
            "pgste": ns.Partition.of("pgste", ste, segments=[lobes[0], mixing, lobes[1]]),
            "dde": ns.Partition.of("dde", t["dde"])}


def _rows(template, G):
    return type(template)(G=np.asarray(G, np.float32), dt=template.dt, rf=template.rf, readout=template.readout)


def _directions():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(5, 3))
    return u / np.linalg.norm(u, axis=1)[:, None]


def _play(template, scales):
    """``template``'s one shape along each direction at each scale of its amplitude: ``(n, n_t, 3)`` rows."""
    prof = np.asarray(template.G, np.float64)[0, :, 2]
    u = _directions()
    return _rows(template, np.concatenate([prof[None, :, None] * (s * u)[:, None, :] for s in scales]))


def acquisitions(ns):
    """``{key: (partition, sequence)}``: each partition played at other amplitudes and directions (and the tensor
    encoding at other weights and rotations)."""
    t = templates(ns)
    u = _directions()
    b = np.r_[np.full(5, 1e9), np.full(5, 3e9), 0.0]
    dirs = np.concatenate([u, u, [[0, 0, 1]]]).tolist()
    c, s = np.cos(0.6), np.sin(0.6)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    Gt = np.asarray(t["tensor"].G, np.float64)
    weights = np.array([[1.0, 0.0, 0.0], [0.3, 0.5, -0.2], [0.0, 0.7, 0.7]])
    tens = np.stack([np.einsum("j,jtc->tc", wv, Gt) for wv in weights])
    Gd = np.asarray(t["dde"].G, np.float64)[0]
    return {
        "box": ("box", ns.pgse(dirs, 4e-3, 10e-3, bvalues=b, TE=TE, n_t=N_T, slew_rate=np.inf)),
        "trap": ("trap", _play(t["trap"], (0.5, 1.0, 1.5))),
        "cos": ("cos", _play(t["cos"], (0.4, 1.0))),
        "tensor": ("tensor", _rows(t["tensor"], np.concatenate([tens, tens @ R.T]))),
        "pgste": ("pgste", ns.pgste(dirs, 3e-3, 10e-3, bvalues=b, n_t=N_T, slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))),
        "dde": ("dde", _rows(t["dde"], np.stack([Gd * a for a in (0.3, 0.7, 1.0)] + [Gd * np.array([[-1.0, 1.0, 1.0]]) * 0.5]))),
    }
