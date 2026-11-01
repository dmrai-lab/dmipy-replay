"""The call table of the replay move's parity record (dmrai-lab/dmipy-sim#834): thirteen groups of calls on the
committed fixtures of ``tests/data``, each a function of a spelling (:mod:`apis`) returning ``{name: value}``.

A value is an array, a number, a string, ``None``, bytes (recorded by their sha256) or a dict/list of these
(flattened by :func:`record`). Timing fields are recorded by their keys only (``seconds/...`` names). Every call states
its backend, so a default that changes with the move (the image and pose backends are NumPy in dmipy-replay) does not
change what is compared."""
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
FIXTURES = ("cylinder", "permeable", "windows", "field")
B = 1e9


def _dirs(n, seed=0):
    u = np.random.default_rng(seed).normal(size=(n, 3))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def _rot(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


R1 = _rot([1.0, 2.0, 0.5], 0.7)
R2 = _rot([0.0, 1.0, 1.0], -1.1)


def _pack(api, name):
    return api.read_rpk(str(DATA / f"{name}.rpk"))


def _pgse(api, T, dirs=((0.0, 0.0, 1.0),), b=B, n_t=None):
    """A pulsed gradient spin echo inside a window of ``T`` seconds: lobes of ``T/5`` at ``T/2`` apart."""
    kw = {} if n_t is None else dict(n_t=n_t)
    return api.pgse([list(d) for d in dirs], 0.2 * T, 0.5 * T, bvalues=[b] * len(dirs), TE=0.95 * T, slew_rate=np.inf, **kw)


def _response(r):
    out = dict(coeffs=r.coeffs, misfit=r.misfit, floor=r.floor, lmax=r.lmax, nmax=r.nmax,
               phase_amplitude=r.phase_amplitude, route=r.route, n_bodies=r.n_bodies, field_lmax=r.field_lmax)
    try:
        out["asymmetry"] = r.asymmetry
    except Exception as exc:                                        # a response that states none says why
        out["asymmetry"] = f"{type(exc).__name__}: {exc}"
    return out


# ------------------------------------------------------------------------------------------------------ the groups
def g01_so3(api):
    """The SO(3) and spherical-harmonic basis, the distributions and the FOD."""
    api.require("so3")
    so3, sh = api.so3, api.sh_analytical
    d = _dirs(200)
    R = np.stack([R1, R2])
    out = {"real_sh": so3.real_sh(8, d), "real_sh_full": so3.real_sh(6, d, full=True),
           "sphere_quadrature": so3.sphere_quadrature(9, 17), "spherical_jn_all": so3.spherical_jn_all(10, np.linspace(0, 30, 61)),
           "legendre": so3.legendre(5, np.linspace(-1, 1, 21)), "so3_index": so3.so3_index(6, 3)}
    for l, M in enumerate(so3.wigner_blocks(6, R)):
        out[f"wigner_blocks/{l}"] = M
    for L in range(5):
        out[f"clebsch_gordan/{L}"] = so3.clebsch_gordan(2, 2, L)
    out["coupling"] = so3.coupling(2, 2)
    out["so3_design"] = so3.so3_design(8, R, 4)
    out["haar_rotations"] = so3.haar_rotations(32, seed=0)
    out["rotations_from_quaternions"] = so3.rotations_from_quaternions(_dirs(4, 1) @ np.eye(3)[:, :3].T @ np.ones((3, 4)) / 3)
    out["rotate_sh"] = so3.rotate_sh(so3.real_sh(6, d[:3]), R1, lmax=6)
    Rq, w, *_ = so3.so3_quadrature(6, 3)
    out["so3_quadrature/R"], out["so3_quadrature/w"] = Rq, w
    out["oversampled_design"] = so3.oversampled_design(4, 2, 3)[-1]
    out["delta_coeffs"] = so3.delta_coeffs(R1, 8, 4)
    out["axis_coeffs"] = so3.axis_coeffs([0.3, 0.4, 0.866], 8, 4)
    out["watson_coeffs"] = so3.watson_coeffs(8.0, (0.0, 0.6, 0.8), 8, 4)
    out["bingham_coeffs"] = so3.bingham_coeffs(R1, (6.0, 2.0), 4, 2, roll_kappa=1.5)
    c = so3.watson_coeffs(8.0, (0.0, 0.6, 0.8), 8, 4)
    out["rotate_coeffs"] = so3.rotate_coeffs(c, R, 8, 4)
    out["truncate_coeffs"] = so3.truncate_coeffs(c, 8, 4, 4, 2)
    out["extend_coeffs"] = so3.extend_coeffs(so3.truncate_coeffs(c, 8, 4, 4, 2), 4, 2, 8, 4)
    out["rebanded"] = so3.rebanded(c, 8, 4, 6, 2)
    out["energy"] = so3.energy(c, 8, 4)
    out["evaluate"] = so3.evaluate(c, 8, R, 4)
    Dz = so3.so3_design(8, R, 4)
    out["project"] = so3.project(Dz, np.ones(len(R)) / len(R), np.ones((len(R), 2)))
    D = api.Distribution
    for name, dist in [("pose", D.pose(R1)), ("axis", D.axis([0.0, 0.6, 0.8])), ("watson", D.watson(8.0)),
                       ("bingham", D.bingham(R1, (6.0, 2.0), roll_kappa=1.0, lmax=4, nmax=2)), ("uniform", D.uniform()),
                       ("rotated", D.watson(8.0).rotated(R2))]:
        out[f"Distribution/{name}/coeffs"] = dist.coeffs
        out[f"Distribution/{name}/source"] = dist.source
    F = api.FOD
    fw = F.watson(6.0, (0.0, 0.6, 0.8), lmax=8)
    out["FOD/watson"], out["FOD/watson/integral"] = fw.coeffs, fw.integral
    out["FOD/watson/evaluate"] = fw.evaluate(d[:20])
    out["FOD/isotropic"] = F.isotropic(8).coeffs
    for basis, legacy in (("tournier07", False), ("tournier07", True), ("descoteaux07", False)):
        f = F.from_sh(fw.coeffs, basis=basis, legacy=legacy)
        out[f"FOD/from_sh/{basis}/{legacy}"] = f.coeffs
        out[f"FOD/from_sh/{basis}/{legacy}/source"] = f.source
    out["Distribution/axis_density"] = D.axis_density(fw).coeffs
    out["sh/cart2sphere"] = sh.cart2sphere(d)
    out["sh/watson_sh"] = sh.watson_sh(np.array([0.0, 0.6, 0.8]), 8.0, l_max=8)
    out["sh/bingham_sh"] = sh.bingham_sh(R1, (6.0, 2.0), l_max=8)
    out["sh/watson_zonal_ratios"] = sh.watson_zonal_ratios(8.0, 8)
    return out


def g02_pose_closed(api):
    """The pose response, closed form: every fixture, with and without tissue, a compartment, the field."""
    api.require("pose")
    out = {}
    backend = dict(backend=api.DEFAULTS["pose"])
    for name in ("cylinder", "permeable", "windows"):
        pk = _pack(api, name)
        seq = _pgse(api, pk.T_window)
        out.update({f"{name}/bare/{k}": v for k, v in _response(pk.pose_response(seq, **backend)).items()})
        if name == "cylinder":
            r = pk.pose_response(seq, tissue=pk.nominal, scanner=3.0, **backend)
            out.update({f"{name}/nominal/{k}": v for k, v in _response(r).items()})
            r = pk.pose_response(seq, keep=(4, 0), **backend)
            out.update({f"{name}/keep40/{k}": v for k, v in _response(r).items()})
            r = pk.pose_response(seq, backend="torch", device="cpu")
            out.update({f"{name}/torch/{k}": v for k, v in _response(r).items()})
        if name == "permeable":
            r = pk.pose_response(seq, compartment=1, **backend)
            out.update({f"{name}/compartment0/{k}": v for k, v in _response(r).items()})
        if name == "windows":
            seq2 = _pgse(api, 2 * pk.T_window)
            out.update({f"{name}/both/{k}": v for k, v in _response(pk.pose_response(seq2, **backend)).items()})
    pk = _pack(api, "field")
    seq = _pgse(api, pk.T_window, b=2e8)
    r = pk.pose_response(seq, tissue=api.Tissue(chi_iso=-1e-7, chi_aniso=-5e-8), scanner=3.0, **backend)
    out.update({f"field/chi/{k}": v for k, v in _response(r).items()})
    return out


def g03_pose_quadrature(api):
    """The quadrature route, by name and for a multi-axis (tensor) encoding."""
    api.require("pose")
    pk = _pack(api, "cylinder")
    T = pk.T_window
    seq = _pgse(api, T, b=5e8)
    out = {f"named/{k}": v for k, v in _response(pk.pose_response(seq, method="quadrature",
                                                                   backend=api.DEFAULTS["pose"])).items()}
    n_t = 401
    dt = 0.95 * T / (n_t - 1)
    G = np.zeros((1, n_t, 3), np.float32)
    i0, i1 = 20, 120
    k = np.arange(i1 - i0)
    G[0, i0:i1, 0] = 0.02 * np.sin(2 * np.pi * k / (i1 - i0))
    G[0, i0:i1, 1] = 0.02 * np.sin(4 * np.pi * k / (i1 - i0))
    G[0, 200:300, 2] = 0.02 * np.sin(2 * np.pi * k / (i1 - i0))
    ref = _pgse(api, T)
    tensor = api.ScannerSequence(G=G, dt=dt, rf=ref.rf, readout=(n_t - 1,))
    out.update({f"tensor/{k}": v for k, v in _response(pk.pose_response(tensor, backend=api.DEFAULTS["pose"])).items()})
    return out


def _classes(api, seq):
    """Three encoding classes of one timing class: ``seq``, its rows at 0.8 of their amplitude, and its rows turned."""
    from dmipy_replay.acquisition.waveforms import rotate_waveform
    scaled = api.ScannerSequence(G=0.8 * np.asarray(seq.G), dt=seq.dt, rf=seq.rf, readout=seq.readout)
    return [seq, scaled, rotate_waveform(seq, R1)]


def g04_pose_batch_series(api):
    """A batch of acquisitions, and the series over the encoding classes of one timing class."""
    api.require("pose", "series")
    pk = _pack(api, "cylinder")
    T = pk.T_window
    seqs = [_pgse(api, T, dirs=[d], b=b) for d, b in (([0, 0, 1], 1e9), ([1, 0, 0], 2e9), ([0.6, 0.8, 0], 5e8))]
    out = {}
    for i, r in enumerate(pk.pose_responses(seqs, backend=api.DEFAULTS["pose"])):
        out.update({f"batch/{i}/{k}": v for k, v in _response(r).items()})
    seq = api.pgse([[0, 0, 1], [1, 0, 0]], 0.2 * T, 0.5 * T, gradient_strengths=[0.05, 0.03], TE=0.95 * T, slew_rate=200.0)
    series = pk.pose_series(_classes(api, seq), backend=api.DEFAULTS["pose"])
    out["series/bounds"] = series.bounds
    for i, r in enumerate(series.responses()):
        out.update({f"series/{i}/{k}": v for k, v in _response(r).items()})
    series = pk.pose_series(_classes(api, seq), backend="torch", device="cpu")
    for i, r in enumerate(series.responses()):
        out.update({f"series_torch/{i}/{k}": v for k, v in _response(r).items()})
    fp = _pack(api, "field")
    seq = api.pgse([[0, 0, 1]], 0.2 * fp.T_window, 0.5 * fp.T_window, gradient_strengths=[0.05], TE=0.95 * fp.T_window,
                   slew_rate=200.0)
    series = fp.pose_series(_classes(api, seq), tissue=api.Tissue(chi_iso=-1e-7, chi_aniso=-5e-8), scanner=3.0,
                            backend=api.DEFAULTS["pose"])
    out["field_series/bounds"] = series.bounds
    for i, r in enumerate(series.responses()):
        out.update({f"field_series/{i}/{k}": v for k, v in _response(r).items()})
    return out


def g05_composition(api):
    """A pack replayed against distributions of poses, and the response's own compose/at/retained."""
    api.require("pose")
    pk = _pack(api, "cylinder")
    seq = _pgse(api, pk.T_window, dirs=[[0, 0, 1], [1, 0, 0]])
    D = api.Distribution
    out = {"watson": pk.replay(seq, orientation=D.watson(8.0)), "uniform": pk.replay(seq, orientation=D.uniform()),
           "pose_distribution": pk.replay(seq, orientation=D.pose(R1)), "pose_matrix": pk.replay(seq, orientation=R1),
           "fod": pk.replay(seq, orientation=api.FOD.watson(6.0, (0.0, 0.6, 0.8)))}
    r = pk.pose_response(seq, backend=api.DEFAULTS["pose"])
    out["compose/watson"] = r.compose(D.watson(8.0))
    out["compose/bingham"] = r.compose(D.bingham(R1, (6.0, 2.0), lmax=6, nmax=0))
    out["at/R1"], out["at/R2"] = r.at(R1), r.at(R2)
    out["retained"] = r.retained(4, 0)
    with tempfile.TemporaryDirectory() as tmp:
        r.save(os.path.join(tmp, "r.npz"))
        back = api.PoseResponse.load(os.path.join(tmp, "r.npz"))
    out.update({f"load/{k}": v for k, v in _response(back).items()})
    return out


def g06_device_kernels(api):
    """The pose expansion's device kernels on fixed inputs, once per device word."""
    api.require("pose")
    pdv = api.pose_device
    rng = np.random.default_rng(6)
    out = {}
    dirs = _dirs(300, 2)
    x = np.abs(rng.normal(size=300)) * 8
    R = np.stack([R1, R2, R1 @ R2])
    Q = rng.normal(size=(300, 2, 3, 3))
    ew = rng.random(300)
    kappa = np.abs(rng.normal(size=(300, 3))) * 8
    field = (0.7, -0.2, np.array([0.0, 0.6, 0.8]), rng.normal(size=300), rng.normal(size=(300, 6)), rng.normal(size=(300, 6)))
    for dev in ("numpy", "jax", "torch:cpu"):
        out[f"{dev}/bessel_tails"] = np.asarray(pdv.bessel_tails(kappa, ew, 12, device=dev))
    for dev in ("numpy", "jax"):                                     # these kernels have no torch route
        out[f"{dev}/real_sh"] = np.asarray(pdv.real_sh(8, dirs, device=dev))
        out[f"{dev}/spherical_jn_all"] = np.asarray(pdv.spherical_jn_all(8, x, device=dev))
        out[f"{dev}/pose_samples"] = np.asarray(pdv.pose_samples(R, Q, ew, float(ew.sum()), device=dev))
        out[f"{dev}/pose_samples/field"] = np.asarray(pdv.pose_samples(R, Q, ew, float(ew.sum()), field=field, device=dev))
    return out


def g07_pathways(api):
    """An RF train as a sum of gates on a pack."""
    api.require("pose", "pathways")
    pk = _pack(api, "windows")
    seq = api.splice([[1.0, 0.0, 0.0]], 15e-3, 25e-3, 3, 10e-3, bvalues=[1e9], TE_prep=80e-3, beta_deg=150.0, n_t_per_echo=40)
    tr = api.train_response(pk, seq, backend=api.DEFAULTS["pose"])
    out = {"n_gates": tr.n_gates, "repr": repr(tr)}
    for key, w in (("weights", tr.weights()), ("weights/b1", tr.weights(b1_scale=0.9, dw=20.0))):
        out.update({f"{key}/{'|'.join(g)}": np.asarray(v) for g, v in w.items()})
    for key, kw in (("at", {}), ("at/echo0", dict(echo=0)), ("at/b1/dw", dict(b1_scale=0.9, dw=20.0))):
        out.update({f"{key}/{k}": v for k, v in _response(tr.at(**kw)).items()})
    return out


def _grid(api, shape, size=2.5e-2):
    return api.Grid(shape=shape, voxel_size_m=(size,) * 3, origin_m=tuple(-0.5 * (n - 1) * size for n in shape),
                    isocenter_m=(0.0, 0.0, 0.0))


def _untimed(obj):
    """``obj`` with every timing (a ``seconds`` entry, at any depth) replaced by its keys, or by ``"<timing>"``."""
    if isinstance(obj, dict):
        return {k: ((sorted(v) if isinstance(v, dict) else "<timing>") if k == "seconds" else _untimed(v))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_untimed(v) for v in obj]
    return obj


def _report(rep):
    """A replay's report with its timings by key only."""
    return {k: (json.dumps(v, sort_keys=True, default=str) if isinstance(v, list) else v) for k, v in _untimed(rep).items()}


def _masked(meta):
    """A pack's or a phantom's metadata without the producing code's commit: the move changes the commit that writes a
    pack, and whether one is recorded at all (a package installed from a checkout records it, one installed from an
    archive does not); versions are frozen."""
    if isinstance(meta, dict):
        return {k: _masked(v) for k, v in meta.items() if k != "commit"}
    if isinstance(meta, list):
        return [_masked(v) for v in meta]
    return meta


def _file(path):
    """A safetensors file (a pack or a phantom) by its parts: the header in its order with every metadata string's
    commit left out (:func:`_masked`), and the sha256 of the data that follows it."""
    blob = Path(path).read_bytes()
    n = int.from_bytes(blob[:8], "little")
    header = json.loads(blob[8:8 + n])
    md = header.get("__metadata__") or {}
    for k, v in md.items():
        try:
            md[k] = json.dumps(_masked(json.loads(v)))
        except ValueError:
            pass
    return {"header": json.dumps(header), "data": hashlib.sha256(blob[8 + n:]).hexdigest()}


def g08_phantom(api):
    """Phantoms composed, partitioned, written and read, and replayed on every route; what a phantom's substrates and
    machine bring to the pose replay (a specimen pose, a closed form's expansion and RF train, the delivered
    gradient of a time-segment layout)."""
    api.require("pose", "pathways", "phantom", "delivery")
    out = {}
    pk = _pack(api, "cylinder")
    seq = _pgse(api, pk.T_window)
    r = pk.pose_response(seq, pose=api.Pose(R1), backend=api.DEFAULTS["pose"])
    out.update({f"specimen_pose/{k}": v for k, v in _response(r).items()})
    form = api.FreeWater(m0=1.0, tissue=api.Tissue(D=2e-9, T2=0.08))
    seq2 = _pgse(api, pk.T_window, dirs=[[0, 0, 1], [1, 0, 0]])
    out.update({f"analytic/{k}": v for k, v in _response(api.analytic_pose_response(form, seq2)).items()})
    train = api.splice([[1.0, 0.0, 0.0]], 15e-3, 25e-3, 3, 10e-3, bvalues=[1e9], TE_prep=80e-3, beta_deg=150.0, n_t_per_echo=40)
    cf = api.closed_form_train(form, train)
    out["closed_train/at"] = np.asarray(cf.at())
    out["closed_train/at/b1"] = np.asarray(cf.at(b1_scale=0.9, echo=1))
    out["closed_train/repr"] = repr(cf)
    from segment_scenarios import acquisitions
    from types import SimpleNamespace
    ns = SimpleNamespace(pgse=api.pgse, pgste=api.pgste, ogse=api.ogse, ScannerSequence=api.ScannerSequence,
                         Partition=api.Partition, Tissue=api.Tissue)
    ts = api.TimeSegmentLayout(str(DATA / "time_segments"))
    d = ts.delivery(acquisitions(ns)["box"][1], api.ScannerLimits.of("siemens_magnetom_prisma_3T"))
    out["delivery/partition"], out["delivery/voxels"], out["delivery/levels"] = d.partition, np.asarray(d.voxels), np.asarray(d.levels)
    out["delivery/q"], out["delivery/g0"], out["delivery/kappa"] = np.asarray(d.q), np.asarray(d.g0), np.asarray(d.kappa)
    cyl = str(DATA / "cylinder.rpk")
    wpk = str(DATA / "windows.rpk")
    grid = _grid(api, (2, 2, 1))
    sub = api.PackSubstrate(cyl, m0=0.7, name="cyl")
    frac = np.ones((2, 2, 1), np.float32)
    ph = api.Phantom.compose(grid, fractions={sub: frac}, orientation={sub: api.Watson(mu=np.broadcast_to([0.0, 0.0, 1.0], (2, 2, 1, 3)), kappa=8.0)},
                             remainder=api.Inert(name="bg"))
    pack = _pack(api, "cylinder")
    seq = _pgse(api, pack.T_window, dirs=[[0, 0, 1], [1, 0, 0]])
    rep = {}
    out["watson/replay"] = ph.replay(seq, packs={"cyl": pack}, report=rep)
    out.update({f"watson/report/{k}": v for k, v in _report(rep).items()})
    out["watson/mode"] = ph.mode
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.rph")
        ph.write(path, id="parity/watson", license="CC0-1.0", citation="parity", embed=True)
        out["watson/rph"] = _file(path)
        back = api.Phantom.read(path)
        out["watson/read/replay"] = back.replay(seq)
        f = api.read_rph(path)
        out["watson/read/n_voxels"] = f.n_voxels
        out["watson/read/voxel_index"] = np.asarray(f.voxel_index)
        out["watson/read/geometric_fraction"] = np.asarray(f.geometric_fraction)
    # frames, replayed through the vector-Bloch route
    rots = np.broadcast_to(R1, (2, 2, 1, 3, 3)).copy()
    rots[0, 0, 0] = R2
    fr = api.Phantom.compose(grid, fractions={sub: frac}, orientation={sub: api.Frames(rots)}, remainder=api.Inert(name="bg"))
    out["frames/replay"] = fr.replay(seq, packs={"cyl": pack})
    out["frames/replay_bloch"] = fr.file.replay_bloch(seq, packs={"cyl": pack})
    # an ODF phantom's RF train
    SH = (6, 6, 2)
    c = np.zeros(SH + (45,), np.float32)
    c[..., 0] = 1.0 / np.sqrt(4 * np.pi)
    c[..., 3] = 0.3
    wm = api.PackSubstrate(wpk, m0=0.7, name="wm")
    odf = api.Phantom.compose(_grid(api, SH), fractions={wm: np.ones(SH, np.float32)},
                              orientation={wm: api.ODF(c, basis="mrtrix3")}, remainder=api.Inert(name="bg"))
    train = api.splice([[1.0, 0, 0]], 15e-3, 25e-3, 3, 10e-3, bvalues=[1e9], TE_prep=80e-3, beta_deg=150.0, n_t_per_echo=40)
    rep = {}
    out["odf/replay_train"] = odf.replay_train(train, packs={"wm": _pack(api, "windows")}, report=rep)
    out.update({f"odf/report/{k}": v for k, v in _report(rep).items()})
    # a partition with a pose
    pgrid = api.Grid.covering(pack.r0, voxel_size_m=(2.5e-6, 2.5e-6, float(np.ptp(pack.r0[:, 2])) + 1e-6), attach="substrate")
    part = api.Phantom.partition(api.PackSubstrate(cyl, m0=1.0, name="cyl"), pgrid, pose=api.Pose(R1))
    out["partition/n_voxels"] = part.n_voxels
    out["partition/voxel_index"] = np.asarray(part.voxel_index)
    out["partition/replay"] = part.replay(seq)
    return out


def g09_study(api):
    """Walker primitives of an acquisition and a study on a pack."""
    api.require("study")
    out = {}
    for name in ("cylinder", "field"):
        pk = _pack(api, name)
        seq = _pgse(api, pk.T_window, dirs=[[0, 0, 1], [1, 0, 0]], b=(B if name == "cylinder" else 2e8))
        tissue = pk.nominal if name == "field" else api.Tissue(T2={p["name"]: 0.07 for p in pk.meta["substrate"]["pools"]})
        scanner = 3.0 if name == "field" else None
        acq = api.Acquisition(seq)
        prim = pk.walker_primitives(acq)
        w, ew, E = prim.signals(tissue, scanner)
        out[f"{name}/primitives/w"], out[f"{name}/primitives/ew"], out[f"{name}/primitives/E"] = w, ew, E
        st = api.Study(api.StudyProtocol([acq, api.Acquisition(_pgse(api, pk.T_window, b=5e8))]),
                       tissues=(None, tissue), scanners=(scanner,))
        out[f"{name}/study"] = pk.study(st)
        out[f"{name}/study/meta"] = json.dumps(st.to_meta(), sort_keys=True, default=str)
    return out


def g10_jax_twins(api):
    """The JAX twins of the replay and the vector-Bloch propagator on both backends."""
    api.require("jax_bloch")
    from dmipy_replay.replay.replay import compile_scheme
    pk = _pack(api, "cylinder")
    seq = _pgse(api, pk.T_window, dirs=[[0, 0, 1], [1, 0, 0]])
    C = np.asarray(pk.position_coeffs if hasattr(pk, "position_coeffs") else pk.arrays["position_coeffs"])
    W = compile_scheme(np.asarray(seq.G_eff, np.float64), float(seq.dt), pk.K, T=pk.T_window)
    w = np.asarray(pk.spin_weights)
    out = {"replay_signal_jax": np.asarray(api.replay_signal_jax(C, w, W)),
           "replay_batch_jax": np.asarray(api.replay_batch_jax(C, w, np.stack([W, 0.5 * W])))}
    for be in ("numpy", "jax"):
        out[f"replay_bloch/{be}"] = np.asarray(pk.replay_bloch(seq, backend=be, b1_scale=0.95))
    return out


def g11_images(api):
    """The columnar layouts' images on both backends and a study's image; the time-segment images on every backend."""
    api.require("study", "columnar_study", "segment_backends")
    out = {}
    from columnar_scenarios import seqs, tissue_of
    from types import SimpleNamespace
    ns = SimpleNamespace(pgse=api.pgse, set_b=api.set_b, Tissue=api.Tissue)
    S = seqs(ns)
    for lay_name in ("columnar_float", "columnar_quant"):
        lay = api.ColumnarPack(str(DATA / lay_name))
        tissue = tissue_of(ns, lay.meta)
        for key, seq in S.items():
            for be in ("numpy", "jax"):
                img, floor, plan = lay.image(seq, backend=be)
                out[f"{lay_name}/{key}/{be}/S"], out[f"{lay_name}/{key}/{be}/floor"] = np.asarray(img), np.asarray(floor)
                out[f"{lay_name}/{key}/{be}/plan"] = json.dumps(_untimed(dict(plan)), sort_keys=True, default=str)
            img, floor, _ = lay.image(seq, settings=[(None, None), (tissue, 3.0)], backend="numpy")
            out[f"{lay_name}/{key}/settings/S"], out[f"{lay_name}/{key}/settings/floor"] = np.asarray(img), np.asarray(floor)
        st = api.Study(api.StudyProtocol([api.Acquisition(s) for s in S.values()]), tissues=(None, tissue), scanners=(None,))
        img, floor, plan = lay.image_study(st)
        out[f"{lay_name}/study/S"], out[f"{lay_name}/study/floor"] = np.asarray(img), np.asarray(floor)
        out[f"{lay_name}/study/plan"] = json.dumps(_untimed(dict(plan)), sort_keys=True, default=str)
    from segment_scenarios import TISSUE_KW, acquisitions
    ns = SimpleNamespace(pgse=api.pgse, pgste=api.pgste, ogse=api.ogse, ScannerSequence=api.ScannerSequence,
                         Partition=api.Partition, Tissue=api.Tissue)
    ts = api.TimeSegmentLayout(str(DATA / "time_segments"))
    for key, (_part, seq) in sorted(acquisitions(ns).items()):
        for be, dev in (("numpy", None), ("jax", None), ("torch", "cpu")):
            try:
                img, floor = ts.image(seq, backend=be, device=dev)
                out[f"segments/{key}/{be}/S"], out[f"segments/{key}/{be}/floor"] = np.asarray(img), np.asarray(floor)
            except Exception as exc:
                out[f"segments/{key}/{be}/refused"] = f"{type(exc).__name__}: {exc}"
    return out


def _pack_record(pk, path=None):
    out = {f"tensor/{k}": hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() + f" {v.dtype} {v.shape}"
           for k, v in sorted(pk.arrays.items()) if v is not None}
    out["meta"] = json.dumps(_masked(pk.meta), sort_keys=True, default=str)
    if path is not None:
        out["file"] = _file(path)
    return out


def g12_writer(api):
    """The writer operations on the fixtures: their tensors, their metadata and their files."""
    api.require("writer")
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        t = lambda n: os.path.join(tmp, n)
        win = _pack(api, "windows")
        tr = api.call("truncate", win, 1, out_path=t("trunc.rpk"))
        out.update({f"truncate/{k}": v for k, v in _pack_record(tr, t("trunc.rpk")).items()})
        cyl = _pack(api, "cylinder")
        pf = api.call("prefix", cyl, cyl.T_window / 2, out_path=t("prefix_half.rpk"))
        out.update({f"prefix_half/{k}": v for k, v in _pack_record(pf, t("prefix_half.rpk")).items()})
        pw = api.call("prefix", win, 0.1, out_path=t("prefix_window.rpk"))
        out.update({f"prefix_window/{k}": v for k, v in _pack_record(pw, t("prefix_window.rpk")).items()})
        rw = api.call("rewindow", cyl, cyl.T_window / 2, out_path=t("rewindow.rpk"))
        out.update({f"rewindow/{k}": v for k, v in _pack_record(rw, t("rewindow.rpk")).items()})
        for name in ("permeable", "cylinder"):
            pk = _pack(api, name)
            out[f"restated/{name}/edge"] = api.call("restate_surface_envelope", pk, out_path=t(f"restated_{name}.rpk"))
            out.update({f"restated/{name}/{k}": v for k, v in _pack_record(pk, t(f"restated_{name}.rpk")).items()})
        out["stated_envelope"] = json.dumps(api.call("stated_envelope", cyl), sort_keys=True, default=str)
        api.call("save", cyl, t("saved.rpk"))
        out["save/file"] = _file(t("saved.rpk"))
        ext = api.call("extend", win, 0.1, seed=7, out_path=t("extended.rpk"), require_gpu=False)
        out.update({f"extend/{k}": v for k, v in _pack_record(ext, t("extended.rpk")).items()})
    return out


def _said(exc):
    """An exception as its type and message, a package's dotted names reduced to their last part: the move renames
    the package a message names, and nothing else of it."""
    said = re.sub(r"dmipy_(sim|replay)(\.\w+)*\.(\w+)", r"<pkg>.\3", str(exc))
    return f"{type(exc).__name__}: " + re.sub(r"\S*/(\w+\.rpk)", r"\1", said)


def g13_refusals(api):
    """What is refused, and how it is said."""
    api.require("pose", "phantom")
    pk = _pack(api, "cylinder")
    seq = _pgse(api, pk.T_window)
    out = {}

    def refusal(key, fn, message=True):
        try:
            fn()
            out[key] = "accepted"
        except Exception as exc:
            out[key] = _said(exc) if message else type(exc).__name__

    refusal("pose_matrix", lambda: pk.pose_response(seq, pose=np.eye(3)))
    refusal("bare_fod", lambda: api.Distribution.axis_density(np.zeros(45)))
    refusal("orientation_object", lambda: pk.replay(seq, orientation=object()))
    refusal("pose_backend", lambda: pk.pose_response(seq, backend="cupy"), message=False)   # D7 adds "numpy" to the list
    refusal("pose_method", lambda: pk.pose_response(seq, method="exact"))
    refusal("fod_basis", lambda: api.FOD.from_sh(np.zeros(45), basis="mrtrix2"))
    refusal("fod_unnormalised", lambda: api.FOD.native(np.ones(45)))
    refusal("phantom_bare_orientation", lambda: api.Phantom.compose(
        _grid(api, (1, 1, 1)), fractions={api.Inert(name="x"): np.ones((1, 1, 1))}, orientation={"x": np.eye(3)}))
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "bad.rpk")
        from dmipy_replay.replay.container import write_safetensors
        write_safetensors(p, {"positions": np.zeros((3, 3), np.float32)}, {"rpk": json.dumps(pk.meta)})
        refusal("header", lambda: api.read_rpk(p))
    return out


GROUPS = {f.__name__: f for f in (g01_so3, g02_pose_closed, g03_pose_quadrature, g04_pose_batch_series, g05_composition,
                                  g06_device_kernels, g07_pathways, g08_phantom, g09_study, g10_jax_twins, g11_images,
                                  g12_writer, g13_refusals)}
