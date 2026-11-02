"""The parity battery: every replay route this package serves, written once against a namespace so that the same
scenario ran on dmipy-sim ``8d7b97f`` (its numbers are ``data/reference.json``, a frozen record) and on dmipy-replay (``test_parity.py``
compares). A scenario is ``(name, pack, fn)`` with ``fn(ns, pack) -> ndarray``; ``ns`` carries the builders and
``Tissue`` of whichever package runs it.
"""
import numpy as np

DIRS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
DIRS = DIRS / np.linalg.norm(DIRS, axis=1, keepdims=True)
B = np.array([0.0, 1e9, 3e9])


def _shells(dirs=DIRS, b=B):
    """Every direction at every b, as the two arrays a builder takes."""
    return np.repeat(dirs, len(b), axis=0), np.tile(b, len(dirs))


def _pgse(ns, delta=0.006, Delta=0.02, b=B, **kw):
    d, bv = _shells(b=b)
    return ns.pgse(d, delta, Delta, bvalues=bv, **kw)


def _tissue(ns, **kw):
    return ns.Tissue(**kw)


def bare(ns, p):
    return p.replay(_pgse(ns), complex_signal=True)


def square_lobes(ns, p):
    return p.replay(_pgse(ns, slew_rate=np.inf), complex_signal=True)


def relaxed(ns, p):
    t = _tissue(ns, T2={"extra": 0.06, "intra": 0.08}, T1={"extra": 1.2, "intra": 0.9})
    return p.replay(_pgse(ns, TE=0.035), tissue=t, complex_signal=True)


def surface(ns, p):
    return p.replay(_pgse(ns), tissue=_tissue(ns, rho2=2e-5), complex_signal=True)


def surface_longitudinal(ns, p):
    d, bv = _shells(b=np.array([0.0, 1e9]))
    seq = ns.pgste(d, 0.004, 0.02, bvalues=bv)
    return p.replay(seq, tissue=_tissue(ns, rho2=2e-5, rho1=1e-5, T1={"extra": 1.0, "intra": 0.8}), complex_signal=True)


def stimulated(ns, p):
    d, bv = _shells()
    return p.replay(ns.pgste(d, 0.004, 0.02, bvalues=bv), complex_signal=True)


def oscillating(ns, p):
    d, bv = _shells(b=np.array([0.0, 3e8]))
    return p.replay(ns.ogse(d, 50.0, 0.02, shape="cosine", bvalues=bv), complex_signal=True)


def posed(ns, p):
    return p.replay(_pgse(ns), orientation=np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0), complex_signal=True)


def posed_matrix(ns, p):
    c, s = np.cos(0.7), np.sin(0.7)
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return p.replay(_pgse(ns), orientation=R, complex_signal=True)


def one_pool(ns, p):
    return p.replay(_pgse(ns), compartment=1, complex_signal=True)


def per_walker(ns, p):
    w, ew, E = p.walker_signals(_pgse(ns, b=np.array([0.0, 2e9])))
    return (ew[:, None] * E).sum(0) / w.sum()


def faster(ns, p):
    return p.replay(_pgse(ns, delta=0.004, Delta=0.012, b=np.array([0.0, 1e9])), tissue=_tissue(ns, D=3e-9), complex_signal=True)


def smaller(ns, p):
    return p.at_length_scale(0.8).replay(_pgse(ns, b=np.array([0.0, 1e9])), complex_signal=True)


def bloch_nominal(ns, p):
    return p.replay_bloch(_pgse(ns, b=np.array([0.0, 1e9])), complex_signal=True)


def bloch_b1(ns, p):
    t = _tissue(ns, T2={"extra": 0.06, "intra": 0.08})
    return p.replay_bloch(_pgse(ns, b=np.array([0.0, 1e9]), TE=0.035), b1_scale=0.85, tissue=t, complex_signal=True)


def bloch_train(ns, p):
    return p.replay_bloch(ns.cpmg(4, 0.008, beta_deg=150.0), tissue=_tissue(ns, T2={"extra": 0.06, "intra": 0.08}),
                          complex_signal=True)


def bloch_offresonance(ns, p):
    return p.replay_bloch(_pgse(ns, b=np.array([0.0, 1e9])), off_resonance_T=2e-8, complex_signal=True)


def gre_field_3T(ns, p):
    return p.replay(ns.gre(0.006), tissue=_tissue(ns, chi_iso=-1e-7), scanner=3.0, complex_signal=True)


def gre_field_7T_aniso(ns, p):
    return p.replay(ns.gre(0.006), tissue=_tissue(ns, chi_iso=-1e-7, chi_aniso=-1e-7), scanner=7.0, complex_signal=True)


def se_field(ns, p):
    d, bv = _shells(b=np.array([0.0, 1e8]))
    seq = ns.pgse(d, 0.0015, 0.004, bvalues=bv, slew_rate=np.inf)
    return p.replay(seq, tissue=_tissue(ns, chi_iso=-1e-7), scanner=7.0, complex_signal=True)


def field_posed(ns, p):
    return p.replay(ns.gre(0.006), tissue=_tissue(ns, chi_iso=-1e-7), scanner=7.0,
                    orientation=np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0), complex_signal=True)


def bloch_field(ns, p):
    return p.replay_bloch(ns.gre(0.006), tissue=_tissue(ns, chi_iso=-1e-7), scanner=7.0, complex_signal=True)


def long_te(ns, p):
    return p.replay(_pgse(ns, delta=0.01, Delta=0.12, b=np.array([0.0, 1e9])), complex_signal=True)


def long_te_relaxed(ns, p):
    t = _tissue(ns, T2={"extra": 0.06, "intra": 0.08}, rho2=1e-5)
    return p.replay(_pgse(ns, delta=0.01, Delta=0.12, b=np.array([0.0, 1e9])), tissue=t, complex_signal=True)


def long_te_bloch(ns, p):
    return p.replay_bloch(_pgse(ns, delta=0.01, Delta=0.12, b=np.array([0.0, 1e9])), b1_scale=0.9, complex_signal=True)


def exchange_relaxed(ns, p):
    t = _tissue(ns, T2={"extra": 0.05, "intra": 0.09})
    return p.replay(_pgse(ns, TE=0.035), tissue=t, complex_signal=True)


def exchange_stimulated(ns, p):
    d, bv = _shells(b=np.array([0.0, 1e9]))
    t = _tissue(ns, T2={"extra": 0.05, "intra": 0.09}, T1={"extra": 1.2, "intra": 0.7})
    return p.replay(ns.pgste(d, 0.004, 0.02, bvalues=bv), tissue=t, complex_signal=True)


# ---- plan before read: what a replay reads, its error, and reads at a stated depth
def _plan(pl):
    """A :class:`ReadPlan` as the numbers it states."""
    return np.array([pl.K, pl.K_stored, pl.band_error, pl.modes, pl.modes_stored, pl.mode_error, pl.bytes_read,
                     pl.bytes_stored, pl.floor, pl.tol], np.float64)


def plan_default(ns, p):
    return _plan(p.plan(_pgse(ns)))


def plan_tight(ns, p):
    return _plan(p.plan(_pgse(ns), tol=0.01))


def plan_carried(ns, p):
    S = p.replay(_pgse(ns), complex_signal=True)
    return np.concatenate([S, _plan(S.plan)])


def full_read(ns, p):
    S = p.replay(_pgse(ns), tol=None, complex_signal=True)
    return np.concatenate([S, _plan(S.plan)])


def explicit_K(ns, p):
    S = p.replay(_pgse(ns), K=12, complex_signal=True)
    return np.concatenate([S, _plan(S.plan)])


def phases_planned(ns, p):
    w, ew, phi = p.walker_phases(_pgse(ns, b=np.array([0.0, 2e9])))
    return np.concatenate([(ew[:, None] * np.exp(1j * phi)).sum(0) / w.sum(), _plan(phi.plan)])


def walker_range(ns, p):
    return p.walkers(100, 700).replay(_pgse(ns), complex_signal=True)


def decoded_twice(ns, p):
    seq = _pgse(ns)
    a = p.replay(seq, complex_signal=True)
    b = p.replay(seq, complex_signal=True)
    p.release_decoded()
    return np.concatenate([a, b, [p.decoded_bytes]])


# ---- the band check at the pack's cap
def band_budget(ns, p):
    b = p.truncation_budget
    return np.array([b["cap"], b["other_terms"], b["budget"]])


def band_of_pgse(ns, p):
    return np.array(p.waveform_band(_pgse(ns)), np.float64)


def band_of_ogse(ns, p):
    d, bv = _shells(b=np.array([0.0, 3e8]))
    return np.array(p.waveform_band(ns.ogse(d, 50.0, 0.02, shape="cosine", bvalues=bv)), np.float64)


def band_beyond(ns, p):
    d, bv = _shells(dirs=DIRS[:1], b=np.array([0.0, 2e8]))
    return p.replay(ns.ogse(d, 2500.0, 0.02, shape="trapezoid", slew_rate=np.inf, bvalues=bv), complex_signal=True)


def _stated_cap(p, cap=0.03):
    """``p`` with a stated numerical cap (``fidelity.target_floor``) beside its certified terms."""
    fid = dict(p.meta["fidelity"], target_floor=cap, err_quadrature=2e-3, err_quantisation=1e-3)
    return type(p)(p.arrays, dict(p.meta, fidelity=fid), source=p.source)


def band_budget_stated(ns, p):
    return band_budget(ns, _stated_cap(p))


def band_of_ogse_stated(ns, p):
    return band_of_ogse(ns, _stated_cap(p))


def plan_stated(ns, p):
    S = _stated_cap(p).replay(_pgse(ns), complex_signal=True)
    return np.concatenate([S, _plan(S.plan)])


# ---- field modes planned at the certified field scale
def field_plan(ns, p):
    return _plan(p.plan(ns.gre(0.006), tissue=_tissue(ns, chi_iso=-1e-7), scanner=7.0))


def field_modes(ns, p):
    S = p.replay(ns.gre(0.006), tissue=_tissue(ns, chi_iso=-1e-7), scanner=7.0, modes=4, complex_signal=True)
    return np.concatenate([S, _plan(S.plan)])


def field_full(ns, p):
    return p.replay(ns.gre(0.006), tissue=_tissue(ns, chi_iso=-1e-7), scanner=7.0, tol=None, complex_signal=True)


def field_beyond_envelope(ns, p):
    return p.replay(ns.gre(0.006), tissue=_tissue(ns, chi_iso=-1e-5), scanner=11.7, complex_signal=True)


SCENARIOS = [
    ("bare", "cylinder", bare), ("square_lobes", "cylinder", square_lobes), ("relaxed", "cylinder", relaxed),
    ("surface", "cylinder", surface), ("surface_longitudinal", "cylinder", surface_longitudinal),
    ("stimulated", "cylinder", stimulated), ("oscillating", "cylinder", oscillating), ("posed", "cylinder", posed),
    ("posed_matrix", "cylinder", posed_matrix), ("one_pool", "cylinder", one_pool), ("per_walker", "cylinder", per_walker),
    ("faster", "cylinder", faster), ("smaller", "cylinder", smaller), ("bloch_nominal", "cylinder", bloch_nominal),
    ("bloch_b1", "cylinder", bloch_b1), ("bloch_train", "cylinder", bloch_train),
    ("bloch_offresonance", "cylinder", bloch_offresonance),
    ("exchange_relaxed", "permeable", exchange_relaxed), ("exchange_stimulated", "permeable", exchange_stimulated),
    ("long_te", "windows", long_te), ("long_te_relaxed", "windows", long_te_relaxed), ("long_te_bloch", "windows", long_te_bloch),
    ("gre_field_3T", "field", gre_field_3T), ("gre_field_7T_aniso", "field", gre_field_7T_aniso), ("se_field", "field", se_field),
    ("field_posed", "field", field_posed), ("bloch_field", "field", bloch_field),
    ("plan_default", "cylinder", plan_default), ("plan_tight", "cylinder", plan_tight),
    ("plan_carried", "cylinder", plan_carried), ("full_read", "cylinder", full_read), ("explicit_K", "cylinder", explicit_K),
    ("phases_planned", "cylinder", phases_planned), ("walker_range", "cylinder", walker_range),
    ("decoded_twice", "cylinder", decoded_twice), ("plan_windows", "windows", plan_default),
    ("plan_exchange", "permeable", plan_carried),
    ("band_budget", "cylinder", band_budget), ("band_of_pgse", "cylinder", band_of_pgse),
    ("band_of_ogse", "cylinder", band_of_ogse), ("band_beyond", "cylinder", band_beyond),
    ("band_budget_stated", "cylinder", band_budget_stated), ("band_of_ogse_stated", "cylinder", band_of_ogse_stated),
    ("plan_stated", "cylinder", plan_stated), ("band_budget_windows", "windows", band_budget), ("band_of_pgse_windows", "windows", band_of_pgse),
    ("field_plan", "field", field_plan), ("field_modes", "field", field_modes), ("field_full", "field", field_full),
    ("field_beyond_envelope", "field", field_beyond_envelope), ("band_budget_field", "field", band_budget),
]
