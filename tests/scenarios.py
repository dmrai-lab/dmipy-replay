"""The parity battery: every replay route this package serves, written once against a namespace so that the same
scenario runs on dmipy-sim (``tools/make_fixtures.py`` records its numbers) and on dmipy-replay (``test_parity.py``
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
    d, bv = _shells(b=np.array([0.0, 1e9]))
    seq = ns.pgse(d, 0.0015, 0.004, bvalues=bv)
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
]
