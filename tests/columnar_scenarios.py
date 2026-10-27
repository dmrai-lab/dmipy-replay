"""The acquisitions the columnar fixtures are imaged with, written once against a namespace (``ns`` carries the
builders and ``Tissue`` of whichever package runs them), shared by the fixture tool and the tests."""
import numpy as np


def seqs(ns):
    return {
        "pgse_short": ns.set_b(ns.pgse([[1, 0, 0], [0, 0, 1], [0.6, 0.8, 0.0]], 0.6e-3, 1.8e-3, gradient_strengths=0.2,
                                       n_t=401, slew_rate=np.inf, TE=4e-3), [1e9, 2e9, 3e9]),
        "pgse_long": ns.set_b(ns.pgse([[1, 0, 0], [0, 1, 0]], 1.2e-3, 2.4e-3, gradient_strengths=0.1, n_t=401,
                                      slew_rate=np.inf, TE=4e-3), [5e8, 1.5e9]),
    }


def tissue_of(ns, meta):
    """A T2 on every pool of the layout's embedded spec."""
    pools = [p["name"] for p in meta["substrate"]["pools"]]
    return ns.Tissue(T2={n: 0.04 + 0.01 * i for i, n in enumerate(pools)})
