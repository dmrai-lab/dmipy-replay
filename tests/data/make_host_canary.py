"""Writes ``host_canary.npz``: ``exp`` of a fixed real and a fixed imaginary argument on the host that wrote
``bound_regression_golden.npz``.

NumPy's vectorised ``exp`` differs in the last bit between CPU instruction sets, so the golden's bit-for-bit comparison is
made only where the reading host's ``exp`` reproduces these values (``tests/test_bound_column.py``); elsewhere it is held to a
relative 1e-14. Run from the repository root: ``python tests/data/make_host_canary.py``.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def arguments():
    return np.linspace(-12.0, 0.0, 2001)


if __name__ == "__main__":
    x = arguments()
    np.savez_compressed(HERE / "host_canary.npz", exp=np.exp(x), cexp=np.exp(1j * x))
