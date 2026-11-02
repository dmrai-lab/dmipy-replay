"""The optional extras of dmipy-replay (``pose``: scipy and threadpoolctl; ``jax``; ``torch``): what a route that needs
one imports through, so that the core imports numpy alone and a call that needs an extra which is not installed is
refused with the extra's pip spelling."""
import importlib

EXTRAS = {"scipy": "pose", "threadpoolctl": "pose", "jax": "jax", "torch": "torch"}
"""The extra that installs each optional top-level module."""


def need(module):
    """The module ``module`` (``"scipy.special"``, ``"jax.numpy"``, ...), imported; refused, naming the extra that
    installs it, when its package is not installed."""
    top = module.split(".")[0]
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        if not str(getattr(exc, "name", "") or "").startswith(top):
            raise
        extra = EXTRAS[top]
        raise ImportError(f"{module} is not installed: this route needs dmipy-replay's {extra!r} extra "
                          f"(pip install 'dmipy-replay[{extra}]')") from exc
