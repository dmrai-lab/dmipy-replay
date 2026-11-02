"""The names dmipy-replay refuses by name, each with its one spelling: a name spelled another way, or a form whose one
copy is dmipy-analytic's.

A module that refuses names defines ``__getattr__`` through :func:`refuse`: ``dmipy_replay.calc_b`` raises an
``AttributeError`` that says ``seq.b()``, ``dmipy_replay.GAMMA`` one that says
``GAMMA (from dmipy_analytic.constants import GAMMA)``, and ``from dmipy_replay import calc_b`` raises an
``ImportError`` that says it too (CPython drops the ``AttributeError`` of a module's ``__getattr__`` inside a ``from``
import and reports only that the name cannot be imported). A name that is none of these is the ordinary
``AttributeError``. A module whose forms all moved to dmipy-analytic (``dmipy_replay.math.sh_analytical``,
``dmipy_replay.acquisition.epg_state``) is a ``ModuleNotFoundError``: there is no module stub."""
import dis
import sys


def _analytic(module, *names):
    """``{name: its spelling}`` of forms that are ``dmipy_analytic.<module>``'s."""
    return {n: f"{n} (from dmipy_analytic.{module} import {n})" for n in names}


SPELLINGS = {
    "dmipy_replay": {"calc_b": "seq.b()", "calc_btensor": "seq.btensor()", **_analytic("constants", "GAMMA")},
    "dmipy_replay.constants": _analytic("constants", "GAMMA", "GAMMA_BAR"),
    "dmipy_replay.acquisition.waveforms": {"calc_b": "seq.b()", "calc_btensor": "seq.btensor()"},
    "dmipy_replay.acquisition.epg": _analytic("epg", "Pulse", "Winding", "Schedule", "Pathway", "enumerate_pathways",
                                              "splice_schedule", "cpmg_schedule", "ste_schedule", "ste_amplitude"),
    "dmipy_replay.replay.so3": _analytic("harmonics", "real_sh", "n_sh_coeffs", "sh_block", "wigner_blocks"),
    "dmipy_replay.spec.substrate": {"Pool": "PoolSpec (from dmipy_replay.spec.substrate import PoolSpec)"},
    "dmipy_replay.replay.columnar": {"Source": "ColumnSource (from dmipy_replay.replay.columnar import ColumnSource)"},
}
"""``{module: {refused name: its one spelling}}``, stated whole: a method of a sequence (``seq.b()``), or a name with
the import that brings it (``PoolSpec (from dmipy_replay.spec.substrate import PoolSpec)``,
``GAMMA (from dmipy_analytic.constants import GAMMA)``)."""


def _from_import():
    """Whether the lookup that ``refuse`` answers is the one of a ``from module import name`` statement: the
    statement's frame is three frames up (``refuse`` is called by a module's ``__getattr__``, which the statement
    runs), standing at its ``IMPORT_FROM`` instruction."""
    try:
        frame = sys._getframe(3)
    except ValueError:
        return False
    return frame.f_code.co_code[frame.f_lasti] == dis.opmap["IMPORT_FROM"]


def refuse(module, name):
    """The exception of ``module.name``: the refusal that says the spelling of a refused name (an ``ImportError`` in a
    ``from module import name`` statement, an ``AttributeError`` otherwise, so ``hasattr`` stays false), the ordinary
    ``AttributeError`` of any other name. Called as ``raise refuse(__name__, name)`` by a module's ``__getattr__``."""
    spelling = SPELLINGS.get(module, {}).get(name)
    if spelling is None:
        return AttributeError(f"module {module!r} has no attribute {name!r}")
    message = f"{module}.{name} does not exist: use {spelling}"
    return ImportError(message, name=module) if _from_import() else AttributeError(message)
