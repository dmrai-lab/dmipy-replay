"""State an envelope in the header of every fixture pack and columnar layout under ``tests/data``.

A replay is held to the envelope its pack states (``replay_envelope.acquisition``: ``bound`` ``"envelope"``, ``name``,
``band_hz``, ``G_max``, ``B0_list``), and the fixtures are walked without one. They are stamped with a ``G_max`` wide
enough for every scenario of ``tests/scenarios.py`` and the band each fixture stores (the class fields below, read by
:func:`stamp_envelope.stamp_meta`), so the numbers they compare are unchanged and only the header gains the envelope.
Run from the repository root after ``tools/make_fixtures.py`` / ``tools/make_columnar_fixtures.py``; stamping twice is
a no-op.

    python tools/stamp_fixtures.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from dmipy_replay.replay.container import read_safetensors, write_safetensors  # noqa: E402
from stamp_envelope import stamp_meta  # noqa: E402

DATA = ROOT / "tests" / "data"

#: the stamped class: an explicit limit point, far above the scenarios' gradients, read as an envelope at each
#: fixture's stored band
FIXTURE_CLASS = dict(bound="scanner_class", scanner="fixture", G_max=10.0)


def stamp(meta):
    """``meta`` with the fixture's envelope in ``replay_envelope.acquisition``."""
    acq = meta.setdefault("replay_envelope", {}).setdefault("acquisition", {})
    if acq.get("bound") != "envelope":
        acq.update(FIXTURE_CLASS)
    return stamp_meta(meta)[0]


def main():
    for path in sorted(DATA.glob("*.rpk")):
        arrays, hdr = read_safetensors(path)
        key = "rpk" if "rpk" in hdr else "json"
        hdr[key] = json.dumps(stamp(json.loads(hdr[key])))
        write_safetensors(path, {k: v.copy() for k, v in arrays.items()}, hdr)
        print("stamped", path.name)
    for path in sorted(DATA.glob("columnar_*/manifest.json")):
        manifest = json.loads(path.read_text())
        stamp(manifest["meta"])
        path.write_text(json.dumps(manifest))
        print("stamped", path.parent.name)


if __name__ == "__main__":
    main()
