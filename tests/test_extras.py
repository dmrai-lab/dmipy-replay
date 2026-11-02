"""A route that needs an optional extra and runs where it is not installed is refused with the extra's pip spelling, and
the modules that hold such routes import with numpy alone (each is run here with scipy hidden from the interpreter)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _without_scipy(code):
    head = (f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n"
            "for m in ('scipy', 'scipy.special', 'scipy.spatial', 'scipy.spatial.transform'):\n    sys.modules[m] = None\n")
    r = subprocess.run([sys.executable, "-I", "-c", head + code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_the_modules_import_without_scipy():
    assert _without_scipy("import dmipy_replay.math.sh_analytical, dmipy_replay.replay.so3, dmipy_replay.replay.fod\n"
                          "print('ok')") == "ok"


def test_a_route_that_needs_scipy_names_the_pose_extra():
    out = _without_scipy("import numpy as np\n"
                         "from dmipy_replay.math.sh_analytical import watson_sh\n"
                         "from dmipy_replay.replay import so3\n"
                         "for f in (lambda: watson_sh(np.array([0.0, 0.0, 1.0]), 4.0), lambda: so3.clebsch_gordan(1, 1, 1),\n"
                         "          lambda: so3.haar_rotations(2)):\n"
                         "    try:\n        f()\n    except ImportError as e:\n        print(e)\n")
    lines = out.splitlines()
    assert len(lines) == 3 and all("pip install 'dmipy-replay[pose]'" in l for l in lines), out
