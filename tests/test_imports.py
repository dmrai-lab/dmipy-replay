import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_relative_imports_resolve():
    """Every relative import, function-local ones included, names something its module defines."""
    r = subprocess.run([sys.executable, "tools/check_imports.py"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
