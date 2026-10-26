"""After ``dedup_sim.py``: a moved name a sim module still USES gets an import from its replay home, and a moved name
still listed in a module's ``__all__`` leaves it (sim does not re-export what it no longer defines). Driven by
pyflakes' undefined names; run with the sim checkout as the working directory, repeatedly until it reports nothing."""
import ast
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dedup_sim import NAMES  # noqa: E402

HOME = {}
for (_mod, name), tgt in NAMES.items():
    HOME[name] = tgt

out = subprocess.run([sys.executable, "-m", "pyflakes", "dmipy_sim", "tests", "examples", "tools"],
                     capture_output=True, text=True).stdout
uses, exports = {}, {}
for line in out.splitlines():
    m = re.match(r"(.+?):\d+:\d+:? undefined name '(\w+)'( in __all__)?", line)
    if not m or m.group(2) not in HOME:
        continue
    (exports if m.group(3) else uses).setdefault(m.group(1), set()).add(m.group(2))

for path, names in uses.items():
    text = pathlib.Path(path).read_text()
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    last = 0
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            last = n.end_lineno
        elif last and not isinstance(n, ast.Expr):
            break
    if last == 0:                                      # after the docstring
        last = tree.body[0].end_lineno if tree.body and isinstance(tree.body[0], ast.Expr) else 0
    by = {}
    for nm in sorted(names):
        by.setdefault(HOME[nm], []).append(nm)
    new = "".join(f"from {mod} import {', '.join(ns)}\n" for mod, ns in sorted(by.items()))
    lines[last:last] = [new]
    pathlib.Path(path).write_text("".join(lines))
    print(f"{path}: import {sorted(names)}")

for path, names in exports.items():
    text = pathlib.Path(path).read_text()
    for nm in names:
        text = re.sub(rf'\s*"{nm}",?', "", text, count=1)
    pathlib.Path(path).write_text(text)
    print(f"{path}: __all__ drops {sorted(names)}")
