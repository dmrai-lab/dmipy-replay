"""Every relative ``from .x import name`` in the package, function-local ones included, names something the module
defines: the check pyflakes cannot make across modules. Run from the repository root."""
import ast
import importlib
import pathlib
import sys

sys.path.insert(0, ".")
bad = []
for path in pathlib.Path("dmipy_replay").rglob("*.py"):
    mod = ".".join(path.with_suffix("").parts).replace(".__init__", "")
    pkg = mod if path.name == "__init__.py" else mod.rsplit(".", 1)[0]
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = pkg.split(".")[: len(pkg.split(".")) - (node.level - 1)]
            target = ".".join(base + ([node.module] if node.module else []))
            try:
                m = importlib.import_module(target)
            except ImportError as e:
                bad.append(f"{path}:{node.lineno}: {target}: {e}")
                continue
            for a in node.names:
                if not hasattr(m, a.name):
                    try:
                        importlib.import_module(f"{target}.{a.name}")
                    except ImportError:
                        bad.append(f"{path}:{node.lineno}: {target} has no {a.name}")
print("\n".join(bad) or "all relative imports resolve")
sys.exit(1 if bad else 0)
