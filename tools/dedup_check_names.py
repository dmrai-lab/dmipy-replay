"""After the dedup tools: every `from dmipy_sim.X import n` in a sim checkout resolves -- catches a package
re-export the move dropped. Run with the sim checkout as the working directory; prints what does not resolve."""
import ast, importlib, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dedup_sim import absolute
seen, bad = {}, {}
for d in ("dmipy_sim", "tests", "examples", "tools", "docs"):
    for p in pathlib.Path(d).rglob("*.py"):
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.ImportFrom):
                mod = absolute(n, str(p))
                if not mod.startswith("dmipy_sim"):
                    continue
                for a in n.names:
                    if a.name == "*": continue
                    key = (mod, a.name)
                    if key not in seen:
                        try:
                            m = importlib.import_module(mod)
                            ok = hasattr(m, a.name)
                            if not ok:
                                try: importlib.import_module(f"{mod}.{a.name}"); ok = True
                                except ImportError: pass
                        except Exception as e:
                            ok = f"module: {type(e).__name__}"
                        seen[key] = ok
                    if seen[key] is not True:
                        bad.setdefault(key, []).append(f"{p}:{n.lineno}")
for (mod, name), where in sorted(bad.items()):
    print(f"{mod}.{name} [{seen[(mod,name)]}] x{len(where)}: {where[:3]}")
print(len(bad), "unresolved")
