"""After ``dedup_sim.py``: a moved name reached as an ATTRIBUTE of a sim module (``cx.band_projection`` with ``from
dmipy_sim.replay import compression as cx``, ``hc.iso_P6``) is reached through its replay home instead: the module
alias of the replay module is imported beside the sim one and the attribute re-pointed. Prints what it would change;
``--apply`` writes. Run with the sim checkout as the working directory."""
import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dedup_sim import NAMES, absolute  # noqa: E402

SUBSETS = {mod for mod, _ in NAMES}


def module_aliases(tree, path):
    """``{alias: sim module}`` for every import of a subset module as a name."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = absolute(node, path)
            for a in node.names:
                full = f"{base}.{a.name}"
                if full in SUBSETS:
                    out[a.asname or a.name] = full
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in SUBSETS and a.asname:
                    out[a.asname] = a.name
    return out


def main(apply):
    total = 0
    for p in [p for d in ("dmipy_sim", "tests", "examples", "tools") for p in pathlib.Path(d).rglob("*.py")]:
        text = p.read_text()
        tree = ast.parse(text)
        aliases = module_aliases(tree, str(p))
        if not aliases:
            continue
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
                home = NAMES.get((aliases[node.value.id], node.attr))
                if home:
                    hits.append((node.lineno, node.col_offset, node.value.id, node.attr, home))
        if not hits:
            continue
        lines = text.splitlines(keepends=True)
        need = {}
        for ln, col, alias, attr, home in sorted(hits, reverse=True):
            new_alias = "_r_" + home.rsplit(".", 1)[1]
            need[new_alias] = home
            line = lines[ln - 1]
            assert line[col:col + len(alias) + 1 + len(attr)] == f"{alias}.{attr}", (p, ln, line)
            lines[ln - 1] = line[:col] + f"{new_alias}.{attr}" + line[col + len(alias) + 1 + len(attr):]
        imports = "".join(f"import {home} as {a}\n" for a, home in sorted(need.items()))
        first = next((n.lineno - 1 for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                      and getattr(n, "module", None) != "__future__"), 0)
        lines[first:first] = [imports]
        total += len(hits)
        print(f"{p}: {len(hits)} ({', '.join(sorted({h[3] for h in hits}))})")
        if apply:
            p.write_text("".join(lines))
    print(f"{total} attribute reads")


if __name__ == "__main__":
    main("--apply" in sys.argv)
