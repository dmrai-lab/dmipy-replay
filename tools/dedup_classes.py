"""After ``dedup_sim.py``: a class whose reader half moved to dmipy-replay becomes a subclass of the replay class in
the sim checkout. Every method and class attribute the replay class defines is deleted from sim's class (it is
inherited) unless sim keeps it as an override (:data:`CLASSES`); the class's bases become the replay class, imported
as ``_<Name>``. Prints what it would change; ``--apply`` writes. Run with the sim checkout as the working directory.
"""
import ast
import pathlib
import sys

REPLAY = pathlib.Path(__file__).resolve().parents[1] / "dmipy_replay"

#: (sim module path, class, replay module path, replay module, the members sim keeps as its own)
CLASSES = [
    ("dmipy_sim/replay/replay.py", "ReplayPack", "replay/replay.py", "dmipy_replay.replay.replay",
     {"open", "_as_distribution", "BLOCH_BACKENDS"}),
    ("dmipy_sim/replay/columnar.py", "ColumnarPack", "replay/columnar.py", "dmipy_replay.replay.columnar",
     {"__init__", "image", "IMAGE_BACKENDS"}),
    ("dmipy_sim/replay/time_segments.py", "TimeSegmentLayout", "replay/time_segments.py", "dmipy_replay.replay.time_segments",
     {"BACKENDS", "DEFAULT_BACKEND"}),
]


def members(cls_node):
    """``{name: (start, end)}`` (0-based line span, decorators included) of a class body's methods and assignments."""
    out = {}
    for n in cls_node.body:
        name = None
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = n.name
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            name = n.target.id
        if name is None:
            continue
        start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])]) - 1
        out.setdefault(name, []).append((start, n.end_lineno))
    return out


def find_class(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def main(apply):
    for sim_path, cls, replay_path, replay_mod, keep in CLASSES:
        p = pathlib.Path(sim_path)
        text = p.read_text()
        tree = ast.parse(text)
        node = find_class(tree, cls)
        theirs = set(members(find_class(ast.parse((REPLAY / replay_path).read_text()), cls)))
        mine = members(node)
        drop = sorted(n for n in mine if n in theirs and n not in keep)
        lines = text.splitlines(keepends=True)
        spans = sorted((s for n in drop for s in mine[n]), reverse=True)
        for s, e in spans:
            while s > node.lineno and lines[s - 1].lstrip().startswith("#") and lines[s - 1].startswith("    #"):
                s -= 1                                       # a comment block above the member goes with it
            del lines[s:e]
            while s < len(lines) and s > 0 and lines[s].strip() == "" and lines[s - 1].strip() == "":
                del lines[s]
        head = lines[node.lineno - 1]
        base = f"_{cls}"
        if f"class {cls}(" in head or f"class {cls}:" in head:
            lines[node.lineno - 1] = f"class {cls}({base}):\n" if head.rstrip().endswith(":") and "(" not in head else \
                head.replace(f"class {cls}(", f"class {cls}({base}, ", 1)
        new = "".join(lines)
        imp = f"from {replay_mod} import {cls} as {base}\n"
        if imp not in new:
            t2 = ast.parse(new)
            first = next(n for n in t2.body if isinstance(n, (ast.Import, ast.ImportFrom)) and
                         getattr(n, "module", None) != "__future__")
            ls = new.splitlines(keepends=True)
            ls[first.lineno - 1:first.lineno - 1] = [imp]
            new = "".join(ls)
        print(f"{sim_path}: {cls}({base}), drops {len(drop)}: {', '.join(drop)}")
        print(f"   keeps {sorted(n for n in mine if n not in drop)}")
        if apply:
            p.write_text(new)


if __name__ == "__main__":
    main("--apply" in sys.argv)
