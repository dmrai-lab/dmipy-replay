"""Bring the definitions dmipy-replay holds of dmipy-sim's subset modules to a sim revision.

For every top-level function, class and assignment (and every method of a class) of a dmipy-replay module that was
moved as a subset (:data:`SOURCES`), the source text at the cut, at the sim revision and here are compared. sim's text
is read as this package's: ``dmipy_sim`` -> ``dmipy_replay`` and every relative import re-pointed at the replay module
that defines the name (:func:`rewrite`). A definition whose text here is its cut text so read is current or out of
date; one that differs beyond that has diverged here and is listed, not touched.

    python tools/dedup_sync.py SIM_CHECKOUT CUT_REV [--rev REV] [--apply] [--add module:NAME ...]

``--apply`` replaces every out-of-date definition with the revision's text; ``--add`` copies a definition sim added
(``replay/replay.py:ReplayPack.plan`` for a method), placed after the definition that precedes it in sim.
"""
import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "dmipy_replay"
sys.path.insert(0, str(Path(__file__).parent))

#: replay module -> the sim modules its definitions came from
SOURCES = {
    "replay/compression.py": ["replay/compression.py"],
    "replay/replay.py": ["replay/replay.py"],
    "replay/field.py": ["replay/bank.py", "fields/hollow_cylinder.py"],
    "replay/storage.py": ["replay/bank.py"],
    "replay/bloch.py": ["replay/trajectories.py", "engine/bloch.py"],
    "replay/events.py": ["coefficient_walk.py"],
    "replay/so3.py": ["replay/so3.py"],
    "replay/columnar.py": ["replay/columnar.py"],
    "replay/hub.py": ["replay/publish.py"],
    "replay/remote.py": ["replay/remote.py"],
    "replay/time_segments.py": ["replay/time_segments.py"],
}

#: whole modules copied as they are (sim path -> replay path, under the package)
WHOLE = {"constants.py", "replay/decoded.py", "replay/planning.py", "replay/_replay_kernel.py", "phantom/grid.py",
         "math/gradient_conversions.py", "spec/substrate.py", "spec/tissue.py"}
WHOLE |= {f"acquisition/{m}.py" for m in ("scanner_sequence", "rf", "timing", "waveforms", "prescription", "epg",
                                          "epg_state", "scanners", "scanner_constants", "maxwell", "solid_harmonics")}
WHOLE |= {f"sequences/{m}.py" for m in ("builders", "assemble", "_helpers")}


#: names a replay module spells differently from its sim source (bank's aliases of the field's channel tuples)
RENAMES = {"replay/field.py": (("_PATH_CHANNELS", "CHANNEL_NAMES"), ("_ISO_PATH_CHANNELS", "ISO_CHANNEL_NAMES"))}


def defs(text):
    """``{name: source}`` of the top-level functions, classes and assignments, and ``Class.method`` of every method
    (a property's setter is ``Class.name#2``)."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    out = {}

    def src(node):
        start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
        return "".join(lines[start:node.end_lineno])

    def put(key, value):
        k, i = key, 2
        while k in out:
            k, i = f"{key}#{i}", i + 1
        out[k] = value

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            put(node.name, src(node))
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        put(f"{node.name}.{sub.name}", src(sub))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    put(t.id, src(node))
    return out


def show(sim, rev, path):
    r = subprocess.run(["git", "-C", sim, "show", f"{rev}:dmipy_sim/{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def norm(s):
    return s.replace("dmipy_sim", "dmipy_replay")


def home_of(sim_module, name, extra=()):
    """The replay module (a path under the package) that defines ``name`` of the sim module ``sim_module`` (a path
    under ``dmipy_sim``), or None. ``extra`` are ``(replay module, name)`` pairs about to be added."""
    if sim_module in WHOLE:
        return sim_module
    if sim_module.endswith("__init__.py"):
        return None
    for mod, srcs in SOURCES.items():
        if sim_module in srcs:
            if name in defs((HERE / mod).read_text()) or (mod, name) in extra:
                return mod
    sub = sim_module[:-3] + f"/{name}.py"                                 # from ..acquisition import rf
    if sub in WHOLE or sub in SOURCES:
        return "MODULE:" + sub
    return None


_IMPORT = re.compile(r"^(?P<indent>[ \t]*)from (?P<dots>\.+)(?P<mod>[\w.]*) import (?P<names>\([^)]*\)|[^\n#]*)", re.M)


def _relative(target, here):
    """The relative spelling of the replay module path ``target`` from the replay module path ``here``."""
    t = target[:-3].split("/")
    h = here[:-3].split("/")[:-1]
    i = 0
    while i < len(h) and i < len(t) - 1 and h[i] == t[i]:
        i += 1
    return "." * (len(h) - i + 1) + ".".join(t[i:])


def rewrite(text, sim_module, replay_module, extra=(), unresolved=None):
    """sim's ``text`` from ``sim_module`` as it reads in ``replay_module``: the package renamed and every relative
    import of a name pointed at the replay module that defines it. A name no replay module defines keeps its import
    and is added to ``unresolved``."""
    def sub(m):
        level = len(m["dots"])
        base = sim_module[:-3].split("/")[:-1]
        base = base[:len(base) - (level - 1)]
        mod = "/".join(base + ([*m["mod"].split(".")] if m["mod"] else []))
        names = [n.strip() for n in m["names"].strip("()").replace("\n", " ").split(",") if n.strip()]
        groups = {}
        for n in names:
            name = n.split(" as ")[0].strip()
            src = None
            for cand in (mod + ".py", mod + "/__init__.py"):
                src = home_of(cand, name, extra)
                if src:
                    break
            if src is None:
                if unresolved is not None:
                    unresolved.append(f"{mod}.{name}")
                tgt = "." * level + m["mod"]
            elif src.startswith("MODULE:"):
                p = src[len("MODULE:"):]
                tgt = _relative(str(Path(p).parent / "__init__.py"), replay_module)
                tgt = tgt[:-len(".__init__")] if tgt.endswith(".__init__") and tgt.strip(".") != "__init__" else tgt[:-len("__init__")]
            else:
                tgt = _relative(src, replay_module)
            groups.setdefault(tgt, []).append(n)
        return "\n".join(f"{m['indent']}from {t} import {', '.join(ns)}" for t, ns in groups.items())
    out = _IMPORT.sub(sub, norm(text))
    for a, b in RENAMES.get(replay_module, ()):
        out = re.sub(rf"\b{a}\b", b, out)
    return out


def _place(text, name, new_src, sim_order, mine):
    """``text`` with ``new_src`` (the definition ``name``) inserted after the nearest definition before it in sim's
    order that ``text`` holds."""
    cls = name.split(".")[0] if "." in name else None
    i = sim_order.index(name)
    for prev in reversed(sim_order[:i]):
        same_scope = (prev.split(".")[0] == cls and "." in prev) if cls else ("." not in prev)
        if same_scope and prev in mine:
            at = text.index(mine[prev]) + len(mine[prev])
            sep = "\n" if cls else "\n\n\n"
            return text[:at] + sep + new_src + text[at:]
    if cls:                                                               # first method: after the class's docstring
        body = mine[cls]
        head = body.split('"""', 2)
        at = text.index(body) + (len(head[0]) + len(head[1]) + 6 + 1 if len(head) == 3 else body.index(":\n") + 2)
        return text[:at] + "\n" + new_src + text[at:]
    return text.rstrip("\n") + "\n\n\n" + new_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim")
    ap.add_argument("cut")
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--add", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None, help="replay modules to report/apply")
    a = ap.parse_args()
    extra = {tuple(s.split(":")) for s in a.add}
    extra |= {(m, n.split(".")[0]) for m, n in list(extra)}
    for mod, srcs in SOURCES.items():
        if a.only is not None and mod not in a.only:
            continue
        if not (HERE / mod).exists():
            continue
        text = (HERE / mod).read_text()
        mine = defs(text)
        cut, new, origin, order = {}, {}, {}, []
        for s in srcs:
            for k, v in defs(show(a.sim, a.cut, s)).items():
                cut.setdefault(k, (v, s))
            for k, v in defs(show(a.sim, a.rev, s)).items():
                if k not in new:
                    new[k] = (v, s)
                    order.append(k)
        print(f"== {mod}")
        unresolved = []
        for name in list(mine):
            if name not in cut or "." not in name and name in {n.split(".")[0] for n in mine if "." in n}:
                continue                                                   # a class is its methods
            if name not in new:
                print(f"  REMOVED in sim   {name}")
                continue
            c = rewrite(cut[name][0], cut[name][1], mod)
            n = rewrite(new[name][0], new[name][1], mod, extra, unresolved)
            if c == n:
                continue
            if mine[name] == c or mine[name] == norm(cut[name][0]):
                print(f"  out of date      {name}")
                if a.apply:
                    text = text.replace(mine[name], n, 1)
            elif mine[name] != n:
                print(f"  DIVERGED         {name}")
        for spec in a.add:
            m, name = spec.split(":")
            if m != mod:
                continue
            mine = defs(text)
            if name in mine:
                print(f"  already here     {name}")
                continue
            src = rewrite(new[name][0], new[name][1], mod, extra, unresolved)
            text = _place(text, name, src, order, mine)
            print(f"  added            {name}")
        for u in sorted(set(unresolved)):
            print(f"  UNRESOLVED import {u}")
        added = [k for k in new if k not in cut and "." not in k and k not in defs(text)]
        if added:
            print(f"  sim added (not here): {', '.join(added)}")
        if a.apply or a.add:
            (HERE / mod).write_text(text)


if __name__ == "__main__":
    main()
