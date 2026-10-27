"""Rewrite a dmipy-sim checkout to import what moved to dmipy-replay (docs/dedup-ledger.md): every
``from <moved module> import ...`` and every import of a moved name from a subset module points at its replay home,
absolute; the moved files and definitions are deleted. No alias is left behind -- a caller of a moved name imports
it from dmipy-replay. Run with the sim checkout as the working directory:

    python /path/to/dmipy-replay/tools/dedup_sim.py [--apply]

Without ``--apply`` it prints what it would change.
"""
import ast
import os
import pathlib
import sys

REPLAY = pathlib.Path(__file__).resolve().parents[1] / "dmipy_replay"

#: whole modules: sim -> replay
MODULES = {f"dmipy_sim.acquisition.{m}": f"dmipy_replay.acquisition.{m}" for m in
           ("scanner_sequence", "rf", "timing", "waveforms", "prescription", "epg", "epg_state", "scanners",
            "scanner_constants", "maxwell", "solid_harmonics")}
MODULES.update({f"dmipy_sim.sequences.{m}": f"dmipy_replay.sequences.{m}" for m in ("builders", "assemble", "_helpers")})
MODULES.update({"dmipy_sim.constants": "dmipy_replay.constants",
                "dmipy_sim.math.gradient_conversions": "dmipy_replay.math.gradient_conversions",
                "dmipy_sim.spec.substrate": "dmipy_replay.spec.substrate",
                "dmipy_sim.spec.tissue": "dmipy_replay.spec.tissue",
                "dmipy_sim.replay._replay_kernel": "dmipy_replay.replay._replay_kernel",
                "dmipy_sim.phantom.grid": "dmipy_replay.phantom.grid"})
FILES = {"dmipy_sim/acquisition/scanner_constants.json", "dmipy_sim/acquisition/scanner_catalogue.schema.json",
         "dmipy_sim/spec/substrate.schema.json"}


def _defs(path):
    tree = ast.parse(pathlib.Path(path).read_text())
    out = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    return out


#: names out of subset modules: (sim module, name) -> replay module
NAMES = {}
for name in _defs(REPLAY / "replay" / "compression.py") - {"log"}:
    NAMES[("dmipy_sim.replay.compression", name)] = "dmipy_replay.replay.compression"
for name in ("frame_from_axis", "has_susc_path", "susc_path_coeffs", "susc_path_field", "susc_path_decode",
             "path_field_integral", "_refuse_trace_component", "_mode_sample_matrix"):
    NAMES[("dmipy_sim.replay.bank", name)] = "dmipy_replay.replay.field"
for name in ("CHANNEL_NAMES", "ISO_CHANNEL_NAMES", "q_of_H", "iso_P6", "iso_weights", "field_terms", "contract"):
    NAMES[("dmipy_sim.fields.hollow_cylinder", name)] = "dmipy_replay.replay.field"
for name in ("_rf_increment", "finite_180_longitudinal_dwell", "_BlochTerms", "_bloch_timeline", "_bloch_replay_terms",
             "_bloch_replay_output"):
    NAMES[("dmipy_sim.replay.trajectories", name)] = "dmipy_replay.replay.bloch"
NAMES[("dmipy_sim.engine.bloch", "_build_crusher")] = "dmipy_replay.replay.bloch"
for name in ("EventList", "events_label_at", "events_at_instants", "events_within"):
    NAMES[("dmipy_sim.coefficient_walk", name)] = "dmipy_replay.replay.events"
NAMES[("dmipy_sim.replay.so3", "rotation_of")] = "dmipy_replay.replay.so3"
for name in ("REFUSED_KEYS", "REFUSED_CHANNEL_KEYS", "REFUSED_COLUMN_KINDS", "REFUSED_TENSORS", "REFUSED_TENSOR_INFIX",
             "_refuse_predating", "ScannerField", "scanner_field", "declared_susceptibility_field", "FieldTerm",
             "field_term", "_pose_matrix", "_duration", "_echo_blocks", "compile_scheme", "_container_of",
             "_compile_continuous", "_band_phase", "_path_field_channels", "_has_aniso", "_signal_factor",
             "_gated_contact", "surface_logweight", "replay_signal", "replay_coefficients"):
    NAMES[("dmipy_sim.replay.replay", name)] = "dmipy_replay.replay.replay"
for name in ("STORAGE_BAND_MARGIN", "storage_band_hz", "mode_count"):
    NAMES[("dmipy_sim.replay.bank", name)] = "dmipy_replay.replay.storage"
for name in ("is_hub_uri", "parse_uri", "fetch", "fetch_windows", "window_prefix", "_coalesce", "_window_tensor_names"):
    NAMES[("dmipy_sim.replay.publish", name)] = "dmipy_replay.replay.hub"
for name in ("from_pulseq", "pulseq_timing"):
    NAMES[("dmipy_sim.sequences.pulseq", name)] = "dmipy_replay.sequences.pulseq"

#: definitions deleted from the subset modules (they live in replay now)
DELETE = {}
for (mod, name), _ in NAMES.items():
    DELETE.setdefault(mod, set()).add(name)
DELETE["dmipy_sim.replay.publish"] |= {"_ST_DTYPES", "_auth_headers", "_range_get", "_fetch_header", "_read_cached_windows",
                                       "_windows_cache_path", "_fetch_windows_to"}
DELETE["dmipy_sim.sequences.pulseq"] |= {"_rf_from_pulseq", "_event_times", "_gradient_live_spans", "_has_adc",
                                         "_decode_rf_events"}

#: a package that star-re-exports a subset module re-exported its moved names too (``from dmipy_sim.replay import
#: compile_scheme``): that spelling is rewritten like the module's own; nothing is deleted from the package
STAR_REEXPORTS = {"dmipy_sim.replay": "dmipy_sim.replay.replay"}
for pkg, mod in STAR_REEXPORTS.items():
    for (m, name), tgt in list(NAMES.items()):
        if m == mod:
            NAMES[(pkg, name)] = tgt


def module_of(path):
    p = pathlib.Path(path).with_suffix("")
    parts = list(p.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def absolute(node, path):
    """The absolute module an ImportFrom names, from the file it is in."""
    if not node.level:
        return node.module or ""
    mod = module_of(path)
    pkg = mod if pathlib.Path(path).name == "__init__.py" else mod.rsplit(".", 1)[0]
    base = pkg.split(".")
    base = base[:len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _alias(a):
    return a.name if a.asname is None else f"{a.name} as {a.asname}"


def _statement(indent, module, aliases, comment):
    body = ", ".join(_alias(a) for a in aliases)
    line = f"{indent}from {module} import {body}"
    if len(line) > 118:
        inner = ",\n".join(f"{indent}    {_alias(a)}" for a in aliases)
        line = f"{indent}from {module} import (\n{inner},\n{indent})"
    return line + (f"  {comment}" if comment else "")


def rewrite_imports(path, text):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        mod = absolute(node, path)
        if not mod.startswith("dmipy_sim"):
            continue
        groups = {}                                         # target module -> aliases, in order
        for a in node.names:
            if mod in MODULES:
                tgt = MODULES[mod]
            elif (mod, a.name) in NAMES:
                tgt = NAMES[(mod, a.name)]
            elif a.name != "*" and f"{mod}.{a.name}" in MODULES:         # from dmipy_sim.acquisition import rf
                groups.setdefault(MODULES[f"{mod}.{a.name}"].rsplit(".", 1)[0], []).append(a)
                continue
            else:
                tgt = None
            groups.setdefault(tgt, []).append(a)
        if list(groups) == [None]:
            continue
        first, last = node.lineno - 1, node.end_lineno - 1
        indent = lines[first][:len(lines[first]) - len(lines[first].lstrip())]
        tail = lines[last].rstrip("\n")
        comment = tail[tail.index("#"):] if "#" in tail and "#" not in ast.get_source_segment(text, node) else ""
        stmts = []
        for tgt, al in groups.items():
            if tgt is None:
                stmts.append(_statement(indent, "." * node.level + (node.module or ""), al, ""))
            else:
                stmts.append(_statement(indent, tgt, al, ""))
        if comment:
            stmts[-1] += f"  {comment}"
        edits.append((first, last, "\n".join(stmts) + "\n"))
    for first, last, new in sorted(edits, reverse=True):
        lines[first:last + 1] = [new]
    text, n = "".join(lines), len(edits)
    # a module named in a string (``__import__("dmipy_sim.spec.substrate")``, ``importlib.import_module``)
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Constant) and node.value in MODULES:
            lines = text.splitlines(keepends=True)
            line = lines[node.lineno - 1]
            seg = line[node.col_offset:node.end_col_offset]
            lines[node.lineno - 1] = (line[:node.col_offset] + seg.replace(node.value, MODULES[node.value])
                                      + line[node.end_col_offset:])
            text, n = "".join(lines), n + 1
    return text, n


def delete_definitions(path, text, names):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    spans = []
    for n in tree.body:
        name = None
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            name = n.name
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
        if name in names:
            start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])]) - 1
            while start > 0 and lines[start - 1].lstrip().startswith("#") and lines[start - 1].strip() != "#":
                start -= 1                                   # the comment block above it goes with it
            end = n.end_lineno
            # a docstring right after an assignment documents it
            if isinstance(n, ast.Assign):
                nxt = tree.body[tree.body.index(n) + 1] if tree.body.index(n) + 1 < len(tree.body) else None
                if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant) and isinstance(nxt.value.value, str):
                    end = nxt.end_lineno
            spans.append((start, end))
    for s, e in sorted(spans, reverse=True):
        del lines[s:e]
    return "".join(lines), len(spans)


def main(apply):
    root = pathlib.Path(".")
    files = [p for d in ("dmipy_sim", "tests", "examples", "tools", "docs") for p in (root / d).rglob("*.py")]
    moved_files = {pathlib.Path(*m.split(".")).with_suffix(".py") for m in MODULES}
    total = 0
    for p in files:
        if p in moved_files:
            continue
        text = p.read_text()
        new, n = rewrite_imports(str(p), text)
        mod = module_of(p)
        if mod in DELETE:
            new, k = delete_definitions(str(p), new, DELETE[mod])
            n += k
        if n:
            total += n
            print(f"{p}: {n}")
            if apply:
                p.write_text(new)
    for p in sorted(moved_files) + sorted(pathlib.Path(f) for f in FILES):
        print(f"delete {p}")
        if apply and p.exists():
            os.remove(p)
    print(f"{total} edits")


if __name__ == "__main__":
    main("--apply" in sys.argv)
