"""The engine of the replay move's parity record (dmrai-lab/dmipy-sim#834): run a group of :mod:`calls` under a spelling
of :mod:`apis`, flatten what it returns, write it as a record, and compare a record with another.

A record is a directory: ``manifest.json`` (the environment it was taken in and the packages' commits) and per group
``<group>.json`` (every value: an array's dtype, shape and sha256, a scalar or a string exactly, bytes by their
sha256) with ``<group>.npz`` (the arrays). Two records agree on a group when every key is in both and every value is
identical: arrays by ``np.array_equal(..., equal_nan=True)`` and their sha256, the rest exactly.

Every group runs in a fresh interpreter with one BLAS and XLA thread, a fixed hash seed, JAX on its CPU and the
packages' disk caches in an empty directory, so that a value cannot come from an earlier call or another order."""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "JAX_PLATFORMS": "cpu",
    "PYTHONHASHSEED": "0", "CUDA_VISIBLE_DEVICES": "",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}
THREADS = {"before": "DMIPY_SIM_CPU_THREADS", "after": "DMIPY_REPLAY_CPU_THREADS"}
"""The variable each spelling's packages read their CPU thread cap from (set to 1; the other one is unset)."""


def _env(cache, spelling):
    env = dict(os.environ, **ENV)
    for s, var in THREADS.items():
        env.pop(var, None)
    env[THREADS[spelling]] = "1"
    for var in ("DMIPY_SIM_CACHE", "DMIPY_REPLAY_CACHE", "DMIPY_SIM_RUN_DIR"):
        env[var] = str(Path(cache) / var.lower())
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), str(HERE.parent)] + [p for p in [env.get("PYTHONPATH")] if p])
    return env


# --------------------------------------------------------------------------------------------- values -> a record
def _sha(b):
    return hashlib.sha256(b).hexdigest()


def flatten(values, prefix=""):
    """``{key: value}`` with every dict and list flattened into ``key/child`` keys."""
    out = {}
    for k, v in values.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "/"))
        elif isinstance(v, (list, tuple)) and any(hasattr(x, "shape") or isinstance(x, (dict, list, tuple)) for x in v):
            out.update(flatten({str(i): x for i, x in enumerate(v)}, key + "/"))
        else:
            out[key] = v
    return out


def encode(values):
    """``(entries, arrays)``: the JSON entry of every value and the arrays to store."""
    import numpy as np
    entries, arrays = {}, {}
    for k, v in flatten(values).items():
        if isinstance(v, (bytes, bytearray)):
            entries[k] = {"kind": "bytes", "sha256": _sha(bytes(v)), "n": len(v)}
        elif hasattr(v, "shape") and hasattr(v, "dtype") and not np.isscalar(v):
            a = np.asarray(v)
            if a.dtype == object:
                entries[k] = {"kind": "repr", "value": repr(a.tolist())}
                continue
            a = np.ascontiguousarray(a)
            name = f"a{len(arrays)}"
            entries[k] = {"kind": "array", "dtype": a.dtype.str, "shape": list(a.shape), "sha256": _sha(a.tobytes()),
                          "npz": name}
            arrays[name] = a
        elif isinstance(v, (np.generic,)):
            entries[k] = {"kind": "scalar", "dtype": v.dtype.str, "value": repr(v.item())}
        elif v is None or isinstance(v, (bool, int, float, complex, str)):
            entries[k] = {"kind": type(v).__name__, "value": v if not isinstance(v, (float, complex)) else repr(v)}
        elif isinstance(v, (list, tuple)):
            entries[k] = {"kind": "list", "value": json.dumps(v, default=repr)}
        else:
            entries[k] = {"kind": "repr", "value": repr(v)}
    return entries, arrays


def run_group(group, spelling):
    """``(status, values or reason, seconds)`` of one group in this interpreter."""
    import apis
    import calls
    api = apis.SPELLINGS[spelling]()
    t0 = time.perf_counter()
    try:
        values = calls.GROUPS[group](api)
    except apis.Pending as exc:
        return "pending", str(exc), time.perf_counter() - t0
    return "ok", values, time.perf_counter() - t0


def _child(group, spelling, out_dir):
    status, values, seconds = run_group(group, spelling)
    out = Path(out_dir)
    if status == "pending":
        (out / f"{group}.json").write_text(json.dumps({"status": "pending", "reason": values, "seconds": seconds}, indent=1))
        return
    import numpy as np
    entries, arrays = encode(values)
    np.savez_compressed(out / f"{group}.npz", **arrays)
    (out / f"{group}.json").write_text(json.dumps({"status": "ok", "seconds": seconds, "entries": entries}, indent=1,
                                                  sort_keys=True))


def _git_commit(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _package(name, module):
    """The installed package's version and the commit it was installed from."""
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    info = {"version": dist.version}
    direct = dist.read_text("direct_url.json")
    if direct:
        d = json.loads(direct)
        info["commit"] = (d.get("vcs_info") or {}).get("commit_id")
        if info["commit"] is None and d.get("url", "").startswith("file://"):
            info["commit"] = _git_commit(d["url"][7:])
    return info


def manifest(spelling):
    """The environment a record is taken in: what decides the last bit of a sum besides the code."""
    code = """
import json, platform, sys
import numpy, scipy
info = dict(python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__)
try:
    import jax, jaxlib; info.update(jax=jax.__version__, jaxlib=jaxlib.__version__)
except ImportError: pass
try:
    import torch; info.update(torch=torch.__version__)
except ImportError: pass
try:
    from threadpoolctl import threadpool_info
    info["blas"] = [dict(api=p.get("internal_api"), version=p.get("version"), threads=p.get("num_threads"),
                         arch=p.get("architecture")) for p in threadpool_info() if p.get("user_api") == "blas"]
except ImportError: pass
print(json.dumps(info))
"""
    with tempfile.TemporaryDirectory() as cache:
        env = _env(cache, spelling)
        libs = json.loads(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                         check=True).stdout)
    cpu = None
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        cpu = platform.processor() or None
    return {"spelling": spelling, "packages": {"dmipy-replay": _package("dmipy-replay", "dmipy_replay"),
                                               "dmipy-sim": _package("dmipy-sim", "dmipy_sim")},
            "libraries": libs, "machine": platform.machine(), "cpu": cpu, "env": dict(ENV, **{THREADS[spelling]: "1"})}


def record(out_dir, spelling, groups=None):
    """Run every group (or ``groups``) under ``spelling``, each in a fresh interpreter, into ``out_dir``."""
    import calls
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest(spelling), indent=1, sort_keys=True))
    for group in groups or list(calls.GROUPS):
        with tempfile.TemporaryDirectory() as cache:
            t0 = time.perf_counter()
            r = subprocess.run([sys.executable, str(HERE / "parity.py"), "_child", group, spelling, str(out)], env=_env(cache, spelling),
                               capture_output=True, text=True)
            if r.returncode:
                (out / f"{group}.json").write_text(json.dumps({"status": "failed", "stderr": r.stderr[-20000:]}, indent=1))
            status = json.loads((out / f"{group}.json").read_text())["status"]
            print(f"{group:24s} {status:8s} {time.perf_counter() - t0:7.1f} s", flush=True)
            if r.returncode:
                print(r.stderr[-4000:], file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------- record vs record
def _load(d, group):
    import numpy as np
    j = json.loads((Path(d) / f"{group}.json").read_text())
    if j.get("status") == "ok":
        with np.load(Path(d) / f"{group}.npz") as z:
            j["arrays"] = {k: z[e["npz"]] for k, e in j["entries"].items() if e["kind"] == "array"}
    return j


def compare(ref_dir, new_dir, group):
    """``(status, differences)`` of one group: ``identical``, ``DIFFERENT``, ``pending`` or ``failed``."""
    import numpy as np
    a, b = _load(ref_dir, group), _load(new_dir, group)
    if a.get("status") != "ok":
        return "failed", [f"the reference did not run: {a.get('stderr', a.get('reason'))}"]
    if b.get("status") == "pending":
        return "pending", [b["reason"]]
    if b.get("status") != "ok":
        return "failed", [b.get("stderr", "")[-3000:]]
    diffs = []
    ea, eb = a["entries"], b["entries"]
    for k in sorted(set(ea) | set(eb)):
        if k not in eb:
            diffs.append(f"{k}: missing")
        elif k not in ea:
            diffs.append(f"{k}: not in the reference")
        elif {**ea[k], "npz": None} != {**eb[k], "npz": None}:
            diffs.append(f"{k}: {json.dumps(ea[k])[:300]} != {json.dumps(eb[k])[:300]}")
        elif ea[k]["kind"] == "array" and not np.array_equal(a["arrays"][k], b["arrays"][k], equal_nan=True):
            diffs.append(f"{k}: the arrays differ")
    return ("identical" if not diffs else "DIFFERENT"), diffs


def compare_records(ref_dir, other_dir, groups=None):
    """Compare two records group by group without running anything; returns the exit status."""
    import calls
    worst = 0
    for group in groups or list(calls.GROUPS):
        status, diffs = compare(ref_dir, other_dir, group)
        print(f"{group:24s} {status:10s}")
        for d in diffs[:10]:
            print("    " + d)
        worst |= status != "identical"
    return worst


def check(ref_dir, spelling, groups=None, require_all=False, keep=None):
    """Run the groups under ``spelling`` and compare them with the record in ``ref_dir``; returns the exit status."""
    import calls
    groups = groups or list(calls.GROUPS)
    with tempfile.TemporaryDirectory() as tmp:
        new = Path(keep or tmp)
        record(new, spelling, groups)
        print()
        worst = 0
        ref_env = json.loads((Path(ref_dir) / "manifest.json").read_text())
        new_env = json.loads((new / "manifest.json").read_text())
        if ref_env["libraries"] != new_env["libraries"] or ref_env["cpu"] != new_env["cpu"]:
            print("note: the environment differs from the record's:", json.dumps(ref_env["libraries"]), ref_env["cpu"],
                  "vs", json.dumps(new_env["libraries"]), new_env["cpu"])
        for group in groups:
            status, diffs = compare(ref_dir, new, group)
            n = len(json.loads((Path(ref_dir) / f"{group}.json").read_text()).get("entries", {}))
            print(f"{group:24s} {status:10s} {n:5d} values")
            for d in diffs[:40]:
                print("    " + d)
            if status in ("DIFFERENT", "failed") or (status == "pending" and require_all):
                worst = 1
        return worst


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "_child":
    sys.path[:0] = [str(HERE), str(HERE.parent)]
    _child(*sys.argv[2:5])
