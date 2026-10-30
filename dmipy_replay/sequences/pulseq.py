"""Pulseq (``.seq``) files read natively: the file's own text, numpy, no pypulseq.

Pulseq (Layton et al., MRM 2017) is the open, vendor-neutral sequence format. A file is plain text in sections --
``[VERSION]``, ``[DEFINITIONS]``, ``[BLOCKS]`` (one row per block: its duration and the ids of its RF, three gradient
and ADC events), ``[RF]``, ``[GRADIENTS]`` (arbitrary and extended-trapezoid gradients), ``[TRAP]``, ``[ADC]``,
``[EXTENSIONS]`` and ``[SHAPES]`` (run-length coded on the derivative) -- and every event is stated in it: a gradient
as its corners or its raster samples, an RF pulse as its amplitude (Hz), magnitude and phase shapes, raster, delay,
frequency and phase offset and (format 1.5) its use. Formats 1.4 and 1.5 are read.

:func:`from_pulseq` turns a file into a :class:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence` (one
measurement, ``G`` of shape ``(1, n_t, 3)`` in T/m): the gradient rasterised exactly onto a uniform grid anchored at
the excitation (piecewise-linear between the file's own points, zero outside its events), every RF pulse an
:class:`~dmipy_replay.acquisition.rf.RFEvent` carrying the PLAYED B1(t) as its envelope -- so its flip is
``gamma int |B1| dt`` of the file, its phase the file's, and a transmit scale (``replay_bloch(b1_scale=)``) acts on
the real pulse -- and the readout at the ADC's centre. A file dmipy wrote (``dmipy_*`` definitions) carries its
exact schedule, grid and budget, which are preferred to what the blocks raster. Units: Pulseq's Hz/m and Hz with
``gamma / 2 pi`` (:data:`~dmipy_replay.constants.GAMMA_BAR`) at the boundary.

Not interpreted, and refused by name rather than dropped: rotation extensions (they change which axis a gradient
plays on). Labels, triggers and other extensions do not change what the spins see and are ignored.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..constants import GAMMA_BAR
from ..acquisition.rf import B1Pulse, RFEvent, RFSchedule
from ..acquisition.timing import SequenceTiming
from ..acquisition.scanner_sequence import ScannerSequence

__all__ = ["read_seq", "from_pulseq", "pulseq_timing", "PulseqFile"]

_USE = {"e": "excitation", "r": "refocusing", "i": "inversion", "s": "saturation", "p": "preparation", "o": "other",
        "u": "undefined"}


# ------------------------------------------------------------------ the file
def decompress_shape(num_samples, data):
    """A ``[SHAPES]`` entry's samples: stored as is when it holds ``num_samples`` values, else the run-length code of
    its derivative (a value written twice is followed by its extra repeat count), integrated."""
    data = np.asarray(data, np.float64)
    n = int(num_samples)
    if data.size == n:
        return data
    out = np.zeros(n)
    pack = unpack = 0
    for m in np.flatnonzero(np.diff(data) == 0.0):      # a repeated value marks a run; a later marker inside one is not
        if m < pack:
            continue
        if m > pack:
            out[unpack:unpack + m - pack] = data[pack:m]
            unpack += m - pack
            pack = m
        rep = int(data[pack + 2] + 2)
        out[unpack:unpack + rep] = data[pack]
        pack += 3
        unpack += rep
    if pack < data.size:
        if data.size - pack != n - unpack:
            raise ValueError(f"a shape decodes to {unpack + data.size - pack} samples and declares {n}")
        out[unpack:] = data[pack:]
    return np.cumsum(out)


class PulseqFile:
    """A ``.seq`` file parsed into its tables: ``version`` ``(major, minor, revision)``, ``definitions`` (strings,
    numbers or lists), ``blocks`` ``{id: (duration_s, rf, gx, gy, gz, adc, ext)}`` in file order, ``rf``,
    ``grad`` (``("trap", ...)`` or ``("grad", ...)``), ``adc`` and ``shapes`` (decoded) by id, and the raster times.
    Nothing is interpreted beyond the units."""

    def __init__(self, text):
        self.version = (0, 0, 0)
        self.definitions, self.blocks, self.rf, self.grad, self.adc, self.shapes = {}, {}, {}, {}, {}, {}
        self.extensions = []
        sections = {}
        cur = None
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                cur = line
                sections.setdefault(cur, [])
                continue
            if cur is not None:
                sections[cur].append(line)
        for line in sections.get("[VERSION]", []):
            k, v = line.split()[:2]
            i = {"major": 0, "minor": 1, "revision": 2}.get(k)
            if i is not None:
                ver = list(self.version)
                ver[i] = int(v.split(".")[0]) if i < 2 else int("".join(c for c in v if c.isdigit()) or 0)
                self.version = tuple(ver)
        if self.version[:2] < (1, 4):
            raise ValueError(f"Pulseq format {'.'.join(map(str, self.version))}: formats 1.4 and 1.5 are read; "
                             "convert an older file with the Pulseq tools")
        v15 = self.version[:2] >= (1, 5)
        for line in sections.get("[DEFINITIONS]", []):
            parts = line.split(None, 1)
            key, val = parts[0], (parts[1] if len(parts) > 1 else "")
            self.definitions[key] = _definition_value(val)
        d = self.definitions
        self.grad_raster = float(d.get("GradientRasterTime", 10e-6))
        self.rf_raster = float(d.get("RadiofrequencyRasterTime", 1e-6))
        self.adc_raster = float(d.get("AdcRasterTime", 100e-9))
        self.block_raster = float(d.get("BlockDurationRaster", 10e-6))
        self._read_shapes(sections.get("[SHAPES]", []))
        for line in sections.get("[BLOCKS]", []):
            x = line.split()
            ids = [int(v) for v in x[2:8]] + [0] * (6 - len(x[2:8]))
            self.blocks[int(x[0])] = (float(x[1]) * self.block_raster, *ids)
        for line in sections.get("[RF]", []):
            x = line.split()
            if v15:              # id amp mag phase time center delay freqPPM phasePPM freq phase use
                self.rf[int(x[0])] = dict(amp=float(x[1]), mag=int(x[2]), phase=int(x[3]), time=int(x[4]),
                                          center=float(x[5]) * 1e-6, delay=float(x[6]) * 1e-6, freq_ppm=float(x[7]),
                                          phase_ppm=float(x[8]), freq=float(x[9]), phase_offset=float(x[10]),
                                          use=_USE.get(x[11] if len(x) > 11 else "u", "undefined"))
            else:                # id amp mag phase time delay freq phase
                self.rf[int(x[0])] = dict(amp=float(x[1]), mag=int(x[2]), phase=int(x[3]), time=int(x[4]),
                                          center=None, delay=float(x[5]) * 1e-6, freq_ppm=0.0, phase_ppm=0.0,
                                          freq=float(x[6]), phase_offset=float(x[7]), use=None)
        for line in sections.get("[GRADIENTS]", []):
            x = line.split()
            if v15:              # id amp first last amp_shape time_shape delay
                self.grad[int(x[0])] = ("grad", dict(amp=float(x[1]), first=float(x[2]), last=float(x[3]),
                                                     shape=int(x[4]), time=int(x[5]), delay=float(x[6]) * 1e-6))
            else:                # id amp amp_shape time_shape delay
                self.grad[int(x[0])] = ("grad", dict(amp=float(x[1]), first=None, last=None, shape=int(x[2]),
                                                     time=int(x[3]), delay=float(x[4]) * 1e-6))
        for line in sections.get("[TRAP]", []):
            x = line.split()     # id amp rise flat fall delay
            self.grad[int(x[0])] = ("trap", dict(amp=float(x[1]), rise=float(x[2]) * 1e-6, flat=float(x[3]) * 1e-6,
                                                 fall=float(x[4]) * 1e-6, delay=float(x[5]) * 1e-6))
        for line in sections.get("[ADC]", []):
            x = line.split()     # id num dwell(ns) delay(us) ...
            self.adc[int(x[0])] = dict(num=int(x[1]), dwell=float(x[2]) * 1e-9, delay=float(x[3]) * 1e-6)
        for line in sections.get("[EXTENSIONS]", []):
            self.extensions.append(line)
        if any(l.lower().startswith("extension") and "rot" in l.lower() for l in self.extensions):
            raise NotImplementedError("this file rotates its gradients (a rotation extension), which changes the axis "
                                      "each gradient plays on; that is not read yet")
        if not v15:
            self._port_14_gradients()

    def _read_shapes(self, lines):
        i = 0
        while i < len(lines):
            k, v = lines[i].split()[:2]
            if k != "shape_id":
                raise ValueError(f"[SHAPES]: expected shape_id, got {lines[i]!r}")
            sid = int(v)
            k2, n = lines[i + 1].split()[:2]
            if k2 != "num_samples":
                raise ValueError(f"[SHAPES]: shape {sid} has no num_samples")
            i += 2
            vals = []
            while i < len(lines) and not lines[i].startswith("shape_id"):
                vals.extend(float(t) for t in lines[i].split())
                i += 1
            self.shapes[sid] = decompress_shape(int(n), vals)

    def _port_14_gradients(self):
        """Format 1.4 states no arbitrary gradient's first and last value: the first is the previous block's last on
        that axis when the event starts at the block's start and the previous one ended at the block boundary
        (else 0), the last the linear extrapolation of the raster samples (an extended trapezoid's last sample);
        the rule of the Pulseq reference reader."""
        prev_last = [0.0, 0.0, 0.0]
        done = set()
        for bid, (dur, _rf, gx, gy, gz, _adc, _ext) in self.blocks.items():
            for j, gid in enumerate((gx, gy, gz)):
                if not gid or self.grad[gid][0] != "grad":
                    prev_last[j] = 0.0
                    continue
                g = self.grad[gid][1]
                if g["delay"] > 0:
                    prev_last[j] = 0.0
                w = self.shapes[g["shape"]] * g["amp"]
                if gid not in done:
                    g["first"] = prev_last[j]
                    if g["time"] != 0:
                        g["last"] = float(w[-1])
                    else:
                        g["last"] = float((3.0 * w[-1] - w[-2]) * 0.5) if w.size > 1 else float(w[-1])
                    done.add(gid)
                end = g["delay"] + (self._grad_times(g)[-1] if g["time"] != 0 else w.size * self.grad_raster)
                prev_last[j] = 0.0 if end + 1e-15 < dur else float(g["last"])

    # ---------------------------------------------------------- events as functions of time
    def _grad_times(self, g):
        n = self.shapes[g["shape"]].size
        if g["time"] == 0:
            return (np.arange(n) + 0.5) * self.grad_raster
        if g["time"] == -1:
            return 0.5 * (np.arange(n) + 1) * self.grad_raster
        return self.shapes[g["time"]] * self.grad_raster

    def grad_points(self, gid):
        """``(t, value)`` of a gradient event from its block's start: the piecewise-linear function it plays (Hz/m),
        its corners for a trapezoid, ``first``, the raster samples and ``last`` for an arbitrary gradient, the points
        of an extended trapezoid."""
        kind, g = self.grad[gid]
        if kind == "trap":
            t = g["delay"] + np.array([0.0, g["rise"], g["rise"] + g["flat"], g["rise"] + g["flat"] + g["fall"]])
            return t, g["amp"] * np.array([0.0, 1.0, 1.0, 0.0])
        w = self.shapes[g["shape"]] * g["amp"]
        tt = self._grad_times(g)
        if g["time"] == 0:
            t = np.concatenate([[0.0], tt, [w.size * self.grad_raster]]) + g["delay"]
            return t, np.concatenate([[g["first"]], w, [g["last"]]])
        return g["delay"] + tt, w

    def rf_signal(self, rid):
        """``(t, signal_Hz)``: a pulse's complex amplitude in Hz (``gamma B1 / 2 pi``) at its raster samples, from the
        pulse's own start (its delay not included)."""
        r = self.rf[rid]
        mag = self.shapes[r["mag"]]
        ph = self.shapes[r["phase"]]
        sig = r["amp"] * mag * np.exp(2j * np.pi * ph)
        if r["time"] == 0:
            t = (np.arange(sig.size) + 0.5) * self.rf_raster
        else:
            t = self.shapes[r["time"]] * self.rf_raster
        return t, sig

    def block_starts(self):
        """``{id: t0}``: every block's start time from the file's ``t = 0``."""
        out, t = {}, 0.0
        for bid, row in self.blocks.items():
            out[bid] = t
            t += row[0]
        return out

    @property
    def duration(self):
        return float(sum(row[0] for row in self.blocks.values()))


def _definition_value(val):
    toks = val.split()
    try:
        nums = [float(t) for t in toks]
        return nums[0] if len(nums) == 1 else nums
    except ValueError:
        return val


def read_seq(src):
    """A ``.seq`` file (a path, its text, or a sequence object that writes one, ``src.write(path)``: a
    ``pypulseq.Sequence``) as a :class:`PulseqFile`."""
    if isinstance(src, PulseqFile):
        return src
    if not isinstance(src, (str, os.PathLike)) and callable(getattr(src, "write", None)):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sequence.seq")
            src.write(path)
            with open(path) as f:
                return PulseqFile(f.read())
    s = os.fspath(src) if not isinstance(src, str) else src
    if "\n" not in s and os.path.isfile(s):
        with open(s) as f:
            s = f.read()
    return PulseqFile(s)


# ------------------------------------------------------------------ the pulses
def _rf_center(t, sig):
    """Pulseq's centre of a pulse with no stated centre (format 1.4): the mid-point of the samples at its peak
    magnitude."""
    a = np.abs(sig)
    top = np.flatnonzero(a >= a.max() * (1.0 - 1e-5))
    return 0.5 * (t[top[0]] + t[top[-1]])


def _pulse_events(f, t_ref):
    """Every RF pulse of ``f`` as ``(RFEvent, use)`` on the clock anchored at ``t_ref``: the played B1(t) as the
    envelope (``signal / GAMMA_BAR`` tesla on the RF raster), its phase offset as the B1 axis, its frequency offset
    as the carrier; the event's instant the middle of its shape (so the envelope spans exactly the samples the
    file plays)."""
    starts = f.block_starts()
    out = []
    for bid, (_dur, rid, *_rest) in f.blocks.items():
        if not rid:
            continue
        r = f.rf[rid]
        t, sig = f.rf_signal(rid)
        if r["time"] != 0:                   # a pulse on its own time shape (a block pulse's two corners): on the raster
            n = int(np.ceil(t[-1] / f.rf_raster - 1e-9))
            tc = (np.arange(n) + 0.5) * f.rf_raster
            sig = np.interp(tc, t, sig.real) + 1j * np.interp(tc, t, sig.imag)
        dur = sig.size * f.rf_raster
        env = B1Pulse(sig / GAMMA_BAR, f.rf_raster)
        flip = float(env.nominal_flip_deg)
        use = r["use"]
        if use is None:                      # format 1.4 states no use: the reference reader's rule
            use = "excitation" if flip < 90.01 else "refocusing"
        t0 = starts[bid] + r["delay"] - t_ref
        ev = RFEvent(t0 + 0.5 * dur, flip_deg=flip, label=use, axis_deg=float(np.degrees(r["phase_offset"])),
                     offset_hz=float(r["freq"]), envelope=env)
        out.append((ev, use))
    return out


def _labelled(events):
    """The role labels a schedule keys on: the first excitation tips down, a later pair stores and recalls, a
    refocusing pulse refocuses; inversion, saturation and preparation pulses keep their use as a label, which
    declares no coherence role."""
    out, n_exc = [], 0
    for ev, use in events:
        if use == "excitation":
            label = ("Mz→Mxy", "store", "recall")[min(n_exc, 2)]
            n_exc += 1
        elif use == "refocusing":
            label = "refocus"
        else:
            label = use
        out.append(RFEvent(ev.t_s, flip_deg=ev.flip_deg, label=label, axis_deg=ev.axis_deg, offset_hz=ev.offset_hz,
                           envelope=ev.envelope))
    return RFSchedule(out) or None


def _excitation_time(f):
    starts = f.block_starts()
    for bid, (_dur, rid, *_rest) in f.blocks.items():
        if not rid:
            continue
        r = f.rf[rid]
        t, sig = f.rf_signal(rid)
        use = r["use"]
        if use is None:
            use = "excitation" if abs(np.sum(sig) * f.rf_raster) * 360.0 < 90.01 else "refocusing"
        if use in ("excitation", "undefined"):            # the reference reader's rule: an undefined pulse excites
            centre = r["center"] if r["center"] is not None else _rf_center(t, sig)
            return starts[bid] + r["delay"] + centre
    return None


# ------------------------------------------------------------------ the acquisition
def _decode_rf_events(s):
    if not s:
        return None
    try:
        return RFSchedule.from_dicts(json.loads(s)) or None
    except (ValueError, TypeError):
        return None


def from_pulseq(src, *, dt=None, readout_s=None):
    """A Pulseq ``.seq`` (a path, its text, a :class:`PulseqFile`, or a sequence object that writes one) as a
    :class:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence` of one measurement.

    The grid is uniform at ``dt`` (the file's ``dmipy_dt``, else its gradient raster) from the first excitation's
    centre (the file's ``t = 0`` when it excites nothing) to the file's end; the gradient on it is the file's own
    piecewise-linear function, zero between events. Every RF pulse is an event carrying the B1(t) it plays; a file
    dmipy wrote carries its exact schedule in ``dmipy_rf_events``, which is preferred (the blocks are its raster).
    The readout is the ADC's centre (``dmipy_echo_idx`` when stated). ``readout_s`` declares the span ``(t0, t1)``,
    from the excitation, over which the gradient is the readout's rather than an encoding's, so a timing budget
    does not refuse it."""
    f = read_seq(src)
    defs = f.definitions
    dt = float(dt if dt is not None else defs.get("dmipy_dt", f.grad_raster))
    t_exc = _excitation_time(f)
    t0 = 0.0 if t_exc is None else float(t_exc)
    n_t = int(defs["dmipy_n_t"]) if "dmipy_n_t" in defs else max(2, int(round((f.duration - t0) / dt)) + 1)
    tq = np.arange(n_t) * dt + t0

    G = np.zeros((n_t, 3), np.float64)
    starts = f.block_starts()
    for j in range(3):
        pts, live = [], np.zeros(n_t, bool)
        for bid, row in f.blocks.items():
            gid = row[2 + j]
            if not gid:
                continue
            t, v = f.grad_points(gid)
            t = t + starts[bid]
            pts.append((t, v))
            # an arbitrary event's span widened by half a raster: its first and last raster samples sit at raster
            # centres, and a grid instant on its edge reads the event's own end value
            pad = 0.5 * f.grad_raster if f.grad[gid][0] == "grad" else 0.0
            live |= (tq >= t[0] - pad) & (tq <= t[-1] + pad)
        if pts:
            tt = np.concatenate([p[0] for p in pts]); vv = np.concatenate([p[1] for p in pts])
            G[:, j] = np.where(live, np.interp(tq, tt, vv, left=0.0, right=0.0), 0.0)
    G = (G / GAMMA_BAR).astype(np.float32)

    meta_events = _decode_rf_events(defs.get("dmipy_rf_events"))
    blk = _pulse_events(f, t0)
    blk_events = _labelled(blk) if blk else None
    native = bool(blk_events) and not (meta_events and len(meta_events) > len(blk_events))
    rf_events = blk_events if native else (meta_events or blk_events)
    if native and meta_events and len(meta_events) == len(blk_events):
        rf_events = meta_events
    if not native and defs.get("dmipy_gradient") != "physical" and rf_events:
        G = G * RFSchedule(rf_events).sign(np.arange(n_t) * dt)[:, None]

    if "dmipy_echo_idx" in defs:
        echo_idx = int(defs["dmipy_echo_idx"])
    else:
        adcs = [(starts[b] + f.adc[a]["delay"], f.adc[a]) for b, (_d, _r, _x, _y, _z, a, _e) in f.blocks.items() if a]
        if adcs:
            s0, a = adcs[0]
            echo_idx = int(round((s0 + 0.5 * a["num"] * a["dwell"] - t0) / dt))
        else:
            echo_idx = n_t - 1
    echo_idx = int(np.clip(echo_idx, 0, n_t - 1))

    timing = prescription = readout_window = None
    sched = RFSchedule(rf_events)
    if defs.get("dmipy_timing"):
        timing = SequenceTiming.from_dict(json.loads(_text(defs["dmipy_timing"])))
    if defs.get("dmipy_prescription"):
        from ..acquisition.prescription import Prescription
        prescription = Prescription.from_dict(json.loads(_text(defs["dmipy_prescription"])))
    elif (native and not meta_events and len(sched) == 2 and sched.refocus_time is not None
          and sched.mixing_time == (None, False) and any(row[5] for row in f.blocks.values())):
        timing = SequenceTiming.from_pulseq(f)
    if defs.get("dmipy_readout_window"):
        readout_window = tuple(json.loads(_text(defs["dmipy_readout_window"])))
    elif readout_s is not None:
        readout_window = (float(readout_s[0]), float(readout_s[1]))
    return ScannerSequence(G=G[None], dt=dt, rf=rf_events, readout=(echo_idx,), timing=timing, family="pulseq",
                           prescription=prescription, readout_window=readout_window)


def _text(v):
    """A definition read back as the text it was written as (a JSON value split on whitespace reads as a list)."""
    return v if isinstance(v, str) else " ".join(str(x) for x in np.atleast_1d(v))


def pulseq_timing(src):
    """The spin-echo timing budget of a ``.seq``: the first RF block is the 90, the second the 180, one ADC reads.
    Returns (seconds) ``t_excite`` / ``t_refocus`` (the pulses' durations), ``TE`` (twice the 90-to-180 centre
    distance), ``t_readout_pre_echo`` (the ADC's start to the echo) and ``readout_duration``."""
    f = read_seq(src)
    starts = f.block_starts()
    rfs, adc = [], None
    for bid, (_dur, rid, _x, _y, _z, aid, _e) in f.blocks.items():
        if rid:
            r = f.rf[rid]
            t, sig = f.rf_signal(rid)
            d = sig.size * f.rf_raster if r["time"] == 0 else float(np.ceil(t[-1] / f.rf_raster) * f.rf_raster)
            rfs.append((starts[bid] + r["delay"] + d / 2.0, d))
        if aid and adc is None:
            a = f.adc[aid]
            adc = (starts[bid] + a["delay"], a["num"] * a["dwell"])
    if len(rfs) < 2:
        raise ValueError(f"pulseq_timing expects >=2 RF blocks (90 excitation + 180 refocus); found {len(rfs)}.")
    if adc is None:
        raise ValueError("pulseq_timing found no ADC (readout) block.")
    (t90c, t90d), (t180c, t180d) = rfs[0], rfs[1]
    TE = 2.0 * (t180c - t90c)
    return {"t_excite": t90d, "t_refocus": t180d, "TE": TE, "t_readout_pre_echo": t90c + TE - adc[0],
            "readout_duration": adc[1]}
