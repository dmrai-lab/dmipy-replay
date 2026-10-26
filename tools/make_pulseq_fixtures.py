"""Write the Pulseq parity files and record how dmipy-sim (through pypulseq) reads each one.

Dev-only: needs dmipy-sim and pypulseq. Writes ``tests/data/seq/<name>.seq`` and ``tests/data/seq/reference.npz``:
per file the rasterised ``G``, ``dt``, the readout index and the RF instants and labels sim's importer finds, plus
pypulseq's own flip angle of every pulse (|sum of the signal| x 360) -- the oracle of the native reader's envelopes,
which sim does not read (it takes a 90 for an excitation and a 180 for a refocusing).
"""
import json
from pathlib import Path

import numpy as np
import pypulseq as pp

import dmipy_sim as d
from dmipy_sim.sequences.pulseq import to_pulseq, from_pulseq

OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "seq"


def foreign_spin_echo(path, *, flip_ref=180.0, sinc=True):
    """A diffusion spin echo written by pypulseq itself: sinc (or block) 90 and 180, trapezoid lobes, an ADC."""
    sys = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s", rf_ringdown_time=20e-6,
                  rf_dead_time=100e-6, adc_dead_time=10e-6)
    seq = pp.Sequence(sys)
    if sinc:
        rf90 = pp.make_sinc_pulse(flip_angle=np.pi / 2, duration=3e-3, time_bw_product=4, system=sys, use="excitation")
        rf180 = pp.make_sinc_pulse(flip_angle=np.deg2rad(flip_ref), duration=4e-3, time_bw_product=4, system=sys,
                                   use="refocusing", phase_offset=np.pi / 2)
    else:
        rf90 = pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3, system=sys, use="excitation")
        rf180 = pp.make_block_pulse(flip_angle=np.deg2rad(flip_ref), duration=2e-3, system=sys, use="refocusing")
    g = pp.make_trapezoid(channel="x", amplitude=30e-3 * sys.gamma, flat_time=8e-3, rise_time=0.3e-3, system=sys)
    gy = pp.make_trapezoid(channel="y", amplitude=-12e-3 * sys.gamma, flat_time=8e-3, rise_time=0.3e-3, system=sys)
    adc = pp.make_adc(num_samples=64, duration=3.2e-3, system=sys)
    gap = 2e-3
    # the 90's centre to the 180's: the rest of the 90's block, a lobe, the gap, the 180's own lead-in
    t90c = rf90.delay + pp.calc_rf_center(rf90)[0]
    t180c = pp.calc_duration(rf90) + pp.calc_duration(g, gy) + gap + rf180.delay + pp.calc_rf_center(rf180)[0]
    TE = 2.0 * (t180c - t90c)
    t_after = t180c + (pp.calc_duration(rf180) - rf180.delay - pp.calc_rf_center(rf180)[0]) + pp.calc_duration(g, gy)
    raster = sys.block_duration_raster
    lead = np.round((t90c + TE - adc.delay - 0.5 * adc.num_samples * adc.dwell - t_after) / raster) * raster
    seq.add_block(rf90)
    seq.add_block(g, gy)
    seq.add_block(pp.make_delay(gap))
    seq.add_block(rf180)
    seq.add_block(g, gy)
    if lead > 0:
        seq.add_block(pp.make_delay(lead))
    seq.add_block(adc)
    seq.write(str(path))


def foreign_arbitrary(path):
    """An arbitrary-gradient waveform (a cosine OGSE lobe on z) around a block 180, written by pypulseq."""
    sys = pp.Opts(max_grad=80, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")
    seq = pp.Sequence(sys)
    n = 800
    w = 20e-3 * sys.gamma * np.sin(np.pi * np.arange(n) / n) * np.cos(2 * np.pi * 4 * np.arange(n) / n)
    gz = pp.make_arbitrary_grad(channel="z", waveform=w, system=sys, first=0, last=0)
    rf90 = pp.make_block_pulse(flip_angle=np.pi / 2, duration=0.5e-3, system=sys, use="excitation")
    rf180 = pp.make_block_pulse(flip_angle=np.pi, duration=1e-3, system=sys, use="refocusing")
    adc = pp.make_adc(num_samples=16, duration=1e-3, system=sys)
    t90c = rf90.delay + pp.calc_rf_center(rf90)[0]
    t180c = pp.calc_duration(rf90) + pp.calc_duration(gz) + rf180.delay + pp.calc_rf_center(rf180)[0]
    t_after = t180c + pp.calc_duration(rf180) - rf180.delay - pp.calc_rf_center(rf180)[0] + pp.calc_duration(gz)
    raster = sys.block_duration_raster
    lead = np.round((2 * t180c - t90c - adc.delay - 0.5 * adc.num_samples * adc.dwell - t_after) / raster) * raster
    pre = max(0.0, -lead)                   # an ADC that would read past the echo: a gap before the 180 moves the echo twice as far
    lead = lead + pre
    seq.add_block(rf90)
    seq.add_block(gz)
    if pre > 0:
        seq.add_block(pp.make_delay(pre))
    seq.add_block(rf180)
    seq.add_block(gz)
    if lead > 0:
        seq.add_block(pp.make_delay(lead))
    seq.add_block(adc)
    seq.write(str(path))


def pypulseq_flips(path):
    s = pp.Sequence()
    s.read(str(path))
    out = []
    for i in range(1, len(s.block_events) + 1):
        rf = getattr(s.get_block(i), "rf", None)
        if rf is not None:
            dt = np.diff(rf.t) if rf.t.size > 2 else np.array([rf.t[-1] - rf.t[0]])
            out.append((float(np.abs(np.sum(rf.signal[:dt.size] * dt))) * 360.0, float(np.sum(np.abs(rf.signal[:dt.size]) * dt)) * 360.0))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    seqs = {"export_pgse": d.pgse([[1, 0, 0]], 0.006, 0.02, bvalues=[1e9], n_t=400),
            "export_pgste": d.pgste([[0, 1, 0]], 0.004, 0.02, bvalues=[1e9], n_t=400),
            "export_ogse": d.ogse([[0, 0, 1]], 50.0, 0.02, shape="cosine", bvalues=[3e8], n_t=400)}
    for name, seq in seqs.items():
        files[name] = OUT / f"{name}.seq"
        to_pulseq(seq, filename=str(files[name]))
    files["foreign_sinc_se"] = OUT / "foreign_sinc_se.seq"
    foreign_spin_echo(files["foreign_sinc_se"])
    files["foreign_block_se_150"] = OUT / "foreign_block_se_150.seq"
    foreign_spin_echo(files["foreign_block_se_150"], flip_ref=150.0, sinc=False)
    files["foreign_arbitrary"] = OUT / "foreign_arbitrary.seq"
    foreign_arbitrary(files["foreign_arbitrary"])
    ref, index = {}, {}
    for name, path in files.items():
        s = from_pulseq(str(path))
        ref[f"{name}/G"] = np.asarray(s.G, np.float64)
        ref[f"{name}/dt"] = np.float64(s.dt)
        ref[f"{name}/readout"] = np.asarray(s.readout, np.int64)
        ref[f"{name}/rf_t"] = np.array([e.t_s for e in s.rf], np.float64)
        fl = np.array(pypulseq_flips(path), np.float64).reshape(-1, 2)
        ref[f"{name}/flips_pypulseq"], ref[f"{name}/nutation_pypulseq"] = fl[:, 0], fl[:, 1]
        index[name] = dict(labels=[e.label for e in s.rf], flips_sim=[e.flip_deg for e in s.rf])
        print(name, s.G.shape, s.dt, s.readout, [(round(e.t_s, 6), e.label, e.flip_deg) for e in s.rf])
    np.savez(OUT / "reference.npz", **ref)
    (OUT / "reference.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
