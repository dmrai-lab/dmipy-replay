"""The time-segment read path holds a sequence to the slew limit it is given: ``fit``, ``image`` and ``delivery`` refuse
a played gradient that steps faster than a stated limit, by ``ScannerSequence.check_slew`` (the measurement, the axis,
the step and the limit named); the limit is the call's ``slew_max``, else a delivery's scanner limits, else the
manifest's ``source.envelope["slew_max"]``; with none stated no slew is checked."""
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_replay.acquisition.scanners import ScannerLimits
from dmipy_replay.replay.time_segments import Partition, TimeSegmentLayout
from segment_scenarios import acquisitions, templates

DATA = Path(__file__).parent / "data" / "time_segments"
NS = SimpleNamespace(pgse=dr.pgse, pgste=dr.pgste, ogse=dr.ogse, ScannerSequence=dr.ScannerSequence, Partition=Partition,
                     Tissue=dr.Tissue)
#: a square pulse of 0.1 T/m on a 0.1 ms grid: one step of 1000 T/m/s (float32 0.1 over 1e-4 s)
BOX = templates(NS)["box"]
REFUSAL = "measurement 0, axis z: the played gradient steps at 1000 T/m/s, above the slew limit 200 T/m/s"
PRISMA = ScannerLimits.of("siemens_magnetom_prisma_3T")          # slew_max 200 T/m/s


def _limits(slew):
    return replace(PRISMA, slew_max=slew)


@pytest.fixture(scope="module")
def lay():
    return TimeSegmentLayout(str(DATA))


@pytest.fixture
def stated(tmp_path):
    """The fixture layout, copied, with ``slew_max`` 200 T/m/s stated in its manifest's envelope."""
    d = tmp_path / "stated"
    shutil.copytree(DATA, d)
    m = json.loads((d / "manifest.json").read_text())
    m["source"]["envelope"]["slew_max"] = 200.0
    (d / "manifest.json").write_text(json.dumps(m))
    return TimeSegmentLayout(str(d))


def test_the_square_pulse_steps_at_1000_t_per_m_per_s():
    assert BOX.peak_slew().max() == pytest.approx(1000.0, rel=1e-7)


@pytest.mark.parametrize("read", ["fit", "image", "delivery"])
def test_a_stated_keyword_limit_refuses_the_step_on_every_read(lay, read):
    call = dict(fit=lambda: lay.fit(BOX, slew_max=200.0), image=lambda: lay.image(BOX, slew_max=200.0),
                delivery=lambda: lay.delivery(BOX, _limits(None), slew_max=200.0))[read]
    with pytest.raises(ValueError, match=REFUSAL):
        call()


def test_a_delivery_is_held_to_its_scanners_slew(lay):
    with pytest.raises(ValueError, match=REFUSAL):
        lay.delivery(BOX, PRISMA)


@pytest.mark.parametrize("read", ["fit", "image", "delivery"])
def test_a_slew_stated_in_the_manifest_refuses_the_step(stated, read):
    call = dict(fit=lambda: stated.fit(BOX), image=lambda: stated.image(BOX),
                delivery=lambda: stated.delivery(BOX, _limits(None)))[read]
    with pytest.raises(ValueError, match=REFUSAL):
        call()


def test_the_keyword_comes_before_the_scanner_and_the_scanner_before_the_manifest(lay, stated):
    lay.delivery(BOX, PRISMA, slew_max=1000.0)                           # keyword 1000 over the scanner's 200
    stated.image(BOX, slew_max=1000.0)                                   # keyword 1000 over the manifest's 200
    stated.delivery(BOX, _limits(1000.0))                                # scanner 1000 over the manifest's 200
    with pytest.raises(ValueError, match=REFUSAL):
        lay.delivery(BOX, _limits(1000.0), slew_max=200.0)               # keyword 200 under the scanner's 1000
    with pytest.raises(ValueError, match="above the slew limit 500 T/m/s"):
        stated.delivery(BOX, _limits(500.0))                             # scanner 500 under the manifest's 200


def test_the_step_is_admitted_at_its_own_slew_and_with_no_limit_stated(lay):
    assert lay.slew_limit() is None                                      # the fixture states none
    for read in (lambda: lay.image(BOX, slew_max=1000.0), lambda: lay.image(BOX)):
        S, _ = read()
        assert np.isfinite(S).any()
    assert lay.fit(BOX, slew_max=1000.0)[0] == lay.fit(BOX)[0] == "box"
    lay.delivery(BOX, _limits(None))


def test_a_limit_given_per_axis_is_read_per_axis(lay):
    lay.fit(BOX, slew_max=[200.0, 200.0, 1000.0])
    with pytest.raises(ValueError, match=REFUSAL):
        lay.fit(BOX, slew_max=[1000.0, 1000.0, 200.0])


def test_a_trapezoid_is_admitted_at_the_slew_it_was_built_at(lay):
    trap = templates(NS)["trap"]                                         # built at 200 T/m/s
    S, _ = lay.image(trap, slew_max=200.0)
    assert np.isfinite(S).any()
    with pytest.raises(ValueError, match="above the slew limit 100 T/m/s"):
        lay.image(trap, slew_max=100.0)


def test_the_idealised_box_acquisition_is_refused_by_a_clinical_scanner_and_delivered_without_a_slew(lay):
    box = acquisitions(NS)["box"][1]                                     # the parity record's delivery call
    with pytest.raises(ValueError, match=r"measurement 0, axis x: the played gradient steps at .* above the slew "
                                         r"limit 200 T/m/s"):
        lay.delivery(box, PRISMA)
    d = lay.delivery(box, _limits(None))
    assert d.partition == "box"
