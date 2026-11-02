"""The timing record of a replay route (dmipy_replay.replay.phases): the seconds of each phase, a route inside a route
joining the outer one, a progress statement before any phase opening the route's own, and nothing outliving the
route's ``with`` block on the thread's stack."""
import time

from dmipy_replay.replay.phases import Phases, current


def test_the_phases_partition_the_route():
    with Phases("route", params=dict(n=1)) as ph:
        assert current() is ph
        ph.phase("a")
        time.sleep(0.01)
        ph.phase("b")
        time.sleep(0.01)
        ph.phase("a")
    s = ph.phase_seconds()
    assert set(s) == {"a", "b"} and s["a"] >= 0.01 and s["b"] >= 0.01
    assert current() is None
    assert ph.phase_seconds() == s                                    # closed: the breakdown no longer grows


def test_a_route_inside_a_route_joins_it():
    with Phases("outer") as outer:
        outer.phase("gather")
        with Phases("inner") as inner:
            assert inner is outer
            inner.phase("pose")
        assert current() is outer
        outer.phase("layers")
    assert set(outer.phase_seconds()) == {"gather", "pose", "layers"}


def test_progress_before_a_phase_is_the_route_itself():
    with Phases("route") as ph:
        ph.progress(1, 2, unit="groups")
    assert set(ph.phase_seconds()) == {"run"} and ph.progressed == (1, 2, "groups")


def test_an_exception_closes_the_route():
    try:
        with Phases("route") as ph:
            ph.phase("a")
            raise RuntimeError("x")
    except RuntimeError:
        pass
    assert current() is None and set(ph.phase_seconds()) == {"a"}
