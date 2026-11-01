"""The wall time of a replay route by phase: what a phantom replay's ``report["seconds"]`` is.

A route that takes long enough to be worth a breakdown (the pose expansion of a batch, a series, a phantom's replay,
its RF train, its vector-Bloch propagation) opens :class:`Phases`; a route opened inside another JOINS it, so that one
call's phases are one breakdown (a phantom replay's pose expansions are phases of the phantom replay). Nothing is
written anywhere: the breakdown lives as long as the object, in memory, and :func:`current` is the innermost open one
of this thread.
"""
import threading
import time

_local = threading.local()


def current():
    """The innermost open :class:`Phases` of this thread, or ``None``."""
    stack = getattr(_local, "stack", None)
    return stack[-1] if stack else None


class Phases:
    """The phases of one replay route and the seconds spent in each.

    ``with Phases(route, params=...) as ph`` opens it (the outer one when a route is already open in this thread:
    the phases of the inner route are the outer's); :meth:`phase` begins a phase and ends the previous one,
    :meth:`progress` states where the current phase is, :meth:`phase_seconds` reads the breakdown."""

    def __init__(self, route, params=None):
        self.route = str(route)
        self.params = dict(params or {})
        self.started = None
        self.progressed = None                       # (done, total, unit) of the latest progress statement
        self._phases = []                            # [name, started, ended or None]
        self._joined = False
        self._lock = threading.Lock()

    def __enter__(self):
        outer = current()
        if outer is not None:                        # a route inside a route: one breakdown, the outer's
            self._joined = True
            return outer
        self.started = time.time()
        stack = getattr(_local, "stack", None)
        if stack is None:
            stack = _local.stack = []
        stack.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._joined:
            return False
        now = time.time()
        with self._lock:
            for ph in self._phases:
                if ph[2] is None:
                    ph[2] = now
        stack = getattr(_local, "stack", None)
        if stack and stack[-1] is self:
            stack.pop()
        return False

    def phase(self, name, **fields):
        """A phase ``name`` begins and the open one ends; ``fields`` describe it and are not kept."""
        now = time.time()
        with self._lock:
            for ph in self._phases:
                if ph[2] is None:
                    ph[2] = now
            self._phases.append([str(name), now, None])

    def progress(self, done, total, *, unit="walkers"):
        """Where the current phase is: ``done`` of ``total`` ``unit``. A progress statement before any phase opens
        the phase ``"run"`` at the route's start."""
        with self._lock:
            if not self._phases:
                self._phases.append(["run", self.started if self.started is not None else time.time(), None])
            self.progressed = (done, total, str(unit))

    def phase_seconds(self):
        """Seconds spent in each phase so far, ``{name: seconds}`` (a phase entered twice sums)."""
        now = time.time()
        out = {}
        with self._lock:
            for name, t0, t1 in self._phases:
                spent = (t1 or now) - t0
                out[name] = out[name] + spent if name in out else spent
        return out
