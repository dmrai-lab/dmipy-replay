"""The occupancy event list (RPK.md 9.5.7): per walker the instants t and labels of every change of pool, exact in
time, and the readers that evaluate it at instants or cut it to an interval."""

import numpy as np

class EventList:
    """One column's events over a walk: per walker the changes of its label, ``(t, label)`` with ``t`` in seconds from
    the walk's start, strictly increasing within a walker; ``offsets`` (n_walkers + 1) int64 delimit walker ``i``'s
    entries ``offsets[i]:offsets[i+1]``. The label held before the first change is the walker's start label, kept
    beside the list (``CoefficientWalk.compartment``, the bound column's start)."""
    __slots__ = ("offsets", "t", "label")

    def __init__(self, offsets, t, label):
        self.offsets = np.asarray(offsets, np.int64)
        self.t = np.asarray(t, np.float64)
        self.label = np.asarray(label)
        if self.offsets.ndim != 1 or self.t.shape != self.label.shape or self.offsets[-1] != self.t.shape[0]:
            raise ValueError("an event list is (offsets (n_w + 1), t (n_ev), label (n_ev)) with offsets[-1] == n_ev")

    @property
    def n_walkers(self):
        return int(self.offsets.shape[0] - 1)

    @property
    def n_events(self):
        return int(self.t.shape[0])

    @property
    def counts(self):
        return np.diff(self.offsets)

    def __iter__(self):
        """The list as its three arrays, ``offsets, t, label = events`` (RPK.md 9.5.7's tensors)."""
        return iter((self.offsets, self.t, self.label))

    def take(self, idx):
        """The lists of the walkers ``idx``, in that order."""
        idx = np.asarray(idx, np.int64)
        counts = self.counts[idx]
        off = np.concatenate([[0], np.cumsum(counts)])
        src = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in idx]) if idx.size else np.zeros(0, np.int64)
        return EventList(off, self.t[src], self.label[src])

    def shifted(self, dt):
        """The same events ``dt`` seconds later on the clock."""
        return EventList(self.offsets, self.t + float(dt), self.label)

    @staticmethod
    def concatenate(lists):
        """The lists of consecutive walker ranges as one."""
        lists = list(lists)
        offs = [lists[0].offsets]
        for ev in lists[1:]:
            offs.append(ev.offsets[1:] + offs[-1][-1])
        return EventList(np.concatenate(offs), np.concatenate([ev.t for ev in lists]), np.concatenate([ev.label for ev in lists]))

    def __iter__(self):
        """``offsets, t, label = events``: the three arrays, in that order."""
        return iter((self.offsets, self.t, self.label))

    def __eq__(self, other):
        return (isinstance(other, EventList) and np.array_equal(self.offsets, other.offsets)
                and np.array_equal(self.t, other.t) and np.array_equal(self.label, other.label))

    def __repr__(self):
        return f"EventList({self.n_walkers} walkers, {self.n_events} events)"

def events_label_at(events, start, t):
    """The label every walker holds at time ``t`` (s): its ``start`` label, or the label of its last change at or
    before ``t``."""
    start = np.asarray(start)
    out = np.array(start, copy=True)
    if events is None or events.n_events == 0:
        return out
    sel = events.t <= float(t)
    if not sel.any():
        return out
    # the last event at or before t per walker: the entries are in time order within a walker, so the last selected wins
    last = {}
    for i, lab in zip(w[sel], events.label[sel]):
        last[i] = lab
    for i, lab in last.items():
        out[i] = lab
    return out

def events_within(events, start, t0, t1):
    """The changes with ``t0 < t <= t1`` per walker and the label held at ``t0``: ``(EventList, label_at_t0)`` --
    one window's list of the walk's (RPK.md 9.5.7)."""
    lab0 = events_label_at(events, start, t0)
    if events is None:
        return None, lab0
    w = np.repeat(np.arange(events.n_walkers), events.counts)
    sel = (events.t > float(t0)) & (events.t <= float(t1))
    counts = np.bincount(w[sel], minlength=events.n_walkers)
    return EventList(np.concatenate([[0], np.cumsum(counts)]), events.t[sel], events.label[sel]), lab0

def events_at_instants(events, start, times):
    """The label of every walker at each of ``times`` (s), ``(n_walkers, len(times))``: a walk's occupancy sampled on
    a reader's grid (never a stored form)."""
    times = np.asarray(times, np.float64).reshape(-1)
    start = np.asarray(start)
    out = np.repeat(start[:, None], times.size, axis=1)
    if events is None or events.n_events == 0:
        return out
    w = np.repeat(np.arange(events.n_walkers), events.counts)
    # for each (walker, time): the last event with t <= time; events are time-ordered within a walker
    for i in range(events.n_walkers):
        a, b = events.offsets[i], events.offsets[i + 1]
        if b == a:
            continue
        pos = np.searchsorted(events.t[a:b], times, side="right") - 1
        has = pos >= 0
        out[i, has] = events.label[a:b][pos[has]]
    return out
