"""The label a walker holds at a time is its start label, or that of its last change at or before the time."""
import numpy as np

from dmipy_replay.replay.events import EventList, events_label_at


def test_the_label_at_a_time_is_the_last_change_at_or_before_it():
    ev = EventList(offsets=[0, 2, 2, 3], t=[1.0, 3.0, 2.0], label=[1, 0, 1])     # walker 1 never changes
    start = np.array([0, 0, 0])
    np.testing.assert_array_equal(events_label_at(ev, start, 0.5), [0, 0, 0])
    np.testing.assert_array_equal(events_label_at(ev, start, 2.0), [1, 0, 1])
    np.testing.assert_array_equal(events_label_at(ev, start, 3.5), [0, 0, 1])
