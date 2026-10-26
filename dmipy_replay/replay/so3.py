"""Rotations of a substrate's pose: the one convention that names a pose from the lab direction of its axis."""
import numpy as np


def rotation_of(axis, frame_axis=(0.0, 0.0, 1.0), roll=0.0):
    """A rotation taking ``frame_axis`` onto ``axis``, with ``roll`` (rad) about ``frame_axis`` applied first.

    Only a convention, and only needed to *name* a pose: nothing in the composition depends on which rotation
    of the family a given axis is paired with, because the distribution carries the azimuth explicitly.
    """
    f = np.asarray(frame_axis, np.float64); f = f / np.linalg.norm(f)
    n = np.asarray(axis, np.float64); n = n / np.linalg.norm(n)
    v = np.cross(f, n)
    s, c = np.linalg.norm(v), float(f @ n)
    if s < 1e-12:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        R = np.eye(3) + K + K @ K * (1.0 / (1.0 + c))
    if roll:
        K = np.array([[0.0, -f[2], f[1]], [f[2], 0.0, -f[0]], [-f[1], f[0], 0.0]])
        R = R @ (np.eye(3) + np.sin(roll) * K + (1.0 - np.cos(roll)) * (K @ K))
    return R
