"""A specimen's pose in the bore: :class:`Pose`, the one type a ``pose=`` argument takes, and :func:`pose_rotation`,
which reads it."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Pose"]


@dataclass(frozen=True)
class Pose:
    """Where the specimen sits in the bore: the rotation taking the substrate frame (the frame the walk was
    stored in) to the scanner frame. Translation is the grid's (its origin), so a pose is a rotation only."""

    rotation: np.ndarray

    def __init__(self, rotation=None):
        R = np.eye(3) if rotation is None else np.asarray(rotation, np.float64).reshape(3, 3)
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-8) or np.linalg.det(R) < 0:
            raise ValueError("a pose is a proper rotation (R R^T = I, det +1)")
        object.__setattr__(self, "rotation", R)

    @property
    def is_identity(self):
        return bool(np.allclose(self.rotation, np.eye(3), atol=1e-12))

    def to_meta(self):
        return {"rotation": self.rotation.tolist()}

    @classmethod
    def from_meta(cls, m):
        return cls(None if m is None else m.get("rotation"))


def pose_rotation(pose):
    """The 3x3 rotation of a ``pose=`` argument: a :class:`Pose` (``None`` is no pose and stays ``None``). A bare matrix,
    or any other object, is refused by name: a pose is spelled ``Pose(R)``."""
    if pose is None:
        return None
    if not isinstance(pose, Pose):
        raise TypeError(f"pose= takes a dmipy_replay.phantom.Pose (or None), got {type(pose).__name__}; "
                        f"a rotation matrix R is spelled Pose(R)")
    return pose.rotation
