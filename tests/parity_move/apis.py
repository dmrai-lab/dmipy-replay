"""The two spellings of every call the move's parity record makes: :class:`Before` (the import paths of dmipy-sim and
dmipy-replay before the replay move, dmrai-lab/dmipy-sim#834) and :class:`After` (the paths after it). The call table
(:mod:`calls`) reads every name through one of them, so the two drivers run the same calls.

A name :class:`After` cannot import yet is :class:`Pending`: the group that needs it is reported as not yet moved,
neither identical nor different."""
import importlib


class Pending(Exception):
    """A name the installed packages do not define yet."""


def _attr(path):
    """The object at the dotted ``path`` (``package.module:attr.attr``)."""
    mod, _, rest = path.partition(":")
    try:
        obj = importlib.import_module(mod)
    except ImportError as exc:
        raise Pending(f"{path}: {exc}") from None
    for part in rest.split(".") if rest else ():
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise Pending(f"{path}: {exc}") from None
    return obj


class _Spelling:
    """Names resolved lazily from :attr:`PATHS`; a method of a pack is resolved on the pack."""

    PATHS = {}
    METHODS = {}
    FEATURES = {}

    def require(self, *features):
        """Raise :class:`Pending` unless every feature is where this spelling puts it."""
        for f in features:
            ok = self.FEATURES.get(f, lambda: True)
            try:
                present = ok()
            except (ImportError, AttributeError, Pending):
                present = False
            if not present:
                raise Pending(f"{f} is not in the installed packages yet")

    def __getattr__(self, name):
        if name not in self.PATHS:
            raise AttributeError(name)
        return _attr(self.PATHS[name])

    def call(self, op, pack, *args, **kw):
        """``op`` (a writer operation or a pack method) on ``pack``."""
        how = self.METHODS[op]
        if how.startswith("."):
            fn = getattr(pack, how[1:], None)
            if fn is None:
                raise Pending(f"{type(pack).__module__}.{type(pack).__name__}{how}")
            return fn(*args, **kw)
        return _attr(how)(pack, *args, **kw)


_COMMON = {
    "pgse": "dmipy_replay:pgse", "pgste": "dmipy_replay:pgste", "ogse": "dmipy_replay:ogse", "splice": "dmipy_replay:splice",
    "set_b": "dmipy_replay:set_b", "ScannerSequence": "dmipy_replay:ScannerSequence", "Tissue": "dmipy_replay:Tissue",
    "ScannerLimits": "dmipy_replay:ScannerLimits", "Grid": "dmipy_replay:Grid",
    "Partition": "dmipy_replay.replay.time_segments:Partition",
}


class Before(_Spelling):
    """The import paths before the move: the pose replay, the phantoms and the study in dmipy-sim."""

    name = "before"
    PATHS = dict(_COMMON, **{
        "read_rpk": "dmipy_sim.replay.replay:read_rpk",
        "so3": "dmipy_sim.replay.so3",
        "sh_analytical": "dmipy_sim.math.sh_analytical",
        "FOD": "dmipy_sim.replay.fod:FOD",
        "Distribution": "dmipy_sim.replay.so3:Distribution",
        "PoseResponse": "dmipy_sim.replay.replay:PoseResponse",
        "analytic_pose_response": "dmipy_sim.replay.replay:analytic_pose_response",
        "pose_device": "dmipy_sim.replay.pose_device",
        "replay_signal_jax": "dmipy_sim.replay.replay:replay_signal_jax",
        "replay_batch_jax": "dmipy_sim.replay.replay:replay_batch_jax",
        "train_response": "dmipy_sim.replay.pathways:train_response",
        "closed_form_train": "dmipy_sim.replay.pathways:closed_form_train",
        "Acquisition": "dmipy_sim.replay.study:Acquisition",
        "StudyProtocol": "dmipy_sim.replay.study:Protocol",
        "Study": "dmipy_sim.replay.study:Study",
        "Phantom": "dmipy_sim.phantom:Phantom",
        "PartitionPhantom": "dmipy_sim.phantom:PartitionPhantom",
        "Pose": "dmipy_sim.phantom:Pose",
        "PackSubstrate": "dmipy_sim.phantom:PackSubstrate",
        "FreeWater": "dmipy_sim.phantom:FreeWater",
        "Inert": "dmipy_sim.phantom:Inert",
        "Watson": "dmipy_sim.phantom:Watson",
        "ODF": "dmipy_sim.phantom:ODF",
        "Frames": "dmipy_sim.phantom:Frames",
        "read_rph": "dmipy_sim.replay.phantom:read_rph",
        "ColumnarPack": "dmipy_sim.replay.columnar:ColumnarPack",
        "TimeSegmentLayout": "dmipy_sim.replay.time_segments:TimeSegmentLayout",
    })
    METHODS = {
        "save": ".save", "truncate": ".truncate", "prefix": ".prefix", "rewindow": ".rewindow",
        "restate_surface_envelope": ".restate_surface_envelope", "stated_envelope": ".stated_envelope",
        "extend": ".extend",
    }
    # the image backends a layout was imaged with by default before the move: JAX in dmipy-sim's subclasses
    DEFAULTS = {"columnar_image": "jax", "segment_image": "jax", "pose": "jax"}


class After(_Spelling):
    """The import paths after the move: the pose replay, the phantoms and the study in dmipy-replay, the writer
    operations dmipy-sim functions taking a pack."""

    name = "after"
    PATHS = dict(_COMMON, **{
        "read_rpk": "dmipy_replay:read_rpk",
        "so3": "dmipy_replay.replay.so3",
        "sh_analytical": "dmipy_replay.math.sh_analytical",
        "FOD": "dmipy_replay.replay.fod:FOD",
        "Distribution": "dmipy_replay.replay.so3:Distribution",
        "PoseResponse": "dmipy_replay.replay.pose:PoseResponse",
        "analytic_pose_response": "dmipy_replay.replay.pose:analytic_pose_response",
        "pose_device": "dmipy_replay.replay.pose_device",
        "replay_signal_jax": "dmipy_replay.replay.jax_routes:replay_signal_jax",
        "replay_batch_jax": "dmipy_replay.replay.jax_routes:replay_batch_jax",
        "train_response": "dmipy_replay.replay.pathways:train_response",
        "closed_form_train": "dmipy_replay.replay.pathways:closed_form_train",
        "Acquisition": "dmipy_replay.replay.study:Acquisition",
        "StudyProtocol": "dmipy_replay.replay.study:StudyProtocol",
        "Study": "dmipy_replay.replay.study:Study",
        "Phantom": "dmipy_replay.phantom:Phantom",
        "PartitionPhantom": "dmipy_replay.phantom:PartitionPhantom",
        "Pose": "dmipy_replay.phantom:Pose",
        "PackSubstrate": "dmipy_replay.phantom:PackSubstrate",
        "FreeWater": "dmipy_replay.phantom:FreeWater",
        "Inert": "dmipy_replay.phantom:Inert",
        "Watson": "dmipy_replay.phantom:Watson",
        "ODF": "dmipy_replay.phantom:ODF",
        "Frames": "dmipy_replay.phantom:Frames",
        "read_rph": "dmipy_replay.replay.phantom:read_rph",
        "ColumnarPack": "dmipy_replay.replay.columnar:ColumnarPack",
        "TimeSegmentLayout": "dmipy_replay.replay.time_segments:TimeSegmentLayout",
    })
    METHODS = {
        "save": "dmipy_sim.replay.container:save_pack",
        "truncate": "dmipy_sim.replay.reencode:truncate_pack",
        "prefix": "dmipy_sim.replay.reencode:prefix_pack",
        "rewindow": "dmipy_sim.replay.reencode:rewindow_pack",
        "restate_surface_envelope": "dmipy_sim.replay.reencode:restate_surface_envelope",
        "stated_envelope": "dmipy_sim.replay.reencode:stated_envelope",
        "extend": "dmipy_sim.replay.continuation:extend_pack",
    }
    DEFAULTS = {"columnar_image": "numpy", "segment_image": "numpy", "pose": "numpy"}
    FEATURES = {
        "so3": lambda: hasattr(_attr("dmipy_replay.replay.so3"), "real_sh") and _attr("dmipy_replay.replay.fod:FOD") is not None,
        "pose": lambda: hasattr(_attr("dmipy_replay:ReplayPack"), "pose_responses"),
        "series": lambda: hasattr(_attr("dmipy_replay:ReplayPack"), "pose_series"),
        "jax_bloch": lambda: "jax" in _attr("dmipy_replay:ReplayPack").BLOCH_BACKENDS,
        "pathways": lambda: _attr("dmipy_replay.replay.pathways:train_response") is not None,
        "study": lambda: hasattr(_attr("dmipy_replay:ReplayPack"), "study"),
        "columnar_study": lambda: (hasattr(_attr("dmipy_replay:ColumnarPack"), "image_study")
                                   and "jax" in _attr("dmipy_replay:ColumnarPack").IMAGE_BACKENDS),
        "segment_backends": lambda: {"jax", "torch"} <= set(_attr("dmipy_replay:TimeSegmentLayout").BACKENDS),
        "delivery": lambda: hasattr(_attr("dmipy_replay:TimeSegmentLayout"), "delivery"),
        "phantom": lambda: _attr("dmipy_replay.phantom:Phantom") is not None,
        "writer": lambda: _attr("dmipy_sim.replay.reencode:truncate_pack") is not None,
    }


SPELLINGS = {"before": Before, "after": After}
