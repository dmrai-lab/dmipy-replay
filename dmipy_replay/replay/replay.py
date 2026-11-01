"""Replay packs (``.rpk``) and the compiled-scheme forward — the shared replay primitive.

A **replay pack** stores the state of one converged Monte-Carlo walk (walker trajectories in the bridge form:
per axis the two endpoints and the lowest ``K`` sine bands of the path pinned at both, plus a spin weight and
optional channels) so the
diffusion-weighted signal for *any* gradient waveform can be reconstructed without re-simulating. It is a
single ``safetensors`` file: the arrays are the tensors, the JSON metadata sits in the ``"rpk"`` header
key. Producers (e.g. the substrate generator) write them; consumers (dmipy-fit compartments, dmipy-design
waveform optimization) replay them.

The forward is exact and cheap. The signal is

    E = < w_i exp(i phi_i) > / < w_i > ,   phi_i(m) = gamma * dt * sum_t G_m(t) . r_i(t)

and because the phase is linear in position it is a contraction of the stored coefficients ``C_i`` (the two
endpoints and the ``K`` bands the walk accumulated, :mod:`dmipy_replay.engine.accumulate`) with the waveform's
projection on the same basis,

    phi_i(m) = sum_{k,c} C_{i,k,c} * W_{m,k,c},

exact for every waveform inside the stored band. The projection ``W`` (= :func:`compile_scheme`) is independent of the walkers; each forward is
then one dense matmul ``C @ W`` + a weighted complex mean (:func:`replay_signal`). This is the SAME math
whether the acquisition is fixed and the substrate varies (fitting) or the substrate is fixed and the
waveform varies (design) — in the latter it is differentiable in ``G``, so it drives gradient-based
waveform/B1 optimization.

Surface relaxivity is exact, via the stored boundary local time (the C2 channel, bridge form): a
per-walker reweight by ``exp((rho2/D) * sum_t chi(t) ell_i(t))``, optionally coherence-gated by an
occupancy schedule ``chi`` (:func:`surface_logweight`). A tissue's ``rho1`` adds the longitudinal
term on the same channel, gated by the complement ``chi_parallel = active - chi`` (on while the
magnetisation is stored along B0, e.g. a stimulated echo's mixing time): the full contact weight is
``exp((rho2/D) * <chi, ell> + (rho1/D) * <chi_parallel, ell>)``, two calls of the same bridge
contraction summed -- no new channel, no new walk (dmipy-sim#574).
"""

import json

from dataclasses import dataclass

from functools import cached_property

import numpy as np

from .compression import c2_bands_K as _cx_bands_K
from .decoded import PackArrays
from .planning import PLAN_TOL, ReadPlan, planned

from ..constants import GAMMA

from ..acquisition.scanner_sequence import Protocol, ScannerSequence

__all__ = ["ReplayPack", "read_rpk", "compile_scheme", "replay_signal", "replay_coefficients", "surface_logweight"]


def read_rpk(path, *, cls=None):
    """Read a replay pack into a :class:`ReplayPack` (arrays + metadata). A pack that embeds no substrate spec, or
    whose spec's pools, tiers and stored field channel disagree (:func:`declared_susceptibility_field`), is
    refused. ``cls`` is the class built, a subclass of :class:`ReplayPack` (the producer's, which adds its routes)."""
    from .container import read_safetensors
    arrays, hdr = read_safetensors(path)
    meta = json.loads(hdr.get("rpk") or hdr.get("json") or "{}")
    _refuse_predating(meta, path, tensors=arrays)
    pack = (cls or ReplayPack)(arrays, meta, source=str(path))
    pack.susceptibility_field
    return pack

REFUSED_KEYS = (("walk_params", "n_t"), ("walk_params", "dt_traj"), ("walk_params", "cell_size"), ("compression", "n_t"),
                ("compression", "dt_traj"), ("compression", "basis"))

REFUSED_CHANNEL_KEYS = ("susceptibility_path",)

REFUSED_COLUMN_KINDS = ("label", "fraction")

REFUSED_TENSORS = ("positions", "compartment", "boundary_local_time", "susc_field_basis", "dlog_boundary_unit",
                   "blt_counts", "blt_cols", "blt_qvals", "blt_dense_q", "susc_field_0", "susc_field_C", "susc_field_S",
                   "dct_coeffs", "modes", "blt_dct_coeffs")

REFUSED_TENSOR_INFIX = "_rle_"


def _refuse_predating(meta, where, tensors=()):
    """Refuse, by name, a pack this revision does not read (RPK.md 9.5.1): a header with ``n_t`` or ``dt_traj`` (in
    ``walk_params``, its window table or ``compression``), ``compression.basis``, ``walk_params.cell_size``, the
    channel ``susceptibility_path``, a field channel carrying ``iso_P_zz`` (the thirteenth component is never stored), the codec ``identity``, no ``walk_params.block``, or a field channel without
    ``walk_params.qv.field_increments`` (its field read charged by a retired rule); a tensor of the registry's
    refused forms among ``tensors`` (their names; a window's ``s{i}/`` prefix is read through). Its tensors are not
    decoded: a coefficient read against the wrong basis is a plausible number, not an error. A pack is never
    converted; it is walked again."""
    wp = meta.get("walk_params") or {}
    cx = meta.get("compression") or {}
    found = [f"{a}.{b}" for a, b in REFUSED_KEYS if b in (wp if a == "walk_params" else cx)]
    found += [f"walk_params.windows.{k}" for k in ("n_t", "dt_traj") if k in (wp.get("windows") or {})]
    chans = cx.get("channels") or {}
    found += [f"compression.channels.{k}" for k in REFUSED_CHANNEL_KEYS if k in chans]
    sfb = chans.get("susc_field_basis") or {}
    found += [f"compression.channels.susc_field_basis.{k}" for k in ("n_t", "dt") if k in sfb]
    if sfb.get("iso_P_zz") == "stored" or "iso_P_zz" in (sfb.get("channels") or []):       # RPK.md 5.3: iso_P_zz is never stored
        found.append("compression.channels.susc_field_basis carrying iso_P_zz")
    for col in (chans.get("compartment") or {}).get("columns") or []:
        if col.get("kind") in REFUSED_COLUMN_KINDS:
            found.append(f"compression.channels.compartment.columns[{col.get('name')}].kind = {col['kind']}")
    if cx.get("method") == "identity":
        found.append("compression.method = identity")
    for key in tensors:
        base = str(key).split("/")[-1]
        if base in REFUSED_TENSORS or REFUSED_TENSOR_INFIX in base:
            found.append(f"the tensor {key}")
    if not found and "block" not in wp:
        found.append("no walk_params.block")
    if not found and "susc_field_basis" in chans and "field_increments" not in (wp.get("qv") or {}):
        found.append("no walk_params.qv.field_increments")
    if found:
        raise ValueError(f"{where}: This pack carries `{found[0]}`, which this revision does not read; re-walk the pack "
                         "with a conformant producer. A pack is never converted."
                         + (f" (Also: {', '.join(found[1:])}.)" if found[1:] else ""))


@dataclass(frozen=True)
class ScannerField:
    """What a replay reads from its one ``scanner=`` knob: the static field's strength in tesla (``None``
    for no field) and the patient-frame unit vector it points along, plus the machine's name for provenance.

    The direction is not a labelling detail. A susceptibility field is not isotropic and neither is the
    phase it produces: an anisotropic susceptibility depends on the angle between the source and B0, so the
    field's direction enters every contraction. A bare strength carries no direction and keeps the
    conventional bore geometry -- B0 along the bore, the patient's head-foot axis -- which is right for every
    cylindrical magnet and wrong for a bi-planar one, where B0 runs across the patient; such a machine is
    given as a :class:`~dmipy_replay.acquisition.scanners.ScannerLimits` and declares ``b0_axis``.
    """
    B0: object = None
    axis: tuple = (0.0, 0.0, 1.0)
    name: str = None


def scanner_field(scanner):
    """Resolve ``scanner=`` -- ``None``, a field strength in tesla, or a
    :class:`~dmipy_replay.acquisition.scanners.ScannerLimits` -- into a :class:`ScannerField`. A name is refused:
    resolving it is the catalogue's job (``ScannerLimits.of``), not a replay's."""
    if scanner is None:
        return ScannerField()
    from ..acquisition.scanners import ScannerLimits
    if isinstance(scanner, ScannerLimits):
        if scanner.field_T is None:
            raise ValueError(f"the catalogue knows no field strength for {scanner.name!r}; give the field in tesla")
        axis = (0.0, 0.0, 1.0)
        if scanner.b0_axis is not None:
            v = np.asarray(scanner.b0_axis, dtype=np.float64)
            axis = tuple(float(x) for x in v / np.linalg.norm(v))
        return ScannerField(B0=float(scanner.field_T), axis=axis, name=scanner.name)
    if isinstance(scanner, str):
        raise TypeError("scanner is a ScannerLimits (ScannerLimits.of('connectom')) or a field strength in tesla, not a name")
    return ScannerField(B0=float(scanner))


def declared_susceptibility_field(meta, *, has_field):
    """The susceptibility field (``"present"`` / ``"absent"``) a pack's (or a layout's) metadata ``meta`` declares
    through its embedded substrate spec's pools, with ``has_field`` whether it stores the field channel (C3):
    :func:`~dmipy_replay.spec.substrate.susceptibility_field_of`, which refuses a pack without an embedded spec and
    every disagreement between magnetic pools, the ``"field"`` tier and the stored channel."""
    from ..spec.substrate import susceptibility_field_of
    return susceptibility_field_of(meta.get("substrate"), stores_field=has_field, name=meta.get("id"))


@dataclass(frozen=True)
class FieldTerm:
    """The susceptibility term of one replay, as :func:`field_term` resolves it: the static field ``B0`` (tesla) and
    its direction ``axis`` in the machine's frame, with the tissue's ``chi_iso`` / ``chi_aniso``. ``B0`` is
    ``None`` when the term is identically zero -- a substrate with nothing magnetic, a tissue with no
    susceptibility, or a field of zero tesla -- and every route then reads no field channel at all."""
    B0: object = None
    axis: tuple = (0.0, 0.0, 1.0)
    chi_iso: float = 0.0
    chi_aniso: float = 0.0

    @property
    def active(self):
        return self.B0 is not None


def field_term(declared, tissue, scanner, envelope=None):
    """The susceptibility term a substrate's field, a tissue and a scanner give together: the one place a tissue's
    ``chi_iso`` / ``chi_aniso`` meets a substrate's susceptibility field ``declared`` (``"present"`` / ``"absent"``:
    a pack's from its spec's pools, :func:`declared_susceptibility_field`; an analytic or inert phantom substrate's
    ``"absent"`` by kind, :data:`~dmipy_replay.phantom.substrates.FIELD_BY_KIND`) and a scanner's ``B0``. Every replay
    route resolves its field here -- :meth:`ReplayPack._prepare` for the routes that read a tissue with the walk,
    :meth:`~dmipy_replay.replay.study.Primitives.reduction_terms` for those that apply it to contracted primitives (the
    study, the columnar image, the time-segment layout) -- so the rule holds on all of them:

    * ``"absent"``: nothing in the substrate is magnetic and its field is identically zero. A non-zero tissue
      ``chi_iso`` or ``chi_aniso`` is refused by name; any ``B0``, of any strength along any axis, is accepted
      and has no effect.
    * ``"present"``: a non-zero tissue chi needs a field to act in, so one given with ``scanner=None`` is refused.
      A zero or absent chi at any ``B0`` is a field term of zero. With a field ``envelope`` (``{"B0_max",
      "B0_chi_max"}``, a pack's :attr:`ReplayPack.field_envelope`), a term at a ``B0`` above ``B0_max`` or a
      ``|B0| (|chi_iso| + |chi_aniso|)`` above the certified ``B0_chi_max`` is refused, naming both.

    Returns a :class:`FieldTerm`, inactive (``B0`` None) whenever the term is zero."""
    from ..spec.substrate import SUSCEPTIBILITY_FIELD
    if declared not in SUSCEPTIBILITY_FIELD:
        raise ValueError(f"a substrate's susceptibility field is one of {SUSCEPTIBILITY_FIELD}; got {declared!r}")
    f = scanner_field(scanner)
    chi = {k: getattr(tissue, k, None) for k in ("chi_iso", "chi_aniso")}
    given = {k: float(v) for k, v in chi.items() if v is not None and float(v) != 0.0}
    if declared == "absent":
        if given:
            raise ValueError(
                f"the tissue gives {', '.join(f'{k} = {v:.4g}' for k, v in given.items())}, and this substrate "
                "declares its susceptibility field absent: nothing in it is magnetic, so there is no source for a "
                "susceptibility to act on. Replay it with a tissue without chi_iso / chi_aniso (a B0 is accepted and "
                "has no effect)")
        return FieldTerm(axis=f.axis)
    if not given:
        return FieldTerm(axis=f.axis)
    if f.B0 is None:
        raise ValueError(
            f"the tissue gives {', '.join(f'{k} = {v:.4g}' for k, v in given.items())} and no scanner field "
            "(scanner=None): a susceptibility acts only in a B0, and replaying it without one drops the substrate's "
            "field from the signal. Give scanner= (a ScannerLimits or a field strength in tesla), or a tissue without "
            "chi_iso / chi_aniso for the field-free signal")
    if float(f.B0) == 0.0:
        return FieldTerm(axis=f.axis)
    term = FieldTerm(B0=float(f.B0), axis=f.axis, chi_iso=given.get("chi_iso", 0.0), chi_aniso=given.get("chi_aniso", 0.0))
    if envelope is not None:
        B0, top = abs(term.B0), float(envelope["B0_max"])
        if B0 > top * (1.0 + 1e-9):
            raise ValueError(f"B0 = {B0:.4g} T is beyond the {top:.4g} T this pack's field tier is certified at "
                             "(replay_envelope.acquisition.B0_list): replay within it")
        scale, cert = B0 * (abs(term.chi_iso) + abs(term.chi_aniso)), float(envelope["B0_chi_max"])
        if scale > cert * (1.0 + 1e-9):
            raise ValueError(f"B0 (|chi_iso| + |chi_aniso|) = {scale:.4g} T is beyond the certified field scale "
                             f"{cert:.4g} T (the largest B0_list entry times replay_envelope.tissue.chi_max): replay "
                             "a smaller susceptibility or field")
    return term


#: Relative tolerance on a waveform's per-axis gradient amplitude against ``G_max``: the waveform's samples are
#: float32, so a lobe built at exactly ``G_max`` reads back to float32's rounding.
GRADIENT_RTOL = 1e-6


def _view_of(meta):
    """How a refusal names the header's envelope: the pack's, or a length-scaled view's moved one."""
    ls = (meta.get("provenance") or {}).get("length_scaled")
    return (f" (length-scaled view, s = {float(ls['s']):.4g}: the pack's limits over s, replay_envelope.acquisition)"
            if ls else " (replay_envelope.acquisition)")


def check_acquisition_envelope(meta, waveform, *, field_active=False):
    """Refuse, by name, a waveform outside the :class:`~dmipy_replay.replay.envelope.Envelope` a pack states (RPK.md 11:
    a request inside the envelope is guaranteed, one outside is surfaced, never extrapolated). The envelope is
    :meth:`Envelope.from_header` of ``meta``; a header that states none (no ``bound``, or a battery of b values) is
    refused.

    * ``G_max``: the largest played gradient (:attr:`ScannerSequence.played_gradient`, the coils' own) on any axis,
      within :data:`GRADIENT_RTOL`.
    * ``max_refocusing_pulses``: where the pack stores a field channel and the request has a field term
      (``field_active``), the schedule's refocusing pulses within any one storage window
      (:attr:`RFSchedule.refocusing_times`, binned by ``walk_params.windows.T``; the whole schedule on a one-window
      pack) at most the depth the field channel of a window is certified at.

    The waveform's band is held to ``band_hz`` where the band is measured, against the pack's bands
    (:meth:`ReplayPack._check_band`, :meth:`~dmipy_replay.replay.columnar.ColumnarPack.bands_for`). The field strength and
    the susceptibility are held to the envelope by :func:`field_term`, where they act: a ``B0`` is certified over the
    range up to the largest ``B0_list`` entry (:attr:`ReplayPack.field_envelope`). No slew is checked: no term of the
    certificate depends on a waveform's slew except through the band its ramps need.

    A length-scaled view (:meth:`ReplayPack.at_length_scale`) carries its moved ``G_max`` in the same header key and is
    held to it by this function; its refusals name the view. Returns the :class:`Envelope`."""
    from .envelope import Envelope
    env = Envelope.from_header(meta)
    via = _view_of(meta)
    played = np.asarray(getattr(waveform, "played_gradient", waveform.G), np.float64)
    g = float(np.abs(played).max(initial=0.0))
    if g > env.G_max * (1.0 + GRADIENT_RTOL):
        raise ValueError(f"the waveform plays {g:.4g} T/m on one axis, beyond the G_max = {env.G_max:.4g} T/m per axis of "
                         f"the envelope {env.name!r} this pack is certified for{via}")
    if field_active and env.max_refocusing_pulses is not None:
        t = waveform.rf.refocusing_times
        win = (meta.get("walk_params") or {}).get("windows") or {}
        per_window = win.get("n") is not None and int(win["n"]) > 1      # a header with no windows is one window
        n = int(np.bincount(np.floor(t / float(win["T"]) * (1.0 + 1e-12)).astype(np.int64)).max(initial=0)) if per_window and t.size \
            else int(t.size)
        if n > env.max_refocusing_pulses:
            where = f" within one {float(win['T']) * 1e3:.4g} ms storage window" if per_window else ""
            raise ValueError(f"the waveform plays {n} refocusing pulses{where} with a field term, beyond the "
                             f"{env.max_refocusing_pulses} this pack's field channel is certified for "
                             "(replay_envelope.acquisition.max_refocusing_pulses)")
    return env



def field_envelope_of(meta):
    """:attr:`ReplayPack.field_envelope` of a header ``meta`` (a pack's, or a columnar layout's)."""
    env = meta.get("replay_envelope") or {}
    tissue = env.get("tissue") or {}
    unknown = set(tissue) - {"rho2_over_D_max", "chi_max"}
    if unknown:
        raise ValueError(f"this pack's replay_envelope.tissue declares {sorted(unknown)}; it takes only "
                         "rho2_over_D_max and chi_max")
    chi, B0s = tissue.get("chi_max"), (env.get("acquisition") or {}).get("B0_list")
    if chi is None or not B0s:
        return None
    return dict(B0_max=float(max(B0s)), chi_max=float(chi), B0_chi_max=float(max(B0s)) * float(chi))


def _numpy_bloch():
    """The numpy vector-Bloch propagator, the reference (:func:`~dmipy_replay.replay.bloch.replay_bloch`)."""
    from .bloch import replay_bloch
    return replay_bloch


def _pose_matrix(pose):
    """A specimen pose as a 3x3 rotation, from a matrix or an object with ``.rotation``; ``None`` stays ``None``."""
    if pose is None:
        return None
    R = getattr(pose, "rotation", pose)
    return np.asarray(R, np.float64).reshape(3, 3)


class ReplayPack:
    """A replay pack: the channel ``arrays`` plus ``meta``, with ``load`` / ``save`` and the one
    consume path, :meth:`replay`. Accessors mirror the walk parameters (``n_t``, ``dt``, ``K``,
    ``n_walkers``) and the tiers carried (``has_relaxation``, ``has_surface``, ``has_field``)."""

    @staticmethod
    def open(uri, workers=8):
        """A pack open by reference. ``hf://owner/name/path.rpk`` is a single pack on the Hub, a
        :class:`~dmipy_replay.replay.remote.RemotePack`: it plans an acquisition from the file's header and fetches by HTTP
        range only the tensors the plan names (``load()`` downloads the whole file). Any other ``uri`` is the columnar layout
        (a directory, or ``hf://owner/name/prefix`` on the Hub), open by reference:
        a :class:`~dmipy_replay.replay.columnar.ColumnarPack`, whose ``view`` gives a :class:`ReplayPack` of the rows
        and bands an acquisition needs and whose ``image`` replays a whole grid in one pass over the rows."""
        from .hub import is_hub_uri
        if is_hub_uri(uri) and uri.endswith(".rpk"):
            from .remote import RemotePack
            return RemotePack(uri)
        from .columnar import open_columnar
        return open_columnar(uri, workers=workers)

    #: the vector-Bloch propagators by name: a factory returning the function (``bloch.replay_bloch``'s arguments)
    BLOCH_BACKENDS = {"numpy": lambda: _numpy_bloch()}

    @staticmethod
    def _as_distribution(orientation):
        """``orientation=`` as a distribution of poses, or None when it names a single pose."""
        return _as_distribution(orientation)

    def __init__(self, arrays, meta, source=None):
        self.arrays = arrays
        self.meta = dict(meta)
        self.source = source

    @property
    def arrays(self):
        """The pack's tensors by name, a :class:`~dmipy_replay.replay.decoded.PackArrays`: assigning a mapping gives the
        pack its tensors with an empty decoded cache."""
        return self._arrays

    @arrays.setter
    def arrays(self, arrays):
        self._arrays = PackArrays(arrays)

    @property
    def decoded_bytes(self):
        """Bytes of decoded coefficient arrays this pack holds (:meth:`release_decoded`)."""
        return self.arrays.decoded.nbytes

    @property
    def decode_budget(self):
        """Bytes of decoded arrays this pack may hold, settable: :data:`~dmipy_replay.replay.decoded.DECODE_BUDGET_BYTES`
        (1 GiB) unless set. The readers (:func:`~dmipy_replay.replay.compression.read_position_coeffs`,
        :func:`~dmipy_replay.replay.bank.susc_path_coeffs`, :func:`~dmipy_replay.replay.compression.bridge_bands`,
        :func:`~dmipy_replay.replay.compression.occupancy_events`) decode each selection of the pack's stored tensors once
        and every later call of :meth:`replay`, :meth:`walker_signals`, :meth:`walker_phases` and the rest reads the
        held read-only array; an array that does not fit beside what is held is decoded on every call, as
        without the cache. The windows of a pack share its budget, and a view, prefix or any other pack built from this
        one holds its own."""
        return self.arrays.decoded.budget

    @decode_budget.setter
    def decode_budget(self, nbytes):
        self.arrays.decoded.budget = int(nbytes)

    def release_decoded(self):
        """Drop every decoded array this pack (and its windows) holds; the next read decodes again."""
        self.arrays.decoded.clear()

    @classmethod
    def load(cls, path, *, windows=None, **kw):
        """Read a ``.rpk`` file: a local path, or ``hf://owner/name/path/to/file.rpk`` fetched from the hub into the
        local cache and checked against the dataset's ``manifest.json`` (:func:`~dmipy_replay.replay.hub.fetch`).

        ``windows`` reads only the pack's leading windows (RPK.md 4.3: a prefix ``range(k)`` of window indices)
        from the hub, by HTTP byte range, through :func:`~dmipy_replay.replay.hub.fetch_windows` -- only ``hf://``
        URIs take it, since a local file is already whole and cheap to read entire. The pack this returns carries
        :attr:`windows_present`; :meth:`_windows` serves only what it holds, and a replay that reaches further is
        refused by name. Extra keywords (``url=``, ``revision=``) are :func:`~dmipy_replay.replay.hub.fetch_windows`'s."""
        from .hub import fetch, fetch_windows, is_hub_uri
        if windows is not None:
            if not is_hub_uri(path):
                raise ValueError(f"windows= reads a pack's leading windows from the hub by byte range; "
                                 f"{path!r} is not an hf:// URI")
            return read_rpk(fetch_windows(path, windows, **kw), cls=cls)
        if kw:
            raise TypeError(f"load() takes {sorted(kw)} only together with windows=")
        return read_rpk(fetch(path) if is_hub_uri(path) else path, cls=cls)

    @property
    def windows(self):
        """The window table ``walk_params.windows``, ``{n, T, walks}``: the walk is stored in ``n`` windows of ``T``
        seconds each, consecutive windows sharing their boundary instant; window 0's tensors
        sit under the channel names and window ``i`` under the key prefix ``s{i}/``; ``walks`` lists the walks
        that produced them, ``{first, last, seed}`` in windows. Every pack declares it, a walk within one window as
        ``n = 1``."""
        wp = self.meta.get("walk_params") or {}
        seg = wp.get("windows")
        if not seg:
            raise ValueError("this pack declares no walk_params.windows (RPK.md 4.3): a pack is its walk in windows of "
                             "one duration, one window for a walk within that duration. It was written before the "
                             "table existed; rebuild it with build_replay_pack, which declares the table for every pack")
        return seg

    @property
    def n_windows(self):
        return int(self.windows["n"])

    @property
    def windows_present(self):
        """How many of this walk's leading windows (``0..k-1``) this pack's arrays actually hold -- ``None`` for
        a whole pack, an ``int`` ``k`` for one built by :meth:`load`'s ``windows=range(k)`` (RPK.md 4.3). The
        walk's own window count, :attr:`n_windows`, stays the parent's regardless: it says how long the walk
        IS, not how much of it this pack was fetched to hold."""
        k = self.meta.get("windows_present")
        return None if k is None else int(k)

    def _window_arrays(self, i):
        """The tensors of window ``i`` under the channel names: window 0's are the unprefixed ones; a later
        window's are its ``s{i}/`` tensors beside what every window shares -- the weights, a static label, the
        field grid, the voxel tables, whatever has no per-window counterpart."""
        arrays = self.arrays
        if int(i) == 0:
            return {k: v for k, v in arrays.items() if "/" not in k}
        p = f"s{int(i)}/"
        own = {k[len(p):]: v for k, v in arrays.items() if k.startswith(p)}
        if not own:
            raise IndexError(f"window {i}: the pack stores no tensors under {p!r}")
        per_window = {k[3:] for k in arrays if k.startswith("s1/")}
        shared = {k: v for k, v in arrays.items() if "/" not in k and k not in per_window}
        return dict(shared, **own)

    def window(self, i):
        """Window ``i`` of the walk as a pack of its own: its bands over the window from the window's start, its own
        certificate (``fidelity.windows[i]``), the same walkers, weights and spec."""
        import copy
        seg = self.windows; n = int(seg["n"]); i = int(i)
        if i < 0 or i >= n:
            raise IndexError(f"window {i} of a pack that stores {n}")
        if n == 1:
            return self
        meta = copy.deepcopy(self.meta)
        wp = meta["walk_params"]
        T_seg = float(seg["T"])
        walk = next((w for w in seg.get("walks") or [] if int(w["first"]) <= i <= int(w["last"])), None)
        seed = walk["seed"] if walk is not None else wp.get("seed")
        wp.update(T_max=T_seg, seed=seed, windows=dict(n=1, T=T_seg, walks=[dict(first=0, last=0, seed=seed)]))
        fid = dict(meta.get("fidelity") or {})
        per = fid.pop("windows", None)
        meta["fidelity"] = dict(per[i]) if per else fid
        meta.setdefault("provenance", {})["window"] = dict(index=i, of=n, parent_id=self.meta.get("id"))
        out = type(self)(self._window_arrays(i), meta, source=self.source)
        out.arrays.decoded = self.arrays.decoded                  # one budget, and a window decoded once, for all of them
        return out

    def _windows(self):
        """``[(pack, t0, T), ...]``: the windows of the walk this pack actually holds, each as a pack, the instant it
        opens on the walk's clock and its duration (the interval its bands are on); ``[(self, 0.0, T)]`` for a
        single-window pack. Bounded by :attr:`windows_present` for a pack loaded with ``windows=`` -- a caller that
        needs a window beyond it is refused by name before it is asked for here (:meth:`_assert_windows_reach`)."""
        n = self.n_windows
        T_w = self.T_window
        if n == 1:
            return [(self, 0.0, T_w)]
        present = self.windows_present
        upto = n if present is None else min(int(present), n)
        return [(self.window(i), i * T_w, T_w) for i in range(upto)]

    def _block_ends(self, T_w):
        """The block ends of a window of ``T_w`` seconds from its opening instant, ``(n,)`` s: where a reader samples
        the stored functions on the walk's own block (``walk_params.block.h``)."""
        return np.arange(int(round(float(T_w) / self.dt)) + 1) * self.dt

    def _assert_windows_reach(self, T_acq):
        """Refuse, by name, an acquisition of duration ``T_acq`` (s, from the walk's start) that needs a window
        this pack was not loaded with (:attr:`windows_present`)."""
        present = self.windows_present
        if present is None or self.n_windows <= present:
            return
        reach = self._span(T_acq)[1]
        if reach > present:
            raise ValueError(f"this acquisition reaches window {reach - 1}; load windows=range({reach})")

    @property
    def rho2_over_D_max(self):
        """The largest ``rho2 / D`` the contact tier serves (``replay_envelope.tissue.rho2_over_D_max``, RPK.md 8.7), or
        ``None`` when the pack states no bound. ``replay_envelope.tissue`` knows only this one key; any other
        key there is refused by name, naming what is accepted."""
        return self._envelope_tissue().get("rho2_over_D_max")

    def _envelope_tissue(self):
        """``replay_envelope.tissue``: it knows ``rho2_over_D_max`` and ``chi_max``; any other key is refused by name."""
        tissue = (self.meta.get("replay_envelope") or {}).get("tissue") or {}
        unknown = set(tissue) - {"rho2_over_D_max", "chi_max"}
        if unknown:
            raise ValueError(f"this pack's replay_envelope.tissue declares {sorted(unknown)}; it takes only "
                             "rho2_over_D_max and chi_max")
        return tissue

    @property
    def field_envelope(self):
        """The field tier's envelope, ``{"B0_max", "chi_max", "B0_chi_max"}``: the largest field strength
        (``replay_envelope.acquisition.B0_list``), the susceptibility scale (``replay_envelope.tissue.chi_max``) and
        their product, the one certified field scale every field term was read at; ``None`` when the pack stores no
        field or states neither. ``B0_list`` lists the strengths the certificate was measured at, and what it covers is
        the range up to the largest: the field term is linear in ``B0 chi`` and the codec, the field read and the mode
        count are all certified at that one largest product, so any ``B0`` in ``[0, max(B0_list)]`` is served. A replay
        beyond it is refused (:func:`field_term`)."""
        return field_envelope_of(self.meta)

    def _check_rho2(self, rho2_over_D):
        """Refuse a relaxivity beyond the contact tier's envelope (:attr:`rho2_over_D_max`), naming both."""
        top = self.rho2_over_D_max
        if top is not None and float(rho2_over_D) > float(top) * (1.0 + 1e-12):
            D = self.diffusivity
            at = "" if D is None else f" (rho2 = {float(top) * float(D):.3g} m/s at the walk's D = {float(D):.3g} m^2/s)"
            raise ValueError(f"rho2 / D = {float(rho2_over_D):.4g} 1/m is beyond this pack's contact envelope, rho2 / D <= "
                             f"{float(top):.4g} 1/m{at}: past it the stored contact band no longer gives a physical "
                             "attenuation within its certificate (replay_envelope.tissue.rho2_over_D_max)")

    @property
    def has_relaxation(self):
        """C1: a compartment channel, so per-pool T2 / T1 given at replay can be applied."""
        from .compression import has_c1
        return has_c1(self.arrays)

    @property
    def substrate(self):
        """The :class:`~dmipy_replay.spec.SubstrateSpec` the walk was driven by, when the pack embeds one."""
        d = self.meta.get("substrate")
        if d is None:
            return None
        from ..spec.substrate import SubstrateSpec
        return SubstrateSpec.from_dict(d)

    @property
    def has_surface(self):
        """C2: the boundary local time in the bridge form."""
        from .compression import has_c2
        return has_c2(self.arrays)

    @property
    def has_field(self):
        """C3: the field channel (the field's modes along the path)."""
        ch = (self.meta.get("compression", {}).get("channels", {}) or {})
        return ch.get("susc_field_basis") is not None

    @property
    def susceptibility_field(self):
        """The pack's susceptibility field: ``"present"`` -- some pool of its embedded spec is magnetic, the spec
        lists the ``"field"`` tier and the pack carries the field channel (C3) -- or ``"absent"`` -- nothing in its
        substrate is magnetic and the field is identically zero. Read through :func:`declared_susceptibility_field`,
        which refuses a pack without an embedded spec and every disagreement; it is the first thing every replay
        route reads. A view of a layout (:meth:`~dmipy_replay.replay.columnar.ColumnarPack.view`) that leaves the
        layout's field rows unread holds the channel all the same (``meta["view"]["field"] == "unread"``); a
        replay that needs the rows then refuses by name."""
        unread = (self.meta.get("view") or {}).get("field") == "unread"
        return declared_susceptibility_field(self.meta, has_field=self.has_field or unread)

    @property
    def diffusivity(self):
        """The walk's diffusivity (m^2/s) when the producer recorded it."""
        return self.meta.get("walk_params", {}).get("diffusivity")

    @property
    def permeability(self):
        """The permeability (m/s) the walk realised across its walls, from the embedded spec: one number when
        every permeable wall and direction shares it, ``None`` when no wall is permeable, and a dict
        ``{"wall_i.side": kappa}`` when they differ."""
        spec = self.substrate
        if spec is None:
            return None
        found = {}
        for i, w in enumerate(getattr(spec, "walls", ()) or ()):
            perm = getattr(w, "permeability", None)
            for side in ("in_to_out", "out_to_in"):
                v = getattr(perm, side, None) if perm is not None else None
                if v:
                    found[f"wall_{i}.{side}"] = float(v)
        if not found:
            return None
        vals = set(found.values())
        return found[next(iter(found))] if len(vals) == 1 else found

    def at_permeability(self, kappa):
        """This pack read at the wall permeability ``kappa`` (m/s): the view at ``a = kappa / permeability``,
        which is also the view at ``a`` times the walked diffusivity, since the same path is the walk at
        ``(a D, a kappa)`` and at no other pair (:meth:`at_diffusivity`). Needs a walk whose permeable walls
        share one permeability; a spec with several is read by its diffusivity instead."""
        k0 = self.permeability
        if k0 is None:
            raise ValueError("this pack's walls are impermeable (or it embeds no spec), so a permeability has no "
                             "walked value to be read against; give the diffusivity instead")
        if isinstance(k0, dict):
            raise ValueError(f"this pack's walls have different permeabilities, {k0}, which one number cannot "
                             f"rescale; read it by diffusivity (at_diffusivity), which scales them all together")
        return self.at_diffusivity(float(self.diffusivity) * float(kappa) / float(k0))

    def at_diffusivity(self, D):
        """This pack read at the bulk diffusivity ``D``: the same arrays on the block grid divided by
        ``a = D / diffusivity`` (dmipy-sim#289).

        THE PAIR, AND THE COMPUTE IT SAVES. Across a permeable wall the walk realised a crossing probability
        per encounter of about ``kappa sqrt(dt / D)``, which the rescale leaves unchanged only when the
        permeability scales with the diffusivity: the view is the walk at ``(a D, a kappa)`` -- the pair with
        the walked ratio ``kappa / D`` -- and at no other. So one walk at the SLOWEST diffusivity a study needs,
        over the LONGEST time, serves every faster setting on that line at its shorter time as a view, and a
        setting off the line needs its own walk. :attr:`~dmipy_replay.spec.Tissue.kappa` states the permeability
        wanted and :attr:`~dmipy_replay.spec.Tissue.D` the diffusivity; given both, they must sit on the line.

        A path walked at ``D0`` and read on a clock divided by ``a`` IS a path walked at ``a D0``: the stored bands
        carry the window's normalisation ``sqrt(2 / T)``, which the view divides by ``sqrt(a)``
        (:func:`~dmipy_replay.replay.compression.rescale_bands`), so nothing is decoded and every channel follows on the
        new clock in its own space -- the gradient's projection, the relaxation gates, the surface term (whose weight
        divides by the new ``D``), the field's gate. This is what a change of the substrate's temperature does,
        about 2.5 per cent per kelvin for water. The same path is the walk at ``(a D0, a kappa)`` across a
        permeable wall and at no other pair, and the pair is recorded in ``provenance.diffusivity_scaled``.

        Only faster than walked is served (``a >= 1``). The certificate was measured at the walked grid: the
        split-half floor and the codec error are the same statistics on the same coefficients, but the in-step
        integration bias goes as ``D dt^2``, which is ``D0 dt0^2 / a`` -- it falls for ``a > 1`` and grows for
        ``a < 1``, past what the walk's block was chosen to keep below the floor. A published pack does
        not carry what would re-certify it, so a slower replay is refused rather than returned uncertified;
        walk the pack at the slower diffusivity. The acquisition must still fit the shortened walk, ``T / a``,
        which the replay checks as it checks every waveform. The view at ``G(t)`` is the walk at ``G(a t) / a``, so the
        envelope moves with it: its ``band_hz`` and its ``G_max`` times ``a``.

        The result is an ordinary :class:`ReplayPack` sharing this one's arrays, so :meth:`prefix`, the pose
        cache and every route read it as they read any pack; :attr:`~dmipy_replay.spec.Tissue.D` on a replay
        resolves to it. The two similarities, what each keeps, moves and refuses, side by side:
        ``docs/replay-guide/views.md``.
        """
        import copy
        D0 = self.diffusivity
        if D0 is None:
            raise ValueError("this pack records no diffusivity (walk_params.diffusivity), so a replay at another "
                             "one has no ratio to read the grid by")
        a = float(D) / float(D0)
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError(f"a diffusivity is a positive number; got {D!r} against the walk's {D0!r}")
        if a == 1.0:
            return self
        if a < 1.0:
            raise ValueError(
                f"a replay at D = {float(D):.3g} m^2/s is slower than the walk's {float(D0):.3g} (a = {a:.3f}), "
                f"which stretches the block grid and grows the in-step integration bias as 1/a beyond what the "
                f"pack's certificate covers. Faster than walked is free; slower needs a walk at that "
                f"diffusivity, or a pack that recorded enough to re-certify a stretched grid, which this one did "
                f"not (dmipy-sim#289)")
        meta = copy.deepcopy(self.meta)
        wp = meta.setdefault("walk_params", {})
        wp["diffusivity"] = float(D)
        wp["T_max"] = float(wp["T_max"]) / a
        wp["block"] = dict(wp["block"], h=float(wp["block"]["h"]) / a)
        if wp.get("windows"):
            wp["windows"] = dict(wp["windows"], T=float(wp["windows"]["T"]) / a)
        cx = meta.get("compression") or {}
        if cx.get("temporal_bandwidth_hz") is not None:
            cx["temporal_bandwidth_hz"] = float(cx["temporal_bandwidth_hz"]) * a
        acq = (meta.get("replay_envelope") or {}).get("acquisition") or {}
        for key in ("band_hz", "G_max", "slew_max"):         # the view at G(t) is the walk at G(a t) / a: every frequency
            if acq.get(key) is not None:                     # and the amplitude move by a (slew_max, a class header's, by a^2)
                acq[key] = float(acq[key]) * (a * a if key == "slew_max" else a)
        pm = (cx.get("channels") or {}).get("susc_field_basis")
        if pm is not None:
            pm["T"] = float(pm["T"]) / a                     # the field's modes are read over the shorter window
            if pm.get("read_interval") is not None:
                pm["read_interval"] = float(pm["read_interval"]) / a
        # the view's situation IS the walk at (a D, a kappa): the embedded spec says so, wall by wall and pool by
        # pool, so that the view's own permeability and diffusivity read as what it realises
        kappa = {}
        sd = meta.get("substrate")
        if isinstance(sd, dict):
            for i, w in enumerate(sd.get("walls") or []):
                perm = w.get("permeability") if isinstance(w, dict) else None
                if isinstance(perm, dict):
                    for side in ("in_to_out", "out_to_in"):
                        v = perm.get(side)
                        if v:
                            kappa[f"wall_{i}.{side}"] = {"walked": float(v), "replayed": float(v) * a}
                            perm[side] = float(v) * a
            for pool in sd.get("pools") or []:
                if isinstance(pool, dict):
                    for key in ("diffusivity", "D"):
                        if isinstance(pool.get(key), (int, float)) and pool[key]:
                            pool[key] = float(pool[key]) * a
        prov = meta.setdefault("provenance", {})
        prov["diffusivity_scaled"] = dict(walked=float(D0), replayed=float(D), a=a,
                                          grid_dt=dict(walked=float(self.dt), replayed=float(self.dt) / a),
                                          permeability=kappa,
                                          certificate="measured at the walked grid; the in-step bias falls as 1/a")
        from .compression import rescale_bands
        # the bands are normalised by sqrt(2 / T): read over T / a they are the stored ones / sqrt(a)
        arrays = rescale_bands(self.arrays, 1.0 / np.sqrt(a))
        c2 = (cx.get("channels") or {}).get("boundary_local_time")
        if c2 is not None and c2.get("T") is not None:
            c2["T"] = float(c2["T"]) / a
        # the events' instants on the shorter clock
        for k in list(arrays):
            if k.split("/")[-1].endswith("_events_t"):
                arrays[k] = np.asarray(arrays[k], np.float64) / a
        out = type(self)(arrays, meta, source=self.source)
        return out

    def at_length_scale(self, s):
        """This pack read as the walk in its substrate at ``s`` times the size (dmipy-sim#542): every stored
        length multiplied by ``s`` -- the position coefficients, the contact channel's boundary local time, the
        field grid's origin and voxel size -- with the diffusivity at ``s^2`` and every wall rate at ``s`` times
        the walked ones. The second similarity of diffusion, in space, beside :meth:`at_diffusivity`'s in time.

        Scaling every length by ``s`` at fixed times leaves the dimensionless walk as it was: the step relative to
        the smallest feature ``sqrt(D dt) / L``, ``kappa L / D``, ``rho L / D`` and ``D T / L^2`` are the walked
        values. So the stored path with every position multiplied by ``s`` IS a walk of the scaled substrate
        (:meth:`~dmipy_replay.spec.substrate.SubstrateSpec.scaled`, the view's embedded spec) at ``s^2 D``, with
        permeability and reactivity ``s kappa``, and the susceptibility field at corresponding points is the
        walked one (a dipole field depends on shape, not size). Nothing is decoded: the coefficients are linear
        in the positions, so the view's length-carrying arrays are the coded ones times ``s`` (float64; an
        integer container's scales), and every other array is shared.

        Each channel then follows in its own space. The gradient channel of the view at ``G`` is this pack at
        ``s G`` (``gamma G . s x``). The surface term at a tissue's ``rho2`` is this pack's at ``rho2 / s``, since
        ``rho2 s ell / (s^2 D) = (rho2 / s) ell / D``. The relaxation gates and the compartment labels are
        unchanged, and so is the field path channel; the grid route reads the scaled positions against the
        scaled grid. :attr:`~dmipy_replay.spec.Tissue.D` on a replay resolves against the view's diffusivity.

        Both directions are exact in the physics; what bounds ``s`` is the certificate. The view at ``G`` is the
        walk at ``s G``, so the envelope the pack is certified for (``replay_envelope.acquisition``) moves to
        ``G_max / s`` -- more than the pack states for ``s < 1``, less for ``s > 1``; its band, a frequency, stays -- and
        :meth:`_check_envelope` holds every replay to the moved limits, the view carrying no rule of its own; the
        contact tier's ``rho2_over_D_max`` moves to ``rho2_over_D_max / s`` the same way. The pair is recorded in
        ``provenance.length_scaled`` (``s`` cumulative over repeated views), with the battery's ``b_max`` over
        ``s^2`` beside it.

        With :meth:`at_diffusivity` (``a >= 1``) a target ``(L, D, T)`` is reached from a walk ``(L0, D0, T0)``
        by ``s = L / L0`` then ``a = D / (s^2 D0)`` when ``D / L^2 >= D0 / L0^2`` and ``D T / L^2 <= D0 T0 / L0^2``:
        one walk per shape at the slowest ``D / L^2`` over the longest ``D T / L^2`` serves every size and every
        faster diffusivity, read with :meth:`prefix` where it is longer than needed. The result is an ordinary
        :class:`ReplayPack`: :meth:`prefix`, the pose cache and every route read it as they read any pack. The two
        similarities, what each keeps, moves and refuses, side by side: ``docs/replay-guide/views.md``.
        """
        import copy
        from .compression import POSITION_AXES
        from ..spec.substrate import SubstrateSpec
        s = float(s)
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError(f"a size scale is a positive number; got {s!r}")
        if s == 1.0:
            return self
        lengths = (set(POSITION_AXES) | {f"{ax}_ends" for ax in POSITION_AXES}
                   | {"pos_band_scale", "blt_start", "blt_endpoint", "blt_bridge_dst", "blt_band_scale", "susc_grid_origin"})
        arrays = {k: (np.asarray(v, np.float64) * s if v is not None and k.rsplit("/", 1)[-1] in lengths else v)
                  for k, v in self.arrays.items()}
        meta = copy.deepcopy(self.meta)
        wp = meta.setdefault("walk_params", {})
        D0 = wp.get("diffusivity")
        if D0 is not None:
            wp["diffusivity"] = float(D0) * s * s
        if isinstance(wp.get("qv"), dict):                                       # every length x s: QV of a length x s^2
            q = wp["qv"]
            if q.get("positions") is not None:
                q["positions"] = [float(x) * s * s for x in q["positions"]]
            if q.get("local_time") is not None:
                q["local_time"] = float(q["local_time"]) * s * s
        gm = ((meta.get("compression") or {}).get("channels") or {}).get("susceptibility_grid")
        if isinstance(gm, dict):
            for key in ("origin", "voxel_size"):
                if gm.get(key) is not None:
                    gm[key] = (np.asarray(gm[key], np.float64) * s).tolist()
        env = meta.get("replay_envelope") or {}
        acq = env.get("acquisition") or {}
        b_max = acq.get("b_max")
        if b_max is not None:
            acq["b_max"] = float(b_max) / (s * s)
        for key in ("G_max", "slew_max"):                                        # the view at G is this pack at s G
            if acq.get(key) is not None:                                         # (slew_max: a scanner-class header's)
                acq[key] = float(acq[key]) / s
        tis = env.get("tissue") or {}
        if tis.get("rho2_over_D_max") is not None:
            tis["rho2_over_D_max"] = float(tis["rho2_over_D_max"]) / s
        if isinstance(meta.get("substrate"), dict):
            meta["substrate"] = SubstrateSpec.from_dict(meta["substrate"]).scaled(s).to_dict()
        prov = meta.setdefault("provenance", {})
        before = prov.get("length_scaled") or {}
        s_total = float(before.get("s", 1.0)) * s
        prov["length_scaled"] = dict(
            s=s_total,
            diffusivity=dict(walked=before.get("diffusivity", {}).get("walked", D0), replayed=wp.get("diffusivity")),
            b_max=dict(walked=before.get("b_max", {}).get("walked", b_max), served=acq.get("b_max")),
            certificate="measured at the walked lengths; the view at G is the walk at s G, served up to G_max / s")
        return type(self)(arrays, meta, source=self.source)

    def _check_envelope(self, waveform, field):
        """The one check every replay route applies to a waveform against the pack's stated envelope
        (:func:`check_acquisition_envelope`): the per-axis gradient and the refocusing depth of the field channel where
        ``field`` (the request's :class:`FieldTerm`) is active; the band is :meth:`_check_band`'s. A length-scaled view is held to
        its moved limits by the same call. The field strength and the susceptibility reach their envelope in
        :func:`field_term`, which the routes call with :attr:`field_envelope`."""
        check_acquisition_envelope(self.meta, waveform, field_active=bool(field.active))

    def _at_tissue(self, tissue):
        """The pack a replay with ``tissue`` reads: this one, or its view at the tissue's diffusivity and
        permeability. Both stated, they must sit on the walk's line ``kappa / D = kappa_walk / D_walk``."""
        D = getattr(tissue, "D", None)
        kappa = getattr(tissue, "kappa", None)
        if kappa is None and (D is None or self.diffusivity is None or float(D) == float(self.diffusivity)):
            return self
        if kappa is None:
            return self.at_diffusivity(D)
        k0 = self.permeability
        if k0 is None or isinstance(k0, dict):
            raise ValueError(
                "a tissue states a wall permeability, and this pack "
                + ("has no permeable wall to read it against" if k0 is None else f"has walls of different permeabilities {k0}")
                + ": a replay cannot make a wall permeable or change one wall alone; walk the spec wanted (dmipy-sim#289)")
        a_k = float(kappa) / float(k0)
        if D is not None and self.diffusivity is not None:
            a_d = float(D) / float(self.diffusivity)
            if abs(a_d - a_k) > 1e-9 * max(a_d, a_k):
                raise ValueError(
                    f"the pair (D = {float(D):.3g} m^2/s, kappa = {float(kappa):.3g} m/s) is not on this walk's line: "
                    f"the walk realised kappa / D = {float(k0) / float(self.diffusivity):.3g} s/m^2 and a rescale "
                    f"keeps that ratio, so it serves (a D_walk, a kappa_walk) for any a >= 1 and no other pair. "
                    f"D asks for a = {a_d:.4g}, kappa for a = {a_k:.4g}; a setting off the line needs its own "
                    f"walk (dmipy-sim#289)")
        return self.at_permeability(kappa)

    @property
    def nominal(self):
        """The embedded spec's values as a :class:`~dmipy_replay.spec.Tissue`: what a paper's replay applies,
        ``replay(seq, tissue=pack.nominal, scanner=pack.nominal_field_T)``; ``None`` for a pack without a spec.
        Its ``T2`` / ``T1`` map the pools that declare one by name (``None`` when none does), so a spec that
        declares a T2 in some pools only gives a tissue the replay refuses, naming the rest."""
        spec = self.substrate
        if spec is None:
            return None
        from ..spec.tissue import Tissue
        return Tissue.from_spec(spec)

    @property
    def nominal_field_T(self):
        """The spec's calibration field (T), the ``scanner`` of the nominal replay; ``None`` when it names none."""
        spec = self.substrate
        return None if spec is None else spec.nominal_field_T

    def positions(self):
        """Every walker's position at the walk's block ends, ``(n_walkers, n_t, 3)`` float64: the stored function (the
        exact endpoints and the bands of each window) evaluated there, the windows joined on their shared instant --
        what the vector-Bloch route propagates through. A reader's sampling, never a stored form."""
        from .compression import positions_at, require_position_method
        require_position_method(self.method)
        parts = [positions_at(w.arrays, dict(T=T_w), self._block_ends(T_w)) for w, _, T_w in self._windows()]
        return np.concatenate([p if j == 0 else p[:, 1:] for j, p in enumerate(parts)], axis=1)

    def _by_pool(self, values, what):
        """A tissue's per-pool ``T2`` / ``T1`` as the list by pool id over every pool of the embedded spec:
        ``None`` for none, else the ``{pool name: seconds}`` mapping resolved through the spec, which refuses a
        missing pool, an unknown name, a number for every pool, and any per-pool value on a pack without a spec
        (RPK.md 8.5: such a pack replays the gradient alone)."""
        if values is None:
            return None
        spec = self.substrate
        if spec is None:
            raise ValueError(f"{what} was given by pool but this pack embeds no substrate spec, so it has no pool "
                             f"names: it replays the gradient alone (RPK.md 8.5)")
        if not isinstance(values, dict):
            raise ValueError(f"{what} on a pack is {{pool name: seconds}} over every pool of its spec "
                             f"{[p.name for p in spec.pools]}; one number is the closed form's (got {values!r})")
        from ..spec.tissue import _by_pool_id
        return [float(v) for v in _by_pool_id(spec, values, what)]

    def replay(self, waveform, *, tissue=None, scanner=None, orientation=None, compartment=None, complex_signal=False,
               tol=PLAN_TOL, K=None, modes=None):
        """The signal of ``waveform`` on this pack: the gradient always, and every other tier whose inputs are given
        and which the pack carries.

        **Plan before read.** Only the position bands and field modes the acquisition needs are decoded: the fewest
        whose truncation costs at most ``tol`` times the pack's Monte-Carlo floor (:meth:`plan`). The signal returned is a
        :class:`~dmipy_replay.replay.planning.Planned` array (an ``ndarray``) whose ``.plan`` is the
        :class:`~dmipy_replay.replay.planning.ReadPlan` of this call: the bands and modes read, their planned error and the
        source of the variances. ``K=`` and ``modes=`` read that many instead (the error of that read is stated the same
        way); ``tol=None`` reads every stored band and mode. The result is bit for bit the replay of a full decode cut at
        the same ``K`` and ``modes``. A pose distribution (``orientation=``) is the pose expansion's and reads every band.

        ``waveform`` is a :class:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence`: its ``G_eff``, the
        effective gradient, is what this route integrates, and its RF schedule says where the 180 is (a schedule
        without one is a gradient echo). It is read on the pack grid (``n_t`` samples of ``dt``, zero outside
        the waveform).

        Three things describe a replay setting, each stated once:

        * ``tissue`` -- **what the material is**: a :class:`~dmipy_replay.spec.Tissue` (pool T2 / T1, the walls'
          transverse ``rho2`` and longitudinal ``rho1``, the bulk D, the field source's chi) or ``None``, the
          bare diffusion signal. T2 / T1 are
          ``{pool name: seconds}`` over every pool of the embedded spec, ``inf`` for no decay. ``pack.nominal``
          is the embedded spec's values, so a paper's replay is ``replay(seq, tissue=pack.nominal,
          scanner=pack.nominal_field_T)``; ``pack.nominal.replace(T2={"intra": 0.08})`` changes one pool.
        * ``orientation`` -- **where the substrate sits**: its pose. **One pose** -- a 3x3 rotation (substrate
          frame -> lab), or the lab direction its axis (``spec.frame.axis``, default z) points along -- is exact by
          pose covariance: the gradient and the field are rotated into the substrate frame together, and no
          expansion is involved. **A distribution of poses** -- a :class:`~dmipy_replay.replay.so3.Distribution`
          (one pose, an axis density, a Watson cone, a Bingham fan) or an :class:`~dmipy_replay.replay.fod.FOD`,
          which is read as an axis density with no statement about the substrate's own azimuth -- goes through
          :meth:`pose_response` and is composed on SO(3). An FOD's basis must be declared (``FOD.from_sh``,
          ``FOD.native``); a bare coefficient array is refused, since the convention cannot be inferred.
        * ``scanner`` -- **what the scanner is**: its static field, as a
          :class:`~dmipy_replay.acquisition.scanners.ScannerLimits` (the catalogue's ``field_T``) or a number in
          tesla, or ``None`` for no field. Its DIRECTION comes from the machine too -- a bi-planar
          magnet's field runs across the patient, not along the bore -- and the pose turns it.

        The tiers follow from those: **gradient** (C0) always, in mode space from the position coefficients;
        **bulk relaxation** (C1) with a T2 / T1 in the tissue, under the waveform's coherence gate, on the
        occupancy channel; **surface relaxivity** (C2) with a rho2 and/or a rho1, scaled by the walk's D (the
        tissue's, else the pack's recorded one), on the boundary local time -- rho2 gated by the coherence
        (transverse) and rho1 by its complement (stored along B0, e.g. a stimulated echo's mixing time); **field** (C3) with a chi in the tissue and a field on
        the scanner, on the path channel (or the stored basis sampled along the decoded path), the 180 the
        waveform's own. A tier whose inputs are given but which the pack does not carry raises rather than
        returning a plausible number.

        ``compartment`` restricts the ensemble mean to one pool id (or a boolean walker mask).
        """
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        if isinstance(waveform, Protocol):                # a multi-TE scheme: each sequence replayed, placed at its rows
            kw = dict(tissue=tissue, scanner=scanner, orientation=orientation, compartment=compartment,
                      complex_signal=complex_signal, tol=tol, K=K, modes=modes)
            parts = [self.replay(seq, **kw) for seq in waveform]
            out = planned(waveform.scatter(parts), tuple(p.plan for p in parts))      # one plan per sequence
            return out
        view = self._at_tissue(tissue)
        if view is not self:
            return view.replay(waveform, tissue=tissue, scanner=scanner, orientation=orientation,
                               compartment=compartment, complex_signal=complex_signal, tol=tol, K=K, modes=modes)
        dist = self._as_distribution(orientation)
        if dist is not None:
            S = self.pose_response(waveform, tissue=tissue, scanner=scanner, compartment=compartment).compose(dist)
            return planned(S if complex_signal else np.abs(S), self._full_plan("the pose expansion reads every band and mode"))
        P = self._prepare(waveform, tissue=tissue, scanner=scanner, orientation=orientation, compartment=compartment,
                          read=dict(tol=tol, K=K, modes=modes))
        E = _signal_factor(self._walker_phases(P, waveform), P["voxel"])
        S = (P["pathway"] * P["ew"][:, None] * E).sum(0) / P["norm"]
        return planned(S if complex_signal else np.abs(S), P["plan"])

    def walker_signals(self, waveform, *, tissue=None, scanner=None, orientation=None, compartment=None,
                       b1_scale=None, off_resonance_T=None, tol=PLAN_TOL, K=None, modes=None):
        """The replay **before the ensemble mean**: ``(w, ew, E)`` with ``w`` the walkers' statistical weights
        ``(n_w,)``, ``ew`` those weights with the relaxation and surface terms applied, and ``E`` the complex
        signal factor of every walker at every measurement, ``(n_w, n_meas)``.

        :meth:`replay` is ``(ew[:, None] * E).sum(0) / w.sum()``; any other grouping of the walkers -- by the
        voxel they started in, which is what partitions one walk into a phantom (RPH.md) -- is the same sum
        over its members with its own normaliser. Knobs resolve as in :meth:`replay`. An unbalanced encoding's
        voxel factor (:meth:`ScannerSequence.voxel_factor`) is carried in ``E`` as a per-measurement scale,
        so the sum above is the voxel's signal. Only the bands and modes the plan names are decoded (``tol=``, ``K=``,
        ``modes=`` as in :meth:`replay`); ``E`` is a :class:`~dmipy_replay.replay.planning.Planned` array whose ``.plan`` is the
        read's :class:`~dmipy_replay.replay.planning.ReadPlan`. With ``b1_scale`` (a scalar or per walker) or
        ``off_resonance_T`` (a scalar or per walker) given, ``E`` comes from the RF-aware route
        (:meth:`replay_bloch`): the transverse magnetisation of each walker at the readout, with the relaxation
        the route applied itself, so ``ew`` is then ``w``.
        """
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        view = self._at_tissue(tissue)
        if view is not self:
            return view.walker_signals(waveform, tissue=tissue, scanner=scanner, orientation=orientation,
                                       compartment=compartment, b1_scale=b1_scale, off_resonance_T=off_resonance_T,
                                       tol=tol, K=K, modes=modes)
        if b1_scale is None and off_resonance_T is None:
            P = self._prepare(waveform, tissue=tissue, scanner=scanner, orientation=orientation, compartment=compartment,
                              read=dict(tol=tol, K=K, modes=modes))
            w = np.asarray(self.spin_weights, np.float64)
            return w, P["pathway"] * P["ew"], planned(_signal_factor(self._walker_phases(P, waveform), P["voxel"]), P["plan"])
        E = self.replay_bloch(waveform, b1_scale=b1_scale, off_resonance_T=off_resonance_T, tissue=tissue, scanner=scanner,
                              orientation=orientation, compartment=compartment, complex_signal=True, per_walker=True)
        w = np.asarray(self.spin_weights, np.float64)
        return w, w, planned(E, self._full_plan("the vector-Bloch route reads every band and mode"))

    def walker_phases(self, waveform, *, tissue=None, scanner=None, orientation=None, compartment=None,
                      weights=None, tol=PLAN_TOL, K=None, modes=None):
        """:meth:`walker_signals` before the complex exponential: ``(w, ew, phi)`` with ``phi`` the accumulated
        phase of every walker at every measurement, ``(n_w, n_meas)`` real, so that ``E = exp(1j * phi)``. A
        consumer that reduces many walkers over its own groups (an image: the walkers of each voxel) takes the
        phase and forms the exponential and the sums where it accumulates them, on its device; the exponential
        over ``(n_w, n_meas)`` is the one host operation of a replay that does not amortise. Knobs resolve as in
        :meth:`replay` (the plan, ``tol=`` / ``K=`` / ``modes=``, is :meth:`replay`'s, on ``phi.plan``); the RF-aware route has no phase (its signal is the magnetisation vector itself) and is
        :meth:`walker_signals` with ``b1_scale`` or ``off_resonance_T``. An unbalanced encoding is refused
        here: its voxel factor is an amplitude per measurement, which a phase cannot carry, and a consumer
        that formed its own sums would silently omit it -- :meth:`walker_signals` carries it in ``E``."""
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        view = self._at_tissue(tissue)
        if view is not self:
            return view.walker_phases(waveform, tissue=tissue, scanner=scanner, orientation=orientation,
                                      compartment=compartment, weights=weights, tol=tol, K=K, modes=modes)
        read = dict(tol=tol, K=K, modes=modes) if weights is None else dict(tol=None, K=None, modes=None)   # a caller's weights are on every band
        P = self._prepare(waveform, tissue=tissue, scanner=scanner, orientation=orientation, compartment=compartment, read=read)
        if np.any(P["voxel"] != 1.0):
            raise ValueError(
                "this encoding leaves a net moment at the readout, so the voxel's extent multiplies the signal by "
                "a factor per measurement (ScannerSequence.voxel_factor). A phase cannot carry an amplitude, and a "
                "consumer forming its own sums from these phases would omit it silently; use walker_signals, "
                "whose E carries it, or replay")
        if weights is not None:
            P["W"] = self._check_weights(weights, P, orientation)
        w = np.asarray(self.spin_weights, np.float64)
        return w, P["pathway"] * P["ew"], planned(self._walker_phases(P, waveform), P["plan"])

    def _check_weights(self, weights, P, orientation):
        """Validate a caller's per-position replay weights, and refuse the cases this route cannot serve.

        Shape alone is not enough to make weights the right ones. It is identical for every window once
        ``K`` and ``n_meas`` match, so weights built on the SEQUENCE's grid rather than the pack's pass a
        shape test and encode a 45 per cent error in b. The dtype matters too: an integer array is silently
        truncated and a complex one silently loses its imaginary part.
        """
        if len(P["windows"]) > 1:
            raise ValueError("weights= are the per-position replay weights of ONE window, and this pack stores its walk in "
                             f"{len(P['windows'])} windows; give the waveform and let the replay read every window")
        W = np.asarray(weights)
        if not np.issubdtype(W.dtype, np.floating):
            raise ValueError(
                f"weights must be real floating point; got {W.dtype}. An integer array is truncated and a "
                f"complex one loses its imaginary part, both silently")
        W = W.astype(np.float64)
        want = (self.n_coeffs * 3, P["G_eff_wf"].shape[0])
        if W.shape != want:
            raise ValueError(
                f"weights are this pack's replay weights for ONE position, {want} -- (n_coeffs * 3, "
                f"n_meas). Got {W.shape}. dmipy_replay.phantom.bore.delivered_weights builds them per voxel, "
                f"and must be given THIS pack's window: K={self.K}, T={self.T_window!r}")
        if not np.all(np.isfinite(W)):
            raise ValueError("weights are not finite, so the phase they produce is not a signal")
        if orientation is not None:
            raise ValueError(
                "weights= and orientation= cannot both be given. A pose is carried by ROTATING the gradient "
                "before it is projected, and supplied weights replace that projection wholesale -- so the "
                "pose would be silently discarded (measured: 48 per cent of the signal). Build the weights "
                "for the posed gradient instead, which reproduces the pose exactly")
        if P["B0"] is not None:
            raise ValueError(
                "weights= cannot be combined with an active susceptibility field yet. The field branches of "
                "this contraction rebuild the gradient from the nominal sequence and do not read supplied "
                "weights, so the result would be bit-identical to passing none -- including for weights of "
                "zero. That is silently wrong in exactly the low-field case this exists for (dmipy-sim#369)")
        return W

    def _walker_phases(self, P, waveform):
        """``(n_w, n_meas)`` accumulated phase of every walker under the prepared acquisition ``P``: the gradient
        from :func:`_band_phase`, and with a field the path channel contracted under the pose's field direction
        (the grid route, a pack without the path channel, samples the stored basis along the decoded path).

        The gradient term is :func:`_band_phase` and nothing else, so it is the same number
        :func:`~dmipy_replay.replay.study.walker_primitives` contracts, on every route and at every ``b``
        (dmipy-sim#484)."""
        phi = _band_phase(P, W=P.get("W"))                                            # (n_w, n_meas)
        if P["B0"] is None or P["M_use"] == 0:                                       # a field term of zero (field_term), or a plan that reads no mode
            return phi
        ch, b0_dir, B0, chi_aniso = P["ch"], P["b0_dir"], P["B0"], P["chi_aniso"]
        gm = ch.get("susceptibility_grid")
        chi_i = float(P["chi_iso"])
        if ch.get("susc_field_basis") is None:
            raise ValueError("this replay has a field term and the pack carries no field channel (C3)")
        from .field import contract# the field route: every term a contraction
        Psi, names = _path_field_channels(P, waveform)
        aniso = chi_aniso if (_has_aniso(gm, names) and chi_aniso) else 0.0
        return phi + contract(Psi, b0_dir, B0=float(B0), chi_iso=chi_i, chi_aniso=aniso)[:, None]

    def replay_bloch(self, waveform, *, b1_scale=None, off_resonance_T=None, tissue=None, scanner=None,
                     orientation=None, compartment=None, complex_signal=False, per_walker=False,
                     crusher_seed=0, backend="numpy"):
        """The RF-aware replay: each walker's magnetisation vector propagated through the actual sequence
        operators on this pack's walk (:func:`~dmipy_replay.replay.bloch.replay_bloch`).

        This route reads the waveform's PHYSICAL ``G`` and applies its pulses; :meth:`replay` reads ``G_eff``
        with the pulses folded in. The magnitude route of :meth:`replay` assumes ideal pulses and reads the
        signal as a phase sum, so it cannot carry anything that acts on the magnetisation vector: a flip angle that is not nominal
        (``b1_scale``), a finite pulse, a pulse train's coherence pathways. Those are what this route is for,
        and it costs a propagation per piece instead of a projection.

        Knobs are the same as :meth:`replay` and resolve the same way; the pose rotates the
        acquisition and the field direction as it does there. ``b1_scale`` scales every flip angle, as a scalar
        or per walker. ``off_resonance_T`` is a **uniform** static field offset (a field-map value, in T) every
        walker precesses in through the actual pulses -- what a macroscopic layer of a phantom is (RPH.md 5.1).
        ``per_walker`` returns every walker's transverse magnetisation at the readout, ``(n_w, n_meas)`` complex,
        unweighted, instead of the ensemble mean (single-readout sequences).

        A sequence's ``crusher`` is applied here (dmipy-sim#305), and so is the voxel itself for an encoding
        that leaves a net moment at the readout (dmipy-sim#375): each walker is placed at its own drawn
        offset in the prescribed voxel, which is what a played spoiler winds across; a declared ``crusher`` is
        for a winding the waveform does not play in ``G``. The declared one is the voxel-scale spoiler, which a
        micron cell cannot produce geometrically -- a gradient cannot wind much beyond 2 pi across it -- so it
        is modelled as the forward engine models it (:func:`~dmipy_replay.engine.bloch._build_crusher`): each
        walker carries a macroscopic coordinate ``u`` in [0, 1) and accrues ``2 pi n_cycles u`` over each
        crusher window, so the ensemble dephases over ``n_cycles`` turns. Without it every coherence pathway
        of a pulse train stays degenerate and recombines, and the echo barely depends on the refocusing flip
        angle, which is not what a train does. ``crusher_seed`` draws those coordinates, so a replay of one
        pack and one sequence is reproducible.

        ``backend`` names the propagator, one of :attr:`BLOCH_BACKENDS` (``"numpy"`` here, the reference; a
        producer's subclass adds its device ones), all taking the same arguments.
        """
        view = self._at_tissue(tissue)
        if view is not self:
            return view.replay_bloch(waveform, b1_scale=b1_scale, off_resonance_T=off_resonance_T, tissue=tissue, scanner=scanner, orientation=orientation, compartment=compartment, complex_signal=complex_signal, per_walker=per_walker, crusher_seed=crusher_seed, backend=backend)
        if backend not in self.BLOCH_BACKENDS:
            raise ValueError(f"backend={backend!r}: this pack's propagators are {sorted(self.BLOCH_BACKENDS)}")
        _rb = self.BLOCH_BACKENDS[backend]()
        from .bloch import _bloch_timeline
        from .compression import occupancy_at, local_time_at, positions_at
        from ._replay_kernel import gate_weights
        P = self._prepare(waveform, tissue=tissue, scanner=scanner, orientation=orientation, compartment=compartment,
                          relaxation=False, surface=False, pathway=False)
        rf = waveform.rf
        if not rf:
            raise ValueError("the Bloch route replays an RF schedule and this sequence carries none. Without a "
                             "pulse there is nothing this route adds over replay().")
        ch, dt = P["ch"], P["dt"]
        n_t = int(self.n_t)
        _bloch_timeline(rf, n_t, dt)                                    # the whole schedule against the whole walk, once
        # the readouts on the pack grid: the sequence's echoes, else its last sample -- the acquisition's own end,
        # never the walk's, so a shorter acquisition on a longer pack is read at its readout
        echoes = _echo_blocks(waveform, dt)
        single = echoes is None
        if single:
            echoes = [int(round((int(waveform.n_t) - 1) * float(waveform.dt) / dt))]
        if per_walker and not single:
            raise ValueError("per_walker reads each walker at the readout of a single-echo sequence")
        T2v, T1v = P["T2"], P["T1"]
        relax = None
        if T2v is not None or T1v is not None:
            col = next(d for d in ch["compartment"]["columns"] if d["name"] == "comp")
            n_ids = 2 if col["kind"] == "fraction" else self._n_pool_ids(col)   # a fractional occupancy is two pools

            # the Bloch route reads a rate as 1/T: "no decay in this pool" is an infinite time
            def per_pool(v, what):
                out = self._by_pool(v, what)
                if out is None:
                    return [np.inf] * n_ids
                if len(out) < n_ids:
                    raise ValueError(f"the compartment channel addresses pool id {n_ids - 1} and the embedded spec "
                                     f"declares {len(out)} pools: the pack is inconsistent")
                return out
            relax = dict(T2_per_comp=per_pool(T2v, "T2"), T1_per_comp=per_pool(T1v, "T1"))
        if P["rho1"] is not None and float(P["rho1"]) != 0.0:
            # the vector-Bloch route propagates the actual M = (Mx, My, Mz) through the real pulses, so Mxy and
            # Mz are already separate at every step; this route's `surface_relaxivity` (below) attenuates Mxy
            # only, the same transverse-only mechanism the forward engine uses (engine/bloch.py). Extending it to
            # a longitudinal wall term means a parallel attenuation on the Mz deviation, a distinct forward-style
            # mechanism from the C2 scalar/closed-form routes this knob otherwise reaches -- refused rather than
            # silently dropped (dmipy-sim#574).
            raise ValueError("replay_bloch does not apply rho1 (longitudinal surface relaxivity): it propagates "
                             "the magnetisation vector through the real pulses and attenuates Mxy only, like the "
                             "forward engine; use replay() for the C2 scalar route, which applies rho1")
        surface = None
        if P["rho2"] is not None and float(P["rho2"]) != 0.0:
            D_walk = self.diffusivity if P["D"] is None else P["D"]
            if D_walk is None:
                raise ValueError("rho2 needs the walk's diffusivity: the pack did not record it, pass D=")
            if not self.has_surface:
                raise ValueError("surface relaxivity was requested but this pack carries no C2 channel")
            self._check_rho2(float(P["rho2"]) / float(D_walk))
            surface = dict(surface_relaxivity=float(P["rho2"]), D=float(D_walk))
        n_w = P["n_w"]
        off = None
        if off_resonance_T is not None and np.any(np.asarray(off_resonance_T, np.float64) != 0.0):
            off = np.asarray(off_resonance_T, np.float64).reshape(-1)
            if off.size not in (1, n_w):
                raise ValueError(f"off_resonance_T is a scalar or one value per walker ({n_w}); got {off.shape}")
        per_block_all = None
        if getattr(waveform, "crusher", None) is not None:
            from .bloch import _build_crusher
            rate, has = _build_crusher(waveform.crusher, P["dt_wf"], P["G"].shape[1])   # rad/step, waveform grid
            if has and np.any(rate):
                # the macroscopic coordinate is STRATIFIED over the ensemble, (i + 1/2) / n_w dealt to the walkers
                # by the seed: a declared winding of a whole number of turns then cancels the crushed pathways
                # exactly, where n_w uniform draws leave them at 1/sqrt(n_w) (dmipy-sim#393: a 0.2 % offset on a
                # crushed spin echo against the enumeration)
                # the rate as a density (rad/s) carried onto the walk's block ends, then back to radians per block: the
                # phase of each window is preserved however the two grids differ. The propagator reads these as SAMPLES
                # at the block ends (a sampled rate, integrated with the path interpolant), so a window takes the whole
                # walk's samples sliced, never a window's own weights, whose shared block end would be halved
                per_block_all = np.asarray(gate_weights(rate / P["dt_wf"], P["dt_wf"], n_t, dt), np.float64).reshape(-1, n_t)[0] * dt
                u = np.random.default_rng(int(crusher_seed)).permutation((np.arange(n_w) + 0.5) / n_w)
        r_v = None
        if waveform.unbalanced and not waveform.voxel_declared:
            # dmipy-sim#375: an encoding that leaves a net moment at the readout winds across the VOXEL, and a
            # micron-scale substrate cannot. Each walker is put at its own drawn place in the voxel -- the walk
            # translated by r_v, which adds exactly gamma int G . r_v dt to its phase, per measurement, with the
            # RF applied by the propagator as to every other phase. The scalar routes take the same average
            # analytically (ScannerSequence.voxel_factor); this is it per walker. The voxel is the scanner's, so
            # r_v is drawn in the lab and turned into the stored frame with the gradient.
            L = np.asarray(waveform.prescription.voxel_size_m, np.float64)     # _prepare refused without one
            r_v = (np.random.default_rng(int(crusher_seed) + 1).random((n_w, 3)) - 0.5) * L
            if orientation is not None:
                r_v = r_v @ np.asarray(self.pose_rotation(orientation), np.float64)
        # the windows of the walk in turn (RPK.md 4.3): each propagated from the state the previous one left every
        # walker in, its own positions decoded and its own channels read, the readouts it holds recorded
        M = None; recs = []
        for seg, t0, T_s in P["windows"]:
            k0 = int(round(t0 / dt))
            ends = self._block_ends(T_s)
            n_s = ends.shape[0]
            pos = positions_at(seg.arrays, dict(T=T_s), ends)
            local = [e - k0 for e in echoes if (k0 < e <= k0 + n_s - 1) or (k0 == 0 and e == 0)]
            kw = dict(weights=P["ew"] / P["norm"], t0=t0, M_init=M, return_state=True,
                      echo_steps=(local or None), echo_per_walker=bool(per_walker and local))
            if b1_scale is not None:
                kw["b1_scale"] = b1_scale
            if relax is not None:
                kw.update(comp_traj=occupancy_at(seg.arrays, col, t0 + np.arange(n_s) * dt), **relax)
            if surface is not None:
                meta = dict(ch.get("boundary_local_time") or {})
                meta.setdefault("K", _cx_bands_K(seg.arrays, meta))
                B = local_time_at(seg.arrays, meta, ends)
                kw.update(dlog_boundary_unit=np.diff(B, axis=1, prepend=B[:, :1]), **surface)
            extra = None
            if P["B0"] is not None:
                extra = GAMMA * dt * seg._field_along_walk(P, n_s)
            if off is not None:
                uniform = GAMMA * dt * np.broadcast_to(off[:, None], (n_w, n_s))
                extra = uniform if extra is None else extra + uniform
            if per_block_all is not None:
                crush = u[:, None] * np.broadcast_to(per_block_all[k0:k0 + n_s], (n_w, n_s))
                extra = crush if extra is None else extra + crush
            if extra is not None:
                kw["extra_phase_per_step"] = extra
            if r_v is not None:
                pos = pos + r_v[:, None, :]
            M, out = _rb(pos, dt, P["G"], P["dt_wf"], rf, **kw)
            if local:
                recs.append(np.asarray(out))
        E = np.concatenate(recs, axis=1)                                          # (n_meas, n_echo[, n_w])
        if per_walker:
            E = E[:, 0, :].T                                                     # (n_w, n_meas)
            return E if complex_signal else np.abs(E)
        S = E[:, 0] if single else E
        return S if complex_signal else np.abs(S)

    def _field_along_walk(self, P, n_s):
        """The susceptibility off-resonance each walker sees on this window's ``n_s`` block ends, the field's stored
        modes read there (sample 0 the field at the window's start, sample ``j`` its mean over block ``j``)."""
        from .field import susc_path_decode, susc_path_field
        if P["B0"] is None:
            return np.zeros((P["n_w"], int(n_s)))                                     # a field term of zero (field_term)
        gm = P["ch"].get("susceptibility_grid")
        pm = P["ch"].get("susc_field_basis")
        if pm is None:
            raise ValueError("this replay has a field term and the pack carries no field channel (C3)")
        b, _ = susc_path_decode(self.arrays, pm, n_t=int(n_s), n_w=P["n_w"])
        return susc_path_field(b, P["b0_dir"], B0=float(P["B0"]), chi_iso=float(P["chi_iso"]),
                               chi_aniso=P["chi_aniso"], has_aniso=_has_aniso(gm, pm.get("channels")))

    def _prepare(self, waveform, *, tissue, scanner, orientation, compartment, relaxation=True, surface=True,
                 pathway=True, read=None, any_field=False):
        """Everything a replay resolves before it reads the bands: the waveform (its physical and its effective gradient,
        rotated into the substrate frame when a pose is given), the tissue's values (none for ``None``), the susceptibility
        term :func:`field_term` resolves from the pack's declared field, the tissue's chi and the scanner's B0 (``B0``
        None when that term is zero; its direction the machine's B0 axis turned by the pose), the per-walker weights
        with the relaxation and surface terms applied, and the compartment selection.

``read`` (``dict(tol=, K=, modes=)``, as :meth:`replay` takes them) plans the read (:meth:`plan`): ``plan``
        is then the :class:`~dmipy_replay.replay.planning.ReadPlan` and ``K_use`` / ``M_use`` the bands and modes the
        contraction decodes; without it ``plan`` is ``None`` and the contraction reads every stored band and mode.
        ``any_field`` plans the modes for a field term to come (the primitives of an acquisition, whose tissue and
        scanner are applied later).

        ``pathway`` asks for the amplitude of the coherence pathway the sequence's readout IS
        (:func:`~dmipy_replay.acquisition.epg.pathway_weight`): 1 for a refocused echo, and a stimulated echo's
        ``0.5 sin a1 sin a2 sin a3`` for a store-and-recall schedule. It is returned BESIDE the weights and
        not folded into them, because ``ew`` means the relaxation and surface terms and a codec oracle checks
        it means only that; each route that forms a signal applies it once. The vector-Bloch route does not:
        it propagates the magnetisation through the actual pulses, so that amplitude is already in its
        answer, and it passes ``pathway=False``. That is not only to avoid applying the factor twice: a
        readout no SINGLE amplitude describes -- a train at any flip but 180 -- is refused by
        :func:`~dmipy_replay.acquisition.epg.pathway_weight`, and the vector route is precisely the one that
        does not need one, since it carries every pathway itself."""
        from .compression import require_position_method, CumulativeGate
        require_position_method(self.method)
        # two gradients: the PHYSICAL one (``G``, for the vector-Bloch route, which applies the pulses itself)
        # and the EFFECTIVE one (``G_eff``, for the scalar routes, the pulses folded in)
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        if not isinstance(waveform, ScannerSequence):
            raise TypeError(f"a replay takes a ScannerSequence (a bare gradient array says nothing about its pulses); "
                            f"got {type(waveform).__name__}")
        self._check_band(waveform)
        G = np.asarray(waveform.G, np.float64)
        G_eff = np.asarray(waveform.G_eff, np.float64)
        dt_wf = float(waveform.dt)
        dt = self.dt
        ch = (self.meta.get("compression", {}).get("channels", {}) or {})
        n_w = self.n_walkers
        w = np.asarray(self.spin_weights, np.float64)
        from ..spec.tissue import Tissue
        if tissue is not None and not isinstance(tissue, Tissue):
            raise TypeError(f"tissue is a Tissue (pack.nominal, Tissue(...)) or None for the bare diffusion signal; "
                            f"got {type(tissue).__name__}")
        t = tissue if tissue is not None else Tissue()
        T2, T1, rho2, rho1, D = t.T2, t.T1, t.rho2, t.rho1, t.D
        field = field_term(self.susceptibility_field, tissue, scanner, self.field_envelope)   # the one rule of tissue chi x pack field x B0
        self._check_envelope(waveform, field)
        if field.active and not self.has_field:
            raise ValueError(f"pack {self.meta.get('id')!r} declares its susceptibility field present and this replay "
                             "has a field term, but the pack holds no field channel (C3) here: read it with its field "
                             "modes (ColumnarPack.view(modes=...)) or replay the pack whole")
        B0, b0_dir = field.B0, field.axis                            # the MACHINE's field; the pose turns it
        chi_iso, chi_aniso = field.chi_iso, field.chi_aniso
        if orientation is not None:
            R = self.pose_rotation(orientation)
            G, G_eff = G @ R, G_eff @ R                                   # R^T g per sample: stored coordinates
            b0_dir = tuple(np.asarray(R, float).T @ np.asarray(b0_dir, float))
        # the windows of the walk (RPK.md 4.3): every tier is a sum over them, each window reading the acquisition
        # from where it sits on the walk's clock; a single window is the walk itself
        # only the windows the acquisition reaches are read: a window that opens at or beyond the readout
        # contributes nothing, and is not touched -- a short acquisition on a long pack reads its first windows alone
        T_acq = _duration(waveform)
        self._assert_windows_reach(T_acq)
        windows = self._windows()[:self._span(T_acq)[1]]
        # the gate on the occupancy / contact channels, on the waveform's own grid: the sequence's coherence gate, or 1
        # over the sequence -- and 0 after its echo, so a short acquisition relaxes to ITS echo, never to the end of a
        # longer walk (the TE-prefix property, #199)
        chi_wf = np.asarray(waveform.chi_perp if waveform.chi_perp is not None else np.ones(G.shape[1]), np.float64).reshape(-1)
        if chi_wf.shape[0] != G.shape[1]:
            raise ValueError(f"the sequence's coherence gate has {chi_wf.shape[0]} samples and its gradient {G.shape[1]}: "
                             "a gate is read on the waveform's own grid")
        gate_wf = (chi_wf, dt_wf)                                      # the gate on its own grid, which every tier reads
        # the gates' running integrals, what the occupancy's runs between events are weighed by (no grid, RPK.md 9.5.7)
        gate_cum = (CumulativeGate(chi_wf, dt_wf), CumulativeGate(np.ones(chi_wf.shape[0]), dt_wf))
        logw = np.zeros(n_w)
        if (T2 is not None or T1 is not None) and relaxation:
            if not self.has_relaxation:
                raise ValueError("T2 / T1 were given but the pack carries no compartment channel (C1); build it "
                                 "from a walk with tiers='all'")
            from .compression import relaxation_logweight_runs, is_current_c1, decode_occupancy
            if not is_current_c1(ch["compartment"]):
                decode_occupancy(self.arrays, ch["compartment"])              # raises with the re-encode message
            col = next(d for d in ch["compartment"]["columns"] if d["name"] == "comp")
            n_ids = self._n_pool_ids(col)
            T2v = self._by_pool(T2, "T2"); T1v = self._by_pool(T1, "T1")
            if T2v is None:
                T2v = [np.inf] * n_ids                                # no T2 decay, T1 only
            if T1v is None:
                T1v = [np.inf] * n_ids                                # no T1 term
            if len(T2v) < n_ids or len(T1v) < n_ids:
                raise ValueError(f"the compartment channel addresses pool id {n_ids - 1} and the embedded spec "
                                 f"declares {min(len(T2v), len(T1v))} pools: the pack is inconsistent")
            for seg, t0, T_s in windows:                                   # on the runs between events, never a track
                logw = logw + relaxation_logweight_runs(seg.arrays, col, T2v, T1v, gate_cum[0], gate_cum[1], t0=t0, t1=t0 + T_s)
        if rho2 is not None and float(rho2) != 0.0 and surface:
            D_walk = self.diffusivity if D is None else D
            if D_walk is None:
                raise ValueError("rho2 needs the walk's diffusivity: the pack did not record it, pass D=")
            self._check_rho2(float(rho2) / float(D_walk))
            for j in range(len(windows)):
                logw = logw + _gated_contact(windows, gate_wf, j, float(rho2) / float(D_walk),
                                             ch.get("boundary_local_time"))                # raises without C2
        if rho1 is not None and float(rho1) != 0.0 and surface:
            # the SAME bridge contraction (surface_logweight), gated by the complement chi_parallel = active - chi:
            # the contact channel is one series, and the C2 gate rho2 * chi + rho1 * chi_parallel is linear in it, so
            # the longitudinal term is a second call summed in rather than a second channel (dmipy-sim#574).
            D_walk = self.diffusivity if D is None else D
            if D_walk is None:
                raise ValueError("rho1 needs the walk's diffusivity: the pack did not record it, pass D=")
            self._check_rho2(float(rho1) / float(D_walk))
            for j in range(len(windows)):
                logw = logw + _gated_contact(windows, gate_wf, j, float(rho1) / float(D_walk),
                                             ch.get("boundary_local_time"), parallel=True)    # raises without C2
        ew = w * np.exp(logw)
        norm = w.sum()
        ew, norm = self._select(compartment, ew, norm, w, ch, n_w)
        plan = K_use = M_use = None
        if read is not None:
            gate = waveform.effective_gate if waveform.gate is None else waveform.gate
            plan = self._read_plan(G_eff, dt_wf, gate, windows, field, any_field=any_field, **read)
            K_use, M_use = plan.K, plan.modes
        from ..acquisition.epg import pathway_weight
        # the voxel's factor for an unbalanced encoding (dmipy-sim#375), or the refusal without a voxel size;
        # read on the LAB waveform, since the voxel is the scanner's and the dot product k . r_v is the same in
        # every frame. The vector-Bloch route does not apply this scale: it places each walker in the voxel.
        voxel = np.asarray(waveform.voxel_factor(), np.float64)
        return dict(G=G, dt=dt, dt_wf=dt_wf, ch=ch, n_w=n_w, w=w, ew=ew, norm=norm, B0=B0,
                    pathway=(pathway_weight(waveform) if pathway else 1.0), voxel=voxel,
                    b0_dir=b0_dir, chi_iso=chi_iso, chi_aniso=chi_aniso, T2=T2, T1=T1, rho2=rho2, rho1=rho1, D=D,
                    G_eff_wf=G_eff, windows=windows, gate_wf=gate_wf, gate_cum=gate_cum, plan=plan, K_use=K_use, M_use=M_use)

    def _n_pool_ids(self, col):
        """How many pool ids the ``comp`` column addresses, read over every window."""
        key = "comp_static" if col["kind"] == "static" else "comp_events_label"
        return int(max(int(np.max(w.arrays[key])) for w, _, _ in self._windows())) + 1

    def plan(self, waveform, *, tissue=None, scanner=None, orientation=None, tol=PLAN_TOL, K=None, modes=None):
        """What a replay of ``waveform`` reads of this pack, before it reads: a :class:`~dmipy_replay.replay.planning.ReadPlan`
        with the position bands ``K`` and field modes ``modes`` the acquisition needs, the truncation error each costs and
        the source of the variances it was weighed with.

        The plan is the fewest bands, and the fewest modes, whose dropped tail costs at most ``tol`` times the pack's
        Monte-Carlo floor (``fidelity.floor_max``): the rule of the columnar layout's planner
        (:meth:`~dmipy_replay.replay.columnar.ColumnarPack.plan`), in :mod:`~dmipy_replay.replay.planning`. The band variances are
        the header's measured record where it holds one, else the free bridge's at the pack's diffusivity -- no band is
        decoded to plan; the mode variances are the header's record, else measured on a pilot of the pack's walkers and
        held in the pack's decoded cache. The modes are weighed at the pack's certified field scale (``field_envelope``: the largest
        ``B0`` and ``chi``) when it states one, else at the term's own, so the replay, its per-walker depth and the primitives
        read the same modes. A pack with neither is refused by name. The bands are planned for the
        acquisition as the pose turns it (``orientation=``, one pose); a field is planned only when ``tissue`` and
        ``scanner`` resolve to a field term. ``K=`` / ``modes=`` state the counts instead (the plan then carries their
        error); ``tol=None`` is the full read, every stored band and mode, an error of zero against the stored pack.
        A replay called with the same arguments reads exactly this plan."""
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        view = self._at_tissue(tissue)
        if view is not self:
            return view.plan(waveform, tissue=tissue, scanner=scanner, orientation=orientation, tol=tol, K=K, modes=modes)
        if self._as_distribution(orientation) is not None:
            return self._full_plan("a pose distribution is the pose expansion's, which reads every band and mode")
        G_eff = np.asarray(waveform.G_eff, np.float64)
        if orientation is not None:
            G_eff = G_eff @ self.pose_rotation(orientation)
        T_acq = _duration(waveform)
        self._assert_windows_reach(T_acq)
        windows = self._windows()[:self._span(T_acq)[1]]
        field = field_term(self.susceptibility_field, tissue, scanner, self.field_envelope)
        self._check_envelope(waveform, field)
        gate = waveform.effective_gate if waveform.gate is None else waveform.gate
        return self._read_plan(G_eff, float(waveform.dt), gate, windows, field, tol=tol, K=K, modes=modes)

    def _full_plan(self, why):
        """The plan of a read of every stored band and mode, an error of zero against the stored pack; ``why`` is its source."""
        K_st = self.K
        pm = ((self.meta.get("compression") or {}).get("channels") or {}).get("susc_field_basis")
        M_st = self._mode_count(pm) if pm is not None else 0
        return self._make_plan(self._windows(), K_st, 0.0, why, "full", M_st, M_st, 0.0, why if M_st else "no field channel", "full",
                               PLAN_TOL, float("nan"))

    def _mode_count(self, pm):
        from .field import path_mode_count
        return path_mode_count(self.arrays if self.n_windows == 1 else self._windows()[0][0].arrays, pm)

    def _make_plan(self, windows, K, eK, src_K, how_K, M, M_st, eM, src_M, how_M, tol, floor,
                   floor_source="none: a full read has no truncation to bound"):
        from .planning import planned_bytes
        read = stored = 0
        for seg, _, _ in windows:
            table = {k: (np.shape(v), np.dtype(v.dtype).itemsize) for k, v in seg.arrays.items()
                     if k.startswith(("pos_", "susc_path"))}
            r, st = planned_bytes(table, seg.meta, K, M)
            read += r; stored += st
        return ReadPlan(K=int(K), K_stored=int(self.K), band_error=float(eK), band_variance=src_K, bands_chosen=how_K,
                        modes=int(M), modes_stored=int(M_st), mode_error=float(eM), mode_variance=src_M, modes_chosen=how_M,
                        tol=float(tol), floor=float(floor), bytes_read=int(read), bytes_stored=int(stored),
                        floor_source=floor_source)

    def _read_plan(self, G_eff, dt_wf, gate, windows, field, *, tol, K, modes, any_field=False):
        """The :class:`~dmipy_replay.replay.planning.ReadPlan` for the effective gradient ``G_eff`` (in the pack's frame) and
        ``gate`` on their grid ``dt_wf``, over ``windows`` (``[(pack, t0, T), ...]``) and the resolved ``field`` term
        (:meth:`plan`). ``any_field`` plans the modes for a field term that is not resolved yet (the primitives of an
        acquisition, whose tissue and scanner come later).

        The modes are planned at the pack's certified field scale (``field_envelope``: the largest ``B0`` and ``chi``) when it
        states one, so the count depends on the acquisition's gate and the pack alone: a replay, its per-walker depth and the
        primitives read the same modes, and the error stated bounds every field term the envelope allows. A pack that states
        no envelope is planned at the term's own scale; for primitives, which have none, every mode is read."""
        from .planning import band_tail, mode_tail, band_variance, fewest, floor_of
        K_st = self.K
        pm = ((self.meta.get("compression") or {}).get("channels") or {}).get("susc_field_basis")
        M_st = self._mode_count(pm) if pm is not None else 0
        if tol is None and K is None and modes is None:
            return self._full_plan("a full read: every stored band and mode (tol=None)")
        floor, floor_source = floor_of(self.meta, self.n_walkers)
        tol_eff = PLAN_TOL if tol is None else float(tol)
        limit = tol_eff * floor
        wins = [(T_s, t0) for _, t0, T_s in windows]
        # ---- the bands
        if K is not None and not 0 <= int(K) <= K_st:
            raise ValueError(f"K = {K} is outside 0 .. {K_st}, the bands this pack stores")
        if K is not None and int(K) == K_st or (K is None and tol is None):
            Kp, eK, src_K, how_K = K_st, 0.0, "all stored bands: a full read of them", ("explicit" if K is not None else "full")
        else:
            var, src_K = band_variance(self.meta, K_st, self.T_window, self.diffusivity)
            from .compression import band_phase_variances
            beyond = 0.0                               # the bands never stored, at the free bridge's variance (closed form)
            for T_w, t0_w in wins:
                pb, total = band_phase_variances(np.asarray(G_eff, np.float64), float(dt_wf), float(T_w), K_st, float(self.diffusivity), t0=t0_w)
                beyond = beyond + 0.5 * np.clip(total - pb.sum(axis=1), 0.0, None)
            tail = (band_tail(G_eff, dt_wf, wins, var) + np.asarray(beyond).reshape(-1, 1)).max(axis=0)
            if K is None:
                Kp, eK = fewest(tail, range(K_st + 1), limit); how_K = "planned"
            else:
                Kp, eK, how_K = int(K), float(tail[int(K)]), "explicit"
        # ---- the modes
        env = self.field_envelope
        if not (field.active or any_field):
            scale = None
        elif env is not None:
            scale = (env["B0_max"], env["chi_max"])
        elif field.active:
            scale = (field.B0, abs(field.chi_iso) + abs(field.chi_aniso))
        else:
            scale = False
        if M_st == 0:
            if field.active:
                raise ValueError(f"pack {self.meta.get('id')!r} declares its susceptibility field present and this replay has a "
                                 "field term, but the pack holds no field channel (C3) here")
            Mp, eM, src_M, how_M = 0, 0.0, "no field channel", "planned"
        elif scale is None:
            Mp, eM, src_M, how_M = 0, 0.0, "no field term: no mode is read", "planned"
        elif scale is False:
            Mp, eM, src_M, how_M = M_st, 0.0, "no certified field scale stated: every stored mode", "full"
        elif (modes is not None and int(modes) == M_st) or (modes is None and tol is None):
            Mp, eM, src_M, how_M = M_st, 0.0, "all stored modes: a full read of them", ("explicit" if modes is not None else "full")
        else:
            if modes is not None and not 0 <= int(modes) <= M_st:
                raise ValueError(f"modes = {modes} is outside 0 .. {M_st}, the field modes this pack stores")
            var, src_M = self._mode_variance(windows[0][0], pm)
            tail = mode_tail(gate, dt_wf, wins, M_st, var, scale[0], scale[1])
            if modes is None:
                Mp, eM = fewest(tail, range(M_st + 1), limit); how_M = "planned"
            else:
                Mp, eM, how_M = int(modes), float(tail[int(modes)]), "explicit"
        return self._make_plan(windows, Kp, eK, src_K, how_K, Mp, M_st, eM, src_M, how_M, tol_eff, floor, floor_source)

    def _mode_variance(self, seg, pm):
        """``((K_f,) variance summed over the channels, source)`` of the field modes a plan weighs: the header's record, else
        measured on a pilot of ``seg``'s walkers (:func:`~dmipy_replay.replay.planning.pilot_size`) and held in the pack's decoded
        cache beside the modes it was measured on. A pack that stores no mode is refused by name."""
        from .planning import stored_mode_variance, pilot_size
        from .compression import walker_rows
        stored = stored_mode_variance(self.meta)
        if stored is not None:
            return stored
        n_p, why = pilot_size(self.meta, self.n_walkers)

        def measure():
            return _measured_mode_variance(walker_rows(seg.arrays, 0, n_p), pm)
        cache = self.arrays.decoded
        tensors = {k: v for k, v in seg.arrays.items() if k.startswith("susc_path")}
        return cache.get("mode_variance", (n_p, json.dumps(pm, sort_keys=True, default=str)), tensors, measure), why

    def _select(self, compartment, ew, norm, w, ch, n_w):
        """Restrict the ensemble mean to ``compartment`` (a pool id or a walker mask)."""
        from .compression import occupancy_events
        if compartment is None:
            return ew, norm
        sel = np.asarray(compartment)
        if sel.dtype != bool:
            if "compartment" not in ch:
                raise ValueError("compartment= by id needs the pack's compartment channel")
            col = next(d for d in ch["compartment"]["columns"] if d["name"] == "comp")
            ids = occupancy_events(self.arrays, col)[0]                   # the pool at the walk's start
            sel = ids.astype(int) == int(sel)
        if sel.shape[0] != n_w:
            raise ValueError(f"compartment mask has {sel.shape[0]} entries for {n_w} walkers")
        if not sel.any():
            raise ValueError("compartment selection matched no walkers")
        return np.where(sel, ew, 0.0), w[sel].sum()

    @property
    def temporal_bandwidth_hz(self):
        """The highest frequency the position bands resolve, ``K / (2 T)`` with ``T`` a window's duration (#199; a
        one-window pack's walk): the pack's temporal band stated as a frequency, which is what a scanner envelope is
        checked against and what a prefix and a rewindowed pack keep."""
        return float(self.K) / (2.0 * float(self.windows["T"]))

    @property
    def truncation_budget(self):
        """What a waveform may cost this pack in band truncation (:func:`truncation_budget`): ``dict(cap, cap_source,
        other_terms, budget)``. The cap is the pack's ``fidelity.target_floor``; a pack that states none is held to its
        Monte-Carlo floor, ``fidelity.floor_max`` or ``1 / sqrt(n_walkers)`` where the certificate states none."""
        fid = self.meta.get("fidelity") or {}
        if fid.get("floor_max"):
            floor, src = float(fid["floor_max"]), "fidelity.floor_max"
        else:
            floor, src = 1.0 / np.sqrt(float(self.n_walkers)), "1/sqrt(n_walkers)"
        return truncation_budget(self.meta, floor=floor, floor_source=src)

    def waveform_band(self, waveform, *, tol=None):
        """The temporal band ``waveform`` needs on this pack, as ``(hz, bands, err)``: the fewest sine bands whose
        dropped phase costs the signal no more than ``tol`` on every measurement, stated as the frequency
        ``bands / (2 T)`` beside the pack's :attr:`temporal_bandwidth_hz` (#277).

        The gradient phase of a walker is the waveform's projection on the bridge basis contracted with the
        walker's bands (:func:`~dmipy_replay.replay.compression.band_projection`); the bands beyond the pack's
        ``K`` are not stored, so the phase they would carry is lost silently. The cost of dropping bands ``j+1...`` is
        bounded by half the phase variance they carry: a dropped phase ``phi`` with a symmetric law moves the signal
        by ``|S_j| (1 - <e^{i phi}>) <= phi variance / 2`` for ``|S_j| <= 1``, so ``err[j] = max over measurements of
        gamma^2 sum_{k > j} sum_d W_{m,k,d}^2 var_k / 2`` is the estimate, an upper bound for any substrate (the
        retained signal is at most 1) and no sharper. ``var_k`` is the free bridge's band variance at the pack's
        diffusivity, ``2 D T^2 / (pi k)^2`` (:func:`~dmipy_replay.replay.compression.band_variances`), and the sum over
        ``k > j`` is the whole bridge's closed form less the retained bands
        (:func:`~dmipy_replay.replay.compression.band_phase_variances`), so it runs to infinity, not to a grid. Free
        diffusion's bands bound every restricted walk's from above: measured on the 20,000-walker cylinder masters
        (2, 5 and 10 um diameter, 100 ms) the variance of every band above k = 100 is 0.43 to 0.97 of the free value
        (and 1.00 on free water), and the tail's correlation with the retained phase is at most 0.04.

        ``tol`` defaults to the pack's truncation budget (:func:`truncation_budget`): its stated numerical cap
        (``fidelity.target_floor``) less its other certified terms, so a waveform within the band costs the pack no more
        than its cap in total; a pack that states no cap is held to its Monte-Carlo floor (``fidelity.floor_max``,
        ``1 / sqrt(n_walkers)`` where the certificate states none). The band is judged over the stretch the waveform
        spans (:meth:`_span`): the windows its duration reaches on a pack stored in windows (RPK.md 4.3), the window on
        a one-window pack, so a short acquisition on a long walk costs the projection of its own duration, not the
        walk's. The bands are sought up to :data:`CONTINUOUS_BAND_CAP` times the pack's; a waveform beyond that returns
        ``inf``.
        """
        from .compression import band_phase_variances
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        T, windows = self._span(_duration(waveform))
        D = self.diffusivity
        if D is None:
            raise ValueError("the pack records no diffusivity, so the band a waveform needs cannot be judged")
        if tol is None:
            tol = self.truncation_budget["budget"]
        G = np.asarray(waveform.G_eff, np.float64)
        K_have = self.K * windows
        K_try = max(64, 2 * K_have)
        K_cap = CONTINUOUS_BAND_CAP * max(K_have, 1)
        while True:
            per_band, total = band_phase_variances(G, float(waveform.dt), T, K_try, D)
            kept = np.concatenate([np.zeros((per_band.shape[0], 1)), np.cumsum(per_band, axis=1)], axis=1)
            err = 0.5 * np.clip(total[:, None] - kept, 0.0, None).max(axis=0)          # err[j]: keeping j bands
            ok = np.flatnonzero(err <= float(tol))
            if len(ok) or K_try >= K_cap:
                break
            K_try = min(2 * K_try, K_cap)
        at = float(err[min(K_have, K_try)])
        if len(ok) == 0:
            return np.inf, np.inf, at
        bands = int(ok[0])
        return bands / (2.0 * T), bands, at

    def _span(self, T_acq):
        """``(T, windows)``: the stretch of this walk an acquisition of duration ``T_acq`` (s, from the walk's start)
        spans and the windows it makes -- on a pack stored in windows (RPK.md 4.3) the whole windows that open
        before ``T_acq``, whether loaded or not; on a one-window pack the window, ``(T_window, 1)``. An acquisition
        longer than the walk spans the walk."""
        T_w = self.T_window
        if self.n_windows == 1:
            return T_w, 1
        windows = max(1, sum(1 for i in range(int(self.n_windows)) if i * T_w < float(T_acq) * (1.0 - 1e-12)))   # a read has at least window 0
        return windows * T_w, windows

    def _check_band(self, waveform):
        """Refuse a waveform whose gradient needs more temporal band than this pack's envelope admits over the stretch
        it spans (#277; :meth:`waveform_band`): the fewer of the bands it stores and ``ceil(2 band_hz T)`` of its
        :attr:`envelope`. Beyond them the replay would return a smooth, plausible, wrong signal. A pack that records no
        diffusivity (a synthetic master) cannot be judged and is not refused."""
        if self.diffusivity is None:
            return
        hz, bands, err = self.waveform_band(waveform)
        T, windows = self._span(_duration(waveform))
        env = self.envelope
        stored = self.K * windows
        admitted = min(stored, env.mode_count(T))
        if bands <= admitted:
            return
        b = self.truncation_budget
        need = (f"more than {CONTINUOUS_BAND_CAP * self.K * windows / (2.0 * T):.4g} Hz over the {T * 1e3:.4g} ms it spans"
                if not np.isfinite(hz) else f"{hz:.4g} Hz ({bands} bands over the {T * 1e3:.4g} ms it spans)")
        if admitted < stored:
            raise ValueError(f"the waveform's gradient reaches beyond the band of the envelope {env.name!r} this pack is "
                             f"certified for: it needs {need} and the envelope admits band_hz = {env.band_hz:.4g} Hz "
                             f"({admitted} bands){_view_of(self.meta)}. Replay a waveform within it")
        raise ValueError(f"the waveform's gradient reaches beyond this pack's temporal band: it needs {need} and the "
                         f"pack resolves {self.temporal_bandwidth_hz:.4g} Hz (K = {self.K} bands over {float(self.windows['T']) * 1e3:.4g} ms), "
                         f"which would drop up to {err:.2e} of the signal into phase the pack does not carry, against the "
                         f"{b['budget']:.2e} it may cost (its cap {b['cap']:.2e} from {b['cap_source']} less {b['other_terms']:.2e} "
                         f"of other certified terms). Walk a pack whose envelope holds this band, or replay a waveform "
                         f"within this one")

    @property
    def envelope(self):
        """The :class:`~dmipy_replay.replay.envelope.Envelope` this pack is certified for
        (:meth:`~dmipy_replay.replay.envelope.Envelope.from_header` of its header): what every replay is held to."""
        from .envelope import Envelope
        return Envelope.from_header(self.meta)

    @property
    def digest(self):
        """A sha256 of the pack's arrays and metadata: its identity for a cache. Computed once per instance."""
        d = getattr(self, "_digest", None)
        if d is None:
            import hashlib, json
            h = hashlib.sha256()
            for k in sorted(self.arrays):
                h.update(k.encode()); h.update(np.ascontiguousarray(self.arrays[k]).tobytes())
            h.update(json.dumps(self.meta, sort_keys=True, default=str).encode())
            d = self._digest = h.hexdigest()
        return d

    @property
    def substrate_frame(self):
        """The substrate's intrinsic frame (RPK.md 4.2): a ``(3, 3)`` right-handed basis in stored coordinates, column
        3 the primary structural axis, ``d_stored = F d_canonical``. From ``walk_params.substrate_frame`` when the
        producer declared it, else from the embedded spec's ``frame`` (axis, optional in-plane), else the identity --
        the pack's positions are then taken to be in the canonical frame already."""
        from .field import frame_from_axis
        wp = self.meta.get("walk_params", {}) or {}
        F = wp.get("substrate_frame")
        if F is None:
            F = self.meta.get("substrate_frame")
        if F is not None:
            F = np.asarray(F, np.float64).reshape(3, 3)
        else:
            spec = self.substrate
            fr = getattr(spec, "frame", None) if spec is not None else None
            if fr is None:
                return np.eye(3)
            F = np.asarray(frame_from_axis(fr.axis, in_plane=getattr(fr, "in_plane", None)), np.float64).reshape(3, 3)
        if not np.allclose(F @ F.T, np.eye(3), atol=1e-6) or np.linalg.det(F) < 0:
            raise ValueError("the pack's substrate_frame is not a proper rotation")
        return F

    @property
    def frame_axis(self):
        """The substrate's own axis in stored coordinates: column 3 of :attr:`substrate_frame`."""
        return self.substrate_frame[:, 2].copy()

    def pose_rotation(self, orientation):
        """The rotation taking STORED coordinates to the lab for one pose: from a 3x3 pose of the canonical
        substrate frame (``R``, so stored -> lab is ``R F^T`` with ``F`` the pack's :attr:`substrate_frame`), or
        from the lab direction the substrate's axis points along (the azimuth left as the frame's). A lab
        waveform reads in stored coordinates as ``G @ pose_rotation(orientation)``: what a consumer compiling
        its own scheme (a fit's compartment model) rotates by."""
        from .so3 import rotation_of
        o = np.asarray(orientation, float)
        if o.shape == (3, 3):
            if not np.allclose(o @ o.T, np.eye(3), atol=1e-6) or np.linalg.det(o) < 0:
                raise ValueError("orientation must be a proper rotation matrix (R R^T = I, det +1)")
            return o @ self.substrate_frame.T
        if o.shape == (3,):
            return rotation_of(o, self.frame_axis)
        raise ValueError("orientation is a (3, 3) rotation, a (3,) axis direction, or a distribution of poses "
                         "(dmipy_replay.replay.so3.Distribution, or an FOD read as an axis density)")

    @cached_property
    def position_coeffs(self):
        """``(n_walkers, K+2, n_axes)``: two exact endpoints, then ``K`` sine bands per axis.

        The leading two entries are NOT bands -- they are ``r(0)`` and ``r(T)-r(0)`` -- so this
        array must never be handed to a band-basis contraction.  Raises on a pre-layout pack
        rather than guessing: see compression.read_position_coeffs.

        Decoded from the per-axis tensors once for the whole pack (a pack's arrays do not change after loading);
        a reader bounded in memory decodes walker ranges instead (:meth:`walkers`). ``K``, ``n_walkers`` and
        ``n_coeffs`` read the stored shapes, not this block.
        """
        from .compression import read_position_coeffs, require_position_method
        require_position_method(self.method)
        return read_position_coeffs(self.arrays, dtype=np.float32, cached=False)

    @property
    def r0(self):
        """``(n_walkers, 3)``: where every walker **started**, in metres, read from the stored coefficients.

        The position codec keeps ``r(0)`` as an exact entry rather than a reconstructed band, so this is the
        stored value to float32 (of order 0.1 nm on a millimetre coordinate), identical at every ``K``, and it
        costs the first column of the endpoint tensors -- no band is decoded. It is what partitions a walk into
        voxels (RPH.md: ``voxel = floor((r0 - origin) / voxel_size)``), which is why it must not depend on the band.
        """
        from .compression import position_starts, require_position_method
        require_position_method(self.method)
        return np.ascontiguousarray(position_starts(self.arrays))

    @property
    def spin_weights(self):
        w = self.arrays.get("spin_weights")
        return np.ones(self.n_walkers) if w is None else w

    @property
    def boundary_local_time(self):
        """The C2 channel's exact cumulative total per walker, or None if the pack has no C2."""
        return self.arrays.get("blt_endpoint")

    def contact(self):
        """The C2 channel read on the walk's block ends, ``(n_walkers, n_t)``: the wall contact (at ``rho2 / D = 1``)
        each block accrued, the first entry 0 (the walk's start ends no block). Each window's cumulative local time is
        its stored function -- the exact endpoints and the bands -- evaluated at the window's block ends
        (:func:`~dmipy_replay.replay.compression.local_time_at`), and the windows are joined on their shared instant. Its
        ``cumsum`` along the block ends is the boundary local time ``L(t)`` a surface relaxivity weights, ``S(t) =
        <exp((rho2/D) L(t))>``. ``None`` when the pack carries no C2 channel."""
        from .compression import local_time_at, has_c2
        if not has_c2(self.arrays):
            return None
        cm = dict((self.meta["compression"].get("channels") or {})["boundary_local_time"])
        parts = []
        for j, (w, _, T_w) in enumerate(self._windows()):
            m = dict(cm, T=T_w); m.setdefault("K", _cx_bands_K(w.arrays, m))
            B = local_time_at(w.arrays, m, self._block_ends(T_w))
            ell = np.diff(B, axis=1, prepend=B[:, :1])
            parts.append(ell if j == 0 else ell[:, 1:])
        return np.concatenate(parts, axis=1).astype(np.float32)

    @property
    def K(self):
        """Retained sine bands -- NOT the stored coefficient count, which is ``K + 2``.

        Reading the width off the array instead would hand ``K + 2`` to compile_scheme, whose
        output would then be the right shape to multiply and the wrong thing to multiply by.
        Declared and stored are cross-checked so they cannot drift apart.
        """
        k = int(self.meta.get("compression", {}).get("K", -1))
        stored = self.n_coeffs
        if k < 0:
            raise ValueError("pack declares no compression.K; refusing to infer it from the "
                             "array width, which counts the two endpoints as well")
        if stored != k + 2:
            raise ValueError(f"pack declares K={k} but stores {stored} coefficients per axis; "
                             f"expected {k + 2} (two endpoints + K bands)")
        return k

    @property
    def n_coeffs(self):
        """Stored coefficients per axis, ``K + 2``, from the stored shapes."""
        from .compression import position_shape
        return position_shape(self.arrays)[1]

    @property
    def n_walkers(self):
        """The walkers the pack holds, from the stored shapes."""
        from .compression import position_shape
        return position_shape(self.arrays)[0]

    def walkers(self, start, stop):
        """Walkers ``start:stop`` as a pack of their own (:func:`~dmipy_replay.replay.compression.walker_rows`): the same
        metadata, ``walk_params.n_walkers`` theirs, every walker-leading tensor a view of this pack's rows and the
        tables whole. A per-walker quantity read from it is that of the same walkers read from the whole pack."""
        from .compression import position_shape, walker_rows
        arrays = walker_rows(self.arrays, start, stop)
        wp = dict(self.meta.get("walk_params") or {}, n_walkers=position_shape(arrays)[0])
        return type(self)(arrays, dict(self.meta, walk_params=wp), source=self.source)

    @property
    def block(self):
        """The walk's block record ``walk_params.block``: ``h`` (s, the block a walk accumulated its bands over), the
        table's order ``m`` and how they were set."""
        return (self.meta.get("walk_params") or {})["block"]

    @property
    def dt(self):
        """The block length ``h`` (s): the grid the occupancy, a gate's per-block reading and a decoded series are on."""
        return float(self.block["h"])

    @property
    def n_t(self):
        """The block ends of the whole walk, ``T_max / h + 1``."""
        return int(round(float(self.meta["walk_params"]["T_max"]) / self.dt)) + 1

    @property
    def T_window(self):
        """One storage window's duration (s), ``walk_params.windows.T``: the interval the bands are on."""
        return float(self.windows["T"])

    @property
    def id(self):
        return self.meta.get("id")

    @property
    def method(self):
        return self.meta.get("compression", {}).get("method")
    license = property(lambda self: self.meta.get("license"))
    citation = property(lambda self: self.meta.get("citation"))
    fidelity = property(lambda self: self.meta.get("fidelity"))
    replay_envelope = property(lambda self: self.meta.get("replay_envelope"))
    provenance = property(lambda self: self.meta.get("provenance"))


def _measured_mode_variance(arrays, pm):
    """``(K_f,)``: the mean square of each field mode over the walkers of ``arrays`` (a pack's tensors, or a pilot's rows of
    them), summed over the channels -- the variance a plan weighs the modes with when the header states none."""
    from .field import susc_path_coeffs
    C = np.asarray(susc_path_coeffs(arrays, pm)[0], np.float64)                                   # (n, n_ch, K_f)
    return (C ** 2).mean(axis=0).sum(axis=0)


def _as_distribution(orientation):
    """``orientation=`` as a distribution of poses, or None when it names a single pose.

    An :class:`~dmipy_replay.replay.fod.FOD` is read as an axis density with no statement about the substrate's own
    azimuth, which is what an ODF says.
    """
    if orientation is None or isinstance(orientation, (list, tuple, np.ndarray)):
        return None
    raise NotImplementedError(f"orientation={type(orientation).__name__}: a distribution of poses is composed through the "
                              "pose expansion, which this release does not carry yet; give one pose (a 3x3 rotation or "
                              "the lab direction of the substrate's axis)")


def _duration(waveform):
    """An acquisition's duration in seconds from the walk's start: its last gradient sample's time."""
    waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
    return (int(np.asarray(waveform.G_eff).shape[1]) - 1) * float(waveform.dt)


def _echo_blocks(waveform, dt_pack):
    """The block-end indices of a multi-echo readout on the walk's clock, or ``None`` when the sequence is read at its last
    sample (the Bloch replay then returns the final magnetisation)."""
    readout = tuple(int(i) for i in waveform.readout)
    if readout == (waveform.n_t - 1,):
        return None
    return [int(round(i * float(waveform.dt) / float(dt_pack))) for i in readout]

_AXIS_MAPS = {}


def compile_scheme(G, dt, K, gyromagnetic_ratio=GAMMA, *, T=None, method=None):
    """Compile an acquisition into its temporal-basis projection ``W``: the exact integral of the waveform
    against the stored path, in mode space.

    ``G`` is the gradient waveform ``(n_meas, n_wf, 3)`` [T/m] on ITS OWN grid ``dt`` [s]; ``T`` the pack's window
    (``pack.T_window``; default the waveform's own extent, ``(G.shape[1] - 1) dt``); ``K`` the pack's retained-mode count; ``method`` the pack's
    position codec, accepted only to let a caller assert it. Shape is ``(n_c (K+2), n_meas)`` for a waveform of
    ``n_c`` components (three for a pack's positions; fewer when a consumer reads only some stored axes and
    supplies the waveform's matching components, in the same order): per component the two gradient moments
    ``M0`` and ``M1``, which a motion-compensated waveform makes vanish, then the sine bands.
    Nothing is resampled: the waveform enters through :func:`~dmipy_replay.replay.compression.band_projection`, exact
    for the stored bands, so an edge anywhere carries exactly its b. Reusable across every pack on one grid (fitting) and every fit
    iteration; in design it is recomputed per candidate."""
    from .compression import require_position_method, band_projection
    require_position_method(method or "bridge_dst")
    G = np.asarray(G, np.float64)
    T = (G.shape[1] - 1) * float(dt) if T is None else float(T)
    W = band_projection(G, dt, T, int(K))
    return (gyromagnetic_ratio * W).reshape(W.shape[0], (int(K) + 2) * W.shape[2]).T



NUMERICAL_TERMS = ("err_quadrature", "err_quantisation", "err_quadrature_surface", "err_rounding", "err_quadrature_field",
                   "err_field_read", "err_quantisation_field", "err_surface", "err_reprojection")


def truncation_budget(meta, *, floor, floor_source):
    """The truncation error a waveform may cost this pack, as ``dict(cap, cap_source, other_terms, budget)``.

    A pack that states its numerical cap (``fidelity.target_floor``, the error budget it was built to) is held to it:
    ``budget = cap - other_terms``, with ``other_terms`` the sum of its certified non-truncation rows
    (:data:`NUMERICAL_TERMS`) or ``err_max - err_truncation`` where that is larger, so a waveform the check admits
    costs the pack at most its cap in total. A pack that states none is held to ``floor`` (named by ``floor_source``),
    the Monte-Carlo noise it is read through; its ``other_terms`` are then 0 and ``budget = floor``."""
    fid = meta.get("fidelity") or {}
    if fid.get("target_floor") is None:
        return dict(cap=float(floor), cap_source=str(floor_source), other_terms=0.0, budget=float(floor))
    cap = float(fid["target_floor"])
    named = sum(float(fid[k]) for k in NUMERICAL_TERMS if fid.get(k) is not None)
    if fid.get("err_max") is not None and fid.get("err_truncation") is None:
        raise ValueError("the certificate states fidelity.err_max but not fidelity.err_truncation: the truncation budget "
                         "cannot split the error into the truncation and the other terms; re-certify the pack")
    left_over = (float(fid["err_max"]) - float(fid["err_truncation"])) if fid.get("err_max") is not None else 0.0
    other = max(named, left_over)
    return dict(cap=cap, cap_source="fidelity.target_floor", other_terms=other, budget=max(cap - other, 0.0))

CONTINUOUS_BAND_CAP = 64

TORCH_GROUPS = 128

SHELL_RTOL = 1e-3

SHELL_SERIES_TOL = 1e-14

SHELL_SERIES_MAX_PHASE = 10.0


def _container_of(ranges):
    """The builder's band-container knob (``((upto, bits), ...)``, or ``None`` for the float container) from the
    ranges a pack's metadata records (``[{"bands": [k0, k1], "bits": b}, ...]``)."""
    if not ranges:
        return None
    return tuple((None if i == len(ranges) - 1 else int(r["bands"][1]), int(r["bits"])) for i, r in enumerate(ranges))


def _compile_continuous(X, dt_wf, K, T, t0, gyromagnetic_ratio=GAMMA):
    """``(K + 2, n)``: ``gamma`` times the projection (:func:`~dmipy_replay.replay.compression.band_projection`) of each of
    the ``n`` courses ``X`` ``(n, n_wf, 1)`` on its own grid ``dt_wf`` onto the bands of the window of ``T`` seconds
    starting at ``t0`` -- what the pose routes contract one axis of the coefficients against."""
    from .compression import band_projection
    return gyromagnetic_ratio * band_projection(np.asarray(X, np.float64), float(dt_wf), float(T), int(K), t0=t0)[:, :, 0].T


def _band_phase(P, W=None):
    """``(n_w, n_meas)`` gradient phase of every walker under the prepared acquisition ``P``: each window's stored
    coefficients against the waveform's projection on that window's bands (:func:`~dmipy_replay.replay.compression.
    band_projection`), summed (RPK.md 4.3).

    The ONE band contraction of a replay. :meth:`ReplayPack._walker_phases` -- and with it
    :meth:`ReplayPack.replay`, :meth:`ReplayPack.walker_signals` and :meth:`ReplayPack.walker_phases` -- and
    :func:`~dmipy_replay.replay.study.walker_primitives` read it, so the per-walker depth and the ensemble depth are
    the same number by construction rather than by agreement (dmipy-sim#484). ``W`` replaces the projection with a
    caller's own per-position weights (:meth:`ReplayPack._check_weights`, single-window only). An acquisition whose
    effective gradient is zero throughout (a free-induction decay, a gradient-free echo) has the phase zero and reads no
    coefficient. With a plan (``P["K_use"]``) only the lowest ``K_use`` bands of each window are decoded and projected:
    bit for bit the contraction of the full decode cut at ``K_use``.
    """
    from .compression import read_position_coeffs, band_projection
    n_w = P["n_w"]
    if W is None and not np.any(P["G_eff_wf"]):
        return np.zeros((n_w, P["G_eff_wf"].shape[0]))            # no gradient: every walker's phase is zero, no band is read
    phi = None
    K_use = P.get("K_use")
    for seg, t0, T_s in P["windows"]:
        if W is not None:
            W_s = W
            C = read_position_coeffs(seg.arrays, dtype=np.float64)
        else:
            K_s = seg.K if K_use is None else K_use
            Wp = band_projection(P["G_eff_wf"], P["dt_wf"], T_s, K_s, t0=t0)
            W_s = (GAMMA * Wp).reshape(Wp.shape[0], -1).T
            C = read_position_coeffs(seg.arrays, dtype=np.float64, bands=K_use)           # the planned prefix: K_use + 2 entries
        phi_s = C.reshape(n_w, C.shape[1] * 3) @ W_s
        phi = phi_s if phi is None else phi + phi_s
    return phi


def _path_field_channels(P, waveform):
    """``(Psi, names)``: the C3 path channel's gated field integrals per walker under the prepared acquisition,
    summed over the windows -- the ONE read of that channel, which
    :func:`~dmipy_replay.fields.hollow_cylinder.field_terms` turns into the isotropic and anisotropic terms for
    :meth:`ReplayPack._walker_phases` and :func:`~dmipy_replay.replay.study.walker_primitives` alike."""
    from .field import path_field_integral
    pm = P["ch"]["susc_field_basis"]
    Psi = names = None
    for seg, t0, T_s in P["windows"]:
        Psi_s, names = path_field_integral(seg.arrays, pm, waveform, T_s, t0=t0, n_w=P["n_w"], modes=P.get("M_use"))
        Psi = Psi_s if Psi is None else Psi + Psi_s
    return Psi, names


def _has_aniso(grid_meta, names):
    """Whether the pack's field channels carry the anisotropic basis: the channel names hold it, and the field
    record, where the pack carries one, says so too."""
    held = names is not None and "aniso_G_xx" in names
    return held and (grid_meta is None or bool(grid_meta.get("has_aniso")))


def _signal_factor(phi, voxel):
    """``E`` of a replay from the walkers' phases: ``exp(i phi)`` times the voxel's factor per measurement
    (:meth:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence.voxel_factor`, 1 for a refocused encoding and
    the spoiler for an unbalanced one, dmipy-sim#375). Every per-walker route forms ``E`` here, so the weighted
    ensemble mean of any of them is :meth:`ReplayPack.replay`."""
    return np.exp(1j * np.asarray(phi)) * np.asarray(voxel, np.float64)[None, :]


def _gated_contact(windows, gate_wf, j, rho_over_D, chan_meta, *, parallel=False):
    """The gated contact term ``(rho / D) int chi dB`` of window ``j`` of a prepared replay: the gate on its own grid
    (``gate_wf``: the transverse gate, or with ``parallel`` its complement within the acquisition) against the contact
    channel's bridge (:func:`surface_logweight`)."""
    seg, t0, _T_s = windows[j]
    chi, dt_chi = gate_wf
    gate = np.clip(1.0 - chi, 0.0, None) if parallel else chi
    return surface_logweight(seg.arrays, rho_over_D, chan_meta, gate, dt_chi, t0=t0)


def surface_logweight(arrays, rho2_over_D, chan_meta=None, gate=None, dt_gate=None, *, t0=0.0):
    """Per-walker surface log-weight ``(rho2/D) int chi(t) dB(t)`` from the C2 channel, ``chi`` the ``gate`` on its own
    grid ``dt_gate`` (from ``t0``, a window's start).

    C2 is stored in the bridge form (``blt_bridge_dst`` + the two exact endpoints), so the
    UNGATED total contact is ``blt_endpoint`` read directly -- it is the exact cumulative
    ``L(T)``, not something reconstructed from bands, which is the whole reason the endpoint is
    held exactly. A gate contracts the bridge with the gate's cosine integrals
    (:func:`~dmipy_replay.replay.compression.surface_logweight_bridge`): no series is built.

    Takes the pack's ``arrays`` rather than one tensor: the channel is three tensors now, and a
    signature that accepted just the coefficient block invited passing the wrong one.
    """
    from .compression import surface_logweight_bridge, has_c2
    if not has_c2(arrays):
        raise ValueError(
            "surface relaxivity was requested but this pack carries no C2 channel "
            "(no 'blt_bridge_dst'). Returning the signal without the requested attenuation would be a "
            "plausible wrong number.")
    if gate is None:
        return float(rho2_over_D) * np.asarray(arrays["blt_endpoint"], np.float64)
    meta = dict(chan_meta or {})
    meta.setdefault("K", _cx_bands_K(arrays, meta))
    return surface_logweight_bridge(arrays, meta, rho2_over_D, gate, dt_gate, t0=t0)   # the bridge contracted, never decoded


def replay_signal(pack, W, *, rho2_over_D=0.0, gate=None, dt_gate=None, complex_signal=False):
    """Replay a compiled scheme ``W`` (from :func:`compile_scheme`) against ``pack`` (a :class:`ReplayPack`
    or a plain arrays dict). Returns ``E`` per measurement (magnitude unless ``complex_signal``).

    ``rho2_over_D`` > 0 activates the exact surface-relaxivity replay via the pack's boundary local time
    (a per-walker signal loss decaying the whole signal, including ``b=0``); ``gate`` (on its own grid ``dt_gate``)
    coherence-gates it.
    """
    a = pack.arrays if isinstance(pack, ReplayPack) else pack
    from .compression import read_position_coeffs
    from .compression import require_position_method
    if isinstance(pack, ReplayPack):
        require_position_method(pack.method)
    C = read_position_coeffs(a, dtype=np.float64)
    N_w = C.shape[0]
    w0 = np.asarray(a.get("spin_weights", np.ones(N_w)), np.float64)
    surface_logw = None
    if rho2_over_D:
        # asked for, so it must happen: a missing C2 channel raises inside surface_logweight
        # rather than being skipped, so a caller never gets a silently unattenuated signal
        # when rho2_over_D was asked for and the pack carries no boundary channel.
        cm = ((pack.meta.get("compression", {}).get("channels", {}) or {}).get("boundary_local_time")
              if isinstance(pack, ReplayPack) else None)
        surface_logw = surface_logweight(a, rho2_over_D, cm, gate, dt_gate)
    return replay_coefficients(C, w0, W, surface_logw=surface_logw, complex_signal=complex_signal)


def replay_coefficients(position_coeffs, spin_weights, W, *, surface_logw=None, complex_signal=False):
    """The compiled replay on host arrays: ``E = <w_eff exp(i C W)> / <w>`` for the position coefficients
    ``C`` ``(n_w, K+2, n_c)`` (:func:`compression.read_position_coeffs`, any subset of the stored axes so long
    as ``W`` was compiled from the matching waveform components), the walkers' weights ``w`` and the compiled
    scheme ``W`` ``(n_c (K+2), n_meas)`` from :func:`compile_scheme`. ``surface_logw`` is the per-walker
    surface-relaxivity log-weight from :func:`surface_logweight`: a signal LOSS, so it weights the numerator
    while the denominator stays ``<w>`` (``E(b=0) < 1``). The one host kernel every consumer -- the pack's
    ``replay``, a fit's per-voxel forward, a lookup-table build -- evaluates; :func:`replay_signal_jax` is its
    traced twin."""
    C = np.asarray(position_coeffs, np.float64)
    N_w, K, n_c = C.shape
    W = np.asarray(W)
    if W.shape[0] != K * n_c:
        raise ValueError(
            f"compiled scheme has {W.shape[0]} rows for {K * n_c} stored coefficients "
            f"({K} per axis = 2 endpoints + {K - 2} bands, {n_c} axes). Compile with "
            f"compile_scheme(G, dt, pack.K, T=pack.T_window) from the same components -- passing the stored "
            f"width instead of pack.K produces a scheme that multiplies cleanly and means nothing.")
    w0 = np.asarray(spin_weights, np.float64)
    phi = C.reshape(N_w, K * n_c) @ W                          # (N_w, n_meas)
    w_eff = w0 if surface_logw is None else w0 * np.exp(np.asarray(surface_logw, np.float64))
    S = (w_eff[:, None] * np.exp(1j * phi)).sum(0) / w0.sum()
    return S if complex_signal else np.abs(S)
