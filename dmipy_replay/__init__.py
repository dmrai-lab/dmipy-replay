"""dmipy-replay: replay Monte-Carlo diffusion-MRI walks from the Hub with any acquisition.

A replay pack (``.rpk``, replay-pack-spec RPK.md) stores what a walk leaves of every walker that no acquisition
changes -- its path as sine bands, its pool, its wall contact, the field it read -- so any gradient waveform, RF
schedule, relaxation, surface relaxivity, field strength and orientation is an inner product against it, with no
walk and no GPU::

    import dmipy_replay as dr
    pack = dr.ReplayPack.load("hf://SubstrateCommons/canonical-pores/packs/cylinder-d4um.rpk")
    seq = dr.pgse([[1, 0, 0]], 0.01, 0.03, bvalues=[1e9, 3e9])
    signal = pack.replay(seq, tissue=pack.nominal, scanner=pack.nominal_field_T)

Every acquisition is a :class:`ScannerSequence` (replay-pack-spec ACQUISITION.md): build one with the builders, or read
a Pulseq ``.seq`` with :func:`from_pulseq`.
"""
from .constants import GAMMA
from .acquisition.scanner_sequence import ScannerSequence, Encoding, Protocol, Prescription
from .acquisition.rf import B1Pulse, RFEvent, RFSchedule
from .acquisition.timing import SequenceTiming
from .acquisition.scanners import ScannerLimits
from .acquisition.waveforms import set_b, calc_b, calc_btensor, btensor_invariants, rotate_waveform, tile_waveform
from .sequences.builders import pgse, pgste, ogse, cpmg, splice, gre, ste, pte
from .sequences.btable import from_btable
from .spec.tissue import Tissue
from .spec.substrate import SubstrateSpec
from .replay.replay import ReplayPack, read_rpk, compile_scheme
from .replay import hub

__version__ = "0.1.0"

__all__ = ["GAMMA", "ScannerSequence", "Encoding", "Protocol", "Prescription", "B1Pulse", "RFEvent", "RFSchedule",
           "SequenceTiming", "ScannerLimits", "set_b", "calc_b", "calc_btensor", "btensor_invariants", "rotate_waveform",
           "tile_waveform", "pgse", "pgste", "ogse", "cpmg", "splice", "gre", "ste", "pte", "from_btable", "Tissue", "SubstrateSpec",
           "ReplayPack", "read_rpk", "compile_scheme", "hub"]
