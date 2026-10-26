"""Prospective baseline: one core, FP32 mixing, no unused auxiliary head."""
from .future_span import FutureSpanConfig
from .mixing_precision_probe import PrecisionMixingProbe
from .parcae_fp16_candidate import ParcaeFP16Candidate


class LeanFP32MixingCandidate(PrecisionMixingProbe):
    def __init__(self,config,**kwargs):
        if kwargs.get('recurrent',True) is not True:raise ValueError('Explicit shared-family cheap control required')
        super().__init__(config,FutureSpanConfig(context=min(1024,config.max_seq)),
                         precision_mode='mix32',**kwargs)
        del self.future_lm_head
        del self.future_positions

    # Keep the same C/coda/FP32 vocabulary path,without an auxiliary projection.
    decode=ParcaeFP16Candidate.decode
