"""Future-span objective with the separately qualified attention precision path."""
from .future_span import FutureSpanCandidate
from .parcae_primitive_sdpa_candidate import PrimitiveSDPAMixin


class FutureSpanMathSDPACandidate(PrimitiveSDPAMixin,FutureSpanCandidate):
    pass
