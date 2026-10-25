"""Exogenous latent randomness indexed only by an available prefix position."""


def prefix_window_state_key(document_id, window):
    """Evaluator-compatible signature; document identity is deliberately unused.

    Every document uses the same exogenous random stream at a given window
    position. The model's registered state seed still differs between training
    seeds. Generating a full1024-position state before slicing makes earlier
    states independent of document length and future continuation contents.
    """
    if type(window) is not int or not 0 <= window <= (2**63 - 1 - 1_000_000_007)//1024:
        raise ValueError('Nonnegative bounded original window position required')
    return 1_000_000_007 + 1024 * window
