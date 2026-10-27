"""Target log probabilities without subtracting near-equal unit probabilities."""
import torch
import torch.nn.functional as F


def stable_target_log_probs(logits,targets):
    """Exactly -CE in real arithmetic;stable correct-class gradient near p=1.

    log(1+sum_{j!=y} exp(logit_j-logit_y)) is evaluated with logsumexp
    and softplus. The target derivative is minus the sum of other-class
    probabilities, rather than the cancellation-prone p_y-1.
    """
    if logits.shape[:-1]!=targets.shape or logits.shape[-1]<2 or targets.dtype!=torch.long:
        raise ValueError('Matching targets and at least two classes required')
    relative=logits-logits.gather(-1,targets.unsqueeze(-1))
    others=relative.scatter(-1,targets.unsqueeze(-1),float('-inf'))
    return -F.softplus(torch.logsumexp(others,-1))
