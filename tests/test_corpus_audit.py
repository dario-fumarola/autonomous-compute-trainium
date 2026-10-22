import numpy as np

from compute_trainium.corpus_audit import NearDuplicateIndex, bottom_k_jaccard, text_sketch


def test_near_duplicate_audit_is_deterministic_and_recognizes_shared_text():
    # Software fixtures only, never a training/evaluation document.
    text = " ".join(f"word{i}" for i in range(500))
    a, aa = text_sketch(text)
    b, bb = text_sketch(text.upper())
    assert np.array_equal(a, b) and bottom_k_jaccard(aa, bb) == 1
    index = NearDuplicateIndex()
    assert index.check(a, aa) is None
    index.add(a, aa)
    assert index.check(b, bb) == "near_duplicate"
    c, cc = text_sketch(" ".join(f"other{i}" for i in range(500)))
    assert bottom_k_jaccard(aa, cc) == 0
    assert index.check(c, cc) is None


def test_crowded_buckets_are_rejected_without_silently_skipping_audit():
    index = NearDuplicateIndex(maximum_candidates=1)
    sig, a = text_sketch("one two three four five six seven")
    index.add(sig, a)
    index.add(sig, a)
    assert index.check(sig, a) == "crowded_near_duplicate_bucket"
