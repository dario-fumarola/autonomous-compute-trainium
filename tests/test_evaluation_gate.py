import pytest
from compute_trainium.evaluation_gate import validate_trained_test_release


def test_test_release_rejects_unfrozen_models_sources_code_and_untrained_paths():
    release={'status':'FROZEN_TEST_RELEASE','corpus_sha256':'source','allowed_checkpoint_unrolls':['model:2'],'evaluator_sha256':'code'}
    expected=dict(corpus_sha='source',checkpoint_sha='model',repetitions=2,evaluator_sha='code',trained_repetitions=2)
    validate_trained_test_release(release,**expected)
    for key,value in [('corpus_sha','other'),('checkpoint_sha','other'),('repetitions',1),('evaluator_sha','other'),('trained_repetitions',1)]:
        with pytest.raises(ValueError):validate_trained_test_release(release,**dict(expected,**{key:value}))
