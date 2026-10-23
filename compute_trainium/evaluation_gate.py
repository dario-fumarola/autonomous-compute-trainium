"""Fail-closed release check for exact frozen trained-path TEST scoring."""

def validate_trained_test_release(release,*,corpus_sha,checkpoint_sha,repetitions,evaluator_sha,trained_repetitions):
    key=checkpoint_sha+':'+str(repetitions)
    if (release.get('status')!='FROZEN_TEST_RELEASE' or release.get('corpus_sha256')!=corpus_sha
        or key not in release.get('allowed_checkpoint_unrolls',[])
        or release.get('evaluator_sha256')!=evaluator_sha or repetitions!=trained_repetitions):
        raise ValueError('TEST release differs in source, checkpoint, trained path or evaluator')
