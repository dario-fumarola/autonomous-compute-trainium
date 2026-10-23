"""New held-out panel gate; never weakens the consumed legacy TEST gate."""


def validate_depth_test_release(release, *, corpus_sha, checkpoint_sha, repetitions, evaluator_sha):
    if release.get('status')!='FROZEN_VARIABLE_DEPTH_TEST_RELEASE':
        raise ValueError('Missing frozen new-study TEST release')
    if release.get('corpus_sha256')!=corpus_sha or release.get('evaluator_sha256')!=evaluator_sha:
        raise ValueError('Held-out source or evaluator changed')
    matches=[m for m in release.get('models',[]) if m['checkpoint_sha256']==checkpoint_sha]
    if len(matches)!=1 or repetitions not in matches[0]['allowed_depths']:
        raise ValueError('Checkpoint/depth not in frozen readout matrix')
