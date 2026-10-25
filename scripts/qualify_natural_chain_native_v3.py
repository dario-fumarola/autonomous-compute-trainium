"""Same task with already-qualified explicit CE;preserve earlier graph break."""
import sys
from compute_trainium.launcher_compat import without_launcher_corpus

if __name__=='__main__':
    sys.argv=[sys.argv[0],*without_launcher_corpus(sys.argv[1:])]
    from compute_trainium import natural_chain_objective
    from compute_trainium.natural_chain_native_loss import answer_parts
    natural_chain_objective.answer_parts=answer_parts
    from scripts.qualify_natural_chain_native import main
    main()
