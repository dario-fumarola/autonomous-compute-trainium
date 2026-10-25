"""CLI-only recovery of a failed pre-update task qualification launch."""
import sys
from compute_trainium.launcher_compat import without_launcher_corpus

if __name__=='__main__':
    sys.argv=[sys.argv[0],*without_launcher_corpus(sys.argv[1:])]
    from scripts.qualify_natural_chain_native import main
    main()
