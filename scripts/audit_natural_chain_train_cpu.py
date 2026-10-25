import sys
from compute_trainium.readout_panel_scope import verify_train_panel,mark_train_report

if __name__=='__main__':
    verify_train_panel(sys.argv[1:])
    from scripts.audit_natural_chain_readout_cpu import main
    main();mark_train_report(sys.argv[1:])
