import os
import sys
from compute_trainium.readout_panel_scope import verify_train_panel,mark_train_report

if __name__=='__main__':
    verify_train_panel(sys.argv[1:])
    from scripts.read_natural_chain_native import main
    main()
    if int(os.environ.get('LOCAL_RANK','0'))==0:mark_train_report(sys.argv[1:])
