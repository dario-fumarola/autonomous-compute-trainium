import json
import pytest
from compute_trainium.future_span_admission import completed_native_receipts


def test_all_predecessor_fits_and_gates_are_required_not_a_between_phase_idle_gap(tmp_path):
    fits=[];gates=[]
    for arm in ['ordinary','fixed1','poisson2']:
        for phase in ['first','finish','equal']:
            identity=f'low-depth-main-r2-{arm}-{phase}'
            row={'identity':identity,'status':'COMPLETE_REGISTERED_LOW_DEPTH_SCIENTIFIC_SEGMENT',
                 'qualification_only':False,'test_accessed':False}
            if phase=='first' and arm!='ordinary':row['adopted_from']={'identity':identity.replace('-r2-','-v1-')}
            fits.append(row)
        for readout in ['stable1B','matched','equal']:
            gates.append({'arm':arm,'readout':readout,'capture_identity':f'gradient-{arm}-{readout.lower()}',
                          'independent_cpu':{'status':'PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT'}})
    for identity in [r.get('adopted_from',{}).get('identity',r['identity']) for r in fits]+[r['capture_identity'] for r in gates]:
        p=tmp_path/'snapshots'/identity;p.mkdir(parents=True);(p/'closure.json').write_text(json.dumps({'exit_code':0}))
    assert not completed_native_receipts(tmp_path,{'records':fits[:-1]},{'records':gates})
    assert not completed_native_receipts(tmp_path,{'records':fits},{'records':gates[:-1]})
    assert completed_native_receipts(tmp_path,{'records':fits},{'records':gates})
    closure=tmp_path/'snapshots'/gates[0]['capture_identity']/'closure.json'
    closure.unlink();assert not completed_native_receipts(tmp_path,{'records':fits},{'records':gates})
    closure.write_text(json.dumps({'exit_code':137}))
    with pytest.raises(ValueError):completed_native_receipts(tmp_path,{'records':fits},{'records':gates})
