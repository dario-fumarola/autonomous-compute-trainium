import json

import pytest

from scripts.run_scale_main import IDENTITY,analyze


def test_primary_sign_and_conventional_endpoint_cannot_be_replaced_by_equal_data(tmp_path):
    for key,nll in [('ordinary',2.),('ordinary-equal',2.03),('recurrent',1.98)]:
        records=[dict(id=d+str(i),domain=d,group=d+str(i),nll_sum=10*nll,real_targets=10)
                 for d in ['web','math','code'] for i in range(6)]
        folder=tmp_path/IDENTITY/'scores'/key;folder.mkdir(parents=True)
        (folder/'document-scores.json').write_text(json.dumps({'records':records}))
    result=analyze(tmp_path,'frozen',{})
    assert result['primary_recurrent_gain_over_ordinary_work']['gain_nats']==pytest.approx(.02)
    assert result['secondary_equal_data_gain']['gain_nats']==pytest.approx(.05)
    assert result['development_mean_meets_point_threshold']
    assert result['one_training_seed_only'] and not result['terminal_claim']
    file=tmp_path/IDENTITY/'scores/ordinary/document-scores.json'
    rows=json.loads(file.read_text());rows['records'][0]['real_targets']=9;file.write_text(json.dumps(rows))
    with pytest.raises(ValueError):analyze(tmp_path,'frozen',{})
