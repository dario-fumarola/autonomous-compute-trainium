import hashlib
import json

import numpy as np

from scripts.prepare_extension_corpus import concatenate


def test_append_preserves_partial_final_copy_and_all_real_target_masks(tmp_path):
    # Numerical software fixtures only; never used as research training data.
    inputs=[]
    for name,offset in [('parent',0),('extension',2048)]:
        path=tmp_path/name;packed=path/'packed';packed.mkdir(parents=True)
        tokens=np.arange(1025,dtype=np.int32)+offset
        domains=np.ones(1025,dtype=np.uint8)
        domains[0]=0
        if name=='parent':tokens[-1]=2;domains[-1]=0
        counts=np.array([[0,0,0],[int(domains[1:].sum()),0,0]],dtype=np.int64)
        for key,array in [('tokens',tokens),('domains',domains),('prefix_counts',counts)]:np.save(packed/(key+'.npy'),array)
        fingerprint=hashlib.sha256(tokens.tobytes()+domains.tobytes()).hexdigest()
        (packed/'manifest.json').write_text(json.dumps({'windows':1,'fingerprint':fingerprint,
            'source_sha256':name+'-source','order_fingerprint':name+'-order'}))
        inputs.append((path,tokens,domains,counts))
    output=tmp_path/'combined';output.mkdir()
    metadata=concatenate(inputs[0][0],inputs[1][0],output)
    tokens=np.load(output/'packed/tokens.npy');domains=np.load(output/'packed/domains.npy')
    np.testing.assert_array_equal(tokens[:1025],inputs[0][1])
    np.testing.assert_array_equal(tokens[1025:],inputs[1][1][1:])
    np.testing.assert_array_equal(domains[:1025],inputs[0][2])
    np.testing.assert_array_equal(domains[1025:],inputs[1][2][1:])
    counts=np.load(output/'packed/prefix_counts.npy')
    np.testing.assert_array_equal(counts[-1],inputs[0][3][-1]+inputs[1][3][-1])
    assert metadata['real_targets']==2047
    assert metadata['prefix_parent']['fingerprint']==hashlib.sha256(inputs[0][1].tobytes()+inputs[0][2].tobytes()).hexdigest()
