import copy
import pytest
from scripts.recover_latent_content_batch import validate


def test_only_execution_fields_change():
    a=dict(identity='old',registered_unix=1,absolute_deadline_unix=100,threads=2,batch_broad=2,
           workers=4,checkpoint='same',conditions=['a','b'],source_sha256={'file':'old'})
    b=copy.deepcopy(a);b.update(identity='new',registered_unix=10,threads=1,batch_broad=8,source_sha256={'file':'new'},recovery={'old':'receipt'})
    validate(a,b)
    for key,value in [('workers',5),('checkpoint','other'),('conditions',['a']),('absolute_deadline_unix',101)]:
        c=copy.deepcopy(b);c[key]=value
        with pytest.raises(AssertionError):validate(a,c)
