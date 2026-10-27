from pathlib import Path
import ast
from scripts.audit_scale_failure_metrics import instrument


def test_failure_observation_insertion_preserves_all_original_audit_statements():
    original=Path('scripts/audit_scale_registered.py')
    before=original.read_bytes()
    audit=instrument(original)
    assert original.read_bytes()==before
    # Compile original and instrumented code without running any model/data.
    namespace={'__name__':'fixture_original','__file__':str(original)}
    exec(compile(before,str(original),'exec'),namespace)
    baseline=namespace['audit']
    assert audit.__code__.co_argcount==baseline.__code__.co_argcount
    assert audit.__code__.co_names==baseline.__code__.co_names
    assert audit.__code__.co_consts.count('Independent complete512 gradient/NLL failed')==1
    assert 'full-gradient-observation.json' in audit.__code__.co_consts
