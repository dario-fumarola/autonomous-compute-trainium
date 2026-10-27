"""Frozen readout-only adapter; all original scientific sources stay immutable."""
import argparse,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--readout-recovery',type=Path,required=True);p.add_argument('--readout-recovery-sha',required=True)
    a,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
    import torch
    from compute_trainium import training
    from compute_trainium.canonical_prediction_checkpoint import canonical_prediction_checkpoint
    from compute_trainium.lossless_zip_delta import create_archive
    from scripts.qualify_native import digest
    from scripts import score_scale_registered
    if training.file_sha256(a.readout_recovery)!=a.readout_recovery_sha:raise AssertionError('Frozen readout recovery changed')
    recovery=json.loads(a.readout_recovery.read_text());code=Path(__file__).resolve().parents[1]
    if any(training.file_sha256(code/f)!=s for f,s in recovery['additional_source_sha256'].items()):raise AssertionError('Frozen added recovery source changed')
    original=training.atomic_checkpoint;observations=[];verified_writes=[];completed=False
    def write(path,value):
        if value.get('purpose')!='scale_retained_prediction_weights':raise ValueError('No unrelated checkpoint serialization can change')
        fixed=canonical_prediction_checkpoint(value);original(path,fixed)
        saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        if digest(saved['model'])!=value['model_digest']:raise AssertionError('Canonical exact disk readback differs')
        verified_writes.append(str(path))
        if Path(path).resolve()==Path(recovery['replacement_retained_path']).resolve():
            prior=Path(recovery['preserved_retained_path'])
            if training.file_sha256(prior)!=recovery['preserved_retained_sha256']:raise AssertionError('Original failed-readout copy changed')
            old=torch.load(prior,map_location='cpu',mmap=True,weights_only=False)
            if digest(dict(old['model']))!=digest(saved['model']):raise AssertionError('Retained prediction tensors changed in recovery')
            del old
            recipe=prior.with_suffix(prior.suffix+'.archive.json');row=create_archive(path,prior,recipe)
            if row['shared_payload_bytes']<.99*row['original_bytes']:raise AssertionError('Only the identical serialized tensor payloads may be shared')
            prior.unlink();observations.append(dict(original_binary_sha256=recovery['preserved_retained_sha256'],
                exact_failed_binary_restorable=True,unchanged_prediction_digest=value['model_digest'],**row))
        del saved
    training.atomic_checkpoint=write
    try:
        score_scale_registered.main();completed=True
    finally:
        training.atomic_checkpoint=original
        case=rest[rest.index('--case')+1];out=a.readout_recovery.parent/(case+'-serialization.json')
        out.write_text(json.dumps(dict(canonical_container_only=True,readout_completed=completed,exact_disk_write_checks_passed=bool(verified_writes),
            observations=observations,original_study_sha256=recovery['original_plan_sha256']),indent=2)+'\n')


if __name__=='__main__':main()
