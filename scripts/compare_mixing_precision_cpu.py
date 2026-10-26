"""One independent full-objective gold vector,exact-point reuse across probes."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('USE_TORCH_XLA','0')
import torch
from compute_trainium.future_span_scientific_state import signature as unused_parent_signature
from compute_trainium import future_span_gradient_point as points
from compute_trainium.mixing_control_reference import construct_reference
from compute_trainium.mixing_control_science import signature
from compute_trainium.parcae_mixing_control import mixing_only_ledger
from compute_trainium.parcae_update_qualification import relative_error
from compute_trainium.queue import write_state
from compute_trainium.training import atomic_checkpoint,file_sha256
from scripts.audit_future_span_qualification import gradient_check


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--probe-plan',type=Path,required=True);p.add_argument('--probe-sha',required=True)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold',type=Path);p.add_argument('--gold-report-sha');a=p.parse_args();started=time.time()
    torch.set_num_threads(12)
    if file_sha256(a.probe_plan)!=a.probe_sha:raise AssertionError('Precision envelope changed')
    plan=json.loads(a.probe_plan.read_text());point=plan['point']
    if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Frozen precision/reference source changed')
    native=json.loads((a.capture/'report.json').read_text())
    if (native['probe_plan_sha256']!=a.probe_sha or native['checkpoint_sha256']!=point['checkpoint_sha256']
            or native['plan_sha256']!=point['parent_plan_sha256'] or native['windows']!=512
            or native['optimizer_updates']!=0 or not native['model_unchanged'] or not native['replica_gradients_equal']):
        raise ValueError('Exact unchanged full-global precision capture required')
    if file_sha256(a.capture/'gradient.pt')!=native['gradient_sha256']:raise AssertionError('Diagnostic vector changed')
    numerical=None
    if a.gold is None:
        from scripts import audit_future_span_gradient as engine
        points.signature=signature;engine.construct_reference=construct_reference;engine.matmul_ledger=mixing_only_ledger
        original=engine.gradient_check
        def export(expected,actual):
            atomic_checkpoint(a.output/'gold-gradient.pt',dict(gradients=expected))
            return original(expected,actual)
        engine.gradient_check=export
        sys.argv=['independent','--root',str(a.root),'--plan',str(a.root/point['parent_plan']),
            '--plan-sha',point['parent_plan_sha256'],'--case',point['case'],
            '--checkpoint',str(a.root/point['checkpoint']),'--checkpoint-sha',point['checkpoint_sha256'],
            '--capture',str(a.capture),'--output',str(a.output),'--upstream',str(a.root/'upstream-parcae-69284c1'),
            '--seconds-limit','1200']
        try:engine.main()
        except RuntimeError:
            file=a.output/'report.json'
            if not file.exists() or json.loads(file.read_text())['status']!='FAIL_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT':raise
        numerical=json.loads((a.output/'report.json').read_text())
        if numerical['status'] not in ['PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT','FAIL_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT']:
            raise ValueError('Reference must complete,even when the native variant fails')
        write_state(a.output/'original-comparison.json',numerical)
        metadata=dict(probe_plan_sha256=a.probe_sha,checkpoint_sha256=point['checkpoint_sha256'],
            parent_plan_sha256=point['parent_plan_sha256'],rows=native['rows'],noise_sha256_by_rank=native['noise_sha256_by_rank'],
            full_state_digest=native['full_state_digest'],source_cursor=native['source_cursor'],training_update=native['training_update'],
            real_targets=native['real_targets'],valid_anchors=native['valid_anchors'],diagnostic_matrix_ops=native['diagnostic_matrix_ops'],
            cpu_ntp_nll=numerical['cpu_ntp_nll'],cpu_future_mean=numerical['cpu_future_mean'],
            gold_gradient_sha256=file_sha256(a.output/'gold-gradient.pt'))
        write_state(a.output/'gold-meta.json',metadata);gold_dir=a.output
    else:
        if not a.gold_report_sha or file_sha256(a.gold/'report.json')!=a.gold_report_sha:raise AssertionError('Pinned completed gold receipt required')
        previous=json.loads((a.gold/'report.json').read_text())
        if previous['status']!='COMPLETE_MIXING_PRECISION_REFERENCE_COMPARISON' or previous['probe_plan_sha256']!=a.probe_sha:
            raise ValueError('Wrong independent reference receipt')
        if (file_sha256(a.gold/'gold-meta.json')!=previous['gold_meta_sha256']
                or file_sha256(a.gold/'gold-gradient.pt')!=previous['gold_gradient_sha256']):raise AssertionError('Independent gold bytes changed')
        metadata=json.loads((a.gold/'gold-meta.json').read_text());gold_dir=a.gold
        for field in ['checkpoint_sha256','full_state_digest','source_cursor','training_update','real_targets',
                      'valid_anchors','diagnostic_matrix_ops','rows','noise_sha256_by_rank']:
            if native[field]!=metadata[field]:raise AssertionError('Reference reuse changed exact objective/state: '+field)
        a.output.mkdir(parents=True,exist_ok=False)
    for rank,sha in enumerate(native['noise_sha256_by_rank']):
        if file_sha256(a.capture/f'h0-rank{rank}.npy')!=sha:raise AssertionError('Actual supplied states changed')
    gold=torch.load(gold_dir/'gold-gradient.pt',map_location='cpu',mmap=True,weights_only=False)['gradients']
    actual=torch.load(a.capture/'gradient.pt',map_location='cpu',mmap=True,weights_only=False)
    delta=gradient_check(gold,actual['gradients'])
    differences=dict(ntp_nll=actual['ntp_nll']-metadata['cpu_ntp_nll'],future_mean=actual['future_mean']-metadata['cpu_future_mean'])
    passed=delta['passes_two_percent_gate'] and all(abs(v)<=.003 for v in differences.values())
    named={k:relative_error({k:gold[k]},{k:actual['gradients'][k]}) for k in gold
           if k.startswith(('transformer.adapter.','transformer.C.'))}
    baseline_dir=a.root/point['failed_capture'];baseline=json.loads((baseline_dir/'report.json').read_text())
    for field in ['checkpoint_sha256','full_state_digest','source_cursor','training_update','real_targets','valid_anchors','rows','noise_sha256_by_rank']:
        if baseline[field]!=native[field]:raise AssertionError('Original failed baseline is another objective')
    if file_sha256(baseline_dir/'gradient.pt')!=baseline['gradient_sha256']:raise AssertionError('Original failed gradient changed')
    old=torch.load(baseline_dir/'gradient.pt',map_location='cpu',mmap=True,weights_only=False)['gradients']
    original_delta=gradient_check(gold,old)
    original_named={k:relative_error({k:gold[k]},{k:old[k]}) for k in named}
    result=dict(status='COMPLETE_MIXING_PRECISION_REFERENCE_COMPARISON',probe_plan_sha256=a.probe_sha,
        precision_override=native['precision_override'],passes_original_gate=passed,gradient=delta,
        named_injection_readout=named,loss_differences=differences,original_failed_gradient=original_delta,
        original_failed_named_injection_readout=original_named,gold_reused=a.gold is not None,
        gold_gradient_sha256=file_sha256(gold_dir/'gold-gradient.pt'),gold_meta_sha256=file_sha256(gold_dir/'gold-meta.json'),
        gold_directory=str(gold_dir.relative_to(a.root)),windows=512,optimizer_updates=0,assigned_chips=0,
        test_accessed=False,terminal_alpha_spent=0,wall_seconds=time.time()-started,
        scope='Diagnostic completion can include a failing variant;passes_original_gate preserves the unchanged2%/0.003-nat criterion. One independent pinned-upstream full512-window FP32 vector,exact objective/latent-state identities for reuse. No new training qualification or scientific fit.')
    write_state(a.output/'report.json',result);print(json.dumps(dict(status=result['status'],variant=result['precision_override'],passes=passed)))


if __name__=='__main__':main()
