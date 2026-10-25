"""Fail closed on incomplete trials;reuse complete fits for read-only scoring."""


def completed_fits(plan,report):
    if (report['status']!='STOPPED_CHEAP_MIXING_TUNE'
            or "unsupported operand type(s) for +: 'PosixPath' and 'str'" not in report.get('error','')
            or report['scores']):
        raise ValueError('Only the proven pre-scoring path-join failure is eligible')
    expected={(key,u) for key in plan['cases'] for u in [2,576]}
    fits={(r['case_id'],r['update']):r for r in report['training_records']}
    gates={(r['case_id'],r['update']):r for r in report['gradient_gates']}
    if (len(fits)!=6 or len(gates)!=6 or len(report['training_records'])!=6
            or len(report['gradient_gates'])!=6 or set(fits)!=expected or set(gates)!=expected):
        raise ValueError('All three complete fits and all six unique numerical gates required')
    result={}
    for key,case in plan['cases'].items():
        for update in [2,576]:
            row=fits[key,update];gate=gates[key,update]['independent_cpu']
            saved=next(s for s in row['checkpoint_records'] if s['update']==update)
            if (row['qualification_only'] or not saved['all_complete_replica_states_equal']
                    or gate['status']!='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'
                    or gate['checkpoint_sha256']!=saved['checkpoint_sha256']
                    or gate['training_update']!=update or gate['windows']!=512
                    or not gate['full_state_readback_exact'] or not gate['model_unchanged']):
                raise ValueError('Original trained state and complete gradient proof differ')
        row=fits[key,576]
        if (row['real_targets']!=case['end_targets'] or row['valid_anchors']!=case['end_anchors']
                or row['estimated_dense_matrix_ops']!=case['training_dense_matrix_ops']):
            raise ValueError('Complete cooled source/work/anchor endpoint required')
        result[key]=row
    return result


def score_arguments(root,source_plan,source_sha,key,native_identity,checkpoint_sha,target):
    checkpoint=root/'runs'/native_identity/'checkpoint-u576.pt'
    return ['--root',str(root),'--plan',str(source_plan),'--plan-sha',source_sha,
        '--checkpoint',str(checkpoint),'--checkpoint-sha',checkpoint_sha,'--case',key,
        '--cohort','tune','--depth','2','--output',str(target),
        '--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
