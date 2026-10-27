"""Reproduce proposal arithmetic; no corpus, model fit, or confirmation access."""
import json
import math
import statistics

from scipy.stats import chi2

from compute_trainium.model import ModelConfig


def audit():
    rejected = []
    for layers, dim, heads in [(12, 960, 15), (16, 832, 13),
                               (24, 704, 11), (28, 640, 10)]:
        try:
            ModelConfig(layers=layers, dim=dim, heads=heads, kv_heads=4,
                        ffn_dim=round(dim * 8 / 3), max_seq=1024)
        except ValueError as error:
            rejected.append(dict(layers=layers, dim=dim, heads=heads,
                                 reason=str(error)))
        else:
            raise AssertionError("Proposed GQA dimensions unexpectedly accepted")
    alternatives = []
    # Illustrative legal dimensions only, not a selected architecture grid.
    for layers, dim, heads, ffn in [(12, 960, 12, 2560),
                                    (16, 832, 16, 2304),
                                    (20, 768, 12, 2048),
                                    (24, 704, 16, 1920),
                                    (28, 640, 16, 1792)]:
        c = ModelConfig(layers=layers, dim=dim, heads=heads, kv_heads=4,
                        ffn_dim=ffn, max_seq=1024)
        alternatives.append(dict(layers=layers, dim=dim, heads=heads,
                                 kv_heads=4, head_dim=dim // heads,
                                 ffn_dim=ffn, parameters=c.parameter_count()))
    ordinary = ModelConfig(max_seq=1024)
    # Logical applications in the proposed 6/8/6, twice-middle architecture.
    loop = ModelConfig(layers=28, max_seq=1024)
    base_work = ordinary.flops(real_targets=1024, expanded_positions=1024)
    loop_work = loop.flops(real_targets=1024, expanded_positions=1024)
    ratio = loop_work['training_matmul_flops'] / base_work['training_matmul_flops']
    body = base_work['forward_matmul_flops'] - base_work['lm_head_forward_flops']
    core_work = body * 8 / 20
    repeat = []
    for epochs in [2, 4, 8, 16]:
        repeat.append(dict(epochs=epochs,
                           marginal_at_endpoint=math.exp(-(epochs - 1) / 15.4),
                           total_effective_fraction=(1 + 15.4 *
                               (1 - math.exp(-(epochs - 1) / 15.4))) / epochs))
    variance = []
    # Rounded paired differences quoted in the supplied review, not fresh data.
    for name, differences in [('family_A_review_rounding', [.0312, .0299, .0305]),
                               ('family_B_review_rounding', [.012016, .013111])]:
        df = len(differences) - 1
        sd = statistics.stdev(differences)
        variance.append(dict(source=name, n=len(differences), paired_sd=sd,
                             normal_sd_one_sided_95_upper=sd *
                             math.sqrt(df / chi2.ppf(.05, df))))
    measured_targets = 52_337_406
    measured_seconds = 555.363395
    rate = measured_targets / measured_seconds
    return dict(
        status='COMPLETE_PROPOSAL_ARITHMETIC_ONLY',
        assigned_chips=0, optimizer_updates=0, new_research_examples=0,
        proposed_invalid_gqa_shapes=rejected,
        illustrative_valid_shapes_not_admitted=alternatives,
        family_A_work=dict(ordinary=base_work, middle_loop=loop_work,
                           whole_neural_matmul_ratio=ratio,
                           matched_loop_targets_at_ordinary_16_4B=16.4e9 / ratio,
                           head_over_eight_block_core=base_work['lm_head_forward_flops'] / core_work,
                           q_point_2_routing_head_over_selected_core=base_work['lm_head_forward_flops'] / (.2 * core_work),
                           excludes='Optimizer, boundaries, elementwise, communication, padding differences and compiler effects'),
        fp16_unshared_logical_KV_bytes_per_token=dict(
            ordinary=2 * 20 * 4 * 64 * 2,
            middle_loop=2 * 28 * 4 * 64 * 2,
            assumption='Separate K/V per logical application; no unqualified cache sharing'),
        muennighoff_15_4_fit_illustration=repeat,
        small_sample_variance_illustrations=variance,
        mean_at_least_delta_gate_power_at_true_delta_under_symmetric_continuous_noise=.5,
        measured_956M_warm_cost=dict(
            targets_per_second=rate,
            ordinary_100B_days=100e9 / rate / 86400,
            hypothetical_3x_10B_days=3 * 10e9 / rate / 86400,
            hypothetical_3x_20B_days=3 * 20e9 / rate / 86400,
            scope='Measured warm path extrapolation, not an admitted budget or new mechanism benchmark'),
        scope='Proposal review only. No changed charter, statistical gate, architecture, data, live source, or scientific admission.')


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, allow_nan=False))
