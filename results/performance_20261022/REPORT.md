# Packed trainer and performance follow-up — October 22

No large scientific fit or LR sweep has launched. These are numerical, recovery,
data-pipeline and throughput checks on natural data. No pause benefit is established.

## Chosen measured configuration

150,436,608 parameters; tied 32k embedding/head; FP32 master parameters and Adam
moments; BF16 autocast; context 1024; two sequences per core; two accumulated
microbatches; 64 ranks across all 16 chips. The global batch has 262,144 target
slots, averaging about 261,707 real targets after masking boundaries. AdamW is
unfused; global clipping at 1 and the finite-gradient check are included.

| Condition | Warm updates | Warm seconds | Real targets/sec | Versus old benchmark |
| --- | ---: | ---: | ---: | ---: |
| Ordinary | 100 | 30.597 | 855,336 | 5.41× |
| 16 pauses per 128 input positions | 100 | 73.954 | 353,879 | 3.14× |

The old rates were 158,031 and 112,575. Context, batch, precision, synchronization
and data handling changed together; this is a performance improvement, not a
controlled ablation of any one change. The new timer includes packing, H2D,
token-correct accumulation, gradient reduction, clipping and AdamW. It excludes
the initial probe/compile, final checkpoint and full-state hash. Probe/setup in
the chosen runs took 55.5 s ordinary and 116.4 s pause, separately from process
startup. A context-512/accumulation-8 candidate measured 526,414 / 222,039.

The qualification sample contains 1,998,577 unique TRAIN targets. Timing explicitly
repeats it; 27.2M seen targets per phase are not 27.2M unique targets. Its final
domain mixture is uneven. Neither its training curves nor its held-out checks
constitute a scientific mechanism comparison.

## Correctness and training path

CPU tests cover short documents and tails, packing and pause alignment, causal
masking, variable-mask accumulation, atomic recovery, WSD, mapped-data integrity,
and controller failure/branch behavior. A 64-rank BF16 tiny-model check matches
the full CPU accumulated global gradient to relative L2 0.00110. The selected
150M ordinary and pause probes have full-gradient relative L2 errors 0.00230 and
0.00298 against the corresponding CPU BF16 reference.

The whole-instance late switch restores the complete ordinary model/optimizer,
checks its digest on CPU and after transfer, validates the new graph, and
performs 104 pause updates in fresh workers. Every rank finishes with identical
full model and optimizer state. Training and DEV never retain their compiled
graphs in the same processes.

Packing scores every real token once, including short documents and document
tails. BOS/EOS/padding carry no target loss. Training allows causal attention
across document boundaries within a window; DEV isolates documents. The global
loss denominator counts real targets across every rank and microbatch. All but
the last microbatch use DDP `no_sync` around forward and backward.

The mapped loader shares packed arrays across workers and checks file hashes,
context, seed and source provenance. A large-corpus freeze still needs streaming
preparation and measured production I/O; the JSON adapter is for qualification.

End-to-end controller and replay receipts accompany this report. The tiny
three-arm endpoints all use 196,322 real targets, matched initial weights and
order, then fresh DEV evaluation. The 150M recovery plan forks two identical
two-update WSD decay branches from the same stable checkpoint and evaluates
full DEV separately. These are software qualification, not replications for
the research claim. TEST remains sealed.

All 23 CPU tests pass. The [150M replay verification](recovery-verification.json)
confirms bit-identical complete model/optimizer state, schedule, losses and
mapped cursor after both two-update branches, with the stable source file
unchanged. Both full DEV evaluations score the same 232 documents / 307,646
real targets identically. Domain NLL is 6.3201 web, 5.1900 math and 4.3000 code;
the fixed weighted NLL is 5.6900. This is a pipeline/learning check after repeated
qualification data, without a matched scientific baseline or seed interval.

## Budget and comparison endpoints

The [revised budget](budget.json) prices three arms × three seeds at 3B real
targets each. Steady training is 14.98 node-hours; separate LR mini-fits add
0.50 h; additional FLOP-endpoint decay branches add 1.29 h. With a 2–3× allowance
for production I/O, restarts, DEV, checkpoints and variation, plan **33.5–50.3
node-hours**, replacing the earlier 118–177 h proposal. This remains a proposed,
roughly 20-token/parameter signal-seeking pilot. A subsequent ordinary/early
paired continuation from 3B to 30B costs another 29.96 steady hours, before its
own overhead allowance; it is not automatically launched.

Every endpoint completes a cosine decay to 10% of peak LR. Decay work is inside
the endpoint budget, and extra branches are charged. At the ordinary 3B FLOP
budget, early and late endpoints are approximately 2.649B and 2.825B real
targets. They each fork their own decay from a stable trunk. The late switch
stays at the same predeclared source-token boundary. Round actual boundaries
to shared complete global updates before scientific launch. An unannealed
checkpoint cannot serve as the corresponding endpoint.

The dense matmul estimate includes attention, backward and all fixed LM-head
slots, including masked slots that still execute. Pauses add about 13.2% of
that estimate but cost about 2.42× wall time. This is why both FLOP and measured
chip-time comparisons are required. The estimate excludes elementwise and
optimizer work and is not a hardware counter.

## Failures, monitoring and remaining gates

The first full late restore exceeded the 416 GiB container limit during pause
compilation. Its failed receipt is preserved. Mapped checkpoint loading and
returning freed CPU allocation arenas before compilation allowed the retry to
pass. The first controller DEV attempt was rejected before accelerator launch
because it inherited that oversized limit. Phase-specific caps fix the
reservation: 160 GiB tiny training, 256 GiB measured 150M training, 64 GiB DEV.

The supervisor checks memory/disk every two seconds, watches compiler failures
and container OOM events, and enforces a wall bound. The required host reserve
is 512 GiB and the filesystem floor is 35 GiB. See the [ledger](resource-ledger.json)
for all attempts and costs. Original reports, including failures, remain intact.
There are 24 attempts, including one failed compile and one rejected preflight.
Assigned time is 9.5169 chip-hours. The separate inclusive node opportunity
window includes preparation, code, idle time and overlapping small probes;
do not add overlapping opportunity windows. Sampled minima were 543.88 GiB
host MemAvailable and 35.36 GiB filesystem free space.

The larger batch exposed a native contiguous-input fallback. Materializing the
small sliced integer inputs on CPU before transfer removes it. Both 104-update
model/optimizer results remain bit-identical to their original counterparts.
The chosen distributed runs record only `aten::_to_copy` in the fallback inventory;
they do not claim zero fallback. Local reproducers and [OSS notes](../../docs/OSS_NOTES.md)
retain the evidence; no upstream submission was sent.
The standalone strided-input reproducer records the prologue CPU-roundtrip
fallback; its contiguous control records none. Both return the correct result.

Before LR sweeps or the main comparison: freeze a larger natural corpus and
its mixture/order, audit duplicates, measure its mapped I/O, and use separate
variant LR mini-fits at this chosen batch. Then freeze schedules, rounded
endpoints and resource budget. Preserve the original 1.7B results and the
archived predecessor. No synthetic research corpus, teacher traces, RL or
larger model has been introduced.
