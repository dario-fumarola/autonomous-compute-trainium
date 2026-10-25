# Bounded original-conclusion driver qualification

The TRAIN-only source view preparation yields1,618 deduplicated natural tasks.
Before any scientific fit,qualify the actual128-position ordered-answer/EOS
loss,full recurrent gradients,DDP,optimizer and full-state recovery. This is a
finite software/learnability qualification,not an efficacy comparison.

Select32 TRAIN tasks by deterministic document hash with at most one per source
group. Use every task at each of three visible-work stages:all original work,
first half,and no work. Stages change after40 and80 updates. Predict only original
answer tokens with mean CE,plus0.1 times per-example task-EOS CE. EOS is artificial
format supervision and separately counted. All mathematical strings are exact
source substrings;no calculated target or generated work is used.

Both cases start from the same completed one-pass NTP seed5231 body at1.005B
natural targets. Remove only the closed auxiliary head;retain identical deployed
parameters and normal prediction path. Reset the optimizer identically. One case
always executes once;the other cycles4,1,2 terminal passes,synchronized across
workers,with full backpropagation. No student outcome selects tasks or depth.

Each eight-chip/32-rank case performs121 qualification updates on32 fixed TRAIN
examples,including100 changing-schedule warm updates,then separately restores
update120 and repeats121. Use the original qualified optimizer and conservative
1e-4 qualification schedule,FP16 body,explicit FP32 attention,FP32 heads/masters/
moments and manual8192 scaling. This is not a tuned scientific curriculum.

Require complete independent upstream FP32 global gradients at updates0/120,
all named families within2%,loss differences within0.003,and independent
observed-gradient optimizer update/moment checks. Verify exact saved model,
optimizer and per-rank RNG restoration and32-replica equality. Record whether
future trajectories match;do not require or assert bit-exact future training
where the native stack has not established that property.

Run both cases on all16 chips with a1,800-second overall cap,600-second native
phase caps and600-second CPU-audit caps. Preserve512GiB MemAvailable and35GiB
free on all output filesystems. Ordinary failure stops owned workers only.
Include failed attempts/compilation/CPU tails in receipts. No DEV,TEST,reserve,
terminal alpha or scientific state adoption follows automatically.

Training-task loss decrease is a learnability observation only. The proposed
actual experiment still requires ordinary NTP alongside the curriculum,paired
one-pass/recurrent controls,a longer one-pass work control,free generation and
fresh eligible evaluation groups. This continuation qualification cannot
establish learning the curriculum from random initialization or a broad NLL win.
