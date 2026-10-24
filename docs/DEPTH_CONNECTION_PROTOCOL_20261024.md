# Extra core execution beyond the trained inexpensive connection

Identity: `depth-connection-v1`. One-pair DEV mechanism experiment; no terminal
TEST, fresh significance, teacher, generated research data or RL.

The completed FP32-from-initialization seed419 comparison shows a0.032355-nat
gain from an inexpensive connection. Its cheap state is C(e)+e; full recurrence
applies C once more to that state before the same coda. The next question is
whether that additional16-block execution adds value, and whether it repays its
cost against strong connection/ordinary training on more text.

Reuse the completed ordinary/connection seed4191.005B endpoints and their exact
initial-weight identity. Train the existing fixed two-pass architecture in FP32
from the same random initialization and source order to update3840. No new
architecture, pass embedding, auxiliary loss, gate, teacher or normalization.
Keep150,436,608 parameters, context1024, global256 windows, data seed23,
stateless routing271828, qualified fused unscaled AdamW(.9,.95), matrix decay0.1,
no embedding/norm decay, clipping1, primitive SDPA and FP32 masters/moments.

Select fixed's peak LR with the same four candidates used by the two completed
references: {3e-4,6e-4,1e-3,1.5e-3}, seed11, update1152,2.5% token warmup,
last10% cosine decay, existing consumed TUNE. Select minimum weighted NLL,
preferring the lower LR within0.0002 nats. Publish the whole grid and independently
audit the selected TUNE model. Check initial digests against the completed
seed11 references. Main seed419 uses the selected peak, original25,122,189-token
warmup, decay from3456 to3840 and the same original natural targets. Its initial
digest must match the completed seed419 pair. Independent whole-DEV/state-copy
audits remain required; native/CPU NLL limit0.0002, local FP32 gradient gate2%.

Complete both strong resource controls: ordinary and connection to update6496,
1,699,926,288 targets. Restore each **stable update1152** checkpoint with all
weights, moments, RNG/cursor preserved. Keep its selected peak6e-4/1e-3 and
original warmup, extend stable training to5846 and decay through6496. Do not
resume the decayed1B endpoint. Their estimated matmuls match fixed1B within
0.0133%; count head/backward and report actual wall/chip costs separately.

Primary descriptive readouts: fixed minus connection performance at equal data;
fixed versus both completed-decay resource controls. Use all real DEV curve
targets with fixed60/20/20 web/math/code weights, one reused seed419. Do not
claim seed significance or final useful-computation success. Fresh confirmation
still needs at least five new paired seeds, new source groups and alpha spending.

Use ownership-checked isolated containers per native phase,160GiB each and
320GiB aggregate. Ordinary phase failures cannot stop unrelated campaigns;
shared host512GiB/filesystem35GiB emergencies retain managed global shutdown.
The cloned SDK/driver/library stack is unchanged and native32-rank execution
has passed full global-gradient checks in separate containers. Scientific
training remains the already qualified FP32 path; the FP16 candidate is not
adopted into this comparison.

Schedule the connection resource control alongside fixed LR selection, then
ordinary's resource control alongside fixed's main fit. Budget from the actual
100-warm-update FP32 rates162,899 fixed /176,468 ordinary,195s per training
launch,100s per readout,600s selected-TUNE audit,30% margin and30-minute CPU tail.
Freeze all source/parent/panel hashes and exact prefix counts before new updates.
Training cutoff23:00 UTC October24; hard cutoff00:00 UTC October25. Refuse a
late launch that no longer fits. Full checkpoints every1024 updates and at
phase ends. Preserve failures and charge all branches, tuning and CPU tails.
