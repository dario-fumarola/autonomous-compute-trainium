# Post-pilot training/generalization diagnostic

The fixed pilot is complete and its primary result is negative:4/135 R answers
versus2/135 One,with no same-R depth1→2 accuracy gain. This diagnostic cannot
replace that held-out result,change any fitted model or make a terminal claim.

Before reading TRAIN predictions,select256 original TRAIN problems from the
same1,618-problem curriculum pool,by SHA256('chain-train-diagnostic-v1:'+ID),
with at most2 per source group. Preserve all original no-work prompts/answers.
No generated examples or labels,no hyperparameter change,no fit. Evaluate the
existing cooled One/bridge,R/bridge and longer One/bridge-work checkpoints.
Use identical prefix-only generation,noise,EOS/scalar verification,24-token
budget and default/secondary depths as the held-out readout. Independently
check full-panel answer NLL in FP32.

Compare generated-answer accuracy and answer NLL with the already frozen DEV
results. A large TRAIN/DEV gap supports a generalization problem for this
finite recipe;low TRAIN free accuracy despite low logged CE suggests inspecting
the loss/generation gap. Neither establishes a unique causal explanation or
validates scaling. Explicitly label all these examples TRAIN in worker/CPU/
public reports. Do not tune on them or reinterpret them as held-out evidence.

Use all16 chips as two owned eight-chip jobs when ready,under1,800 seconds,
600-second native/CPU phases,512GiB MemAvailable,35GiB free and10,000 inode
reserves. Preserve all model/source artifacts and costs. No optimizer updates,
TEST targets,reserve panels or terminal alpha.
