# Before-curriculum answer baseline

Existing One/NTP and R/NTP models,for both5231/7283 seeds,are read without updates
on the135-task new discovery panel. One uses its trained one-pass path;R uses
predeclared supported depths1,2,4. Every path produces0/135 valid terminated
scalar answers and0/135 EOS terminations within the fixed24-token budget.
First-answer-token accuracy is at most3/135. Private outputs mostly repeat task
formatting. This indicates an unlearned interface,not demonstrated inability to
perform arithmetic. All later relevant controls must receive the same format
training;absolute gains from this baseline cannot identify recurrence's value.

One answer NLL is3.379587/3.371680. R at depth2 scores3.476326/3.447029. These
new-task answer losses are different from broad language-model NLL. Additional
depth is not a remedy for this untrained format. No training or inference-cost
superiority claim is made.

All four independent whole-panel FP32 audits pass;maximum aggregate native/CPU
difference0.000398 nats. All prediction weights and replicas remain unchanged.
Free generation uses only visible prompt tokens and prefix-derived latent noise;
no true answer length or hidden work controls inputs,budget or stopping. EOS and
the complete generated scalar are required by the frozen criterion. Teacher-
forced answer scoring occurs after generation. Full-window uncached generation
and cold compilation are charged;these are not production decoding benchmarks.

The new panel is now consumed discovery DEV. No TEST or terminal alpha. This
readout establishes the starting interface/capability for a future controlled
curriculum;it is not a positive result for the research objective.
