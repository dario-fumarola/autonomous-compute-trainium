# What the earlier branch screen directly supervised

The TRAIN-only audit exactly reproduces the saved endpoint's target-selection
RNG:8192 views,159 possible natural targets/view,32 selected targets/view,
262144 selected labels.8177/8192 decoded views round-trip exactly through the
pinned tokenizer;the15 mismatches are excluded from span claims.

The strict arithmetic-chain parser finds two source-authored chains. Only one
conclusion token and two intermediate-step tokens were selected during training.
The strict whole-fragment Python parser accepts27 views,finds four unambiguous
uses,and maps two target tokens;neither was selected.8150 fragments fail Python
parsing,including non-code text and incomplete code fragments.

These are narrow lower bounds,not a census of all computation opportunities.
They do not prove that other tokens are unpredictable or that the negative
branch results were caused by missing supervision. They do show that the old
small,short-window screen offered very little direct supervision of these
particular verified procedures. No model score,held-out read,optimizer update
or accelerator work occurs. CPU duration5.01seconds. Original results unchanged.
