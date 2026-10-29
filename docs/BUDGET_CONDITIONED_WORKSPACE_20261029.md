# Remaining-budget cue: bounded mechanism diagnostic

The fixed-one replay fit recovers0.276900nats versus the sampled model's one-step
exit at exactly equal data and nominal matrix work. It still loses0.030125 to
equal-data ordinary and0.292091 to work-matched ordinary. This motivates testing
interference between supported depths; it does not establish its cause.
The separate TRAIN gradient-role decomposition finds nearly orthogonal,not
opposing embedding-role gradients. Untying is not justified by cancellation.

The proposed change supplies a learned512-dimensional cue to each proposal pass
according to the externally assigned remaining budget (one or two). The final
prediction pass is unshifted. The two cues start at zero,consume no RNG draws,
and add1024 parameters:62,801,408 total. All previous body/table initialization
is unchanged. No additional objective,target reweighting,teacher,source data,
router or policy gradient is introduced. This is supplied budget conditioning,
not autonomous allocation. The unchanged optimizer gives the2D cue matrix the
same0.1 weight decay as other matrices in a future scientific screen.

The initial function at each supported0/1/2 budget agrees exactly with the old
model; the zero-budget path remains identical even with learned cues. Shared
FP64 gradients at nonzero depth agree to summation-order roundoff (observed
5.6e-17),and zero-budget shared gradients/optimizer updates remain bit-identical.
The full Native scientific tolerances are unchanged. Cues receive a separate
gradient-family audit so their small size cannot hide a numerical discrepancy.

Two new qualifications must pass before a scientific registration:

1. All64 ranks: independent SDPA/CE gradients,AdamW updates/moments,replicas and
   separate-process model/optimizer/RNG replay on disposable small arrays.
2. Actual62.8M shape,batch4/context512: cold local comparisons at every supported
   depth,100 warm updates,exact next-update replay,and full-global mature CPU
   audit on256 natural TRAIN windows. Disposable weights never initialize science.

Each stage has its own fresh identity and one-hour controller budget; a failure
stops it. Keep512GiB host reserve,35GiB disk reserve,10000 inodes,all failed costs
and scientific states. The actual-shape artifact allowance is3GiB: two complete
states and two gradient captures consume less than2GiB before source/receipts.
No protected evaluation source is read by qualification.

If numerics pass,a separate frozen one-fit screen may compare the same sampled
0/1/2 schedule with the existing unconditioned sampled fit and fixed-one fit.
Retain the ordinary equal-data/equal-work and uniform controls. Reusing these
checkpoints is a controlled ablation,not seed replication. Do not replace the
stronger ordinary comparison with a favorable within-workspace comparison.
The scientific endpoint,budgets,control hashes and advancement tests must be
registered before training. No larger model,router or confirmation follows
automatically. The independent protected source reserve remains sealed.
