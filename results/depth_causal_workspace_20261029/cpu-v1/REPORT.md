# Exact depth-causal software reference passes

All871CPU tests pass, with two skips, in156.78seconds. Thirty new checks cover
full-mask/cached/true sequential output and full-gradient agreement, exact
lower-stage invariance, future-token isolation, ordinary-path optimizer parity,
GQA, explicit attention, independent qualification factories and the matrix
ledger. A deliberately detached cache preserves forward scores but fails the
expected gradient-path check. Test arrays are software fixtures only.

The model retains51,747,328parameters and the original initializer. It uses
one exact body execution per stage, reusing differentiable earlier-stage KV.
No router, learned budget embedding, new target weighting or solver is present.
The prospective mixed0/1/2schedule would require49,866,412,847,529,984training
matrix FLOPs for the original128Mtarget schedule:1.138times strongest ordinary,
versus4.454times for the previous persistent mixed-budget graph. These are
executed-matmul ledger values, not measured speed or predictive efficiency.

No scientific fit is admitted. The preceding exact learning readout must close
before the separately registered disposable actual-shape qualification.
Both fresh reserves remain unscored.

[Checks and source hashes](report.json) ·
[Reference rationale](../../../docs/DEPTH_CAUSAL_WORKSPACE_REFERENCE_20261029.md) ·
[Prospective qualification](../../../docs/DEPTH_CAUSAL_QUALIFICATION_20261029.md).
