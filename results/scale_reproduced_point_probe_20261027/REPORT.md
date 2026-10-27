# Reproduced-point precision diagnosis — complete

On the same unchanged ineligible update-62 state, exact 512 source windows and identical Native-generated recurrent noise, both manual loss scales (65536 and 8192) produce nonfinite forward NLL and gradients. Zero optimizer updates occur. Every saved parameter and buffer is finite. The independent pinned upstream FP32 reference has finite NLL 8.126648281332095 and finite full gradients (norm 2.6577374316097524).

Loss scaling alone does not resolve this reproduced failure: the forward computation is already nonfinite. This points toward a forward precision/range problem; it does not identify a vendor defect or certify any repaired training history. The original failed state remains unavailable. Native-generated noise is reused byte for byte in the CPU reference; host and Native Torch random generators are not assumed equivalent.

All observations, costs and original deadlines remain preserved. Public JSON renders nonfinite numerical fields as null; the accompanying finiteness flags distinguish them from absent measurements. The raw private reports remain unchanged. See [receipt](report.json) and [registration](plan.json).
