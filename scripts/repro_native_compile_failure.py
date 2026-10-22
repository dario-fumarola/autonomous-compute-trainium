"""Check that an intentional compiler failure raises before a result is returned."""

import json
import os

# Use only in a separate test process and cache directory.
os.environ["NEURON_CC_FLAGS"] = "--intentional-invalid-option-for-failure-propagation-test"

import torch

expected_input = torch.arange(15, dtype=torch.float32).reshape(3, 5)
expected = expected_input @ expected_input.T
device_input = expected_input.to("neuron")
try:
    actual = (device_input @ device_input.T).cpu()
    torch.neuron.synchronize()
except RuntimeError as error:
    print(json.dumps({"status": "RAISED", "error": str(error)}), flush=True)
else:
    print(json.dumps({"status": "BUG_NO_EXCEPTION", "expected": expected.tolist(),
                      "actual": actual.tolist(), "max_abs_error": (actual-expected).abs().max().item()}), flush=True)
    raise SystemExit(1)
