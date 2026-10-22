# PyTorch and Trainium observations

Record issues only after reproducing them against a pinned version. Separate documented limits, configuration errors and backend defects. Keep a small local reproducer, exact environment, expected/actual behavior and affected training result for any contribution candidate. Do not submit upstream issues or PRs autonomously.

## Verified setup constraint

The [native quickstart](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/get-started/quickstart-torch-neuron.html) limits standalone wheels to Ubuntu 26.04. This host is Ubuntu 24.04 with glibc 2.39; the inspected CPython 3.12 native wheel requires GLIBC_2.43. That is a documented packaging constraint, not a newly discovered correctness bug. Use an isolated Ubuntu 26.04 container and preserve the host's working XLA environment.

The [native release notes](https://github.com/aws-neuron/torch-neuronx/blob/main/release-notes/2.33.0.md) specify static compiled shapes. Use fixed windows and masks; eager execution is for correctness and fallback discovery.

Upstream source: aws-neuron/torch-neuronx at 50b14171affaa365f1df6aa015232427336f6a4d. TorchTitan fork: aws-neuron/torchtitan-neuron at 3f447e28a4cf2f8ba939024bfdd427ee8084ee8c. The upstream [AI policy](https://github.com/aws-neuron/torch-neuronx/blob/main/AI_POLICY.md) requires human involvement in contributions. Local research code and reproducible evidence come first.
