# Primary-source follow-up and bounded software check

The [paper v2](https://arxiv.org/html/2510.00219v2) includes a150M–772M natural-text suite at2.5B tokens. AppendixC lists a150M baseline at4.8B tokens/loss3.27 and forking at2.4B/3.17. This is relevant to our exposure-control question. The1.9B recipe adds synthetic mid-training sources. Section5.1 separately evaluates autoregressive budget scaling. These results motivate inspection;they do not substitute for our conventional frontier or complete cost ledger.

The public PyTorch code inspected here is [commit7f4b853](https://github.com/stanfordnlp/thoughtbubbles/blob/7f4b853f370853c5b51bc784b0a5cb7dca601092/model.py),dated October2,2025,before the paper revision. This check does not reproduce the later JAX implementation or the reported model scores.

The exact release `ForkingBlock.fork` method is extracted by AST with its body unchanged. A fixed two-dimensional LayerNorm/Linear software fixture changes only later source-position inputs,holding the prefix embeddings,cumulative scores,budget and weights fixed. The earlier position receives3 streams in one case and1 in the other. Two prefix-only calls have exactly equal outputs. Four operator calls take1.718 seconds and1.692 CPU seconds,with zero model forwards,optimizer updates,research examples or accelerator chips. [Registration](registration.json) · [Result](report.json) · [Fixture](operator_fixture.py).

This proves suffix dependence of the checked full-block allocation operator. It does not measure a language-model quality effect,contradict recomputed-prefix inference or establish that all versions of the method are noncausal. A causal attention mask alone cannot guarantee prefix invariance of an earlier allocation selected by a full-block ranking. Any port must specify and test the complete causal allocation/cache behavior before its NLL is compared with ordinary8.

Future feasibility would need prefix/suffix invariance under the actual inference policy,output-mixture and head-work accounting,padded attention/gather costs,and independent gradient/state qualification. Our failed frozen uniform-depth selector does not bound jointly trained forked trajectories. No implementation port,scientific run,fresh reserve,TEST or alpha is admitted by this review. The active final-quarter comparison remains unchanged.

To reproduce the software fixture in a fresh directory with CPU PyTorch:

```bash
research_fixture_dir="$(mktemp -d)"
cp results/thoughtbubbles_review_20261026/operator_fixture.py "$research_fixture_dir/"
cp results/thoughtbubbles_review_20261026/registration.json "$research_fixture_dir/"
curl -fsSL https://raw.githubusercontent.com/stanfordnlp/thoughtbubbles/7f4b853f370853c5b51bc784b0a5cb7dca601092/model.py -o "$research_fixture_dir/upstream-model.py"
PYTHONDONTWRITEBYTECODE=1 python -B "$research_fixture_dir/operator_fixture.py"
```
