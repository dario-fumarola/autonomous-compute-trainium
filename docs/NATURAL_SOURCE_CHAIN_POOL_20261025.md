# Reconstruct a larger source-chain candidate pool

The frozen initial checker found143 source-authored chains in20,000 accepted
TRAIN documents. Before any student score,build a larger feasibility pool from
all accepted TRAIN documents in the same six raw shards18–23. Do not change
arithmetic,eligibility,first-chain selection or source-span rules to obtain
more candidates. No DEV,TEST or reserve data is opened.

The accelerated extractor replaces repeated suffix searches with one maximal
math-token scan per document. Its arithmetic checker is unchanged. Require
exactly identical candidate identities,source spans and fragments on the entire
original20,000-document cohort,including every original negative. Stop if this
gate fails;do not silently substitute the faster rule.

Use four CPU processes and a1,200-second cap,with the existing512GiB memory and
35GiB filesystem reserves. Match accepted URL/content-hash/source-group records
against pinned raw documents before extracting any candidate. A calculator
only verifies existing source equalities;its values are never written as
answers or steps. Keep all source fragments private.

Record an exact arithmetic-AST grouping with original literal values,so TeX,
parenthesis and decimal-format variants do not masquerade as different tasks.
Also record a number-blinded operation-tree group for later composition audits.
These are grouping metadata,not rewritten training examples. Preserve all raw
candidate records;do not choose favorable examples using a student's score.

This still produces no training views or curriculum. Inspect source candidates,
freeze normalization,stopping rules,duplication exclusions,source-group splits
and a verifier before admitting any student run. The numeric-expression task
is narrower than natural word-problem understanding or broad text prediction;
the separate word-problem audit remains partial. Any from-scratch,latent-
computation or efficiency claim still needs its declared controls and fresh
confirmation. No new model,teacher,RL,synthetic research corpus or alpha use.
