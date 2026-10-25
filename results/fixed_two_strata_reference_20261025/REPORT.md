# Independent reference and position-loss reader parity

The existing One/NTP work-control model from seed7283 is fixed before the new
seed5231 learning-curve outcomes. It is used only to define descriptive
difficulty bins,never as a teacher or a replacement primary comparator.

Recomputing the full consumed DEV curve panel reproduces the original scalar
NLL **exactly:2.5813822563558504**,difference0.0 over603 documents and754,750
real targets. Model weights remain unchanged. The derived scorer retains the
same arithmetic and additionally writes private position losses and a private
contiguous document index. Only aggregates/hashes are public.

Actual CPU-only wall time195.35s,zero chips,zero learning updates,no TEST or
terminal alpha. It overlaps the separate TUNE reservation interval;do not add
that interval again. MAIN can reuse this completed reference without another
neural forward. [Exact parity](report.json) · [Frozen source/provenance](registration.json).
