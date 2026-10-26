# First CPU check retained; corrected fixture uses a new preparation

The actual four-window random/mature model comparisons pass: removing the unused
head preserves ordered loss, every prediction gradient and the checked optimizer
updates exactly. This attempt takes67.279664 CPU seconds, with zero native chips
and no scientific state adoption.

A separate inventory unit fixture fails because it used the library default
32768-token vocabulary while asserting the production32000-token inventory.
The model check itself uses the actual production configuration. Preserve that
failed unit test and source. A new preparation reruns the CPU checks after the
fixture explicitly selects32000. This first manifest is retired from admission;
no native worker used it.

[Original CPU receipt](report.json) · [Fixture failure](test-fixture-failure.json) ·
[Original fixture](test-fixture-before-correction.txt) · [Replacement](../preparation-v2/REPORT.md).
