# Closed-snapshot bytecode cleanup

Removed 4,219 regenerable CPython cache files, totaling 34,046,482 bytes, from
122 completed snapshots. All 3,856 referenced source files and their original
launch/closure receipts verify unchanged. Free file entries increase from
16,626 to 21,089. Application takes 2.266 CPU wall seconds; no model, optimizer,
source or compiler-cache file is removed. [Receipt](applied.json).
