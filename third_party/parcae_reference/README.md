Pinned MIT-licensed upstream injection operator for CPU value/gradient checks:
[original file](https://github.com/sandyresearch/parcae/blob/69284c13746e849104f738d6d1a347b1f457df76/parcae_lm/modules/injection.py).
Copyright and full license are preserved in LICENSE. Provenance records the
connector-retrieved Git objects. No upstream package or accelerator kernels are
installed. The production trainer does not import this reference module.
Upstream trailing whitespace is preserved so the reference stays byte-identical.

The byte-identical [released optimizer](https://github.com/sandyresearch/parcae/blob/69284c13746e849104f738d6d1a347b1f457df76/recpre/optim/muon_adamw.py)
is also preserved as `muon_adamw.py` for independent CPU update/state checks.
It is a reference fixture; the native optimizer candidate does not import it.
