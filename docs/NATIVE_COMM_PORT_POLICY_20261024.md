# Explicit per-phase native collective ports

The first new-driver integration attempt fails before model initialization or
any optimizer update: the root collective listener reports an address already
in use. Preserve the failed source, plan and receipts. The installed backend
chooses a free ephemeral port by bind(0), reads the port and closes that socket
before NRT binds it. That creates a possible allocation race. The log establishes
the bind collision, not which other connection acquired the port.

Use the supported per-process NEURON_RT_ROOT_COMM_ID configuration on this
single instance. Register a distinct loopback root port for every native phase,
outside the host kernel's ephemeral range. A persistent advisory-lock-protected
registry prevents reuse by managed jobs, including sequential phases with old
connections in TIME_WAIT. Probe availability before registering; never change
system-wide networking, host drivers, SDK or XLA. The interval between probing
and NRT binding is not atomically protected from unrelated applications; choosing
outside ephemeral allocation removes the observed automatic-allocation hazard.

The worker checks that the port belongs to its exact registered output identity
before initializing collectives. Initial, resumed and short-branch phases use
different ports. All model, loss, data, optimizer, numerical gates and budgets
are unchanged. Driver integration revision2 has a new identity and source
freeze; it does not silently retry the stopped execution or claim a model fix.

[Official runtime variable reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/api/nrt-env-var-reference.html) ·
[Runtime developer guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.0/neuron-runtime/guides/nrt-developer-guide.html).

This is a local operational correction and potential upstream rendezvous
improvement. No issue, patch or message was submitted upstream.
