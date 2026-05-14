Floquet roadmap
===============

The Floquet extension is **out of scope** for v0.2.0. Its design is recorded
in ``ROADMAP_FLOQUET.md`` at the repository root and summarised here.

Scope
-----

Periodic Liouvillian ``L(t + T) = L(t)`` -- extension of the GKSL / QMS
core to periodically driven open quantum systems.

Diagnostics extension (planned FD1-FD5)
---------------------------------------

* **FD1** -- Floquet gap (effective generator eigenvalue gap)
* **FD2** -- Stroboscopic slowdown score
* **FD3** -- Branch-sensitivity index
* **FD4** -- Micromotion norm
* **FD5** -- CPTP-violation proximity

Hard exclusions
---------------

* No blind Magnus-as-Lindbladian claims (Wolf, Eisert, Cubitt, Cirac,
  PRL 101, 150402 (2008)).
* No universal Markovianity claim for periodically driven systems.
* No thermodynamic-limit theorems.
* Non-Markovian extensions remain out of scope.

Entry criteria
--------------

Floquet work begins only after:

1. Core v0.2.0 public release completed.
2. All P0 evidence locks resolved.
3. Peer-review feedback incorporated.
4. Core test suite remains green.
