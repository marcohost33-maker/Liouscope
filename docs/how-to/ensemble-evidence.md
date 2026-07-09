# How to supply EnsembleEvidence for A11 confirmation

**Goal:** let a diagnostic run claim more than `UNDEFINED` for the A11
(non-normal quantum-Mpemba) mechanism when the steady state is maximally
mixed.

## Why there is a floor at all

A Mpemba statement is inherently *comparative*: "the initially-farther state
relaxes faster" only means something relative to a reference family of
initial states. A single-state run on a maximally mixed steady state
($\rho_{ss} = I/d$) cannot carry that comparison, so LiouScope reports A11 as
**UNDEFINED / EXPLORATION** in that situation — by design (issue #78,
decision E0706-13). A bare caller assertion is not evidence:

```python
lp.diagnose(L, ensemble_confirmation=True)   # raises — fail-closed by contract
```

## Construct the evidence object

The only thing that lifts the floor is a validated, immutable
`liouscope.EnsembleEvidence` describing a passing reference-family
comparison. All fields are mandatory and validated at construction:

```python
import liouscope as lp

evidence = lp.EnsembleEvidence(
    manifest_sha256="<64-hex digest of the evidence manifest>",
    initial_state_family="haar-pure-d4",
    ordering_parameter="trace-distance-to-steady-state",
    run_ids=("<64-hex run id A>", "<64-hex run id B>"),      # >= 2, unique
    input_hashes=("<64-hex input hash A>", "<64-hex input hash B>"),
    relaxation_metric="trace_distance",
    comparison_test="paired-crossing-time",
    uncertainty_method="bca-bootstrap-B1000",
    software_version=lp.__version__,
    gate_status="PASS",
    reason_code=lp.ENSEMBLE_MPEMBA_CONFIRMED,
    producer_attestation_sha256="<64-hex digest>",
    reviewer_attestation_sha256="<64-hex distinct digest>",
)

report = lp.diagnose(L, rho_initial=rho0, ensemble_evidence=evidence, seed=42)
```

Construction fails closed on malformed digests, duplicate run IDs, fewer than
two reference runs, identical producer/reviewer attestations, or an unknown
`gate_status`.

## What the gate actually checks

Only one combination suppresses the floor:

```python
evidence.permits_claim_floor_override
# True  iff  gate_status == "PASS" and reason_code == ENSEMBLE_MPEMBA_CONFIRMED
```

`FAIL`, `REVIEW`, any other reason code, or missing evidence keep the floor in
place. The floor is a *floor*, not a switch: it is never weakened to
`CANDIDATE` without evidence and never strengthened to an exclusion.

## Provenance guarantees

When evidence participates in a run:

- its canonical SHA-256 (`evidence.sha256`) enters the run `input_hash`, so a
  manifest from an evidence-backed run is distinguishable from a bare run;
- the full payload is preserved in `DiagnosticReport.extras` and remains
  JSON-serialisable, so the claim can be audited from the report alone.

One honest limitation: the object binds the claim to digests, runs and two
distinct attestations, but it cannot prove *organisational* independence of
producer and reviewer — that remains a repository-review and release-policy
control, outside the type system.
