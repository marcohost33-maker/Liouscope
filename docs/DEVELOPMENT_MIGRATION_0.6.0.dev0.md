# Development migration: 0.6.0.dev0

Status: Unreleased development line.

The immutable v0.5.0 release remains the citable release. The default branch now reports 0.6.0.dev0 so VCS installations and run manifests do not reuse a released version identifier for materially different code.

## Ensemble evidence API

A caller-controlled Boolean is no longer accepted as evidence. The public API now requires an immutable `liouscope.EnsembleEvidence` value through `ensemble_evidence`.

The evidence contract validates and binds the evidence-manifest SHA-256, initial-state family and ordering parameter, paired run IDs and input hashes, relaxation metric, comparison test, uncertainty method, generating software version, gate status and reason code, and distinct producer and reviewer attestation SHA-256 digests.

Only a PASS gate with reason `ENSEMBLE_MPEMBA_CONFIRMED` can lift the single-state maximally-mixed A11 insufficient-evidence floor. The canonical evidence digest enters the run input hash and the complete payload is retained in `DiagnosticReport.extras`.

The object is a provenance contract, not proof of organizational independence or scientific correctness. Repository review and release promotion remain separate gates.

## Manifest migration

`MANIFEST_SCHEMA_VERSION` moved from 1.3.0 to 1.4.0 because the input-hash derivation domain now includes the canonical ensemble-evidence digest when structured evidence is supplied. Run IDs and input hashes must be compared only within the same manifest schema version. Existing 1.3.0 manifests remain valid historical records but do not re-derive under 1.4.0 when ensemble evidence participates.

`MANIFEST_SCHEMA_VERSION` then moves from 1.4.0 to 1.5.0 because `compute_input_hash` now absorbs each input object as a length-framed, type-tagged field instead of a bare `repr`/byte concatenation. The old encoding was not injective: distinct input tuples could collide when their serialised forms concatenated to the same byte stream (for example `compute_input_hash(12, 3)` and `compute_input_hash(1, 23)`; issue #97 item 4). Within `diagnose()` the fixed arity and types made this practically unreachable, but `compute_input_hash` is public API, so the derivation is hardened. As with every schema step, 1.4.0 run IDs / input hashes do not re-derive under 1.5.0; compare provenance keys only within one schema version.

## PyPI release workflow

Manual workflow dispatch is build and QA only and receives no OIDC token. The publish job runs only for a published GitHub Release and only when the repository publication flag is enabled. Before upload, both jobs verify source version, built-wheel version, release tag, checked-out commit and release event SHA.

## Compatibility

Calls that omit the former Boolean, or explicitly pass false, preserve single-state behavior. Passing true now fails closed with migration guidance. This intentional change belongs to the next development minor line and does not alter released v0.5.0.
