# Security Policy

LiouScope is a scientific Python package. This document describes how to
report security issues and which versions receive security support.

## Supported versions

| Version | Status       |
|---------|--------------|
| 0.2.x   | Supported    |
| < 0.2.0 | Not supported |

Security fixes ship in patch releases on the latest minor (currently 0.2.x).
Backports to earlier minors are not provided.

## Reporting a vulnerability

If you discover a security issue, **please do not open a public GitHub
issue**. Instead, use one of the private channels:

- GitHub Security Advisories: open a private advisory at
  <https://github.com/marcohost33-maker/Liouscope/security/advisories/new>.
- Email: report to the maintainer listed in `CITATION.cff` with the subject
  prefix `[liouscope-security]`.

We aim to acknowledge reports within 5 business days and to publish a
coordinated patch within 30 days for confirmed issues.

## Scope

The following classes of issues are in scope:

- Arbitrary code execution triggered by data input to public APIs
  (`diagnose`, `build_liouvillian`, `liouscope.io.export.load_report`).
- Deserialisation vulnerabilities in IO helpers.
- Supply-chain integrity (tampered wheels, malicious dependency injection).
- Insecure default configurations.

Out of scope:

- Mathematical correctness reports (file a regular issue or PR).
- Performance issues (file a regular issue).
- Findings reachable only by passing already-trusted Python objects
  containing executable code.

## Supply-chain controls

- The release pipeline uses PyPI Trusted Publishing (OpenID Connect,
  short-lived tokens; no long-lived API tokens). See
  `.github/workflows/pypi.yml`.
- Dependency changes in pull requests are scanned via the GitHub
  `dependency-review-action`. See `.github/workflows/dependency-review.yml`.
- The repository is monitored continuously by OpenSSF Scorecard.
  See `.github/workflows/scorecard.yml`.

## Reproducibility as a security property

LiouScope emits a SHA-256 run-id on every `diagnose()` call (see
`REPRODUCIBILITY.md`). If a published result cannot be reproduced bit-by-bit
from a pinned release, treat that as a security-impacting integrity issue
and report through the private channels above.
