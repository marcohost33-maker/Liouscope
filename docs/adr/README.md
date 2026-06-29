# Architecture Decision Records (ADRs)

This directory holds LiouScope's Architecture Decision Records — short,
append-only documents that capture a significant decision, its context, and its
consequences. The format is a lightweight [MADR][madr]-style template.

ADRs are **immutable once Accepted**: to change a decision, add a new ADR that
supersedes the old one (update the `Superseded by:` / `Supersedes:` headers on
both), rather than rewriting history.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-python-support-policy.md) | Scientific-Python support policy (SPEC 0) | Accepted |

## Conventions

- Filename: `NNNN-kebab-case-title.md`, zero-padded sequential number.
- Required headers: `Status`, `Date`, `Deciders`, `Supersedes`, `Superseded by`.
- Status values: `Proposed` → `Accepted` / `Rejected` → `Superseded` / `Deprecated`.
- Keep each ADR to one decision. Cross-link related ADRs and the roadmap issue.

[madr]: https://adr.github.io/madr/
