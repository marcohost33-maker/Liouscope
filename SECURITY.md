# Security Policy

## Supported versions

LiouScope is research software. Only the most recent minor version on `main` receives security
fixes. There is no LTS branch and no backport policy.

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a vulnerability

If you believe you have found a security-relevant issue, **please do not open a public issue**.

Use GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/marcohost33-maker/Liouscope/security) of this repo
2. Click **Report a vulnerability**
3. Provide:
   - A description of the issue
   - Steps to reproduce
   - The affected commit SHA or release tag
   - Suggested mitigation if you have one

We aim to acknowledge the report within **7 working days** and to either ship a patch or publish
an advisory within **30 days** of the initial report. If you do not receive a response, please
escalate through the GitHub profile contact information.

## Scope

In scope:

- Code under `src/liouscope/`
- CI workflows under `.github/workflows/`
- The published Python package on PyPI (when released)
- The run-manifest schema (`MANIFEST_SCHEMA.json`)

Out of scope:

- Upstream NumPy / SciPy / ARPACK / Python interpreter vulnerabilities (report to those projects
  directly)
- Findings that require physical access to the machine running LiouScope
- Issues in third-party forks
- Numerical-stability bugs that do not constitute a security vulnerability (open a normal issue)

## Supply-chain hardening

LiouScope follows the OpenSSF SHA-pinning recommendation: every external GitHub Action in
`.github/workflows/` is pinned to a 40-character commit SHA, with a `# v1.2.3` comment indicating
the semantic version. A regression test enforces this.

Python dependencies are intentionally minimal (`numpy`, `scipy`); transitive supply-chain risk is
audited via Dependabot security updates (`.github/dependabot.yml`).
