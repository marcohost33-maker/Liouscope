# CLAUDE.md — LiouScope

This file exists because Claude Code does not natively read `AGENTS.md`
([anthropics/claude-code#6235](https://github.com/anthropics/claude-code/issues/6235)).
The single-source-of-truth for agent instructions is `AGENTS.md` at the
repository root.

@AGENTS.md

---

## Claude-Code-specific notes (in addition to AGENTS.md)

- When using Claude Code's hierarchical layering, this file is the project-level
  layer. Path-scoped overrides (e.g. for a `benchmarks/` deep-dive) belong in a
  sibling `AGENTS.md` inside that directory, not here.
- The 2026-05-16 incident is summarised in AGENTS.md (Working
  agreements §2). For Claude Code's branch / history operations, re-read that
  section before any `git push --force`, branch delete, or ref-PATCH.
- For tool-permission preferences specific to Claude Code, see
  `~/.claude/settings.json` (user-global) — do not commit machine-specific
  permissions here.

---

*Thin import layer. Do not duplicate AGENTS.md content here — edit AGENTS.md instead.*
