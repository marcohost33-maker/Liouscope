#!/usr/bin/env python3
"""Regression tests for the agent/stacked-PR workflow contract parser."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import check_agent_branch_triggers as contract


class TriggerContractParserTests(unittest.TestCase):
    def fixture(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8")
        tmp.write(text)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        return Path(tmp.name)

    def test_comment_cannot_hide_pr_branch_filter(self) -> None:
        path = self.fixture(
            "on:\n  pull_request:\n  # comment at trigger indentation\n"
            "    branches: [main]\n"
        )
        self.assertFalse(contract._pull_request_is_unfiltered(path))

    def test_path_and_type_filters_are_not_unfiltered(self) -> None:
        for filter_line in ("paths: ['src/**']", "paths-ignore: ['docs/**']", "types: [opened]"):
            with self.subTest(filter_line=filter_line):
                path = self.fixture(f"on:\n  pull_request:\n    {filter_line}\n")
                self.assertFalse(contract._pull_request_is_unfiltered(path))

    def test_head_ref_text_outside_checkout_does_not_count(self) -> None:
        path = self.fixture(
            "jobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n"
            f"      # ref: {contract.PR_HEAD_REF}\n"
            "      - uses: actions/checkout@deadbeef\n"
            "        with:\n          persist-credentials: false\n"
        )
        self.assertEqual(contract._checkout_ref_values(path), [None])
        self.assertFalse(contract._checks_out_exact_pr_head(path))

    def test_named_checkout_ref_is_bound_to_checkout_step(self) -> None:
        path = self.fixture(
            "jobs:\n  test:\n    steps:\n      - name: Checkout\n"
            "        uses: actions/checkout@deadbeef\n        with:\n"
            f"          ref: {contract.PR_HEAD_REF}\n"
        )
        self.assertEqual(contract._checkout_ref_values(path), [contract.PR_HEAD_REF])
        self.assertTrue(contract._checks_out_exact_pr_head(path))

    def test_job_level_uses_ignores_comments_and_step_payloads(self) -> None:
        path = self.fixture(
            "jobs:\n  smoke:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      # uses: ./.github/workflows/ci-python-local.yml\n"
            "      - run: echo 'uses: ./.github/workflows/ci-python-local.yml'\n"
        )
        self.assertEqual(contract._job_level_uses_values(path), [])


if __name__ == "__main__":
    unittest.main()
