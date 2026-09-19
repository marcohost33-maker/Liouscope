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


    def test_block_scalar_cannot_impersonate_checkout(self) -> None:
        path = self.fixture(
            "jobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - run: |\n"
            "          uses: actions/checkout@deadbeef\n"
            "          with:\n"
            f"            ref: {contract.PR_HEAD_REF}\n"
        )
        self.assertEqual(contract._checkout_ref_values(path), [])

    def test_required_merge_job_cannot_be_replaced_by_skipped_job(self) -> None:
        path = self.fixture(
            "jobs:\n"
            "  test:\n"
            "    if: ${{ github.event_name == 'push' }}\n"
            "    steps:\n"
            "      - uses: actions/checkout@deadbeef\n"
            "        with:\n          persist-credentials: false\n"
            "  test-head:\n"
            f"    if: {contract.PR_ONLY_IF}\n"
            "    steps:\n"
            "      - uses: actions/checkout@deadbeef\n"
            f"        with:\n          ref: {contract.PR_HEAD_REF}\n"
        )
        old = contract.EVIDENCE_JOBS.get(path.name)
        contract.EVIDENCE_JOBS[path.name] = ("test", "test-head")
        self.addCleanup(
            lambda: contract.EVIDENCE_JOBS.__setitem__(path.name, old)
            if old is not None
            else contract.EVIDENCE_JOBS.pop(path.name, None)
        )
        errors = contract._evidence_job_errors(path)
        self.assertTrue(any("required job must not be conditional" in e for e in errors))

    def test_job_parser_binds_default_and_head_checkout_to_distinct_jobs(self) -> None:
        path = self.fixture(
            "jobs:\n"
            "  test:\n"
            "    steps:\n"
            "      - uses: actions/checkout@deadbeef\n"
            "        with:\n          persist-credentials: false\n"
            "  test-head:\n"
            f"    if: {contract.PR_ONLY_IF}\n"
            "    steps:\n"
            "      - uses: actions/checkout@deadbeef\n"
            f"        with:\n          ref: {contract.PR_HEAD_REF}\n"
        )
        specs = contract._job_specs(path)
        self.assertEqual(specs["test"]["refs"], [None])
        self.assertEqual(specs["test-head"]["refs"], [contract.PR_HEAD_REF])


if __name__ == "__main__":
    unittest.main()
