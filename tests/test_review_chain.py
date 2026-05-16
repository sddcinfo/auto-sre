"""Tests for autosre.review.chain — provider chain executor."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from autosre.review import chain

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


class TestChainFromEnv:
    def test_unset_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AUTOSRE_REVIEW_CHAIN", raising=False)
        assert chain._chain_from_env() is None

    def test_empty_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOSRE_REVIEW_CHAIN", "")
        assert chain._chain_from_env() is None

    def test_single_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOSRE_REVIEW_CHAIN", "local")
        assert chain._chain_from_env() == ["local"]

    def test_multiple_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOSRE_REVIEW_CHAIN", "local,codex,claude")
        assert chain._chain_from_env() == ["local", "codex", "claude"]

    def test_drops_unknown_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOSRE_REVIEW_CHAIN", "local,bogus,codex")
        assert chain._chain_from_env() == ["local", "codex"]

    def test_all_invalid_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOSRE_REVIEW_CHAIN", "bogus,fake")
        assert chain._chain_from_env() is None

    def test_whitespace_handling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOSRE_REVIEW_CHAIN", " local , codex ")
        assert chain._chain_from_env() == ["local", "codex"]


class TestParseResponseJson:
    def test_empty_string(self) -> None:
        assert chain._parse_response_json("") == (None, None)

    def test_whitespace_only(self) -> None:
        assert chain._parse_response_json("   \n\t") == (None, None)

    def test_no_json(self) -> None:
        assert chain._parse_response_json("no braces here") == (None, None)

    def test_plain_json_with_findings(self) -> None:
        raw = '{"findings": [{"severity": "P0", "title": "bug"}], "questions": []}'
        findings, questions = chain._parse_response_json(raw)
        assert findings == [{"severity": "P0", "title": "bug"}]
        assert questions is None  # empty list → None

    def test_json_with_questions(self) -> None:
        raw = '{"findings": [], "questions": ["Is it idempotent?"]}'
        findings, questions = chain._parse_response_json(raw)
        assert findings is None
        assert questions == ["Is it idempotent?"]

    def test_json_wrapped_in_markdown(self) -> None:
        raw = '```json\n{"findings": [{"severity": "P2", "title": "minor"}]}\n```'
        findings, _ = chain._parse_response_json(raw)
        assert findings == [{"severity": "P2", "title": "minor"}]

    def test_json_with_narrative_prefix(self) -> None:
        raw = 'Here is my review:\n\n{"findings": [{"severity": "P1", "title": "x"}]}\n\nEnd.'
        findings, _ = chain._parse_response_json(raw)
        assert findings == [{"severity": "P1", "title": "x"}]

    def test_malformed_json(self) -> None:
        raw = '{"findings": [broken'
        assert chain._parse_response_json(raw) == (None, None)


class TestFormatFindings:
    def test_empty_findings(self) -> None:
        result = chain.ChainResult(
            provider="local",
            findings=None,
            questions=None,
            raw_output="",
            elapsed_seconds=1.0,
        )
        assert result.format_findings() == ""

    def test_single_p0(self) -> None:
        result = chain.ChainResult(
            provider="local",
            findings=[
                {
                    "severity": "P0",
                    "title": "SQL injection",
                    "description": "Unsanitized input",
                    "recommendation": "Use parameterized queries",
                },
            ],
            questions=None,
            raw_output="",
            elapsed_seconds=1.0,
        )
        out = result.format_findings()
        assert "local" in out
        assert "P0: SQL injection" in out
        assert "Unsanitized input" in out
        assert "parameterized queries" in out
        assert "1 P0" in out
        assert "BLOCKING" in out

    def test_mixed_severities(self) -> None:
        result = chain.ChainResult(
            provider="local",
            findings=[
                {"severity": "P0", "title": "crit"},
                {"severity": "P1", "title": "high"},
                {"severity": "P1", "title": "high2"},
                {"severity": "P2", "title": "med"},
            ],
            questions=None,
            raw_output="",
            elapsed_seconds=1.0,
        )
        out = result.format_findings()
        assert "1 P0" in out
        assert "2 P1" in out
        assert "1 P2" in out

    def test_has_findings_property(self) -> None:
        empty = chain.ChainResult(
            provider="",
            findings=None,
            questions=None,
            raw_output="",
            elapsed_seconds=0.0,
        )
        assert not empty.has_findings

        populated = chain.ChainResult(
            provider="local",
            findings=[{"severity": "P0", "title": "t"}],
            questions=None,
            raw_output="",
            elapsed_seconds=0.0,
        )
        assert populated.has_findings


def _mk_completed(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["dummy"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class TestRunChain:
    @patch("autosre.review.chain.shutil.which", return_value="/usr/bin/codex")
    @patch("autosre.review.chain.subprocess.run")
    def test_first_provider_succeeds(
        self,
        mock_run: MagicMock,
        _mock_which: MagicMock,
    ) -> None:
        mock_run.return_value = _mk_completed(
            stdout='{"findings": [{"severity": "P1", "title": "x", "description": "d"}]}',
        )
        result = chain.run_chain("prompt", chain=["codex"])
        assert result.provider == "codex"
        assert result.findings == [{"severity": "P1", "title": "x", "description": "d"}]
        assert len(result.attempts) == 1
        assert result.attempts[0].success

    @patch("autosre.review.chain.shutil.which", return_value="/usr/bin/test")
    @patch("autosre.review.chain.subprocess.run")
    def test_fallback_on_first_failure(
        self,
        mock_run: MagicMock,
        _mock_which: MagicMock,
    ) -> None:
        # First call (codex) fails; second (gemini) succeeds.
        mock_run.side_effect = [
            _mk_completed(stderr="codex down", returncode=1),
            _mk_completed(stdout='{"findings": [{"severity": "P2", "title": "t"}]}'),
        ]
        result = chain.run_chain("prompt", chain=["codex", "gemini"])
        assert result.provider == "gemini"
        assert len(result.attempts) == 2
        assert not result.attempts[0].success
        assert result.attempts[1].success

    @patch("autosre.review.chain.shutil.which", return_value="/usr/bin/test")
    @patch("autosre.review.chain.subprocess.run")
    def test_all_providers_fail(
        self,
        mock_run: MagicMock,
        _mock_which: MagicMock,
    ) -> None:
        mock_run.return_value = _mk_completed(stderr="broken", returncode=1)
        result = chain.run_chain("prompt", chain=["codex", "gemini", "claude"])
        assert result.provider == ""
        assert result.findings is None
        assert len(result.attempts) == 3
        assert all(not a.success for a in result.attempts)

    @patch("autosre.review.chain.shutil.which", return_value=None)
    def test_provider_not_on_path_is_skipped(self, _mock_which: MagicMock) -> None:
        result = chain.run_chain("prompt", chain=["codex"])
        assert result.provider == ""
        assert len(result.attempts) == 1
        assert result.attempts[0].error == "not on PATH"

    def test_unknown_provider_name_is_skipped(self) -> None:
        result = chain.run_chain("prompt", chain=["bogus"])
        assert result.provider == ""
        assert len(result.attempts) == 1
        assert result.attempts[0].error is not None
        assert "unknown provider" in result.attempts[0].error

    @patch("autosre.review.chain.shutil.which", return_value="/usr/bin/test")
    @patch("autosre.review.chain.subprocess.run")
    def test_timeout_recorded_as_failure(
        self,
        mock_run: MagicMock,
        _mock_which: MagicMock,
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=30)
        result = chain.run_chain("prompt", chain=["codex"])
        assert result.provider == ""
        assert len(result.attempts) == 1
        assert result.attempts[0].error is not None
        assert "timed out" in result.attempts[0].error


class TestProviderCmds:
    def test_local_entry_exists(self) -> None:
        assert "local" in chain.PROVIDER_CMDS
        cmd = chain.PROVIDER_CMDS["local"]
        # First element is sys.executable — always present
        assert len(cmd) >= 3
        assert cmd[1] == "-m"
        assert cmd[2] == "autosre.review._local_provider_runner"

    def test_upstream_entries_present(self) -> None:
        assert "codex" in chain.PROVIDER_CMDS
        assert "gemini" in chain.PROVIDER_CMDS
        assert "claude" in chain.PROVIDER_CMDS

    def test_default_chains_plan_leads_with_codex(self) -> None:
        # codex (gpt-5.4 xhigh) is primary; local is offline fallback.
        assert chain.DEFAULT_CHAINS["plan"][0] == "codex"
        assert "local" in chain.DEFAULT_CHAINS["plan"]

    def test_codex_wired_for_gpt54_xhigh(self) -> None:
        cmd = chain.PROVIDER_CMDS["codex"]
        assert cmd[0:2] == ["codex", "exec"]
        # Model + reasoning effort are set via -c overrides so the review
        # pipeline doesn't depend on the user's ~/.codex/config.toml.
        assert 'model="gpt-5.4"' in cmd
        assert 'model_reasoning_effort="xhigh"' in cmd
        # --skip-git-repo-check so the review can run from any directory.
        assert "--skip-git-repo-check" in cmd

    def test_codex_timeout_generous(self) -> None:
        # gpt-5.4 xhigh on a plan can take 5-15 min; upstream 600s is too tight.
        assert chain.PROVIDER_TIMEOUTS["codex"] >= 900

    def test_max_chain_seconds_accommodates_codex(self) -> None:
        # MAX_CHAIN_SECONDS must exceed codex timeout + some margin for fallbacks.
        codex_budget = chain.PROVIDER_TIMEOUTS["codex"] + 120
        assert codex_budget <= chain.MAX_CHAIN_SECONDS
