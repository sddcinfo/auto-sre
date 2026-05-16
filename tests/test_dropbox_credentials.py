"""Tests for autosre.dropbox.credentials — no literal passwords in argv."""

from __future__ import annotations

import io
import os
import stat
from pathlib import Path

import pytest

from autosre.dropbox.credentials import (
    MIN_PASSWORD_LENGTH,
    PasswordError,
    read_password_file,
    read_stdin,
    resolve_password,
    verify_password_file_mode,
    write_password_file,
)

GOOD_PW = "TestDropbox12"  # 13 chars, over the 12 minimum


class TestWritePasswordFile:
    def test_creates_file_with_mode_0600(self, tmp_path: Path) -> None:
        target = tmp_path / "pw"
        write_password_file(target, GOOD_PW)

        assert target.exists()
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert target.owner() == Path.home().owner()

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "pw"
        write_password_file(target, "FirstPass1234")
        write_password_file(target, "SecondPass567")
        assert "SecondPass567" in target.read_text()
        assert "FirstPass1234" not in target.read_text()


class TestReadPasswordFile:
    def test_accepts_0600_file(self, tmp_path: Path) -> None:
        target = tmp_path / "pw"
        write_password_file(target, GOOD_PW)

        assert read_password_file(target) == GOOD_PW

    def test_rejects_permissive_mode(self, tmp_path: Path) -> None:
        target = tmp_path / "pw"
        write_password_file(target, GOOD_PW)
        target.chmod(0o644)

        with pytest.raises(PasswordError, match="permissive mode"):
            read_password_file(target)

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(PasswordError, match="not found"):
            read_password_file(tmp_path / "nope")

    def test_accepts_bare_password(self, tmp_path: Path) -> None:
        target = tmp_path / "pw"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.write(fd, b"BarePassw0rd1234\n")
        os.close(fd)

        assert read_password_file(target) == "BarePassw0rd1234"

    def test_rejects_too_short(self, tmp_path: Path) -> None:
        target = tmp_path / "pw"
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.write(fd, b"short\n")
        os.close(fd)

        with pytest.raises(PasswordError, match="at least"):
            read_password_file(target)


class TestReadStdin:
    def test_reads_one_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO(f"{GOOD_PW}\n"))

        assert read_stdin() == GOOD_PW

    def test_strips_trailing_newline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO(f"{GOOD_PW}\n"))

        result = read_stdin()

        assert "\n" not in result

    def test_rejects_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO(""))

        with pytest.raises(PasswordError, match="no input"):
            read_stdin()

    def test_rejects_short(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("short\n"))

        with pytest.raises(PasswordError, match="at least"):
            read_stdin()


class TestResolvePassword:
    def test_stdin_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO(f"{GOOD_PW}\n"))

        assert resolve_password(from_stdin=True, from_file=None) == GOOD_PW

    def test_file_path(self, tmp_path: Path) -> None:
        target = tmp_path / "pw"
        write_password_file(target, GOOD_PW)

        assert resolve_password(from_stdin=False, from_file=target) == GOOD_PW

    def test_rejects_both(self, tmp_path: Path) -> None:
        target = tmp_path / "pw"
        write_password_file(target, GOOD_PW)

        with pytest.raises(PasswordError, match="pick one"):
            resolve_password(from_stdin=True, from_file=target)


class TestVerifyPasswordFileMode:
    def test_ok_on_0600(self, tmp_path: Path) -> None:
        target = tmp_path / "pw"
        write_password_file(target, GOOD_PW)

        ok, detail = verify_password_file_mode(target)

        assert ok
        assert "0o600" in detail or "600" in detail

    def test_flags_loose_mode(self, tmp_path: Path) -> None:
        target = tmp_path / "pw"
        write_password_file(target, GOOD_PW)
        target.chmod(0o644)

        ok, detail = verify_password_file_mode(target)

        assert not ok
        assert "expected 0600" in detail

    def test_flags_missing(self, tmp_path: Path) -> None:
        ok, detail = verify_password_file_mode(tmp_path / "nope")

        assert not ok
        assert "missing" in detail


def test_min_length_matches_filebrowser() -> None:
    """Sanity check — filebrowser enforces 12-char minimum, we must match."""
    assert MIN_PASSWORD_LENGTH == 12
