"""Tests for autosre.dropbox.proxy — HMAC sign/verify + helper parsing."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from autosre.dropbox import proxy
from autosre.dropbox.config import DropboxConfig

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def config_with_tmp_password(tmp_path: Path) -> DropboxConfig:
    """A DropboxConfig whose config_dir is a tmp path with a password file."""
    from autosre.dropbox.credentials import write_password_file

    cfg = DropboxConfig(
        data_dir=tmp_path,
        files_dir=tmp_path / "files",
        config_dir=tmp_path / "config",
        tls_dir=tmp_path / "tls",
        state_dir=tmp_path / "state",
    )
    (tmp_path / "config").mkdir(parents=True)
    write_password_file(cfg.password_file, "DropboxTest1234")
    return cfg


class TestHmacSignVerify:
    def test_round_trip(self) -> None:
        secret = b"a" * 32
        exp = int(time.time()) + 3600
        token = proxy._sign(secret, exp)
        assert proxy._verify(secret, token)

    def test_verify_rejects_wrong_secret(self) -> None:
        good = b"a" * 32
        bad = b"b" * 32
        exp = int(time.time()) + 3600
        token = proxy._sign(good, exp)
        assert not proxy._verify(bad, token)

    def test_verify_rejects_expired(self) -> None:
        secret = b"a" * 32
        token = proxy._sign(secret, int(time.time()) - 10)
        assert not proxy._verify(secret, token)

    def test_verify_rejects_tampered(self) -> None:
        secret = b"a" * 32
        exp = int(time.time()) + 3600
        token = proxy._sign(secret, exp)
        tampered = token[:-1] + ("0" if token[-1] != "0" else "1")
        assert not proxy._verify(secret, tampered)

    def test_verify_rejects_garbage(self) -> None:
        secret = b"a" * 32
        assert not proxy._verify(secret, "not-a-token")
        assert not proxy._verify(secret, "")
        assert not proxy._verify(secret, "12345")
        assert not proxy._verify(secret, ".")


class TestLoadPassword:
    def test_reads_admin_pw_line(self, config_with_tmp_password: DropboxConfig) -> None:
        assert proxy._load_password(config_with_tmp_password) == "DropboxTest1234"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        cfg = DropboxConfig(
            data_dir=tmp_path,
            config_dir=tmp_path / "missing",
            files_dir=tmp_path / "files",
            tls_dir=tmp_path / "tls",
            state_dir=tmp_path / "state",
        )
        assert proxy._load_password(cfg) == ""


class TestLoadSecret:
    def test_seeds_on_first_call(self, tmp_path: Path) -> None:
        (tmp_path / "config").mkdir(parents=True)
        cfg = DropboxConfig(
            data_dir=tmp_path,
            config_dir=tmp_path / "config",
            files_dir=tmp_path / "files",
            tls_dir=tmp_path / "tls",
            state_dir=tmp_path / "state",
        )
        secret = proxy._load_secret(cfg)
        assert len(secret) == 32
        assert cfg.secret_file.exists()
        import stat as _stat

        assert _stat.S_IMODE(cfg.secret_file.stat().st_mode) == 0o600

    def test_reuses_on_subsequent_calls(self, tmp_path: Path) -> None:
        (tmp_path / "config").mkdir(parents=True)
        cfg = DropboxConfig(
            data_dir=tmp_path,
            config_dir=tmp_path / "config",
            files_dir=tmp_path / "files",
            tls_dir=tmp_path / "tls",
            state_dir=tmp_path / "state",
        )
        first = proxy._load_secret(cfg)
        second = proxy._load_secret(cfg)
        assert first == second


class TestBuildResponse:
    def test_includes_content_length(self) -> None:
        response = proxy._build_response("200 OK", b"hello")
        assert b"Content-Length: 5\r\n" in response

    def test_no_server_header(self) -> None:
        """Stealth: no Server: header, ever."""
        response = proxy._build_response("200 OK", b"hello")
        assert b"Server:" not in response

    def test_cache_control_no_store(self) -> None:
        response = proxy._build_response("200 OK", b"hello")
        assert b"Cache-Control: no-store\r\n" in response

    def test_extra_headers_appended(self) -> None:
        response = proxy._build_response(
            "303 See Other",
            b"",
            [("Location", "/"), ("Set-Cookie", "x=y")],
        )
        assert b"Location: /\r\n" in response
        assert b"Set-Cookie: x=y\r\n" in response


class TestParseCookies:
    def test_single_cookie(self) -> None:
        head = b"GET / HTTP/1.1\r\nHost: example\r\nCookie: dbx=token123\r\n"
        assert proxy._parse_cookies(head) == {"dbx": "token123"}

    def test_multiple_cookies_same_header(self) -> None:
        head = b"GET / HTTP/1.1\r\nCookie: a=1; b=2; c=3\r\n"
        cookies = proxy._parse_cookies(head)
        assert cookies == {"a": "1", "b": "2", "c": "3"}

    def test_no_cookies(self) -> None:
        head = b"GET / HTTP/1.1\r\nHost: example\r\n"
        assert proxy._parse_cookies(head) == {}


class TestLoginHtmlIsBare:
    def test_no_title(self) -> None:
        assert b"<title></title>" in proxy.LOGIN_HTML

    def test_no_css(self) -> None:
        assert b"<style" not in proxy.LOGIN_HTML
        assert b"class=" not in proxy.LOGIN_HTML

    def test_no_labels(self) -> None:
        assert b"<label" not in proxy.LOGIN_HTML

    def test_only_password_input(self) -> None:
        assert b"type=password" in proxy.LOGIN_HTML
        assert b"type=text" not in proxy.LOGIN_HTML
        assert b"type=email" not in proxy.LOGIN_HTML

    def test_has_autofocus_no_autocomplete(self) -> None:
        assert b"autofocus" in proxy.LOGIN_HTML
        assert b"autocomplete=off" in proxy.LOGIN_HTML

    def test_posts_to_auth_path(self) -> None:
        assert proxy.LOGIN_PATH.encode() in proxy.LOGIN_HTML
