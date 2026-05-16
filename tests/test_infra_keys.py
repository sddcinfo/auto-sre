"""Tests for autosre.infra.keys module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from autosre.infra.keys import (
    DEFAULT_KEY_NAME,
    MANAGED_BLOCK_BEGIN,
    MANAGED_BLOCK_END,
    VALID_KEY_TYPES,
    SSHKeyManager,
    SSHKeyPair,
    add_to_agent,
    agent_keys,
    detect_agent_socket,
    ensure_config_uses_agent,
)


@pytest.fixture
def key_dir(tmp_path: Path) -> Path:
    return tmp_path / ".ssh"


@pytest.fixture
def manager(key_dir: Path) -> SSHKeyManager:
    return SSHKeyManager(key_dir=key_dir)


def _fake_keygen(pair_private: Path) -> subprocess.CompletedProcess[str]:
    """Simulate ssh-keygen by writing placeholder files at the expected paths."""
    pair_private.write_text("PRIVATE")
    Path(f"{pair_private}.pub").write_text("ssh-ed25519 AAAA test@host\n")
    return subprocess.CompletedProcess(args=["ssh-keygen"], returncode=0, stdout="", stderr="")


class TestPathFor:
    def test_default_paths(self, manager: SSHKeyManager, key_dir: Path) -> None:
        pair = manager.path_for("my_key")
        assert pair.private_key == key_dir / "my_key"
        assert pair.public_key == key_dir / "my_key.pub"
        assert pair.name == "my_key"

    def test_rejects_slashes(self, manager: SSHKeyManager) -> None:
        with pytest.raises(ValueError, match="Invalid key name"):
            manager.path_for("sub/key")

    def test_rejects_empty(self, manager: SSHKeyManager) -> None:
        with pytest.raises(ValueError, match="Invalid key name"):
            manager.path_for("")

    def test_rejects_parent(self, manager: SSHKeyManager) -> None:
        with pytest.raises(ValueError, match="Invalid key name"):
            manager.path_for("..")

    def test_pub_suffix_for_dotted_name(self, manager: SSHKeyManager, key_dir: Path) -> None:
        # Regression: with_suffix(".pub") would eat an existing dotted segment.
        pair = manager.path_for("id_ed25519")
        assert pair.public_key == key_dir / "id_ed25519.pub"


class TestSSHKeyPair:
    def test_exists_false_when_missing(self, tmp_path: Path) -> None:
        pair = SSHKeyPair(
            name="k",
            private_key=tmp_path / "k",
            public_key=tmp_path / "k.pub",
        )
        assert pair.exists is False

    def test_exists_true_when_both_present(self, tmp_path: Path) -> None:
        priv = tmp_path / "k"
        pub = tmp_path / "k.pub"
        priv.write_text("x")
        pub.write_text("y")
        pair = SSHKeyPair(name="k", private_key=priv, public_key=pub)
        assert pair.exists is True

    def test_read_public_key_strips(self, tmp_path: Path) -> None:
        priv = tmp_path / "k"
        pub = tmp_path / "k.pub"
        priv.write_text("x")
        pub.write_text("ssh-ed25519 AAAA test\n")
        pair = SSHKeyPair(name="k", private_key=priv, public_key=pub)
        assert pair.read_public_key() == "ssh-ed25519 AAAA test"


class TestGenerate:
    @patch("autosre.infra.keys.shutil.which", return_value="/usr/bin/ssh-keygen")
    @patch("autosre.infra.keys.subprocess.run")
    def test_generates_new_key(
        self,
        mock_run,
        _mock_which,
        manager: SSHKeyManager,
        key_dir: Path,
    ) -> None:
        expected = key_dir / DEFAULT_KEY_NAME
        mock_run.side_effect = lambda *a, **kw: _fake_keygen(expected)

        pair = manager.generate()

        assert pair.private_key == expected
        assert pair.exists
        assert key_dir.exists()
        assert (key_dir.stat().st_mode & 0o777) == 0o700
        assert (pair.private_key.stat().st_mode & 0o777) == 0o600

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ssh-keygen"
        assert "-t" in cmd and "ed25519" in cmd
        # passphrase-less: -N ""
        n_idx = cmd.index("-N")
        assert cmd[n_idx + 1] == ""

    @patch("autosre.infra.keys.shutil.which", return_value="/usr/bin/ssh-keygen")
    @patch("autosre.infra.keys.subprocess.run")
    def test_idempotent_without_force(
        self,
        mock_run,
        _mock_which,
        manager: SSHKeyManager,
        key_dir: Path,
    ) -> None:
        expected = key_dir / "kp"
        mock_run.side_effect = lambda *a, **kw: _fake_keygen(expected)

        first = manager.generate("kp")
        first_mtime = first.private_key.stat().st_mtime_ns

        mock_run.reset_mock()
        second = manager.generate("kp")

        assert second.private_key == first.private_key
        assert second.private_key.stat().st_mtime_ns == first_mtime
        mock_run.assert_not_called()

    @patch("autosre.infra.keys.shutil.which", return_value="/usr/bin/ssh-keygen")
    @patch("autosre.infra.keys.subprocess.run")
    def test_force_overwrites(
        self,
        mock_run,
        _mock_which,
        manager: SSHKeyManager,
        key_dir: Path,
    ) -> None:
        expected = key_dir / "kp"
        mock_run.side_effect = lambda *a, **kw: _fake_keygen(expected)
        manager.generate("kp")
        mock_run.reset_mock()

        manager.generate("kp", force=True)
        assert mock_run.call_count == 1

    @patch("autosre.infra.keys.shutil.which", return_value="/usr/bin/ssh-keygen")
    @patch("autosre.infra.keys.subprocess.run")
    def test_comment_passed(
        self,
        mock_run,
        _mock_which,
        manager: SSHKeyManager,
        key_dir: Path,
    ) -> None:
        expected = key_dir / DEFAULT_KEY_NAME
        mock_run.side_effect = lambda *a, **kw: _fake_keygen(expected)

        manager.generate(comment="autosre@gb10")
        cmd = mock_run.call_args[0][0]
        assert "-C" in cmd
        assert cmd[cmd.index("-C") + 1] == "autosre@gb10"

    def test_rejects_unknown_type(self, manager: SSHKeyManager) -> None:
        with pytest.raises(ValueError, match="Unsupported key type"):
            manager.generate(key_type="dsa")

    @patch("autosre.infra.keys.shutil.which", return_value=None)
    def test_missing_ssh_keygen(self, _mock_which, manager: SSHKeyManager) -> None:
        with pytest.raises(RuntimeError, match="ssh-keygen not found"):
            manager.generate()

    @patch("autosre.infra.keys.shutil.which", return_value="/usr/bin/ssh-keygen")
    @patch("autosre.infra.keys.subprocess.run")
    def test_keygen_failure_raises(
        self,
        mock_run,
        _mock_which,
        manager: SSHKeyManager,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ssh-keygen"],
            returncode=1,
            stdout="",
            stderr="boom",
        )
        with pytest.raises(RuntimeError, match="ssh-keygen failed: boom"):
            manager.generate()

    def test_valid_key_types_constant(self) -> None:
        assert "ed25519" in VALID_KEY_TYPES
        assert "rsa" in VALID_KEY_TYPES


class TestListKeys:
    def test_empty_dir(self, manager: SSHKeyManager) -> None:
        assert manager.list_keys() == []

    def test_lists_pairs(self, manager: SSHKeyManager, key_dir: Path) -> None:
        key_dir.mkdir(parents=True)
        (key_dir / "a").write_text("priv")
        (key_dir / "a.pub").write_text("pub")
        (key_dir / "b").write_text("priv")
        (key_dir / "b.pub").write_text("pub")
        # Orphan pub without matching private — should be skipped.
        (key_dir / "orphan.pub").write_text("pub")

        names = [p.name for p in manager.list_keys()]
        assert names == ["a", "b"]


class TestCopyIdCommand:
    def test_builds_command(self, manager: SSHKeyManager, key_dir: Path) -> None:
        key_dir.mkdir(parents=True)
        pub = key_dir / "k.pub"
        pub.write_text("ssh-ed25519 AAAA x")
        (key_dir / "k").write_text("priv")
        pair = manager.path_for("k")

        cmd = manager.copy_id_command("testuser@10.0.0.1", pair)
        assert cmd == ["ssh-copy-id", "-i", str(pub), "testuser@10.0.0.1"]

    def test_rejects_missing_user(self, manager: SSHKeyManager) -> None:
        pair = manager.path_for("k")
        with pytest.raises(ValueError, match="user@host"):
            manager.copy_id_command("10.0.0.1", pair)


class TestDetectAgentSocket:
    def test_gcr_socket_found(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = tmp_path / "run"
        gcr_dir = runtime / "gcr"
        gcr_dir.mkdir(parents=True)
        sock = gcr_dir / "ssh"
        sock.touch()  # A regular file is enough for the .exists() check.

        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
        monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)

        assert detect_agent_socket() == sock

    def test_falls_back_to_ssh_auth_sock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # XDG_RUNTIME_DIR points at a dir with no gcr/ssh socket.
        runtime = tmp_path / "run"
        runtime.mkdir()
        env_sock = tmp_path / "other.sock"
        env_sock.touch()

        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
        monkeypatch.setenv("SSH_AUTH_SOCK", str(env_sock))

        assert detect_agent_socket() == env_sock

    def test_neither_set_returns_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = tmp_path / "empty"
        runtime.mkdir()
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
        monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)

        assert detect_agent_socket() is None

    def test_ssh_auth_sock_missing_path_ignored(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = tmp_path / "run"
        runtime.mkdir()
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
        monkeypatch.setenv("SSH_AUTH_SOCK", str(tmp_path / "nonexistent.sock"))

        assert detect_agent_socket() is None


class TestAgentKeys:
    @patch("autosre.infra.keys.shutil.which", return_value="/usr/bin/ssh-add")
    @patch("autosre.infra.keys.subprocess.run")
    def test_exit_zero_returns_lines(
        self,
        mock_run,
        _mock_which,
        tmp_path: Path,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ssh-add", "-l"],
            returncode=0,
            stdout="256 SHA256:abc key1 (ED25519)\n256 SHA256:def key2 (ED25519)\n",
            stderr="",
        )
        sock = tmp_path / "sock"
        sock.touch()

        lines = agent_keys(sock)
        assert len(lines) == 2
        assert "SHA256:abc" in lines[0]

        # Verify env was passed with our socket.
        env = mock_run.call_args.kwargs["env"]
        assert env["SSH_AUTH_SOCK"] == str(sock)

    @patch("autosre.infra.keys.shutil.which", return_value="/usr/bin/ssh-add")
    @patch("autosre.infra.keys.subprocess.run")
    def test_exit_one_returns_empty(
        self,
        mock_run,
        _mock_which,
        tmp_path: Path,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ssh-add", "-l"],
            returncode=1,
            stdout="The agent has no identities.\n",
            stderr="",
        )
        assert agent_keys(tmp_path / "sock") == []

    @patch("autosre.infra.keys.shutil.which", return_value="/usr/bin/ssh-add")
    @patch("autosre.infra.keys.subprocess.run")
    def test_exit_two_raises(
        self,
        mock_run,
        _mock_which,
        tmp_path: Path,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ssh-add", "-l"],
            returncode=2,
            stdout="",
            stderr="Could not open a connection to your authentication agent.\n",
        )
        with pytest.raises(RuntimeError, match="rc=2"):
            agent_keys(tmp_path / "sock")


class TestAddToAgent:
    @staticmethod
    def _make_pair(key_dir: Path) -> SSHKeyPair:
        key_dir.mkdir(parents=True, exist_ok=True)
        priv = key_dir / "k"
        pub = key_dir / "k.pub"
        priv.write_text("PRIVATE")
        pub.write_text("ssh-ed25519 AAAA test@host\n")
        return SSHKeyPair(name="k", private_key=priv, public_key=pub)

    @patch("autosre.infra.keys.shutil.which", return_value="/usr/bin/ssh-add")
    @patch("autosre.infra.keys._public_key_fingerprint", return_value="SHA256:zzz")
    @patch("autosre.infra.keys.agent_keys", return_value=[])
    @patch("autosre.infra.keys.subprocess.run")
    def test_success(
        self,
        mock_run,
        _mock_agent_keys,
        _mock_fp,
        _mock_which,
        tmp_path: Path,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ssh-add"], returncode=0, stdout="", stderr=""
        )
        pair = self._make_pair(tmp_path / "keys")
        sock = tmp_path / "sock"
        sock.touch()

        added = add_to_agent(pair, sock)
        assert added is True
        # Verify the command and SSH_AUTH_SOCK env
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ssh-add"
        assert cmd[1] == str(pair.private_key)
        env = mock_run.call_args.kwargs["env"]
        assert env["SSH_AUTH_SOCK"] == str(sock)

    @patch("autosre.infra.keys.shutil.which", return_value="/usr/bin/ssh-add")
    @patch("autosre.infra.keys._public_key_fingerprint", return_value="SHA256:zzz")
    @patch("autosre.infra.keys.agent_keys", return_value=[])
    @patch("autosre.infra.keys.subprocess.run")
    def test_failure_raises(
        self,
        mock_run,
        _mock_agent_keys,
        _mock_fp,
        _mock_which,
        tmp_path: Path,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ssh-add"], returncode=1, stdout="", stderr="nope"
        )
        pair = self._make_pair(tmp_path / "keys")
        sock = tmp_path / "sock"
        sock.touch()

        with pytest.raises(RuntimeError, match="ssh-add failed: nope"):
            add_to_agent(pair, sock)

    @patch("autosre.infra.keys.shutil.which", return_value="/usr/bin/ssh-add")
    @patch("autosre.infra.keys._public_key_fingerprint", return_value="SHA256:zzz")
    @patch(
        "autosre.infra.keys.agent_keys",
        return_value=["256 SHA256:zzz autosre_ed25519 (ED25519)"],
    )
    @patch("autosre.infra.keys.subprocess.run")
    def test_idempotent_when_fingerprint_present(
        self,
        mock_run,
        _mock_agent_keys,
        _mock_fp,
        _mock_which,
        tmp_path: Path,
    ) -> None:
        # When the fingerprint already appears in `ssh-add -l` output,
        # ssh-add must not be invoked. Choice: we pass the fingerprint
        # through _public_key_fingerprint as a mock, then check that
        # subprocess.run (for ssh-add) is never called.
        pair = self._make_pair(tmp_path / "keys")
        sock = tmp_path / "sock"
        sock.touch()

        added = add_to_agent(pair, sock)
        assert added is False
        mock_run.assert_not_called()

    @patch("autosre.infra.keys.shutil.which", return_value="/usr/bin/ssh-add")
    def test_missing_key_raises(self, _mock_which, tmp_path: Path) -> None:
        pair = SSHKeyPair(
            name="missing",
            private_key=tmp_path / "missing",
            public_key=tmp_path / "missing.pub",
        )
        with pytest.raises(RuntimeError, match="does not exist"):
            add_to_agent(pair, tmp_path / "sock")


class TestEnsureConfigUsesAgent:
    def test_creates_new_file_with_0600(self, tmp_path: Path) -> None:
        config = tmp_path / "config"
        sock = tmp_path / "gcr-ssh"

        modified = ensure_config_uses_agent(sock, config_path=config)
        assert modified is True
        assert config.exists()
        assert (config.stat().st_mode & 0o777) == 0o600

        content = config.read_text()
        assert MANAGED_BLOCK_BEGIN in content
        assert MANAGED_BLOCK_END in content
        assert f"IdentityAgent {sock}" in content
        assert "AddKeysToAgent yes" in content

    def test_preserves_existing_content(self, tmp_path: Path) -> None:
        config = tmp_path / "config"
        existing = "Host example\n    HostName example.com\n    User alice\n"
        config.write_text(existing)
        sock = tmp_path / "gcr-ssh"

        modified = ensure_config_uses_agent(sock, config_path=config)
        assert modified is True

        content = config.read_text()
        assert "Host example" in content
        assert "User alice" in content
        assert MANAGED_BLOCK_BEGIN in content

    def test_managed_block_appears_before_other_content(self, tmp_path: Path) -> None:
        config = tmp_path / "config"
        existing = "Host example\n    HostName example.com\n"
        config.write_text(existing)
        sock = tmp_path / "gcr-ssh"

        ensure_config_uses_agent(sock, config_path=config)

        content = config.read_text()
        begin_idx = content.index(MANAGED_BLOCK_BEGIN)
        host_idx = content.index("Host example")
        assert begin_idx < host_idx

    def test_idempotent_second_call(self, tmp_path: Path) -> None:
        config = tmp_path / "config"
        sock = tmp_path / "gcr-ssh"

        first = ensure_config_uses_agent(sock, config_path=config)
        second = ensure_config_uses_agent(sock, config_path=config)
        assert first is True
        assert second is False

    def test_replaces_old_managed_block_on_socket_change(self, tmp_path: Path) -> None:
        config = tmp_path / "config"
        old_sock = tmp_path / "old-sock"
        new_sock = tmp_path / "new-sock"

        ensure_config_uses_agent(old_sock, config_path=config)
        modified = ensure_config_uses_agent(new_sock, config_path=config)
        assert modified is True

        content = config.read_text()
        assert f"IdentityAgent {new_sock}" in content
        assert f"IdentityAgent {old_sock}" not in content
        # And only one managed block remains.
        assert content.count(MANAGED_BLOCK_BEGIN) == 1
        assert content.count(MANAGED_BLOCK_END) == 1

    def test_does_not_touch_user_host_star_block(self, tmp_path: Path) -> None:
        config = tmp_path / "config"
        user_block = "Host *\n    ServerAliveInterval 60\n    ServerAliveCountMax 3\n"
        config.write_text(user_block)
        sock = tmp_path / "gcr-ssh"

        ensure_config_uses_agent(sock, config_path=config)
        content = config.read_text()
        assert "ServerAliveInterval 60" in content
        assert "ServerAliveCountMax 3" in content
        assert MANAGED_BLOCK_BEGIN in content

    def test_mode_enforced_when_already_up_to_date(self, tmp_path: Path) -> None:
        config = tmp_path / "config"
        sock = tmp_path / "gcr-ssh"

        ensure_config_uses_agent(sock, config_path=config)
        # Deliberately change mode to something permissive.
        config.chmod(0o644)

        modified = ensure_config_uses_agent(sock, config_path=config)
        assert modified is False
        assert (config.stat().st_mode & 0o777) == 0o600
