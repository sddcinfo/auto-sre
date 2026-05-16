"""Tests for autosre.infra.types module."""

import pytest

from autosre.infra.types import CLUSTER_MODELS, SOLO_FALLBACK_MODEL, GB10Node, NodeRole


class TestNodeRole:
    def test_head_value(self) -> None:
        assert NodeRole.HEAD.value == "head"

    def test_worker_value(self) -> None:
        assert NodeRole.WORKER.value == "worker"

    def test_from_string(self) -> None:
        assert NodeRole("head") is NodeRole.HEAD
        assert NodeRole("worker") is NodeRole.WORKER

    def test_invalid_role(self) -> None:
        with pytest.raises(ValueError, match="not a valid"):
            NodeRole("invalid")


class TestGB10Node:
    def test_defaults(self) -> None:
        node = GB10Node(hostname="gb10-1", ip="192.168.1.101")
        assert node.ssh_user == "root"
        assert node.ssh_key is None
        assert node.role is NodeRole.WORKER

    def test_custom_values(self) -> None:
        node = GB10Node(
            hostname="gb10-head",
            ip="10.0.0.1",
            ssh_user="admin",
            ssh_key="/home/admin/.ssh/custom_key",
            role=NodeRole.HEAD,
        )
        assert node.hostname == "gb10-head"
        assert node.ip == "10.0.0.1"
        assert node.ssh_user == "admin"
        assert node.ssh_key == "/home/admin/.ssh/custom_key"
        assert node.role is NodeRole.HEAD

    def test_ssh_target(self) -> None:
        node = GB10Node(hostname="gb10-1", ip="192.168.1.101")
        assert node.ssh_target == "root@192.168.1.101"

    def test_ssh_target_custom_user(self) -> None:
        node = GB10Node(hostname="gb10-1", ip="10.0.0.1", ssh_user="admin")
        assert node.ssh_target == "admin@10.0.0.1"


class TestGB10NodeSerialization:
    def test_to_dict_minimal(self) -> None:
        node = GB10Node(hostname="gb10-1", ip="192.168.1.101")
        d = node.to_dict()
        assert d == {
            "hostname": "gb10-1",
            "ip": "192.168.1.101",
            "ssh_user": "root",
            "role": "worker",
        }
        assert "ssh_key" not in d

    def test_to_dict_with_ssh_key(self) -> None:
        node = GB10Node(
            hostname="gb10-1",
            ip="192.168.1.101",
            ssh_key="/root/.ssh/id_ed25519",
            role=NodeRole.HEAD,
        )
        d = node.to_dict()
        assert d["ssh_key"] == "/root/.ssh/id_ed25519"
        assert d["role"] == "head"

    def test_from_dict_minimal(self) -> None:
        data = {"hostname": "gb10-2", "ip": "192.168.1.102"}
        node = GB10Node.from_dict(data)
        assert node.hostname == "gb10-2"
        assert node.ip == "192.168.1.102"
        assert node.ssh_user == "root"
        assert node.ssh_key is None
        assert node.role is NodeRole.WORKER

    def test_from_dict_full(self) -> None:
        data = {
            "hostname": "gb10-head",
            "ip": "10.0.0.1",
            "ssh_user": "admin",
            "ssh_key": "/keys/id_rsa",
            "role": "head",
        }
        node = GB10Node.from_dict(data)
        assert node.hostname == "gb10-head"
        assert node.ssh_user == "admin"
        assert node.ssh_key == "/keys/id_rsa"
        assert node.role is NodeRole.HEAD

    def test_roundtrip(self) -> None:
        original = GB10Node(
            hostname="gb10-1",
            ip="192.168.1.101",
            ssh_user="dgx",
            ssh_key="/home/dgx/.ssh/key",
            role=NodeRole.HEAD,
        )
        restored = GB10Node.from_dict(original.to_dict())
        assert restored.hostname == original.hostname
        assert restored.ip == original.ip
        assert restored.ssh_user == original.ssh_user
        assert restored.ssh_key == original.ssh_key
        assert restored.role == original.role


class TestConstants:
    def test_solo_fallback_model(self) -> None:
        assert SOLO_FALLBACK_MODEL == "nemotron-nano"

    def test_cluster_models(self) -> None:
        assert "nemotron-super" in CLUSTER_MODELS
        assert "qwen3.6-122b" in CLUSTER_MODELS
        assert "qwen3.6-397b" in CLUSTER_MODELS
        assert "nemotron-nano" not in CLUSTER_MODELS
