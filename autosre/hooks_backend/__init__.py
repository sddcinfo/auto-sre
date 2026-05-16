"""Hooks backend — guard and stop-check logic for Claude Code hooks.

The guard module loads rules from a YAML file and evaluates Bash commands
against them. The stop_check module inspects the current repo state and
builds a session-end checklist.
"""
