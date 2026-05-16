"""Claude Code hook scripts (PreToolUse / PostToolUse / Stop / …).

Each module in this package is a standalone hook script that
``autosre claude`` wires into its temp ``--settings`` file via a
``[sys.executable, "-m", "autosre.claude_hooks.<name>"]`` command. Each
module:

- Reads Claude Code hook JSON from stdin.
- Writes the hook response JSON to stdout.
- Exits 0 on success. Exit codes >0 are reserved for explicit error
  signaling per the Claude Code hook protocol.

Fail-open policy: any unexpected exception should log to
``autosre.paths.hook_errors_log()`` and return an ``allow`` response.
The guard hook is the sole exception — it fails CLOSED on errors.
"""
