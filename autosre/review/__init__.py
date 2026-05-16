"""Plan-review subsystem for autosre.

Subprocess-based ``run_chain()`` executor plus the iteration-tracking
CLI command. Plan review uses a simple dict-of-dicts output shape with
``P0``/``P1``/``P2`` severity strings per ``review/chain.py``'s
``ChainResult.findings``.
"""
