"""vLLM V1 scheduler priority-preemption pre-hook.

Motivation
----------
vLLM V1 priority scheduling (``--scheduling-policy=priority``) only reorders
the waiting queue. It does NOT preempt a running request to admit a
higher-priority waiting one when ``len(running) == max_num_seqs``. See the
upstream scheduler loop in
``vllm/v1/core/sched/scheduler.py``:

    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        if len(self.running) == self.max_num_running_reqs:
            break

At that point the scheduler silently gives up and waits for a running slot
to free naturally. On a shared coder serving long Claude Code requests,
this means a latency-critical waiting request (e.g. meeting-scribe live
translation at priority -10) can sit behind 8 multi-minute coder requests
at priority 10 — for minutes.

The in-tree "priority preemption" path at ``scheduler.py``:~474 only fires
when ``kv_cache_manager.allocate_slots`` returns ``None``, i.e. under KV
cache exhaustion. With the coder's typical working set that never happens,
so ``vllm:num_preemptions_total`` stays at 0.

What this patch does
--------------------
Monkey-patches ``Scheduler.schedule`` with a pre-hook that runs before
every scheduling step. Under priority policy, while ``len(running) >=
max_num_running_reqs`` and the waiting queue's highest-priority request
has strictly lower priority value (= strictly higher scheduling priority)
than the lowest-priority running request, evict the lowest-priority
running request using the existing ``_preempt_request`` helper (which
moves it back to the waiting queue with ``num_computed_tokens = 0``).
The next ``_original_schedule`` call then admits the high-priority
waiter into the freshly-vacated slot.

Ordering is ``(priority, arrival_time)`` — same tuple vLLM itself uses
(``scheduler.py`` line 478) — so ties break by age, matching FCFS fairness
within a priority band.

Deployment
----------
Mounted read-only into the vLLM container as
``/opt/autosre/vllm_priority_preempt.py`` and activated by setting
``PYTHONSTARTUP`` to that path in the container env. ``PYTHONSTARTUP`` is
evaluated by the Python interpreter at startup *before* it runs
``vllm serve``, so by the time ``Scheduler.__init__`` runs the pre-hook
is already installed. The host source lives at
``autosre/backends/vllm_priority_preempt.py`` so the file is part of the
autosre package — no external asset to manage, no separate install step.
"""

from __future__ import annotations

import logging
import os
import time

try:
    from vllm.v1.core.sched.request_queue import SchedulingPolicy
    from vllm.v1.core.sched.scheduler import Scheduler
except ImportError:
    # Not running inside a vLLM V1 environment — nothing to patch. We
    # want this file to be importable in tests and dev shells even when
    # vllm isn't installed.
    Scheduler = None
    SchedulingPolicy = None


_logger = logging.getLogger("autosre.vllm_priority_preempt")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _logger.addHandler(_handler)
_logger.setLevel(
    logging.INFO if os.environ.get("AUTOSRE_PRIORITY_PREEMPT_DEBUG") else logging.WARNING
)


def _install_priority_preempt_hook() -> bool:
    """Install the pre-hook on Scheduler.schedule. Returns True on success."""
    if Scheduler is None or SchedulingPolicy is None:
        return False

    if getattr(Scheduler.schedule, "_autosre_priority_patched", False):
        return True

    _original_schedule = Scheduler.schedule

    def _patched_schedule(self):  # type: ignore[no-untyped-def]
        # Fast path: only act under priority scheduling with both waiting
        # and running requests to trade. The hot-path cost when priority
        # is not active is one attribute lookup + one boolean.
        if (
            getattr(self, "policy", None) == SchedulingPolicy.PRIORITY
            and self.waiting
            and self.running
            and len(self.running) >= self.max_num_running_reqs
        ):
            now = time.monotonic()
            # Preempt in a loop: several high-priority requests may be
            # waiting and we want them all to get slots. Bounded by
            # len(self.running) so we can't infinite-loop even if a bug
            # in peek_request caused no progress.
            max_preemptions = len(self.running)
            preempted_this_step = 0
            while (
                self.waiting
                and len(self.running) >= self.max_num_running_reqs
                and preempted_this_step < max_preemptions
            ):
                try:
                    waiting_top = self.waiting.peek_request()
                except (IndexError, StopIteration):
                    break

                # Running request with the LOWEST scheduling priority
                # (largest numerical priority value). Tie-break by newer
                # arrival_time so the youngest loser gets evicted first,
                # preserving fairness for older running requests.
                running_min = max(
                    self.running,
                    key=lambda r: (r.priority, r.arrival_time),
                )

                waiting_key = (waiting_top.priority, waiting_top.arrival_time)
                running_key = (running_min.priority, running_min.arrival_time)

                # No priority inversion → nothing to preempt. Fall out
                # and let the normal scheduler admit what it can.
                if waiting_key >= running_key:
                    break

                self.running.remove(running_min)
                self._preempt_request(running_min, now)
                preempted_this_step += 1

                # WARNING rather than INFO so this always shows up in
                # `docker logs` without the debug env var — priority
                # preemption is a rare event worth surfacing in ops.
                _logger.warning(
                    "priority-preempt: evicted running req=%s pri=%d for waiting req=%s pri=%d",
                    running_min.request_id,
                    running_min.priority,
                    waiting_top.request_id,
                    waiting_top.priority,
                )

        return _original_schedule(self)

    _patched_schedule._autosre_priority_patched = True  # type: ignore[attr-defined]  # noqa: SLF001
    Scheduler.schedule = _patched_schedule
    _logger.warning("autosre: vLLM V1 priority-preemption pre-hook installed on Scheduler.schedule")
    return True


_install_priority_preempt_hook()
