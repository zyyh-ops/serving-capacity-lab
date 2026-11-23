import json
import logging
import os

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

logger = logging.getLogger("week17.day3")


def short_id(request_id):
    return request_id[-12:]


class TraceScheduler(Scheduler):

    def schedule(self):
        if not hasattr(self, "_day3_step"):
            self._day3_step = 0

        self._day3_step += 1

        pool = self.kv_cache_manager.block_pool

        # Observe state before scheduling.
        before_running = list(self.running)
        free_before = pool.get_num_free_blocks()

        # Execute the ORIGINAL vLLM scheduler.
        output = super().schedule()

        # Validate scheduling invariants.
        scheduled = output.num_scheduled_tokens
        total = output.total_num_scheduled_tokens

        assert sum(scheduled.values()) == total
        assert 0 <= total <= self.max_num_scheduled_tokens
        assert all(n > 0 for n in scheduled.values())

        # A running request may have been preempted.
        preempted = [
            short_id(req.request_id)
            for req in before_running
            if req.status == RequestStatus.PREEMPTED
        ]

        # For newly admitted requests, nonzero computed
        # tokens can reveal prefix cache reuse.
        prefix_hit_tokens = {
            short_id(req.req_id): req.num_computed_tokens
            for req in output.scheduled_new_reqs
            if req.num_computed_tokens > 0
        }

        record = {
            "step": self._day3_step,
            "running": len(self.running),
            "waiting": len(self.waiting),
            "free_blocks_before": free_before,
            "free_blocks_after": pool.get_num_free_blocks(),
            "scheduled_tokens": {
                short_id(req_id): n
                for req_id, n in scheduled.items()
            },
            "total_scheduled_tokens": total,
            "new_requests": [
                short_id(req.req_id)
                for req in output.scheduled_new_reqs
            ],
            "prefix_hit_tokens": prefix_hit_tokens,
            "preempted": preempted,
        }

        limit = int(os.getenv("W17_TRACE_LIMIT", "500"))

        if self._day3_step <= limit or preempted:
            logger.warning(
                "W17D3_TRACE %s",
                json.dumps(record, ensure_ascii=False)
            )

        return output
