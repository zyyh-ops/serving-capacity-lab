
import json
import logging
import os

from vllm.v1.worker.gpu_worker import Worker

logger = logging.getLogger("week17.day4")


class TraceWorker(Worker):

    def execute_model(self, scheduler_output):
        # Run the original vLLM implementation.
        output = super().execute_model(scheduler_output)

        step = getattr(self, "_w17_step", 0) + 1
        self._w17_step = step

        limit = int(os.getenv("W17_DAY4_TRACE_LIMIT", "80"))
        total = scheduler_output.total_num_scheduled_tokens

        if step > limit or total == 0:
            return output

        runner = self.model_runner
        batch = runner.input_batch

        req_ids = list(batch.req_ids)
        n = len(req_ids)

        # These are CPU-side metadata buffers already
        # maintained by GPUModelRunner.
        query_start_loc = (
            runner.query_start_loc.np[:n + 1].tolist()
        )
        seq_lens = runner.seq_lens.np[:n].tolist()

        per_req = [
            scheduler_output.num_scheduled_tokens[req_id]
            for req_id in req_ids
        ]

        assert sum(per_req) == total
        assert query_start_loc[0] == 0
        assert query_start_loc[-1] == total

        record = {
            "step": step,
            "num_reqs": n,
            "req_ids": [x[-12:] for x in req_ids],
            "scheduled_tokens": per_req,
            "total_scheduled_tokens": total,
            "query_start_loc": query_start_loc,
            "seq_lens": seq_lens,
            "last_query_indices_derived": [
                x - 1 for x in query_start_loc[1:]
            ],
        }

        logger.warning(
            "W17D4_RUNNER %s",
            json.dumps(record, ensure_ascii=False),
        )

        return output
