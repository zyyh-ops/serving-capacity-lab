
import argparse

import torch
from vllm import LLM, SamplingParams


def main(mode):
    graph_mode = 1 if mode == "piecewise" else 0

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        dtype="auto",
        max_model_len=512,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        compilation_config={
            "level": 3,
            "cudagraph_mode": graph_mode,
        },
    )

    sampling = SamplingParams(
        temperature=0,
        max_tokens=96,
        ignore_eos=True,
    )

    prompts = [
        (
            "Explain the role of CUDA Graph and "
            "GPU kernel launch overhead. "
        ) * 8
    ]

    # Warmup, excluded from Nsight capture.
    for _ in range(5):
        llm.generate(prompts, sampling, use_tqdm=False)

    torch.cuda.synchronize()

    # Only profile the following inference.
    torch.cuda.profiler.start()

    try:
        outputs = llm.generate(
            prompts,
            sampling,
            use_tqdm=False,
        )
        torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()

    for output in outputs:
        print(
            "output_tokens:",
            len(output.outputs[0].token_ids),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["no_graph", "piecewise"],
        required=True,
    )

    args = parser.parse_args()
    main(args.mode)
