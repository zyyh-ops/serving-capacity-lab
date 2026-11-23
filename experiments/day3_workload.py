
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
URL = "http://127.0.0.1:8000/v1/completions"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

LINE = (
    "The inference scheduler assigns token budgets "
    "and manages paged KV cache for active requests.\n"
)


def count_tokens(prompt):
    return len(tokenizer.encode(
        prompt, add_special_tokens=False
    ))


def make_prompt(header, target_tokens=96):
    prompt = header + "\n"
    while count_tokens(prompt) < target_tokens:
        prompt += LINE
    return prompt


def request(prompt, max_tokens=32, ignore_eos=False):
    n_prompt = count_tokens(prompt)

    assert n_prompt + max_tokens <= 256, (
        f"Context too long: {n_prompt} + {max_tokens}"
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
    }

    t0 = time.perf_counter()
    with httpx.Client(timeout=240.0) as client:
        response = client.post(URL, json=payload)
        response.raise_for_status()
        data = response.json()
    return {
        "request_id": data["id"],
        "latency_s": time.perf_counter() - t0,
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "finish_reason": data["choices"][0]["finish_reason"],
        "text": data["choices"][0]["text"],
    }


def prefix_test():
    prompt = make_prompt(
        "Explain how a KV cache helps a decoder.",
        target_tokens=96,
    )

    # Cold request.
    first = request(prompt, max_tokens=24)

    # Identical prompt, same engine process.
    second = request(prompt, max_tokens=24)

    return {
        "scenario": "prefix_cache",
        "prompt_tokens": count_tokens(prompt),
        "cold": first,
        "warm": second,
        "same_output": first["text"] == second["text"],
    }


def pressure_test():
    prompts = [
        make_prompt(
            f"Independent workload {i:02d}.",
            target_tokens=96,
        )
        for i in range(8)
    ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda p: request(
                p,
                max_tokens=112,
                ignore_eos=True,
            ),
            prompts,
        ))

    return {
        "scenario": "kv_pressure",
        "results": results,
        "unique_request_ids": (
            len({x["request_id"] for x in results}) == 8
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["prefix", "pressure"],
        required=True,
    )

    args = parser.parse_args()
    if args.scenario == "prefix":
        report = prefix_test()
    else:
        report = pressure_test()
    print(json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ))