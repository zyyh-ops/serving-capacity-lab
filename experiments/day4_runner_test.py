
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
URL = "http://127.0.0.1:8000/v1/completions"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

FILLER = (
    "The GPU inference engine schedules tokens, "
    "updates KV cache, and computes attention.\n"
)


def token_count(text):
    return len(tokenizer.encode(
        text, add_special_tokens=False
    ))


def make_prompt(name, target):
    text = f"Request {name}. Explain GPU inference.\n"

    while token_count(text) < target:
        text += FILLER

    return text


def complete(prompt, *, temperature=0.0, top_k=-1,
             max_tokens=24, barrier=None):

    assert token_count(prompt) + max_tokens <= 256

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "ignore_eos": True,
    }

    if barrier is not None:
        barrier.wait()

    start = time.perf_counter()

    with httpx.Client(timeout=180.0) as client:
        response = client.post(URL, json=payload)
        response.raise_for_status()
        data = response.json()

    return {
        "request_id": data["id"],
        "elapsed_s": time.perf_counter() - start,
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "finish_reason": data["choices"][0]["finish_reason"],
        "text": data["choices"][0]["text"],
    }


def test_sampler():
    prompt = make_prompt("SAMPLER", 48)

    greedy = complete(
        prompt,
        temperature=0.0,
    )

    topk1 = complete(
        prompt,
        temperature=0.8,
        top_k=1,
    )

    return {
        "greedy": greedy,
        "topk1": topk1,
        "same_text": greedy["text"] == topk1["text"],
    }


def test_mixed_batch():
    prompts = [
        make_prompt("A", 48),
        make_prompt("B", 176),
        make_prompt("C", 72),
    ]

    barrier = threading.Barrier(3)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(
                complete,
                prompt,
                barrier=barrier,
            )
            for prompt in prompts
        ]

        results = [f.result() for f in futures]

    assert len({
        x["request_id"] for x in results
    }) == 3

    assert all(
        x["completion_tokens"] == 24
        for x in results
    )

    return results


if __name__ == "__main__":
    sampler = test_sampler()
    mixed = test_mixed_batch()

    assert sampler["same_text"], (
        "Greedy and top-k=1 outputs differ."
    )

    report = {
        "model": MODEL,
        "sampler": sampler,
        "mixed_batch": mixed,
        "checks": {
            "greedy_topk1_same": True,
            "unique_request_ids": True,
            "all_requests_complete": True,
        },
    }

    print(json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ))
