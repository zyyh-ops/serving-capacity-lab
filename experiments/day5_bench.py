
import argparse
import asyncio
import hashlib
import json
import statistics
import time

import httpx

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "Qwen/Qwen3-0.6B"

PROMPT = (
    "Explain GPU inference scheduling, "
    "paged KV cache and CUDA Graph. "
) * 8

OUTPUT_TOKENS = 96


def digest(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def payload(prompt, stream=False):
    return {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,
        "stream": stream,
    }


async def complete(client, prompt):
    t0 = time.perf_counter()

    resp = await client.post(
        URL,
        json=payload(prompt),
    )
    resp.raise_for_status()

    data = resp.json()
    elapsed = time.perf_counter() - t0

    return {
        "elapsed_ms": elapsed * 1000,
        "request_id": data["id"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "finish_reason": data["choices"][0]["finish_reason"],
        "hash": digest(data["choices"][0]["text"]),
    }


async def run_batch(client, batch_size):
    prompts = [
        PROMPT + f"\nRequest variant: {i}."
        for i in range(batch_size)
    ]

    start = time.perf_counter()

    outputs = await asyncio.gather(*[
        complete(client, prompt)
        for prompt in prompts
    ])

    wall_s = time.perf_counter() - start
    tokens = sum(x["completion_tokens"] for x in outputs)

    assert len({x["request_id"] for x in outputs}) == batch_size
    assert all(x["finish_reason"] == "length" for x in outputs)
    assert all(
        x["completion_tokens"] == OUTPUT_TOKENS
        for x in outputs
    )

    return {
        "batch_size": batch_size,
        "wall_ms": wall_s * 1000,
        "output_tokens": tokens,
        "e2e_output_tokens_per_s": tokens / wall_s,
        "request_latencies_ms": [
            x["elapsed_ms"] for x in outputs
        ],
        "output_hashes": [
            x["hash"] for x in outputs
        ],
    }


async def stream_once(client):
    start = time.perf_counter()
    first_text_ms = None
    chunks = 0
    text_parts = []
    finish_reason = None

    async with client.stream(
        "POST",
        URL,
        json=payload(PROMPT, stream=True),
    ) as resp:
        resp.raise_for_status()

        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue

            raw = line[6:]
            if raw == "[DONE]":
                break

            event = json.loads(raw)

            for choice in event.get("choices", []):
                piece = choice.get("text", "")
                if piece:
                    if first_text_ms is None:
                        first_text_ms = (
                            time.perf_counter() - start
                        ) * 1000
                    chunks += 1
                    text_parts.append(piece)

                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]

    assert first_text_ms is not None

    return {
        "first_content_ms": first_text_ms,
        "elapsed_ms": (time.perf_counter() - start) * 1000,
        "content_chunks": chunks,
        "finish_reason": finish_reason,
        "hash": digest("".join(text_parts)),
    }


async def main(mode):
    limits = httpx.Limits(
        max_connections=16,
        max_keepalive_connections=16,
    )

    async with httpx.AsyncClient(
        timeout=180.0,
        limits=limits,
    ) as client:

        # Warmup: excluded from measurements.
        for _ in range(5):
            await run_batch(client, 1)

        for _ in range(3):
            await run_batch(client, 8)

        single = [
            await run_batch(client, 1)
            for _ in range(10)
        ]

        concurrent = [
            await run_batch(client, 8)
            for _ in range(5)
        ]

        streaming = [
            await stream_once(client)
            for _ in range(5)
        ]

    report = {
        "mode": mode,
        "model": MODEL,
        "max_tokens": OUTPUT_TOKENS,
        "single_batch": single,
        "concurrent_batch": concurrent,
        "streaming": streaming,
        "summary": {
            "single_e2e_p50_ms": statistics.median(
                x["wall_ms"] for x in single
            ),
            "batch8_e2e_p50_ms": statistics.median(
                x["wall_ms"] for x in concurrent
            ),
            "batch8_output_tok_s_p50": statistics.median(
                x["e2e_output_tokens_per_s"]
                for x in concurrent
            ),
            "first_content_p50_ms": statistics.median(
                x["first_content_ms"]
                for x in streaming
            ),
        },
    }

    print(json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    asyncio.run(main(args.mode))
