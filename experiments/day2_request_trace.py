import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-0.6B"


def payload(prompt, stream=False):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 96,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def non_stream(prompt):
    start = time.perf_counter()

    with httpx.Client(timeout=120.0) as client:
        response = client.post(URL, json=payload(prompt))
        response.raise_for_status()
        data = response.json()

    elapsed = time.perf_counter() - start

    return {
        "request_id": data["id"],
        "mode": "non_stream",
        "elapsed_s": elapsed,
        "finish_reason": data["choices"][0]["finish_reason"],
        "content": data["choices"][0]["message"]["content"],
        "usage": data.get("usage"),
    }


def streaming(prompt):
    start = time.perf_counter()
    first_content_s = None
    response_id = None
    content_parts = []
    content_events = 0
    finish_reason = None

    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST", URL, json=payload(prompt, stream=True)
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue

                raw = line[6:]
                if raw == "[DONE]":
                    break

                event = json.loads(raw)
                response_id = event.get("id", response_id)

                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""

                    if text:
                        if first_content_s is None:
                            first_content_s = (
                                time.perf_counter() - start
                            )
                        content_parts.append(text)
                        content_events += 1

                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]

    return {
        "request_id": response_id,
        "mode": "stream",
        "elapsed_s": time.perf_counter() - start,
        "time_to_first_content_s": first_content_s,
        "content_events": content_events,
        "finish_reason": finish_reason,
        "content": "".join(content_parts),
    }


def main():
    prompt = "请简要解释什么是连续批处理。"

    baseline = non_stream(prompt)
    streamed = streaming(prompt)

    prompts = [
        "请解释Prefill和Decode的区别。",
        "请解释KV Cache的作用。",
        "请解释Continuous Batching。",
    ]

    with ThreadPoolExecutor(max_workers=3) as pool:
        concurrent_results = list(pool.map(non_stream, prompts))

    ids = [r["request_id"] for r in concurrent_results]

    assert len(set(ids)) == len(ids), "Duplicate request IDs"

    for result in [baseline, streamed, *concurrent_results]:
        assert result["request_id"]
        assert result["finish_reason"] in {"stop", "length"}
        assert result["content"], "Empty generation"

    assert baseline["usage"]["completion_tokens"] > 0
    assert streamed["content_events"] > 0
    assert streamed["time_to_first_content_s"] is not None

    report = {
        "environment": {
            "model": MODEL,
            "api": URL,
            "temperature": 0,
            "max_tokens": 96,
        },
        "single_non_stream": baseline,
        "single_stream": streamed,
        "concurrent": concurrent_results,
        "checks": {
            "unique_concurrent_request_ids": True,
            "non_empty_outputs": True,
            "streaming_received": True,
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
