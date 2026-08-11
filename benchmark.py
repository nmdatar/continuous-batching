from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


PROMPTS = [
    "Continuous batching improves inference because",
    "The most important property of a scheduler is",
    "A tiny language model can teach us about",
    "When several users send prompts at once",
]


@dataclass(slots=True)
class RequestMetrics:
    latency_ms: float
    ttft_ms: float | None
    itl_ms: list[float]
    tokens: int


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


async def generate(
    client: httpx.AsyncClient, url: str, prompt: str, max_new_tokens: int
) -> RequestMetrics:
    started = time.perf_counter()
    token_times: list[float] = []
    tokens = 0
    async with client.stream(
        "POST",
        f"{url.rstrip('/')}/generate",
        json={"prompt": prompt, "max_new_tokens": max_new_tokens},
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            event = json.loads(line)
            now = time.perf_counter()
            if event["type"] == "token":
                token_times.append(now)
                tokens += 1
    ended = time.perf_counter()
    ttft = (token_times[0] - started) * 1000 if token_times else None
    itl = [(b - a) * 1000 for a, b in zip(token_times, token_times[1:])]
    return RequestMetrics((ended - started) * 1000, ttft, itl, tokens)


async def run_level(
    url: str, concurrency: int, request_count: int, token_lengths: list[int]
) -> dict:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int, client: httpx.AsyncClient) -> RequestMetrics:
        async with semaphore:
            return await generate(
                client,
                url,
                PROMPTS[index % len(PROMPTS)],
                token_lengths[index % len(token_lengths)],
            )

    timeout = httpx.Timeout(300.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        started = time.perf_counter()
        metrics = await asyncio.gather(
            *(one(index, client) for index in range(request_count))
        )
        elapsed = time.perf_counter() - started

    latencies = [item.latency_ms for item in metrics]
    ttfts = [item.ttft_ms for item in metrics if item.ttft_ms is not None]
    itls = [gap for item in metrics for gap in item.itl_ms]
    total_tokens = sum(item.tokens for item in metrics)
    return {
        "concurrency": concurrency,
        "requests": request_count,
        "tokens": total_tokens,
        "wall_time_s": round(elapsed, 4),
        "throughput_requests_s": round(request_count / elapsed, 3),
        "throughput_tokens_s": round(total_tokens / elapsed, 3),
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p99_ms": percentile(ttfts, 0.99),
        "itl_p50_ms": percentile(itls, 0.50),
        "itl_p99_ms": percentile(itls, 0.99),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p99_ms": percentile(latencies, 0.99),
        "latency_mean_ms": statistics.fmean(latencies),
    }


async def benchmark(args: argparse.Namespace) -> list[dict]:
    targets = dict(item.split("=", 1) for item in args.url)
    levels = [int(value) for value in args.concurrency.split(",")]
    token_lengths = [int(value) for value in args.max_new_tokens.split(",")]
    if not targets or not levels or not token_lengths:
        raise ValueError("URLs, concurrency levels, and token lengths cannot be empty")
    if any(value < 1 or value > 256 for value in token_lengths):
        raise ValueError("max-new-tokens values must be between 1 and 256")
    results: list[dict] = []
    for name, url in targets.items():
        print(f"\n{name} ({url})")
        if not args.no_warmup:
            async with httpx.AsyncClient(timeout=300.0) as client:
                await generate(client, url, PROMPTS[0], min(token_lengths[0], 8))
        for level in levels:
            result = await run_level(
                url, level, args.requests_per_level, token_lengths
            )
            result.update({"server": name, "url": url})
            results.append(result)
            print(
                f"  c={level:<3} tok/s={result['throughput_tokens_s']:<8.2f} "
                f"TTFT p50/p99={result['ttft_p50_ms']:.1f}/{result['ttft_p99_ms']:.1f} ms "
                f"latency p50/p99={result['latency_p50_ms']:.1f}/{result['latency_p99_ms']:.1f} ms"
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark tiny batching servers")
    parser.add_argument(
        "--url", action="append", required=True, metavar="NAME=URL",
        help="repeat for each server, e.g. --url v1=http://127.0.0.1:8001",
    )
    parser.add_argument("--concurrency", default="1,2,4,8,16")
    parser.add_argument("--requests-per-level", type=int, default=32)
    parser.add_argument(
        "--max-new-tokens", default="8,16,32,64",
        help="comma-separated output lengths; mixed lengths expose slot refill behavior",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        results = asyncio.run(benchmark(args))
    except (ValueError, httpx.HTTPError) as exc:
        raise SystemExit(f"benchmark failed: {exc}") from exc
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
