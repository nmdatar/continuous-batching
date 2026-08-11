# Tiny continuous-batching inference server

This project builds the same token-streaming HTTP service three ways so that the
effect of scheduling is easy to see:

| Version | Scheduler | What happens to a newly arrived request? |
| --- | --- | --- |
| `v1_serial` | One request at a time | It waits for every older request to finish. |
| `v2_dynamic_batch` | Batch requests for 10 ms | It joins the next fixed batch, then that batch runs to completion. |
| `v3_continuous_batch` | Refill active slots after every decode step | It can enter as soon as another sequence finishes. |

This is an educational implementation, not a production engine. In particular,
it recomputes each sequence on every decoding step rather than managing a paged
KV cache. That makes the scheduling code small enough to study while preserving
the key admission and iteration behavior we want to measure.

## 1. Set up

Python 3.10+ is recommended.

```bash
cd continuous-batching
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first launch downloads `sshleifer/tiny-gpt2` from Hugging Face. Override it
with `MODEL_ID`, for example `MODEL_ID=distilgpt2`. The default is intentionally
tiny enough to run on a laptop CPU.

## 2. Run one version

Run commands from this directory so the shared package is importable:

```bash
# Terminal 1: choose one server
uvicorn v1_serial.app:app --port 8001
uvicorn v2_dynamic_batch.app:app --port 8002
uvicorn v3_continuous_batch.app:app --port 8003

# Terminal 2: watch its newline-delimited JSON token stream
curl -N http://127.0.0.1:8003/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Continuous batching is", "max_new_tokens":24}'
```

Keep Uvicorn at its default single worker: multiple workers would create
independent model replicas and queues, obscuring the scheduler comparison.

Useful environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_ID` | `sshleifer/tiny-gpt2` | Hugging Face causal language model |
| `DEVICE` | auto | `cpu`, `cuda`, or `mps` |
| `MAX_BATCH_SIZE` | `8` | V2 batch size / V3 active slots |
| `BATCH_WINDOW_MS` | `10` | V2 collection window |
| `MAX_CONTEXT_TOKENS` | `512` | Prompt + output safety limit |

Every server also exposes `GET /health` and `GET /stats`. `queued` and `active`
in `/stats` are useful while watching the schedulers under load.

## 3. Benchmark the versions

Start all three servers in separate terminals, then run:

```bash
python benchmark.py \
  --url v1=http://127.0.0.1:8001 \
  --url v2=http://127.0.0.1:8002 \
  --url v3=http://127.0.0.1:8003 \
  --concurrency 1,2,4,8,16 \
  --requests-per-level 32 \
  --max-new-tokens 8,16,32,64 \
  --output results.json
```

The benchmark reports:

- **Throughput:** completed generated tokens / wall-clock benchmark time.
- **TTFT:** request start to receipt of its first token.
- **Inter-token latency (ITL):** time between consecutive streamed tokens.
- **Request latency:** request start to the final event.
- **P50/P99:** nearest-rank percentiles over requests (or token gaps for ITL).

Run a short warm-up before trusting a comparison; the benchmark does this by
default. The mixed output lengths make V2 retain partially empty fixed batches
while V3 refills freed slots. The exact same request mix is sent to each version.
Use `--max-new-tokens 32` as a useful uniform-length control. Tiny models can make
HTTP and tokenization overhead relatively prominent, which is itself a useful
warning about interpreting microbenchmarks.

## Suggested implementation order

1. Read and run [`v1_serial/scheduler.py`](v1_serial/scheduler.py). Increase
   concurrency and observe head-of-line blocking.
2. Move to [`v2_dynamic_batch/scheduler.py`](v2_dynamic_batch/scheduler.py).
   Change `BATCH_WINDOW_MS` and compare throughput against TTFT.
3. Study [`v3_continuous_batch/scheduler.py`](v3_continuous_batch/scheduler.py).
   Mix short and long `max_new_tokens` values and observe slot refill behavior.
4. Add a KV cache, sampling, prompt-length bucketing, or a real metrics backend.
