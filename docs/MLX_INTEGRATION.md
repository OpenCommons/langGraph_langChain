# MLX integration

Apple [MLX](https://github.com/ml-explore/mlx) is an alternative chat backend to
Ollama on Apple Silicon. Ollama is unchanged and remains the default.

## Why exclusive selection

Memory is the limiting factor: both runtimes load weights into unified memory.
So exactly **one** backend runs, chosen in `.env` with `USE_MLX` — there is no
per-request switching. A request for a model of the inactive backend fails with
`ModelBackendError` ("model not available in active backend").

| `USE_MLX` | Chat backend | Runtime started by `make start` |
|---|---|---|
| `false` (default) | `ChatOllama` → `OLLAMA_BASE_URL` (`:11434`) | Ollama |
| `true` | `ChatOpenAI` → `MLX_BASE_URL` (`:8081/v1`) | `mlx_lm.server` |

MLX runs as a **native host process** (it needs Metal; it cannot run in a Linux
container). The API works both ways:

- in Docker: `MLX_BASE_URL=http://host.docker.internal:8081/v1` (compose default)
- natively on the Mac: `http://localhost:8081/v1` (`config.py` default)

Embeddings stay on Ollama; MLX embeddings are future work, so RAG ingestion/
retrieval needs Ollama running for embeddings while MLX is active.

## Setup

`make setup` asks "Use MLX instead of Ollama for inference? [y/n]". Yes installs
`services/api/requirements-mlx.txt` into `.venv-mlx` and sets `USE_MLX=true`.
Or set it by hand: `USE_MLX=true` in `.env`, then
`python3 -m venv .venv-mlx && .venv-mlx/bin/pip install -r services/api/requirements-mlx.txt`.

## Switching

```bash
make use-mlx      # stop stack, USE_MLX=true, start (starts MLX, skips Ollama)
make use-ollama   # stop stack, USE_MLX=false, start
make mlx-start ID=RadixArk/Muse-Glimmer-q4-MLX
make mlx-status
make mlx-stop
```

`GET /models` and `GET /health` report `active_backend`; `/models` lists only the
active backend's models and includes `host_ram_gb`.

## Known models

| Registry id | Repo | Notes |
|---|---|---|
| `muse-glimmer` | `RadixArk/Muse-Glimmer-q4-MLX` | 4-bit starter model; `min_ram_gb: 16` (estimate, unverified) |

Add more by appending entries with `"backend": "mlx"` to
`config/models.registry.json`; `tag` is the Hugging Face repo id.

## Troubleshooting

- **`mlx_lm.server not found`** — run `make setup` (choose MLX) or install `requirements-mlx.txt`.
- **Not ready in 15 s** — the first start downloads the model. Retry with
  `MLX_START_TIMEOUT=300 make mlx-start`; logs are in `/tmp/mlx.log`.
- **API can't reach MLX from Docker** — check `curl localhost:8081/v1/models` on the host and `MLX_BASE_URL`.
- **`/health` says degraded** — the active backend is unreachable (`mlx_status` / `ollama_status`).
