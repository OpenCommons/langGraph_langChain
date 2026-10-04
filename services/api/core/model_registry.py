"""Model registry: what the stack can run, what it selected, what is installed.

Three facts about a model live in three different places, and debugging model
problems means joining them:

  *available*  ``config/models.registry.json`` — curated metadata (this module)
  *selected*   ``.env`` → ``config.Settings.llm_model`` / ``embed_model``
  *installed*  the live Ollama host (``GET /api/tags``)

``load_registry()`` reads the first. ``resolve_models()`` joins all three and is
what ``GET /models`` serves, so a missing pull, a typo'd tag, and a model too
big for the host are all distinguishable from one response.

The registry is data, not policy: nothing here downloads or selects a model.
``scripts/lib/models_registry.sh`` reads the same file for ``setup.sh``
validation and ``make model-pull``.

Exactly one backend is active, chosen by ``USE_MLX``: Ollama (default) or Apple
MLX. Only the active backend's models are listed and its runtime is queried.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
import time

import httpx
from pydantic import BaseModel, Field

from config import get_settings

_CACHE_TTL_S = 30.0
_cache: dict = {"at": 0.0, "registry": None, "path": None}

_TAGS_TIMEOUT_S = 3.0


def _default_paths() -> tuple[pathlib.Path, ...]:
    """Candidate registry locations, most specific first.

    Two layouts exist. In a checkout this file sits four levels below the repo
    root (services/api/core/model_registry.py). In the container services/api
    is mounted at /app, so the same file is only two levels below / and the
    repo-root walk has nowhere to go — hence the guarded ``parents`` index.
    Compose sets MODELS_REGISTRY_PATH there anyway; this is the fallback.
    """
    parents = pathlib.Path(__file__).resolve().parents
    paths = [pathlib.Path("/app/config/models.registry.json")]
    if len(parents) > 3:
        paths.insert(0, parents[3] / "config" / "models.registry.json")
    return tuple(paths)


class ModelSource(BaseModel):
    kind: str  # "ollama-library" | "modelfile" | "huggingface-mlx"
    ref: str | None = None
    modelfile: str | None = None
    gguf: str | None = None
    download_url: str | None = None


class ModelEntry(BaseModel):
    id: str
    tag: str
    aliases: list[str] = Field(default_factory=list)
    name: str
    publisher: str
    family: str
    role: str  # "llm" | "embedding"
    backend: str = "ollama"  # "ollama" | "mlx"
    source: ModelSource
    parameters: str | None = None
    quantization: str | None = None
    size_gb: float | None = None
    context_length: int | None = None
    embed_dim: int | None = None
    capabilities: list[str] = Field(default_factory=list)
    min_ram_gb: int | None = None
    install: str = "on-demand"  # "selected" | "on-demand" | "modelfile"
    status: str = "supported"  # "supported" | "experimental" | "broken"
    license: str | None = None
    notes: str | None = None


class RuntimeSpec(BaseModel):
    """The local AI runtime this registry targets, and the version it pins.

    ``pinned_version`` is a floor, not an equality — see the runtime block in
    config/models.registry.json and scripts/lib/ollama_guard.sh.
    """

    name: str = "ollama"
    pinned_version: str | None = None
    latest_known: str | None = None
    checked_at: str | None = None
    release_feed: str | None = None
    download_url: str | None = None


class ModelRegistry(BaseModel):
    version: str
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    roles: dict = Field(default_factory=dict)
    models: list[ModelEntry] = Field(default_factory=list)

    def by_id(self, model_id: str) -> ModelEntry | None:
        return next((m for m in self.models if m.id == model_id), None)

    def for_backend(self, backend: str) -> list[ModelEntry]:
        return [m for m in self.models if m.backend == backend]

    def by_tag(self, tag: str) -> ModelEntry | None:
        """Look up by Ollama tag, honouring aliases and the implicit ``:latest``."""
        wanted = {tag, tag.removesuffix(":latest")}
        for m in self.models:
            known = {m.tag, *m.aliases}
            if wanted & (known | {t.removesuffix(":latest") for t in known}):
                return m
        return None

    def default_for(self, role: str) -> str | None:
        entry = self.roles.get(role) or {}
        return entry.get("default")


def registry_path() -> pathlib.Path:
    """Resolve the registry file.

    Order: ``MODELS_REGISTRY_PATH`` env (compose sets it), then the same key
    from ``.env`` via settings, then the repo-root / container layouts.
    """
    override = os.getenv("MODELS_REGISTRY_PATH") or get_settings().models_registry_path
    if override:
        return pathlib.Path(override)
    candidates = _default_paths()
    return next((p for p in candidates if p.is_file()), candidates[0])


def load_registry(force_refresh: bool = False) -> ModelRegistry:
    """Parse the registry file, cached for a short TTL.

    Raises FileNotFoundError when the file is missing and ValueError when it
    does not parse — a malformed registry is a deploy error worth surfacing,
    not something to paper over with an empty model list.
    """
    path = registry_path()
    now = time.monotonic()
    cached = _cache["registry"]
    fresh = (
        not force_refresh
        and isinstance(cached, ModelRegistry)
        and _cache["path"] == str(path)
        and now - _cache["at"] < _CACHE_TTL_S
    )
    if fresh and isinstance(cached, ModelRegistry):
        return cached

    if not path.is_file():
        raise FileNotFoundError(
            f"Model registry not found at {path}. Set MODELS_REGISTRY_PATH or mount config/ into the container."
        )

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model registry at {path} is not valid JSON: {exc}") from exc

    # Strip the `_comment` documentation keys used throughout config/.
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    registry = ModelRegistry.model_validate(raw)

    _cache.update({"at": now, "registry": registry, "path": str(path)})
    return registry


async def installed_tags(base_url: str | None = None) -> tuple[list[str], str]:
    """Tags currently pulled on the Ollama host.

    Returns ``(tags, status)`` where status is ``"ok"`` or ``"unreachable: ..."``
    so callers can tell "not installed" apart from "could not check".
    """
    s = get_settings()
    url = base_url or s.ollama_base_url
    try:
        async with httpx.AsyncClient(timeout=_TAGS_TIMEOUT_S) as c:
            r = await c.get(f"{url}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])], "ok"
    except Exception as exc:
        return [], f"unreachable: {str(exc)[:160]}"


async def mlx_installed_models(base_url: str | None = None) -> tuple[list[str], str]:
    """Model ids served by the live ``mlx_lm.server`` (OpenAI ``GET /v1/models``).

    Same ``(ids, status)`` contract as ``installed_tags``.
    """
    s = get_settings()
    url = (base_url or s.mlx_base_url).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=_TAGS_TIMEOUT_S) as c:
            r = await c.get(f"{url}/models")
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])], "ok"
    except Exception as exc:
        return [], f"unreachable: {str(exc)[:160]}"


async def installed_version(base_url: str | None = None) -> str | None:
    """Version reported by the live Ollama host, or None when unreachable."""
    s = get_settings()
    url = base_url or s.ollama_base_url
    try:
        async with httpx.AsyncClient(timeout=_TAGS_TIMEOUT_S) as c:
            r = await c.get(f"{url}/api/version")
            r.raise_for_status()
            return r.json().get("version")
    except Exception:
        return None


def version_tuple(v: str) -> tuple[int, ...]:
    """Parse a semver-ish string into a comparable tuple; unparseable → (0,)."""
    parts = re.findall(r"\d+", v or "")
    return tuple(int(p) for p in parts[:3]) or (0,)


def _tag_matches(entry_tag: str, installed: list[str]) -> bool:
    """Ollama reports bare tags as ``name:latest``; match either spelling."""
    wanted = {entry_tag, f"{entry_tag}:latest"} if ":" not in entry_tag else {entry_tag}
    return any(t in wanted for t in installed)


def host_ram_gb() -> float | None:
    """Physical RAM of the host running the API, or None when undetectable.

    In Docker this is the container's view, which on Docker Desktop is the VM
    allocation rather than the Mac's RAM — set ``HOST_RAM_GB`` to correct it.
    """
    override = os.getenv("HOST_RAM_GB")
    if override:
        try:
            return float(override)
        except ValueError:
            return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    except (ValueError, OSError, AttributeError):
        return None


def active_backend() -> str:
    """The exclusive inference backend selected by ``USE_MLX``."""
    return "mlx" if get_settings().use_mlx else "ollama"


async def resolve_models(role: str | None = None, backend: str | None = None) -> dict:
    """Join registry metadata with .env selection and live runtime state.

    ``backend`` defaults to the active one (``USE_MLX``). Only that runtime is
    queried; entries of the other backend are filtered out.
    """
    registry = load_registry()
    s = get_settings()
    backend = backend or active_backend()
    if backend == "mlx":
        (installed, runtime_status), running_version = await mlx_installed_models(), None
    else:
        (installed, runtime_status), running_version = await asyncio.gather(
            installed_tags(), installed_version()
        )
    ram = host_ram_gb()

    if backend == "mlx":
        selected = {"llm": s.llm_model}
    else:
        selected = {"llm": s.llm_model, "embedding": s.embed_model}

    entries = [
        m for m in registry.for_backend(backend) if role is None or m.role == role
    ]
    resolved = []
    for m in entries:
        selected_for = [r for r, tag in selected.items() if registry.by_tag(tag) is m]
        resolved.append(
            {
                **m.model_dump(),
                "installed": _tag_matches(m.tag, installed),
                "selected": bool(selected_for),
                "selected_as": selected_for or None,
                "fits_host": None if ram is None or m.min_ram_gb is None else ram >= m.min_ram_gb,
            }
        )

    # A .env selection pointing at a tag nobody registered is the failure mode
    # this endpoint exists to catch — surface it rather than silently omitting.
    unregistered = {r: tag for r, tag in selected.items() if registry.by_tag(tag) is None}

    # The pin is a floor: a host ahead of it is fine, a host behind it is the
    # thing worth reporting, since that is what setup.sh will act on.
    pinned = registry.runtime.pinned_version
    behind_pin = (
        None
        if running_version is None or pinned is None
        else version_tuple(running_version) < version_tuple(pinned)
    )

    runtime_key = {"status": runtime_status}
    return {
        "registry_version": registry.version,
        "registry_path": str(registry_path()),
        "active_backend": backend,
        "backend_status": {
            "active": backend,
            "reason": "exclusive selection to conserve memory",
        },
        "runtime": {
            **registry.runtime.model_dump(),
            "running_version": running_version,
            "behind_pin": behind_pin,
        },
        **(
            {"mlx": {**runtime_key, "base_url": s.mlx_base_url}}
            if backend == "mlx"
            else {"ollama": {**runtime_key, "base_url": s.ollama_base_url}}
        ),
        "host_ram_gb": None if ram is None else round(ram, 1),
        "selected": selected,
        "unregistered_selections": unregistered or None,
        "defaults": {r: registry.default_for(r) for r in ("llm", "embedding")},
        "count": len(resolved),
        "models": resolved,
    }
