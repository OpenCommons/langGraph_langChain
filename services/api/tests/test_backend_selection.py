"""Exclusive Ollama / MLX backend selection (USE_MLX).

No network: the registry is a tmp fixture and the runtimes' installed-state
probes are monkeypatched.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from core import llm_factory
from core import model_registry as mr

REGISTRY = {
    "version": "9.9",
    "roles": {"llm": {"default": "llama"}},
    "models": [
        {
            "id": "llama",
            "tag": "llama:8b",
            "name": "Llama",
            "publisher": "T",
            "family": "t",
            "role": "llm",
            "source": {"kind": "ollama-library", "ref": "library/llama"},
        },
        {
            "id": "muse",
            "tag": "Org/Muse-q4-MLX",
            "name": "Muse",
            "publisher": "T",
            "family": "t",
            "role": "llm",
            "backend": "mlx",
            "source": {"kind": "huggingface-mlx", "ref": "Org/Muse-q4-MLX"},
        },
    ],
}


@pytest.fixture(autouse=True)
def _registry(tmp_path, monkeypatch):
    path = tmp_path / "models.registry.json"
    path.write_text(json.dumps(REGISTRY))
    monkeypatch.setenv("MODELS_REGISTRY_PATH", str(path))
    monkeypatch.setattr(mr, "_cache", {"at": 0.0, "registry": None, "path": None})


def _backend(monkeypatch, use_mlx: bool):
    s = mr.get_settings()
    monkeypatch.setattr(s, "use_mlx", use_mlx)
    monkeypatch.setattr(s, "mlx_model", "muse")
    monkeypatch.setattr(s, "llm_model", "llama:8b")
    monkeypatch.setattr(s, "mlx_base_url", "http://localhost:8081/v1")
    monkeypatch.setattr(s, "models_registry_path", "")


def test_ollama_backend_returns_chat_ollama(monkeypatch):
    _backend(monkeypatch, False)
    assert isinstance(llm_factory.get_chat_model("llama"), ChatOllama)
    assert isinstance(llm_factory.get_chat_model(), ChatOllama)


def test_mlx_backend_returns_chat_openai(monkeypatch):
    _backend(monkeypatch, True)
    for m in (llm_factory.get_chat_model("muse"), llm_factory.get_chat_model()):
        assert isinstance(m, ChatOpenAI)
        assert m.model_name == "Org/Muse-q4-MLX"
        assert str(m.openai_api_base) == "http://localhost:8081/v1"


def test_ollama_model_rejected_when_mlx_active(monkeypatch):
    _backend(monkeypatch, True)
    with pytest.raises(llm_factory.ModelBackendError, match="not available in active backend"):
        llm_factory.get_chat_model("llama")


def test_mlx_model_rejected_when_ollama_active(monkeypatch):
    _backend(monkeypatch, False)
    with pytest.raises(llm_factory.ModelBackendError, match="not available in active backend"):
        llm_factory.get_chat_model("muse")


def test_unknown_model_rejected_when_mlx_active(monkeypatch):
    _backend(monkeypatch, True)
    with pytest.raises(llm_factory.ModelBackendError):
        llm_factory.get_chat_model("nonsense:1b")


def _patch_runtimes(monkeypatch):
    async def _tags(base_url=None):
        return ["llama:8b"], "ok"

    async def _version(base_url=None):
        return "0.32.9"

    async def _mlx(base_url=None):
        return ["Org/Muse-q4-MLX"], "ok"

    monkeypatch.setattr(mr, "installed_tags", _tags)
    monkeypatch.setattr(mr, "installed_version", _version)
    monkeypatch.setattr(mr, "mlx_installed_models", _mlx)
    monkeypatch.setattr(mr, "host_ram_gb", lambda: 32.0)


def test_resolve_models_lists_only_ollama_models(monkeypatch):
    _backend(monkeypatch, False)
    _patch_runtimes(monkeypatch)
    out = asyncio.run(mr.resolve_models())
    assert out["active_backend"] == "ollama"
    assert out["backend_status"]["active"] == "ollama"
    assert [m["id"] for m in out["models"]] == ["llama"]
    assert "ollama" in out and "mlx" not in out


def test_resolve_models_lists_only_mlx_models(monkeypatch):
    _backend(monkeypatch, True)
    _patch_runtimes(monkeypatch)
    out = asyncio.run(mr.resolve_models())
    assert out["active_backend"] == "mlx"
    assert [m["id"] for m in out["models"]] == ["muse"]
    assert out["models"][0]["backend"] == "mlx"
    assert out["models"][0]["installed"] is True
    assert "mlx" in out and "ollama" not in out


def test_resolve_models_explicit_backend_filter(monkeypatch):
    _backend(monkeypatch, False)
    _patch_runtimes(monkeypatch)
    out = asyncio.run(mr.resolve_models(backend="mlx"))
    assert [m["id"] for m in out["models"]] == ["muse"]
