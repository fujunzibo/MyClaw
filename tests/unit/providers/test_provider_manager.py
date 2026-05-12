# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name,unused-argument,protected-access
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from agentscope_runtime.engine.schemas.exception import (
    ModelNotFoundException,
)

import qwenpaw.providers.provider_manager as provider_manager_module
from qwenpaw.constant import ONEROUTER_PROVIDER_ID, ONEROUTER_FALLBACK_MODEL_ID
from qwenpaw.exceptions import ProviderError
from qwenpaw.config.config import ModelSlotConfig
from qwenpaw.providers.openai_provider import OpenAIProvider
from qwenpaw.providers.provider import ModelInfo
from qwenpaw.providers.provider_manager import ProviderManager, PROVIDER_ONEROUTER


@pytest.fixture
def isolated_secret_dir(monkeypatch, tmp_path):
    secret_dir = tmp_path / ".qwenpaw.secret"
    monkeypatch.setattr(provider_manager_module, "SECRET_DIR", secret_dir)
    return secret_dir


def test_only_onerouter_builtin_registered(isolated_secret_dir) -> None:
    manager = ProviderManager()
    assert list(manager.builtin_providers.keys()) == [ONEROUTER_PROVIDER_ID]
    p = manager.get_provider(ONEROUTER_PROVIDER_ID)
    assert p is not None
    assert isinstance(p, OpenAIProvider)
    assert p.freeze_url is True


async def test_add_custom_provider_is_disabled(isolated_secret_dir) -> None:
    manager = ProviderManager()
    with pytest.raises(ProviderError, match="Custom LLM providers are disabled"):
        await manager.add_custom_provider(
            OpenAIProvider(
                id="custom-openai",
                name="Custom OpenAI",
                base_url="https://custom.example/v1",
                api_key="sk-custom",
                models=[ModelInfo(id="custom-model", name="Custom Model")],
            ),
        )


async def test_activate_model_persists_active_model(
    isolated_secret_dir,
    monkeypatch,
) -> None:
    manager = ProviderManager()

    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(id="ok", request=kwargs)

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
    )

    monkeypatch.setattr(
        OpenAIProvider,
        "_client",
        lambda self, timeout=5: fake_client,
    )

    await manager.activate_model(ONEROUTER_PROVIDER_ID, "gpt-5")

    assert manager.active_model is not None
    assert manager.active_model.provider_id == ONEROUTER_PROVIDER_ID
    assert manager.active_model.model == "gpt-5"

    reloaded = ProviderManager()
    assert reloaded.active_model is not None
    assert reloaded.active_model.provider_id == ONEROUTER_PROVIDER_ID
    assert reloaded.active_model.model == "gpt-5"


async def test_activate_provider_invalid_provider_raises(
    isolated_secret_dir,
) -> None:
    manager = ProviderManager()

    with pytest.raises(ProviderError, match="Provider 'missing' not found"):
        await manager.activate_model("missing", "gpt-5")


def test_update_provider_for_unknown_returns_false(
    isolated_secret_dir,
) -> None:
    manager = ProviderManager()

    ok = manager.update_provider("unknown-provider", {"api_key": "sk-x"})

    assert ok is False


def test_load_provider_missing_returns_none(isolated_secret_dir) -> None:
    manager = ProviderManager()

    loaded = manager.load_provider("not-found", is_builtin=False)

    assert loaded is None


def test_resolve_effective_llm_model_id_strips_legacy_baichuan(
    monkeypatch,
) -> None:
    monkeypatch.delenv("QWENPAW_ONEROUTER_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("QWENPAW_ONEROUTER_OMIT_MODEL_IN_REQUEST", "false")
    assert (
        ProviderManager.resolve_effective_llm_model_id(
            ONEROUTER_PROVIDER_ID,
            "Baichuan-M3-Plus",
        )
        == ONEROUTER_FALLBACK_MODEL_ID
    )


def test_resolve_onerouter_omit_legacy_baichuan_returns_empty(
    monkeypatch,
) -> None:
    monkeypatch.delenv("QWENPAW_ONEROUTER_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("QWENPAW_ONEROUTER_OMIT_MODEL_IN_REQUEST", "true")
    assert (
        ProviderManager.resolve_effective_llm_model_id(
            ONEROUTER_PROVIDER_ID,
            "Baichuan-M3-Plus",
        )
        == ""
    )


def test_resolve_onerouter_omit_empty_returns_empty(monkeypatch) -> None:
    monkeypatch.delenv("QWENPAW_ONEROUTER_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("QWENPAW_ONEROUTER_OMIT_MODEL_IN_REQUEST", "true")
    assert (
        ProviderManager.resolve_effective_llm_model_id(
            ONEROUTER_PROVIDER_ID,
            "",
        )
        == ""
    )


def test_resolve_onerouter_omit_preserves_stored_id(monkeypatch) -> None:
    monkeypatch.delenv("QWENPAW_ONEROUTER_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("QWENPAW_ONEROUTER_OMIT_MODEL_IN_REQUEST", "true")
    assert (
        ProviderManager.resolve_effective_llm_model_id(
            ONEROUTER_PROVIDER_ID,
            "gpt-5",
        )
        == "gpt-5"
    )


def test_onerouter_chat_model_uses_omit_class(monkeypatch) -> None:
    from qwenpaw.providers.openai_chat_model_compat import (
        OpenAIChatModelCompat,
        OpenAIChatModelOmitModelField,
    )

    p = OpenAIProvider(
        id=ONEROUTER_PROVIDER_ID,
        name="OneRouter",
        base_url="https://onerouter.tech/v1",
        api_key="sk-test",
        models=[ModelInfo(id="gpt-5", name="gpt-5")],
    )
    monkeypatch.delenv("QWENPAW_ONEROUTER_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("QWENPAW_ONEROUTER_OMIT_MODEL_IN_REQUEST", "true")
    inst = p.get_chat_model_instance("")
    assert isinstance(inst, OpenAIChatModelOmitModelField)
    monkeypatch.setenv("QWENPAW_ONEROUTER_DEFAULT_MODEL", "from-env")
    inst2 = p.get_chat_model_instance("gpt-5")
    assert isinstance(inst2, OpenAIChatModelCompat)


def test_resolve_effective_llm_respects_env_default(monkeypatch) -> None:
    monkeypatch.setenv("QWENPAW_ONEROUTER_DEFAULT_MODEL", "my-router-model")
    assert (
        ProviderManager.resolve_effective_llm_model_id(
            ONEROUTER_PROVIDER_ID,
            "Baichuan-M3-Plus",
        )
        == "my-router-model"
    )


def test_builtin_onerouter_constants() -> None:
    assert PROVIDER_ONEROUTER.id == ONEROUTER_PROVIDER_ID
    assert "onerouter" in PROVIDER_ONEROUTER.base_url


def test_migrate_legacy_file_maps_active_to_onerouter(
    isolated_secret_dir,
) -> None:
    isolated_secret_dir.mkdir(parents=True, exist_ok=True)
    legacy = {
        "providers": {
            "dashscope": {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key": "sk-test-legacy-secret",
                "extra_models": [{"id": "qwen-plus", "name": "Qwen Plus"}],
                "chat_model": "",
            },
        },
        "custom_providers": {},
        "active_llm": {"provider_id": "dashscope", "model": "qwen3-max"},
    }
    legacy_file = isolated_secret_dir / "providers.json"
    legacy_file.write_text(
        json.dumps(legacy, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manager = ProviderManager()

    assert not legacy_file.exists()
    assert manager.active_model is not None
    assert manager.active_model.provider_id == ONEROUTER_PROVIDER_ID
    assert manager.active_model.model == "qwen3-max"
