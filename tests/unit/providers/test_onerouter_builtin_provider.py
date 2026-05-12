# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name,unused-argument,protected-access
"""Tests for the built-in OneRouter OpenAI-compatible provider."""
from __future__ import annotations

import pytest

import qwenpaw.providers.provider_manager as provider_manager_module
from qwenpaw.constant import ONEROUTER_PROVIDER_ID
from qwenpaw.providers.openai_provider import OpenAIProvider
from qwenpaw.providers.provider_manager import (
    PROVIDER_ONEROUTER,
    ProviderManager,
)


def test_onerouter_provider_is_openai_compatible() -> None:
    assert isinstance(PROVIDER_ONEROUTER, OpenAIProvider)


def test_onerouter_provider_config() -> None:
    assert PROVIDER_ONEROUTER.id == ONEROUTER_PROVIDER_ID
    assert PROVIDER_ONEROUTER.name == "OneRouter"
    assert PROVIDER_ONEROUTER.freeze_url is True
    assert PROVIDER_ONEROUTER.support_model_discovery is True


@pytest.fixture
def isolated_secret_dir(monkeypatch, tmp_path):
    secret_dir = tmp_path / ".qwenpaw.secret"
    monkeypatch.setattr(provider_manager_module, "SECRET_DIR", secret_dir)
    return secret_dir


def test_onerouter_registered_in_provider_manager(isolated_secret_dir) -> None:
    manager = ProviderManager()
    p = manager.get_provider(ONEROUTER_PROVIDER_ID)
    assert p is not None
    assert p.id == ONEROUTER_PROVIDER_ID
