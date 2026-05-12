# -*- coding: utf-8 -*-
"""A Manager class to handle all providers, including built-in and custom ones.
It provides a unified interface to manage providers, such as listing available
providers, adding/removing custom providers, and fetching provider details."""

import asyncio
import os
from typing import Dict, List
import logging
import json

from pydantic import BaseModel

from agentscope.model import ChatModelBase
from agentscope_runtime.engine.schemas.exception import (
    ModelNotFoundException,
)

from ..constant import (
    SECRET_DIR,
    ONEROUTER_PROVIDER_ID,
    ONEROUTER_DEFAULT_BASE_URL,
    ONEROUTER_FALLBACK_MODEL_ID,
    EnvVarLoader,
    onerouter_omit_model_in_request,
)
from ..config.config import ModelSlotConfig
from ..exceptions import ProviderError
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .provider import (
    ModelInfo,
    Provider,
    ProviderInfo,
)
from .openrouter_provider import OpenRouterProvider
from ..security.secret_store import (
    PROVIDER_SECRET_FIELDS,
    decrypt_dict_fields,
    encrypt_dict_fields,
    is_encrypted,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Built-in provider definitions and their default models.
# -------------------------------------------------------


PROVIDER_ONEROUTER = OpenAIProvider(
    id=ONEROUTER_PROVIDER_ID,
    name="OneRouter",
    base_url=ONEROUTER_DEFAULT_BASE_URL,
    api_key_prefix="",
    models=[],
    freeze_url=True,
    require_api_key=True,
    support_model_discovery=True,
)


class ProviderManager:  # pylint: disable=too-many-public-methods
    """A manager class to handle all providers,
    including built-in and custom ones."""

    _instance = None

    def __init__(self) -> None:
        # Initialize provider manager, load providers from registry and store
        # any necessary state (e.g., cached models).
        self.builtin_providers: Dict[str, Provider] = {}
        self.custom_providers: Dict[str, Provider] = {}
        self.plugin_providers: Dict[str, Dict] = {}  # Plugin providers
        self.active_model: ModelSlotConfig | None = None
        self.root_path = SECRET_DIR / "providers"
        self.builtin_path = self.root_path / "builtin"
        self.custom_path = self.root_path / "custom"
        self.plugin_path = self.root_path / "plugin"  # Plugin provider configs
        self._prepare_disk_storage()
        self._init_builtins()
        try:
            self._migrate_legacy_providers()
        except Exception as e:
            logger.warning("Failed to migrate legacy providers: %s", e)
        self._init_from_storage()
        self._apply_onerouter_env_config()
        self._normalize_active_model_for_locked_gateway()
        self._apply_default_annotations()

    def _prepare_disk_storage(self):
        """Prepare directory structure"""
        for path in [
            self.root_path,
            self.builtin_path,
            self.custom_path,
            self.plugin_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(path, 0o700)  # Restrict permissions for security
            except Exception:
                pass

    def _init_builtins(self):
        self._add_builtin(PROVIDER_ONEROUTER)

    def _add_builtin(self, provider: Provider):
        self.builtin_providers[provider.id] = provider

    async def list_provider_info(self) -> List[ProviderInfo]:
        tasks = [
            provider.get_info() for provider in self.builtin_providers.values()
        ]
        tasks += [
            provider.get_info() for provider in self.custom_providers.values()
        ]
        # Add plugin providers - directly return their ProviderInfo
        for plugin_provider in self.plugin_providers.values():
            provider_info = plugin_provider["info"]
            # Plugin providers store ProviderInfo directly, no need to
            # instantiate
            tasks.append(self._get_plugin_provider_info(provider_info))

        provider_infos = await asyncio.gather(*tasks)
        return list(provider_infos)

    async def _get_plugin_provider_info(
        self,
        provider_info: ProviderInfo,
    ) -> ProviderInfo:
        """Helper to return plugin provider info as async task."""
        return provider_info

    @staticmethod
    def _normalize_provider_id(provider_id: str) -> str:
        """Normalize provider ID for backward compatibility.

        Maps legacy 'copaw-local' to 'qwenpaw-local'.
        """
        if provider_id == "copaw-local":
            return "qwenpaw-local"
        return provider_id

    def get_provider(self, provider_id: str) -> Provider | None:
        # Return a provider instance by its ID. This will be used to create
        # chat model instances for the agent.
        # Normalize provider ID for backward compatibility
        provider_id = self._normalize_provider_id(provider_id)
        # Check plugin providers first
        if provider_id in self.plugin_providers:
            plugin_provider = self.plugin_providers[provider_id]
            provider_info = plugin_provider["info"]
            provider_class = plugin_provider["class"]
            # Instantiate with **dict unpacking for Pydantic BaseModel
            return provider_class(**provider_info.model_dump())
        if provider_id in self.builtin_providers:
            return self.builtin_providers[provider_id]
        if provider_id in self.custom_providers:
            return self.custom_providers[provider_id]
        return None

    async def get_provider_info(self, provider_id: str) -> ProviderInfo | None:
        provider = self.get_provider(provider_id)
        return await provider.get_info() if provider else None

    def get_active_model(self) -> ModelSlotConfig | None:
        # Return the currently active provider/model configuration.
        return self.active_model

    def update_provider(self, provider_id: str, config: Dict) -> bool:
        # Update the configuration of a provider (e.g., base URL, API key).
        # This will be called when the user edits a provider's settings in the
        # UI. It should update the in-memory provider instance and persist the
        # changes to providers.json.
        # Normalize provider ID for backward compatibility
        provider_id = self._normalize_provider_id(provider_id)
        provider = self.get_provider(provider_id)
        if not provider:
            return False
        provider.update_config(config)

        # Determine save location
        is_builtin = provider_id in self.builtin_providers
        is_plugin = provider_id in self.plugin_providers

        if is_plugin:
            # Update plugin provider info in memory (convert Provider to
            # ProviderInfo)
            provider_info = ProviderInfo(**provider.model_dump())
            self.plugin_providers[provider_id]["info"] = provider_info
            # Save to plugin path (separate from builtin)
            self._save_plugin_provider(provider)
        else:
            self._save_provider(provider, is_builtin=is_builtin)

        return True

    def start_local_model_resume(self, local_manager) -> None:
        """Schedule background restore of the active local model server."""
        task = asyncio.create_task(
            self._resume_local_model(local_manager),
            name="qwenpaw-local-model-resume",
        )
        task.add_done_callback(self._on_local_model_resume_done)

    @staticmethod
    def _on_local_model_resume_done(task: asyncio.Task[None]) -> None:
        """Log unexpected failures from background local model restore."""
        if task.cancelled():
            return

        exc = task.exception()
        if exc is not None:
            logger.warning(
                "Background local model restore failed: %s",
                exc,
                exc_info=exc,
            )
        logger.info("Background local model restore completed")

    async def fetch_provider_models(
        self,
        provider_id: str,
        save: bool = True,
    ) -> List[ModelInfo]:
        """Fetch the list of available models from a provider.

        Args:
            provider_id: The ID of the provider to fetch models from.
            save: If True, save the discovered models to the provider
                configuration. Defaults to True.

        Returns:
            List of ModelInfo objects representing available models.
        """
        provider_id = self._normalize_provider_id(provider_id)
        provider = self.get_provider(provider_id)
        if not provider:
            return []
        try:
            models = await provider.fetch_models()
            if save:
                provider.extra_models = models
                # Save provider config to appropriate location
                is_plugin = provider_id in self.plugin_providers
                if is_plugin:
                    provider_info = ProviderInfo(**provider.model_dump())
                    self.plugin_providers[provider_id]["info"] = provider_info
                    self._save_plugin_provider(provider)
                else:
                    self._save_provider(
                        provider,
                        is_builtin=provider_id in self.builtin_providers,
                    )
            return models
        except Exception as e:
            logger.warning(
                "Failed to fetch models for provider '%s': %s",
                provider_id,
                e,
            )
            return []

    def _resolve_custom_provider_id(self, provider_id: str) -> str:
        """Resolve provider ID conflicts for a custom provider."""
        base_id = provider_id
        if base_id in self.builtin_providers:
            base_id = f"{base_id}-custom"

        resolved_id = base_id
        while (
            resolved_id in self.builtin_providers
            or resolved_id in self.custom_providers
        ):
            resolved_id = f"{resolved_id}-new"

        return resolved_id

    async def add_custom_provider(self, provider_data: ProviderInfo):
        raise ProviderError(
            message=(
                "Custom LLM providers are disabled. "
                "Use the built-in OneRouter OpenAI-compatible gateway only."
            ),
        )

    def remove_custom_provider(self, provider_id: str) -> bool:
        # Remove a custom provider by its ID. This will update the
        # providers.json file and remove the provider from the UI.
        if provider_id in self.custom_providers:
            del self.custom_providers[provider_id]
            provider_path = self.custom_path / f"{provider_id}.json"
            if provider_path.exists():
                os.remove(provider_path)
            return True
        return False

    async def activate_model(self, provider_id: str, model_id: str):
        # Set the active provider and model for the agent. This will update
        # providers.json and determine which provider/model is used when the
        # agent creates chat model instances.
        # Normalize provider ID for backward compatibility
        provider_id = self._normalize_provider_id(provider_id)
        provider = self.get_provider(provider_id)
        if not provider:
            raise ProviderError(
                message=f"Provider '{provider_id}' not found.",
            )
        if not provider.has_model(model_id):
            if provider_id == ONEROUTER_PROVIDER_ID and (model_id or "").strip():
                mid = model_id.strip()
                added, err = await provider.add_model(
                    ModelInfo(id=mid, name=mid),
                )
                if not added:
                    raise ProviderError(message=err)
                self._save_provider(
                    provider,
                    is_builtin=provider_id in self.builtin_providers,
                )
            else:
                raise ModelNotFoundException(
                    model_name=f"{provider_id}/{model_id}",
                    details={"provider_id": provider_id, "model_id": model_id},
                )
        self.active_model = ModelSlotConfig(
            provider_id=provider_id,
            model=model_id,
        )
        self.save_active_model(self.active_model)

        self.maybe_probe_multimodal(provider_id, model_id)

    def maybe_probe_multimodal(self, provider_id: str, model_id: str) -> None:
        """Schedule multimodal probing for a model if capability is unknown."""
        provider = self.get_provider(provider_id)
        # Auto-probe multimodal if not yet probed
        for model in provider.models + provider.extra_models:
            if model.id == model_id and model.supports_multimodal is None:
                asyncio.create_task(
                    self._auto_probe_multimodal(provider_id, model_id),
                )
                break

    async def _auto_probe_multimodal(
        self,
        provider_id: str,
        model_id: str,
    ) -> None:
        """Background probe that doesn't block model activation."""
        try:
            result = await self.probe_model_multimodal(provider_id, model_id)
            logger.info(
                "Auto-probe for %s/%s: image=%s, video=%s",
                provider_id,
                model_id,
                result.get("supports_image"),
                result.get("supports_video"),
            )
        except Exception as e:
            logger.warning("Auto-probe multimodal failed: %s", e)

    async def add_model_to_provider(
        self,
        provider_id: str,
        model_info: ModelInfo,
    ) -> ProviderInfo:
        provider_id = self._normalize_provider_id(provider_id)
        provider = self.get_provider(provider_id)
        if not provider:
            raise ProviderError(
                message=f"Provider '{provider_id}' not found.",
            )
        added, error_message = await provider.add_model(model_info)
        if not added:
            raise ProviderError(
                message=error_message,
                details={
                    "provider_id": provider_id,
                    "model_id": model_info.id,
                },
            )

        # Save provider config to appropriate location
        is_plugin = provider_id in self.plugin_providers
        if is_plugin:
            provider_info = ProviderInfo(**provider.model_dump())
            self.plugin_providers[provider_id]["info"] = provider_info
            self._save_plugin_provider(provider)
        else:
            self._save_provider(
                provider,
                is_builtin=provider_id in self.builtin_providers,
            )
        return await provider.get_info()

    async def update_model_config(
        self,
        provider_id: str,
        model_id: str,
        config: Dict,
    ) -> ProviderInfo:
        """Update per-model configuration and persist to disk."""
        provider_id = self._normalize_provider_id(provider_id)
        provider = self.get_provider(provider_id)
        if not provider:
            raise ProviderError(
                message=f"Provider '{provider_id}' not found.",
            )
        if not provider.update_model_config(model_id, config):
            raise ModelNotFoundException(
                model_name=f"{provider_id}/{model_id}",
                details={"provider_id": provider_id, "model_id": model_id},
            )

        # Save provider config to appropriate location
        is_plugin = provider_id in self.plugin_providers
        if is_plugin:
            provider_info = ProviderInfo(**provider.model_dump())
            self.plugin_providers[provider_id]["info"] = provider_info
            self._save_plugin_provider(provider)
        else:
            self._save_provider(
                provider,
                is_builtin=provider_id in self.builtin_providers,
            )
        return await provider.get_info()

    async def delete_model_from_provider(
        self,
        provider_id: str,
        model_id: str,
    ) -> ProviderInfo:
        provider_id = self._normalize_provider_id(provider_id)
        provider = self.get_provider(provider_id)
        if not provider:
            raise ProviderError(
                message=f"Provider '{provider_id}' not found.",
            )
        await provider.delete_model(model_id=model_id)

        # Save provider config to appropriate location
        is_plugin = provider_id in self.plugin_providers
        if is_plugin:
            provider_info = ProviderInfo(**provider.model_dump())
            self.plugin_providers[provider_id]["info"] = provider_info
            self._save_plugin_provider(provider)
        else:
            self._save_provider(
                provider,
                is_builtin=provider_id in self.builtin_providers,
            )
        return await provider.get_info()

    async def probe_model_multimodal(
        self,
        provider_id: str,
        model_id: str,
        image_only: bool = False,
    ) -> dict:
        """Probe a model's multimodal capabilities and persist the result.

        Args:
            provider_id: Provider identifier.
            model_id: Model identifier.
            image_only: When True, skip the video probe for a faster result.
                Only ``supports_image`` will be accurate; ``supports_video``
                will remain at its previous value (not updated).
        """
        provider_id = self._normalize_provider_id(provider_id)
        provider = self.get_provider(provider_id)
        if not provider:
            return {"error": f"Provider '{provider_id}' not found"}

        result = await provider.probe_model_multimodal(
            model_id,
            image_only=image_only,
        )

        # Update the model's capability flags.
        # For image_only probes, leave supports_video untouched so a
        # subsequent full probe can fill it in correctly.
        for model in provider.models + provider.extra_models:
            if model.id == model_id:
                model.supports_image = result.supports_image
                if not image_only:
                    model.supports_video = result.supports_video
                    model.supports_multimodal = result.supports_multimodal
                else:
                    # Partial update: derive supports_multimodal from
                    # image alone; video will be updated by the full probe.
                    if result.supports_image:
                        model.supports_multimodal = True
                model.probe_source = "probed"
                break

        # Compare probe result against expected baseline
        from .capability_baseline import (
            ExpectedCapabilityRegistry,
            compare_probe_result,
        )

        registry = ExpectedCapabilityRegistry()
        expected = registry.get_expected(provider_id, model_id)
        if expected:
            discrepancies = compare_probe_result(
                expected,
                result.supports_image,
                result.supports_video,
            )
            for d in discrepancies:
                logger.warning(
                    "Probe discrepancy: %s/%s %s expected=%s actual=%s (%s)",
                    d.provider_id,
                    d.model_id,
                    d.field,
                    d.expected,
                    d.actual,
                    d.discrepancy_type,
                )

        # Persist to disk
        is_plugin = provider_id in self.plugin_providers
        if is_plugin:
            provider_info = ProviderInfo(**provider.model_dump())
            self.plugin_providers[provider_id]["info"] = provider_info
            self._save_plugin_provider(provider)
        else:
            self._save_provider(
                provider,
                is_builtin=provider_id in self.builtin_providers,
            )
        return {
            "supports_image": result.supports_image,
            "supports_video": result.supports_video,
            "supports_multimodal": result.supports_multimodal,
            "image_message": result.image_message,
            "video_message": result.video_message,
        }

    def _save_provider(
        self,
        provider: Provider,
        is_builtin: bool = False,
        skip_if_exists: bool = False,
    ):
        """Save a provider configuration to disk.

        Sensitive fields (``api_key``) are encrypted before writing.
        """
        provider_dir = self.builtin_path if is_builtin else self.custom_path
        provider_path = provider_dir / f"{provider.id}.json"
        if skip_if_exists and provider_path.exists():
            return
        data = encrypt_dict_fields(
            provider.model_dump(),
            PROVIDER_SECRET_FIELDS,
        )
        with open(provider_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        try:
            os.chmod(provider_path, 0o600)
        except OSError:
            pass

    def _save_plugin_provider(self, provider: Provider):
        """Save a plugin provider configuration to disk.

        Sensitive fields (``api_key``) are encrypted before writing.
        """
        provider_path = self.plugin_path / f"{provider.id}.json"
        data = encrypt_dict_fields(
            provider.model_dump(),
            PROVIDER_SECRET_FIELDS,
        )
        with open(provider_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        try:
            os.chmod(provider_path, 0o600)
        except OSError:
            pass

    def load_provider(
        self,
        provider_id: str,
        is_builtin: bool = False,
    ) -> Provider | None:
        """Load a provider configuration from disk.

        Encrypted fields are transparently decrypted.  If a legacy
        plaintext ``api_key`` is detected it is re-encrypted in place.
        """
        provider_dir = self.builtin_path if is_builtin else self.custom_path
        provider_path = provider_dir / f"{provider_id}.json"
        if not provider_path.exists():
            return None
        try:
            with open(provider_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            needs_rewrite = self._maybe_migrate_plaintext(
                data,
                PROVIDER_SECRET_FIELDS,
            )
            data = decrypt_dict_fields(data, PROVIDER_SECRET_FIELDS)
            provider = self._provider_from_data(data)

            if needs_rewrite:
                try:
                    self._save_provider(
                        provider,
                        is_builtin=is_builtin,
                        skip_if_exists=False,
                    )
                except Exception as enc_err:
                    logger.debug(
                        "Deferred plaintext→encrypted migration"
                        " for provider '%s': %s",
                        provider_id,
                        enc_err,
                    )

            return provider
        except Exception as e:
            logger.warning(
                "Failed to load provider '%s' from %s: %s",
                provider_id,
                provider_path,
                e,
            )
            return None

    @staticmethod
    def _maybe_migrate_plaintext(
        data: dict,
        secret_fields: frozenset[str],
    ) -> bool:
        """Return ``True`` when *data* contains plaintext secret fields
        that should be re-encrypted on disk."""
        for field in secret_fields:
            value = data.get(field)
            if isinstance(value, str) and value and not is_encrypted(value):
                return True
        return False

    def _provider_from_data(self, data: Dict) -> Provider:
        """Deserialize provider data to a concrete provider type."""
        provider_id = str(data.get("id", ""))
        chat_model = str(data.get("chat_model", ""))

        if provider_id == "openrouter":
            return OpenRouterProvider.model_validate(data)
        if provider_id == "anthropic" or chat_model == "AnthropicChatModel":
            return AnthropicProvider.model_validate(data)
        if provider_id == "gemini" or chat_model == "GeminiChatModel":
            return GeminiProvider.model_validate(data)
        if provider_id == "ollama":
            return OllamaProvider.model_validate(data)
        return OpenAIProvider.model_validate(data)

    def save_active_model(self, active_model: ModelSlotConfig):
        """Save the active provider/model configuration to disk."""
        active_path = self.root_path / "active_model.json"
        with open(active_path, "w", encoding="utf-8") as f:
            json.dump(
                active_model.model_dump(),
                f,
                ensure_ascii=False,
                indent=2,
            )
        try:
            os.chmod(active_path, 0o600)
        except OSError:
            pass

    def clear_active_model(self, provider_id: str | None = None) -> bool:
        """Clear the active provider/model configuration.

        If provider_id is provided, only clear when it matches the current
        active provider.
        """
        if self.active_model is None:
            return False
        # Normalize provider ID for backward compatibility
        if provider_id is not None:
            provider_id = self._normalize_provider_id(provider_id)
        if (
            provider_id is not None
            and self.active_model.provider_id != provider_id
        ):
            return False

        self.active_model = None
        active_path = self.root_path / "active_model.json"
        try:
            active_path.unlink()
        except (FileNotFoundError, OSError):
            pass
        return True

    def load_active_model(self) -> ModelSlotConfig | None:
        """Load the active provider/model configuration from disk."""
        active_path = self.root_path / "active_model.json"
        if not active_path.exists():
            return None
        try:
            with open(active_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return ModelSlotConfig.model_validate(data)
        except Exception:
            return None

    def _migrate_copaw_config(self) -> None:
        """Migrate copaw-local provider config to qwenpaw-local."""
        # 1. Migrate active model configuration (only provider_id)
        if (
            self.active_model
            and self.active_model.provider_id == "copaw-local"
        ):
            self.active_model.provider_id = "qwenpaw-local"
            self.save_active_model(self.active_model)
            logger.info(
                "Migrated active model provider from "
                "'copaw-local' to 'qwenpaw-local'",
            )

        # 2. Migrate stored provider config file
        copaw_config_path = self.builtin_path / "copaw-local.json"
        if not copaw_config_path.exists():
            return

        try:
            # Load old config and apply to new provider instance
            with open(copaw_config_path, "r", encoding="utf-8") as f:
                old_config = json.load(f)

            # Get the new built-in provider instance
            provider = self.builtin_providers.get("qwenpaw-local")
            if not provider:
                return

            # Apply migrated configuration (preserve extra_models as-is)
            if "extra_models" in old_config:
                provider.extra_models = [
                    ModelInfo.model_validate(model)
                    for model in old_config["extra_models"]
                ]
            if "base_url" in old_config:
                provider.base_url = old_config["base_url"]
            if "generate_kwargs" in old_config:
                provider.generate_kwargs = old_config["generate_kwargs"]

            # Save using standard persistence logic (with encryption)
            self._save_provider(provider, is_builtin=True)

            # Remove old config file
            copaw_config_path.unlink()
            logger.info(
                "Migrated provider config from "
                "'copaw-local.json' to 'qwenpaw-local.json'",
            )
        except Exception as exc:
            logger.warning("Failed to migrate copaw-local config: %s", exc)

    # pylint: disable=too-many-branches
    def _migrate_legacy_providers(self):
        """Migrate from legacy providers.json format to the new structure."""
        legacy_path = SECRET_DIR / "providers.json"
        if legacy_path.exists() and legacy_path.is_file():
            with open(legacy_path, "r", encoding="utf-8") as f:
                legacy_data = json.load(f)
            builtin_providers = legacy_data.get("providers", {})
            custom_providers = legacy_data.get("custom_providers", {})
            active_model = legacy_data.get("active_llm", {})
            # Migrate built-in providers
            for provider_id, config in builtin_providers.items():
                provider = self.get_provider(provider_id)
                if not provider:
                    logger.warning(
                        "Legacy provider '%s' not found in"
                        " registry, skipping migration for this provider.",
                        provider_id,
                    )
                    continue
                if "api_key" in config:
                    provider.api_key = config["api_key"]
                if "extra_models" in config:
                    provider.extra_models = [
                        ModelInfo.model_validate(model)
                        for model in config["extra_models"]
                    ]
                if not provider.freeze_url and "base_url" in config:
                    provider.base_url = config["base_url"]
                self._save_provider(provider, is_builtin=True)
            # Migrate custom providers
            for provider_id, data in custom_providers.items():
                custom_provider = OpenAIProvider(
                    id=provider_id,
                    name=data.get("name", provider_id),
                    base_url=data.get("base_url", ""),
                    api_key=data.get("api_key", ""),
                    is_custom=True,
                )
                if "models" in data:
                    # migrate models to extra_models field
                    custom_provider.extra_models = [
                        ModelInfo.model_validate(model)
                        for model in data["models"]
                    ]
                if "chat_model" in data:
                    custom_provider.chat_model = data["chat_model"]
                self._save_provider(custom_provider, is_builtin=False)
            # Migrate active model (only provider_id, not model)
            if active_model:
                try:
                    # Convert legacy copaw-local provider_id
                    if active_model.get("provider_id") == "copaw-local":
                        active_model["provider_id"] = "qwenpaw-local"
                    self.active_model = ModelSlotConfig.model_validate(
                        active_model,
                    )
                    self.save_active_model(self.active_model)
                except Exception:
                    logger.warning(
                        "Failed to migrate active model, using default.",
                    )
            # Remove legacy file after migration
            try:
                os.remove(legacy_path)
            except Exception:
                logger.warning(
                    "Failed to remove legacy providers.json after migration.",
                )

    def _init_from_storage(self):
        """Initialize all providers and active model from disk storage."""
        # Load built-in providers
        for builtin in self.builtin_providers.values():
            provider = self.load_provider(builtin.id, is_builtin=True)
            if provider:
                # inherit user-configured base_url only when freeze_url=False
                if not builtin.freeze_url:
                    builtin.base_url = provider.base_url
                builtin.api_key = provider.api_key
                builtin_model_ids = {m.id for m in builtin.models}
                builtin.extra_models = [
                    m
                    for m in provider.extra_models
                    if m.id not in builtin_model_ids
                ]
                builtin.generate_kwargs.update(provider.generate_kwargs)
                # Restore per-model generate_kwargs for built-in models.
                # Collect from both stored built-in models and promoted
                # extra_models (models that were user-added but are now
                # part of the built-in list).
                stored_model_kwargs: dict = {}
                for m in provider.models:
                    if m.generate_kwargs:
                        stored_model_kwargs[m.id] = m.generate_kwargs
                for m in provider.extra_models:
                    if m.id in builtin_model_ids and m.generate_kwargs:
                        stored_model_kwargs.setdefault(
                            m.id,
                            m.generate_kwargs,
                        )
                if stored_model_kwargs:
                    for model in builtin.models:
                        if model.id in stored_model_kwargs:
                            model.generate_kwargs = stored_model_kwargs[
                                model.id
                            ]
        # Custom LLM providers are disabled (single OpenAI-compatible gateway).
        # Load active model config
        active_model = self.load_active_model()
        if active_model:
            self.active_model = active_model

        # Migrate copaw-local to qwenpaw-local for backwards compatibility
        self._migrate_copaw_config()

    def _apply_onerouter_env_config(self) -> None:
        """Apply API key and base URL from environment to the sole gateway."""
        prov = self.builtin_providers.get(ONEROUTER_PROVIDER_ID)
        if not prov:
            return
        url = EnvVarLoader.get_str("QWENPAW_ONEROUTER_BASE_URL", "").strip()
        prov.base_url = (url or ONEROUTER_DEFAULT_BASE_URL).rstrip("/")
        key = EnvVarLoader.get_str("QWENPAW_ONEROUTER_API_KEY", "").strip()
        if not key:
            key = EnvVarLoader.get_str("ONEROUTER_API_KEY", "").strip()
        if not key:
            # Common when using OpenAI-compatible gateways with generic .env naming
            key = EnvVarLoader.get_str("OPENAI_API_KEY", "").strip()
        if key:
            prov.api_key = key

    def _normalize_active_model_for_locked_gateway(self) -> None:
        """Force global active LLM to the built-in OneRouter gateway."""
        oid = ONEROUTER_PROVIDER_ID
        prov = self.get_provider(oid)
        if not prov:
            return
        default_model = EnvVarLoader.get_str(
            "QWENPAW_ONEROUTER_DEFAULT_MODEL",
            "",
        ).strip()

        def _onerouter_allows_blank_model() -> bool:
            return onerouter_omit_model_in_request() and not default_model

        def _slot_ok(slot: ModelSlotConfig | None) -> bool:
            if not slot or not slot.provider_id:
                return False
            p = self.get_provider(slot.provider_id)
            if p is None or p.id != oid:
                return False
            if (slot.model or "").strip():
                return True
            return _onerouter_allows_blank_model()

        if not _slot_ok(self.active_model):
            raw = ""
            if self.active_model and self.active_model.model:
                raw = self.active_model.model.strip()
            merged = default_model or raw
            mid = ProviderManager.resolve_effective_llm_model_id(oid, merged)
            self.active_model = ModelSlotConfig(provider_id=oid, model=mid)
            self.save_active_model(self.active_model)
        elif self.active_model and self.active_model.provider_id != oid:
            raw = (
                self.active_model.model.strip()
                if self.active_model.model
                else ""
            )
            merged = default_model or raw
            mid = ProviderManager.resolve_effective_llm_model_id(oid, merged)
            self.active_model = ModelSlotConfig(provider_id=oid, model=mid)
            self.save_active_model(self.active_model)
        else:
            mid = ProviderManager.resolve_effective_llm_model_id(
                oid,
                self.active_model.model,
            )
            if (self.active_model.model or "").strip() != mid:
                self.active_model = ModelSlotConfig(provider_id=oid, model=mid)
                self.save_active_model(self.active_model)

    def _apply_default_annotations(self):
        """Apply doc-based default annotations for unprobed models.

        Models that already carry static annotations (supports_image /
        supports_video set at definition time) only need the derived
        supports_multimodal flag computed.  Models with no annotations
        at all fall back to the ExpectedCapabilityRegistry.
        """
        from .capability_baseline import ExpectedCapabilityRegistry

        registry = ExpectedCapabilityRegistry()
        for provider in self.builtin_providers.values():
            for model in provider.models:
                # Already fully annotated (e.g. by a prior probe) → skip
                if model.supports_multimodal is not None:
                    continue

                # Static annotations present → compute derived flag only
                if (
                    model.supports_image is not None
                    or model.supports_video is not None
                ):
                    model.supports_multimodal = bool(
                        model.supports_image or model.supports_video,
                    )
                    continue

                # No annotations at all → fall back to registry
                expected = registry.get_expected(provider.id, model.id)
                if expected:
                    model.supports_image = expected.expected_image
                    model.supports_video = expected.expected_video
                    model.supports_multimodal = bool(
                        expected.expected_image or expected.expected_video,
                    )
                    model.probe_source = "documentation"

    async def _resume_local_model(self, local_manager) -> None:
        """Resume the active local model server from the previous run."""
        if self.get_provider("qwenpaw-local") is None:
            return

        def _clear_local_provider():
            self.update_provider(
                "qwenpaw-local",
                {
                    "base_url": "",
                    "extra_models": [],
                },
            )

        local_models = self.get_provider("qwenpaw-local").extra_models
        model_id = local_models[0].id if local_models else None
        if model_id is None:
            return

        installed, _ = local_manager.check_llamacpp_installation()
        if not installed:
            logger.info(
                "Skipping local model restore because"
                " llama.cpp is not installed.",
            )
            _clear_local_provider()
            return

        if not local_manager.is_model_downloaded(model_id):
            logger.warning(
                "Skipping local model restore because"
                " model is not downloaded: %s",
                model_id,
            )
            _clear_local_provider()
            return

        try:
            setup_result = await local_manager.setup_server(model_id)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Failed to restore local model server for %s: %s",
                model_id,
                exc,
            )
            _clear_local_provider()
            return

        self.update_provider(
            "qwenpaw-local",
            {
                "base_url": f"http://127.0.0.1:{setup_result.port}/v1",
                "extra_models": [setup_result.model_info],
            },
        )

    def register_plugin_provider(
        self,
        provider_id: str,
        provider_class,
        label: str,
        base_url: str,
        metadata: Dict,
    ):
        """Register a plugin provider.

        Args:
            provider_id: Provider ID
            provider_class: Provider class
            label: Display label
            base_url: API base URL
            metadata: Additional metadata
        """
        # Get default models from provider class if available
        default_models = []
        if hasattr(provider_class, "get_default_models"):
            try:
                default_models = provider_class.get_default_models()
            except Exception as e:
                logger.warning(
                    f"Failed to get default models for {provider_id}: {e}",
                )

        # Create ProviderInfo
        provider_info = ProviderInfo(
            id=provider_id,
            name=label,
            base_url=base_url,
            api_key="",  # Will be configured by user
            chat_model=metadata.get("chat_model", "OpenAIChatModel"),
            models=default_models,  # Add default models
            is_custom=False,  # Mark as non-custom (like builtin, cannot be
            # deleted)
            require_api_key=metadata.get("require_api_key", True),
            meta=metadata.get("meta", {}),  # Pass meta from plugin
        )

        # Check if there's a saved configuration for this plugin provider
        saved_config_path = self.plugin_path / f"{provider_id}.json"
        if saved_config_path.exists():
            try:
                with open(saved_config_path, "r", encoding="utf-8") as f:
                    saved_config = json.load(f)

                # Decrypt sensitive fields
                saved_config = decrypt_dict_fields(
                    saved_config,
                    PROVIDER_SECRET_FIELDS,
                )

                # Merge saved config (api_key, base_url, extra_models, etc)
                if "api_key" in saved_config:
                    provider_info.api_key = saved_config["api_key"]
                if "base_url" in saved_config:
                    provider_info.base_url = saved_config["base_url"]
                if "generate_kwargs" in saved_config:
                    provider_info.generate_kwargs = saved_config[
                        "generate_kwargs"
                    ]
                # Load extra_models from saved config
                if "extra_models" in saved_config:
                    provider_info.extra_models = [
                        ModelInfo.model_validate(
                            model.model_dump()
                            if isinstance(model, BaseModel)
                            else model,
                        )
                        for model in saved_config["extra_models"]
                    ]
                logger.info(
                    f"✓ Loaded saved config for plugin provider: "
                    f"{provider_id} "
                    f"({len(provider_info.extra_models)} extra model(s))",
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load saved config for {provider_id}: {e}",
                )

        # Register to internal dict
        self.plugin_providers[provider_id] = {
            "info": provider_info,
            "class": provider_class,
        }

        logger.info(
            f"✓ Registered plugin provider: {provider_id} "
            f"with {len(default_models)} default model(s)",
        )

    @staticmethod
    def get_instance() -> "ProviderManager":
        """Get the singleton instance of ProviderManager."""
        if ProviderManager._instance is None:
            ProviderManager._instance = ProviderManager()
        return ProviderManager._instance

    @staticmethod
    def resolve_effective_llm_model_id(
        provider_id: str,
        configured_model_id: str,
    ) -> str:
        """Resolve model id for OneRouter (OpenAI-compatible chat completions).

        Priority:
        1. ``QWENPAW_ONEROUTER_DEFAULT_MODEL`` when set (always returned; sent
           as the JSON ``model`` field).
        2. When ``onerouter_omit_model_in_request()`` is True: stored id if
           non-empty and not legacy Baichuan; otherwise ``""`` (no ``model`` in
           the HTTP body — gateway routes from the key).
        3. When omit is False: ``ONEROUTER_FALLBACK_MODEL_ID`` for empty or
           legacy Baichuan ids; else the stored id.
        """
        if provider_id != ONEROUTER_PROVIDER_ID:
            return configured_model_id
        forced = EnvVarLoader.get_str(
            "QWENPAW_ONEROUTER_DEFAULT_MODEL",
            "",
        ).strip()
        if forced:
            return forced
        mid = (configured_model_id or "").strip()
        if onerouter_omit_model_in_request():
            if not mid or "baichuan" in mid.lower():
                return ""
            return mid
        if not mid or "baichuan" in mid.lower():
            return ONEROUTER_FALLBACK_MODEL_ID
        return mid

    @staticmethod
    def get_active_chat_model() -> ChatModelBase:
        """Get the currently active provider/model configuration."""
        manager = ProviderManager.get_instance()
        model = manager.get_active_model()
        if model is None or model.provider_id == "":
            raise ProviderError(
                message="No active model configured.",
            )
        allow_blank = (
            model.provider_id == ONEROUTER_PROVIDER_ID
            and onerouter_omit_model_in_request()
            and not EnvVarLoader.get_str(
                "QWENPAW_ONEROUTER_DEFAULT_MODEL",
                "",
            ).strip()
        )
        if model.model == "" and not allow_blank:
            raise ProviderError(
                message="No active model configured.",
            )
        provider = manager.get_provider(model.provider_id)
        if provider is None:
            raise ProviderError(
                message=f"Active provider '{model.provider_id}' not found.",
            )
        mid = ProviderManager.resolve_effective_llm_model_id(
            model.provider_id,
            model.model,
        )
        return provider.get_chat_model_instance(mid)
