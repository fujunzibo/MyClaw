# -*- coding: utf-8 -*-
"""OpenAI chat model compatibility wrappers."""

from __future__ import annotations

import inspect
import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Literal, Type

import warnings

from agentscope.model import OpenAIChatModel
from agentscope.model._model_response import ChatResponse
from agentscope.model._openai_model import _format_audio_data_for_qwen_omni
from agentscope.tracing import trace_llm
from pydantic import BaseModel

from qwenpaw.constant import ONEROUTER_FALLBACK_MODEL_ID
from qwenpaw.local_models.tag_parser import (
    parse_tool_calls_from_text,
    text_contains_tool_call_tag,
)


def _openai_completion_json_body(body: dict[str, Any]) -> dict[str, Any]:
    """Strip OpenAI SDK sentinels for ``AsyncOpenAI.post`` JSON bodies."""
    try:
        from openai._types import Omit
        from openai._utils import strip_not_given
    except ImportError:
        return {k: v for k, v in body.items() if k != "model"}

    cleaned = strip_not_given(dict(body))
    return {k: v for k, v in cleaned.items() if not isinstance(v, Omit) and k != "model"}


async def _post_chat_completions_via_http(
    client: Any,
    body: dict[str, Any],
) -> Any:
    """POST ``/chat/completions`` without ``chat.completions.create``.

    The OpenAI Python SDK (1.x) requires a ``model`` argument on
    ``create()``, but some OpenAI-compatible gateways accept HTTP bodies
    without a ``model`` field.  ``AsyncOpenAI.post`` sends the JSON as given.
    """
    from openai._base_client import make_request_options
    from openai._streaming import AsyncStream
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

    payload = _openai_completion_json_body(body)
    stream = bool(payload.get("stream"))
    # Older openai-python releases omit ``security`` on ``make_request_options``;
    # bearer auth still comes from ``AsyncOpenAI`` client defaults.
    try:
        sig = inspect.signature(make_request_options)
        if "security" in sig.parameters:
            req_opts = make_request_options(security={"bearer_auth": True})
        else:
            req_opts = make_request_options()
    except TypeError:
        req_opts = {}

    return await client.post(
        "/chat/completions",
        cast_to=ChatCompletion,
        body=payload,
        stream=stream,
        stream_cls=AsyncStream[ChatCompletionChunk],
        options=req_opts,
    )


def _clone_with_overrides(obj: Any, **overrides: Any) -> Any:
    """Clone a stream object into a mutable namespace with overrides."""
    data = dict(getattr(obj, "__dict__", {}))
    data.update(overrides)
    return SimpleNamespace(**data)


def _sanitize_tool_call(tool_call: Any) -> Any | None:
    """Normalize a tool call for parser safety, or drop it if unusable."""
    if not hasattr(tool_call, "index"):
        return None

    function = getattr(tool_call, "function", None)
    if function is None:
        return None

    has_name = hasattr(function, "name")
    has_arguments = hasattr(function, "arguments")

    raw_name = getattr(function, "name", "")
    if isinstance(raw_name, str):
        safe_name = raw_name
    elif raw_name is None:
        safe_name = ""
    else:
        safe_name = str(raw_name)

    raw_arguments = getattr(function, "arguments", "")
    if isinstance(raw_arguments, str):
        safe_arguments = raw_arguments
    elif raw_arguments is None:
        safe_arguments = ""
    else:
        try:
            safe_arguments = json.dumps(raw_arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            safe_arguments = str(raw_arguments)

    if (
        has_name
        and has_arguments
        and isinstance(raw_name, str)
        and isinstance(
            raw_arguments,
            str,
        )
    ):
        return tool_call

    safe_function = SimpleNamespace(
        name=safe_name,
        arguments=safe_arguments,
    )
    return _clone_with_overrides(tool_call, function=safe_function)


def _sanitize_chunk(chunk: Any) -> Any:
    """Drop/normalize malformed tool-calls in a streaming chunk."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return chunk

    sanitized_choices: list[Any] = []
    changed = False

    for choice in choices:
        delta = getattr(choice, "delta", None)
        if delta is None:
            sanitized_choices.append(choice)
            continue

        raw_tool_calls = getattr(delta, "tool_calls", None)
        if not raw_tool_calls:
            sanitized_choices.append(choice)
            continue

        choice_changed = False
        sanitized_tool_calls: list[Any] = []
        for tool_call in raw_tool_calls:
            sanitized = _sanitize_tool_call(tool_call)
            if sanitized is not tool_call:
                choice_changed = True
            if sanitized is not None:
                sanitized_tool_calls.append(sanitized)

        if choice_changed:
            changed = True
            sanitized_delta = _clone_with_overrides(
                delta,
                tool_calls=sanitized_tool_calls,
            )
            sanitized_choice = _clone_with_overrides(
                choice,
                delta=sanitized_delta,
            )
            sanitized_choices.append(sanitized_choice)
            continue

        sanitized_choices.append(choice)

    if not changed:
        return chunk
    return _clone_with_overrides(chunk, choices=sanitized_choices)


def _sanitize_stream_item(item: Any) -> Any:
    """Sanitize either plain stream chunks or structured stream items."""
    if hasattr(item, "chunk"):
        chunk = item.chunk
        sanitized_chunk = _sanitize_chunk(chunk)
        if sanitized_chunk is chunk:
            return item
        return _clone_with_overrides(item, chunk=sanitized_chunk)

    return _sanitize_chunk(item)


class _SanitizedStream:
    """Proxy OpenAI async stream that sanitizes each emitted item and
    captures ``extra_content`` from tool-call chunks (used by Gemini
    thinking models to carry ``thought_signature``)."""

    def __init__(self, stream: Any):
        self._stream = stream
        self._ctx_stream: Any | None = None
        self.extra_contents: dict[str, Any] = {}

    async def __aenter__(self) -> "_SanitizedStream":
        self._ctx_stream = await self._stream.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> bool | None:
        return await self._stream.__aexit__(exc_type, exc, tb)

    def __aiter__(self) -> "_SanitizedStream":
        return self

    async def __anext__(self) -> Any:
        if self._ctx_stream is None:
            raise StopAsyncIteration
        item = await self._ctx_stream.__anext__()
        self._capture_extra_content(item)
        return _sanitize_stream_item(item)

    def _capture_extra_content(self, item: Any) -> None:
        """Store ``extra_content`` keyed by tool-call id."""
        chunk = getattr(item, "chunk", item)
        choices = getattr(chunk, "choices", None) or []
        for choice in choices:
            delta = getattr(choice, "delta", None)
            if not delta:
                continue
            for tc in getattr(delta, "tool_calls", None) or []:
                tc_id = getattr(tc, "id", None)
                if not tc_id:
                    continue
                extra = getattr(tc, "extra_content", None)
                if extra is None:
                    model_extra = getattr(tc, "model_extra", None)
                    if isinstance(model_extra, dict):
                        extra = model_extra.get("extra_content")
                if extra:
                    self.extra_contents[tc_id] = extra


class OpenAIChatModelCompat(OpenAIChatModel):
    """OpenAIChatModel with robust parsing for malformed tool-call chunks
    and transparent ``extra_content`` (Gemini thought_signature) relay."""

    # pylint: disable=too-many-branches
    async def _parse_openai_stream_response(
        self,
        start_datetime: datetime,
        response: Any,
        structured_model: Type[BaseModel] | None = None,
    ) -> AsyncGenerator[ChatResponse, None]:
        sanitized_response = _SanitizedStream(response)

        # Stable tag-extracted tool-call blocks across streaming chunks.
        # Keyed by positional strings so IDs stay consistent as chunks
        # accumulate.  Two sources: "thinking" blocks and plain "text" blocks.
        _think_tool_calls: dict[str, dict] = {}
        _text_tool_calls: dict[str, dict] = {}

        async for parsed in super()._parse_openai_stream_response(
            start_datetime=start_datetime,
            response=sanitized_response,
            structured_model=structured_model,
        ):
            # Attach extra_content (Gemini thought_signature) to tool_use
            # blocks.
            if sanitized_response.extra_contents:
                for block in parsed.content:
                    if block.get("type") != "tool_use":
                        continue
                    tool_id = block.get("id")
                    if not isinstance(tool_id, str):
                        continue
                    ec = sanitized_response.extra_contents.get(tool_id)
                    if ec:
                        block["extra_content"] = ec

            # Check whether the response already carries structured tool_use
            # blocks (either from the model or from extra_content above).
            has_tool_use = any(
                b.get("type") == "tool_use" for b in parsed.content
            )

            if has_tool_use:
                # Structured tool calls arrived — discard any tag-derived
                # ones, so we don't produce duplicates.
                _think_tool_calls.clear()
                _text_tool_calls.clear()
            else:
                # --- 1. Scan thinking blocks ---
                for block in parsed.content:
                    if block.get("type") != "thinking":
                        continue
                    thinking_text = block.get("thinking") or ""
                    if not text_contains_tool_call_tag(thinking_text):
                        continue

                    think_parsed = parse_tool_calls_from_text(thinking_text)
                    if not think_parsed.tool_calls:
                        continue

                    # Keep only the text before the first <tool_call>.
                    # Everything after is the model's simulated continuation
                    # (may include </tool_response>, </think> artefacts).
                    block["thinking"] = think_parsed.text_before.strip()

                    _think_tool_calls = {
                        f"thinking_{i}": {
                            "type": "tool_use",
                            "id": f"think_call_{i}",
                            "name": ptc.name,
                            "input": ptc.arguments,
                            "raw_input": ptc.raw_arguments,
                        }
                        for i, ptc in enumerate(think_parsed.tool_calls)
                    }

                # --- 2. Scan text/content blocks ---
                # Some models emit <tool_call> tags directly in their
                # response text instead of (or in addition to) thinking.
                new_content: list | None = None
                for i, block in enumerate(parsed.content):
                    if block.get("type") != "text":
                        continue
                    text = block.get("text") or ""
                    if not text_contains_tool_call_tag(text):
                        continue

                    text_parsed = parse_tool_calls_from_text(text)
                    # Keep only text_before; discard the tag block and
                    # everything after (same rationale as thinking).
                    clean_text = text_parsed.text_before.strip()
                    block["text"] = clean_text

                    if text_parsed.tool_calls:
                        _text_tool_calls = {
                            f"text_{j}": {
                                "type": "tool_use",
                                "id": f"text_call_{j}",
                                "name": ptc.name,
                                "input": ptc.arguments,
                                "raw_input": ptc.raw_arguments,
                            }
                            for j, ptc in enumerate(text_parsed.tool_calls)
                        }

                    # If the text block is now empty, mark it for removal.
                    if not clean_text:
                        if new_content is None:
                            new_content = list(parsed.content)
                        new_content[i] = None  # type: ignore[index]

                if new_content is not None:
                    parsed.content = [b for b in new_content if b is not None]

                extra = list(_think_tool_calls.values()) + list(
                    _text_tool_calls.values(),
                )
                if extra:
                    parsed.content = list(parsed.content) + extra

            yield parsed


class OpenAIChatModelOmitModelField(OpenAIChatModelCompat):
    """OpenAI-compatible chat without a ``model`` field in the request body.

    Some gateways (e.g. OneRouter) infer the model server-side from the API
    key, matching minimal ``curl`` / ``Invoke-WebRequest`` bodies that only
    send ``messages``.
    """

    # pylint: disable=too-many-branches
    @trace_llm
    async def __call__(  # type: ignore[override]
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "required"] | str | None = None,
        structured_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        if not isinstance(messages, list):
            raise ValueError(
                "OpenAI `messages` field expected type `list`, "
                f"got `{type(messages)}` instead.",
            )
        if not all("role" in msg and "content" in msg for msg in messages):
            raise ValueError(
                "Each message in the 'messages' list must contain a 'role' "
                "and 'content' key for OpenAI API.",
            )

        if self.model_name and "omni" in self.model_name.lower():
            _format_audio_data_for_qwen_omni(messages)

        kwargs_out = {
            "messages": messages,
            "stream": self.stream,
            **self.generate_kwargs,
            **kwargs,
        }
        if self.reasoning_effort and "reasoning_effort" not in kwargs_out:
            kwargs_out["reasoning_effort"] = self.reasoning_effort

        if tools:
            kwargs_out["tools"] = self._format_tools_json_schemas(tools)

        if tool_choice:
            if tool_choice == "any":
                warnings.warn(
                    '"any" is deprecated and will be removed in a future '
                    "version.",
                    DeprecationWarning,
                )
                tool_choice = "required"
            self._validate_tool_choice(tool_choice, tools)
            kwargs_out["tool_choice"] = self._format_tool_choice(tool_choice)

        if self.stream:
            kwargs_out["stream_options"] = {"include_usage": True}

        start_datetime = datetime.now()
        _sdk_structured_model = (
            (self.model_name or "").strip() or ONEROUTER_FALLBACK_MODEL_ID
        )

        if structured_model:
            import openai

            if tools or tool_choice:
                from agentscope._logging import logger as _log

                _log.warning(
                    "structured_model is provided. Both 'tools' and "
                    "'tool_choice' parameters will be overridden and "
                    "ignored. The model will only perform structured output "
                    "generation without calling any other tools.",
                )
            kwargs_out.pop("stream", None)
            kwargs_out.pop("tools", None)
            kwargs_out.pop("tool_choice", None)

            if self._structured_output_fallback:
                response = await self._structured_via_tool_call(
                    kwargs_out,
                    structured_model,
                    start_datetime,
                )
                if isinstance(response, AsyncGenerator):
                    return response
            else:
                kwargs_out["response_format"] = structured_model
                try:
                    if not self.stream:
                        response = await self.client.chat.completions.parse(
                            **kwargs_out,
                            model=_sdk_structured_model,
                        )
                    else:
                        response = self.client.chat.completions.stream(
                            **kwargs_out,
                            model=_sdk_structured_model,
                        )
                        return self._structured_stream_with_fallback(
                            start_datetime,
                            response,
                            structured_model,
                            kwargs_out,
                        )
                except openai.BadRequestError as e:
                    from agentscope._logging import logger as _log

                    _log.warning(
                        "response_format structured output failed (%s: %s), "
                        "falling back to tool-call based structured output. "
                        "Subsequent calls will use tool-call directly.",
                        type(e).__name__,
                        e,
                    )
                    self._structured_output_fallback = True
                    response = await self._structured_via_tool_call(
                        kwargs_out,
                        structured_model,
                        start_datetime,
                    )
                    if isinstance(response, AsyncGenerator):
                        return response
        else:
            response = await _post_chat_completions_via_http(
                self.client,
                kwargs_out,
            )

        if self.stream:
            return self._parse_openai_stream_response(
                start_datetime,
                response,
                structured_model,
            )

        parsed_response = self._parse_openai_completion_response(
            start_datetime,
            response,
            structured_model,
        )

        return parsed_response

    async def _structured_via_tool_call(  # type: ignore[override]
        self,
        kwargs: dict,
        structured_model: Type[BaseModel],
        start_datetime: datetime,
    ) -> Any:
        """Like ``OpenAIChatModel._structured_via_tool_call`` but HTTP POST
        without a ``model`` field (OpenAI SDK ``create`` requires ``model``).
        """
        from agentscope._utils._common import _create_tool_from_base_model

        kwargs = dict(kwargs)
        kwargs.pop("response_format", None)
        format_tool = _create_tool_from_base_model(structured_model)
        kwargs["tools"] = self._format_tools_json_schemas([format_tool])
        kwargs["tool_choice"] = self._format_tool_choice(
            format_tool["function"]["name"],
        )
        if self.stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        response = await _post_chat_completions_via_http(self.client, kwargs)
        if self.stream:
            return self._parse_openai_stream_response(
                start_datetime,
                response,
                structured_model,
            )
        return response
