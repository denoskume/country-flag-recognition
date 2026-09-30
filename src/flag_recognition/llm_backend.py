"""Provider-agnostic LLM client for Flag Intelligence.

Default backend: Ollama.
The adapter intentionally mimics the small subset of the OpenAI Responses API
used by report_writer.py so the existing report pipeline can stay unchanged.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import requests


def llm_backend_name() -> str:
    return os.getenv("FLAG_INTELLIGENCE_LLM_BACKEND", "ollama").strip().lower()


def llm_model_name() -> str:
    if llm_backend_name() == "ollama":
        return os.getenv("FLAG_INTELLIGENCE_OLLAMA_MODEL", "qwen3:8b").strip()
    return os.getenv("FLAG_INTELLIGENCE_WRITER_MODEL", "gpt-5.6-sol").strip()


def llm_auth_token() -> str:
    if llm_backend_name() == "ollama":
        return "ollama-local"
    return os.getenv("OPENAI_API_KEY", "").strip()


def ollama_base_url() -> str:
    return os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    ).strip().rstrip("/")


def _ollama_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    bearer = os.getenv("OLLAMA_AUTH_BEARER", "").strip()
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    return headers


def _messages_from_input(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]

    messages: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user").strip() or "user"
            content = item.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        text_part = part.get("text") or part.get("content")
                        if text_part:
                            parts.append(str(text_part))
                text = "\n".join(parts)
            else:
                text = str(content or "")
            messages.append({"role": role, "content": text})

    if not messages:
        messages.append({"role": "user", "content": str(value or "")})
    return messages


def _schema_from_text_config(text_config: Any) -> dict[str, Any] | None:
    if not isinstance(text_config, dict):
        return None
    fmt = text_config.get("format")
    if not isinstance(fmt, dict):
        return None
    if fmt.get("type") != "json_schema":
        return None
    schema = fmt.get("schema")
    return schema if isinstance(schema, dict) else None


class _OllamaResponses:
    def __init__(self, timeout: float | None = None):
        self.timeout = float(
            timeout
            if timeout is not None
            else os.getenv("FLAG_INTELLIGENCE_OLLAMA_TIMEOUT", "180")
        )

    def create(
        self,
        *,
        model: str | None = None,
        input: Any,
        text: Any = None,
        max_output_tokens: int | None = None,
        reasoning: Any = None,
        tools: Any = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> SimpleNamespace:
        selected_model = llm_model_name()
        messages = _messages_from_input(input)
        schema = _schema_from_text_config(text)

        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "think": os.getenv(
                "FLAG_INTELLIGENCE_OLLAMA_THINK",
                "false",
            ).strip().lower() in {"1", "true", "yes", "on"},
            "options": {
                "temperature": 0.2,
            },
        }

        if max_output_tokens is not None:
            payload["options"]["num_predict"] = max(16, int(max_output_tokens))

        if schema is not None:
            payload["format"] = schema

        try:
            response = requests.post(
                f"{ollama_base_url()}/api/chat",
                headers=_ollama_headers(),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama request failed at {ollama_base_url()} "
                f"after timeout={self.timeout:.0f}s: {exc}"
            ) from exc
        data = response.json()

        message = data.get("message") if isinstance(data, dict) else None
        output_text = ""
        if isinstance(message, dict):
            output_text = str(message.get("content") or "")
        elif isinstance(data, dict):
            output_text = str(data.get("response") or "")

        if not output_text.strip():
            thinking = ""
            if isinstance(message, dict):
                thinking = str(message.get("thinking") or "")
            raise RuntimeError(
                "Ollama returned no answer content"
                + (f"; thinking_chars={len(thinking)}" if thinking else "")
            )

        return SimpleNamespace(output_text=output_text)


class FlagIntelligenceClient:
    """Compatibility client for Ollama or optional OpenAI backend."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int = 0,
    ):
        backend = llm_backend_name()
        if backend == "ollama":
            self.responses = _OllamaResponses(timeout=timeout)
            return

        if backend == "openai":
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
                timeout=timeout,
                max_retries=max_retries,
            )
            self.responses = client.responses
            return

        raise RuntimeError(
            "Unsupported FLAG_INTELLIGENCE_LLM_BACKEND="
            f"{backend!r}. Use 'ollama' or 'openai'."
        )


def probe_llm_backend() -> tuple[bool, str]:
    backend = llm_backend_name()

    if backend == "ollama":
        try:
            response = requests.get(
                f"{ollama_base_url()}/api/tags",
                headers=_ollama_headers(),
                timeout=8.0,
            )
            response.raise_for_status()
            data = response.json()
            models = data.get("models", []) if isinstance(data, dict) else []
            names = {
                str(item.get("name") or "")
                for item in models
                if isinstance(item, dict)
            }
            wanted = llm_model_name()
            if wanted in names or any(
                name.split(":")[0] == wanted.split(":")[0]
                for name in names
            ):
                return True, f"ollama ok: {wanted}"
            available = ", ".join(sorted(name for name in names if name)[:8])
            return False, (
                f"Ollama is reachable but model {wanted!r} is not installed. "
                f"Available: {available or 'none'}"
            )
        except Exception as exc:
            return False, (
                f"Ollama unavailable at {ollama_base_url()}: "
                f"{type(exc).__name__}: {str(exc)[:400]}"
            )

    if backend == "openai":
        token = llm_auth_token()
        if not token:
            return False, "OPENAI_API_KEY missing"
        try:
            client = FlagIntelligenceClient(
                api_key=token,
                timeout=15.0,
                max_retries=0,
            )
            response = client.responses.create(
                model=llm_model_name(),
                input="Reply with OK only.",
                max_output_tokens=16,
            )
            if str(response.output_text or "").strip():
                return True, "openai ok"
            return False, "empty OpenAI API response"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {str(exc)[:500]}"

    return False, f"unsupported backend: {backend}"
