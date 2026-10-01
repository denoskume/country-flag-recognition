"""Provider-agnostic LLM client for Flag Intelligence.

Default backend: GroqCloud.
The adapter exposes the subset of the OpenAI Responses API used by
report_writer.py so the report pipeline does not need provider-specific code.
"""

from __future__ import annotations

import os
import re
import time
from types import SimpleNamespace
from typing import Any

import requests


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_GROQ_FALLBACK_MODEL = "openai/gpt-oss-120b"


def llm_backend_name() -> str:
    backend = os.getenv("FLAG_INTELLIGENCE_LLM_BACKEND", "groq").strip().lower()
    if backend == "ollama":
        return "groq"
    return backend or "groq"


def llm_model_name() -> str:
    backend = llm_backend_name()
    if backend == "groq":
        return os.getenv(
            "FLAG_INTELLIGENCE_GROQ_MODEL",
            DEFAULT_GROQ_MODEL,
        ).strip()
    if backend == "openai":
        return os.getenv(
            "FLAG_INTELLIGENCE_WRITER_MODEL",
            "gpt-5.6-luna",
        ).strip()
    raise RuntimeError(
        "Unsupported FLAG_INTELLIGENCE_LLM_BACKEND="
        f"{backend!r}. Use 'groq' or 'openai'."
    )


def llm_auth_token() -> str:
    backend = llm_backend_name()
    if backend == "groq":
        return os.getenv("GROQ_API_KEY", "").strip()
    if backend == "openai":
        return os.getenv("OPENAI_API_KEY", "").strip()
    return ""


def llm_credentials_available() -> bool:
    """Return whether the configured backend has credentials without probing the network."""
    return bool(llm_auth_token())


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "rate_limit" in text or "status=429" in text or "error code: 429" in text


def _retry_after_seconds(exc: Exception) -> float:
    """Read Groq's retry-after hint while keeping interactive waits bounded."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("retry-after") or headers.get("Retry-After")
    delay: float | None = None
    if raw is not None:
        try:
            delay = float(raw)
        except (TypeError, ValueError):
            delay = None

    if delay is None:
        match = re.search(
            r"(?:try again in|retry after)\s*([0-9]+(?:\.[0-9]+)?)\s*s",
            str(exc),
            flags=re.IGNORECASE,
        )
        if match:
            delay = float(match.group(1))

    if delay is None or delay <= 0:
        delay = 1.0

    max_wait = float(os.getenv("FLAG_INTELLIGENCE_RATE_LIMIT_MAX_WAIT", "3"))
    return max(0.25, min(delay, max(0.25, max_wait)))


def _extract_review_draft(kwargs: dict[str, Any]) -> str:
    """Recover already model-generated prose when only the review pass is rate-limited."""
    tools = kwargs.get("tools") or []
    has_browser_review = any(
        isinstance(tool, dict) and tool.get("type") == "browser_search"
        for tool in tools
    )
    if not has_browser_review:
        return ""

    payload = kwargs.get("input")
    messages = payload if isinstance(payload, list) else []
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or "")
        marker = "DRAFT SECTION:\n"
        if marker in content:
            return content.split(marker, 1)[1].strip()
    return ""


class _ResilientResponses:
    """Small reliability layer around Responses API calls.

    Groq rate limits are model-scoped in practice. A request gets one bounded retry on
    the primary model, then the same request is attempted on a compatible backup model.
    A review-stage outage never destroys prose that was already generated successfully.
    """

    def __init__(self, raw_responses: Any, *, backend: str):
        self._raw = raw_responses
        self._backend = backend

    def _attempt(self, kwargs: dict[str, Any], *, attempts: int = 2):
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._raw.create(**kwargs)
            except Exception as exc:
                last_exc = exc
                if not _is_rate_limit_error(exc) or attempt >= attempts - 1:
                    raise
                time.sleep(_retry_after_seconds(exc))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM request failed without an exception")

    def create(self, **kwargs):
        if self._backend != "groq":
            return self._raw.create(**kwargs)

        primary_kwargs = dict(kwargs)
        primary_model = str(primary_kwargs.get("model") or DEFAULT_GROQ_MODEL)

        try:
            return self._attempt(primary_kwargs, attempts=2)
        except Exception as primary_exc:
            if not _is_rate_limit_error(primary_exc):
                raise

            fallback_model = os.getenv(
                "FLAG_INTELLIGENCE_GROQ_FALLBACK_MODEL",
                DEFAULT_GROQ_FALLBACK_MODEL,
            ).strip()

            if fallback_model and fallback_model != primary_model:
                fallback_kwargs = dict(primary_kwargs)
                fallback_kwargs["model"] = fallback_model
                try:
                    return self._attempt(fallback_kwargs, attempts=2)
                except Exception as fallback_exc:
                    if not _is_rate_limit_error(fallback_exc):
                        raise

            draft = _extract_review_draft(primary_kwargs)
            if draft:
                print(
                    "[Flag Intelligence LLM] Review deferred after Groq rate limits; "
                    "preserving the already generated draft section."
                )
                return SimpleNamespace(output_text=draft)

            raise RuntimeError(
                "The language service is temporarily busy after retrying both Groq models."
            ) from primary_exc


class FlagIntelligenceClient:
    """Compatibility client for GroqCloud or OpenAI."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int = 0,
    ):
        from openai import OpenAI

        backend = llm_backend_name()

        if backend == "groq":
            token = api_key or os.getenv("GROQ_API_KEY", "")
            if not str(token).strip():
                raise RuntimeError("GROQ_API_KEY missing")
            client = OpenAI(
                api_key=token,
                base_url=GROQ_BASE_URL,
                timeout=timeout,
                max_retries=0,
            )
            self.responses = _ResilientResponses(
                client.responses,
                backend="groq",
            )
            return

        if backend == "openai":
            token = api_key or os.getenv("OPENAI_API_KEY", "")
            if not str(token).strip():
                raise RuntimeError("OPENAI_API_KEY missing")
            client = OpenAI(
                api_key=token,
                timeout=timeout,
                max_retries=max_retries,
            )
            self.responses = _ResilientResponses(
                client.responses,
                backend="openai",
            )
            return

        raise RuntimeError(
            "Unsupported FLAG_INTELLIGENCE_LLM_BACKEND="
            f"{backend!r}. Use 'groq' or 'openai'."
        )


def probe_llm_backend() -> tuple[bool, str]:
    backend = llm_backend_name()
    token = llm_auth_token()

    if not token:
        key_name = "GROQ_API_KEY" if backend == "groq" else "OPENAI_API_KEY"
        return False, f"{key_name} missing"

    if backend == "groq":
        try:
            response = requests.post(
                f"{GROQ_BASE_URL}/responses",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": llm_model_name(),
                    "input": "Reply with OK only.",
                    "max_output_tokens": 16,
                },
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("status") in {"completed", "in_progress"}:
                return True, f"groq ok: {llm_model_name()}"
            return False, f"unexpected Groq response: {str(data)[:300]}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {str(exc)[:500]}"

    try:
        client = FlagIntelligenceClient(
            api_key=token,
            timeout=10.0,
            max_retries=0,
        )
        response = client.responses.create(
            model=llm_model_name(),
            input="Reply with OK only.",
            max_output_tokens=16,
        )
        if str(response.output_text or "").strip():
            return True, f"{backend} ok: {llm_model_name()}"
        return False, f"empty {backend} API response"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:500]}"