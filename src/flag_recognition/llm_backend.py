"""Provider-agnostic LLM client for Flag Intelligence.

Default backend: GroqCloud.
The adapter exposes the subset of the OpenAI Responses API used by
report_writer.py so the report pipeline does not need provider-specific code.
"""

from __future__ import annotations

import os

import requests


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


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
                max_retries=max_retries,
            )
            self.responses = client.responses
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
            self.responses = client.responses
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
