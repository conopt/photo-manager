"""Opt-in image inference through an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel, Field, SecretStr

from cull.config import VLM_JPEG_QUALITY


class RemoteVlmConfig(BaseModel):
    """Connection settings; credentials are never included in diagnostics."""

    base_url: str = Field(repr=False)
    model: str
    api_key: SecretStr | None = None
    timeout: float = 120.0


class RemoteVlmError(RuntimeError):
    """An endpoint or response failed, without retaining its body or credentials."""


def remote_config(model: str) -> RemoteVlmConfig | None:
    """Read optional endpoint settings; an unset URL retains local MLX inference."""
    base_url = os.environ.get("PHOTO_MANAGER_VLM_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        return None
    try:
        parsed = urlsplit(base_url)
        valid = (
            parsed.scheme in {"http", "https"} and parsed.hostname
            and not parsed.username and not parsed.password
            and not parsed.query and not parsed.fragment
        )
        parsed.port  # Validate malformed port numbers, too.
        timeout = float(os.environ.get("PHOTO_MANAGER_VLM_TIMEOUT", "120"))
        if not valid or not 0 < timeout < float("inf") or not model.strip():
            raise ValueError
    except ValueError:
        raise ValueError(
            "Invalid remote VLM settings: use an HTTP(S) base URL without credentials, "
            "query or fragment, a nonempty --model, and a finite positive timeout."
        ) from None
    key = os.environ.get("PHOTO_MANAGER_VLM_API_KEY")
    return RemoteVlmConfig(
        base_url=base_url, model=model,
        api_key=SecretStr(key) if key else None, timeout=timeout,
    )


class _NoRedirect(HTTPRedirectHandler):
    """Do not forward image payloads or authorization to a redirected endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def generate_remote(config: RemoteVlmConfig, call: Any, images: list[Any]) -> str:
    """Send already resized, upright images and return the assistant text."""
    content: list[dict[str, Any]] = [{"type": "text", "text": call.prompt}]
    for image in images:
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=VLM_JPEG_QUALITY)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        })
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": call.max_tokens,
        "temperature": call.temperature,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if config.api_key is not None:
        headers["Authorization"] = f"Bearer {config.api_key.get_secret_value()}"
    request = Request(
        config.base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST",
    )
    try:
        with build_opener(_NoRedirect()).open(request, timeout=config.timeout) as response:
            data = json.load(response)
    except HTTPError as exc:
        status = exc.code
        exc.close()
        raise RemoteVlmError(f"Remote VLM request failed (HTTP {status}).") from None
    except (URLError, TimeoutError, OSError, HTTPException):
        raise RemoteVlmError("Remote VLM connection failed or timed out.") from None
    except (ValueError, UnicodeError):
        raise RemoteVlmError("Remote VLM returned invalid JSON.") from None
    try:
        text = data["choices"][0]["message"]["content"]
        if isinstance(text, list):
            text = "".join(part["text"] for part in text if part.get("type") == "text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        raise RemoteVlmError("Remote VLM returned no assistant text.") from None
    return text
