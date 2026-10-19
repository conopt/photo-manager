"""Real HTTP and image encoding against an in-process, credential-free server."""

from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import pytest
from PIL import Image

from cull.remote_vlm import RemoteVlmError, remote_config
from cull.vlm_registry import VLMResolutionError, run_vlm_preflight
from cull.vlm_session import VlmGenerateInput, vlm_session


@pytest.fixture
def endpoint(monkeypatch):
    state = {"requests": [], "status": 200, "delay": 0, "response": {"choices": [{"message": {"content": "ok"}}]}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["requests"].append({
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
            })
            time.sleep(state["delay"])
            self.send_response(state["status"])
            if state["status"] == 302:
                self.send_header("Location", "/redirected")
            self.end_headers()
            body = state["response"]
            try:
                self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())
            except BrokenPipeError:
                pass  # Expected when the timeout test has already disconnected.

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("PHOTO_MANAGER_VLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1/")
    monkeypatch.delenv("PHOTO_MANAGER_VLM_API_KEY", raising=False)
    monkeypatch.delenv("PHOTO_MANAGER_VLM_TIMEOUT", raising=False)
    # A configured endpoint must not resolve a local model or load MLX.
    def local_forbidden(*args):
        raise AssertionError("local model access attempted")
    monkeypatch.setattr("cull.vlm_session.resolve_alias", local_forbidden)
    monkeypatch.setattr("cull.vlm_session._get_mlx_vlm", local_forbidden)
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_remote_multimodal_request_and_session_lifecycle(endpoint, tmp_path, monkeypatch):
    paths = [tmp_path / "first.jpg", tmp_path / "second.hif"]
    Image.new("RGB", (2048, 1024), "red").save(paths[0])
    exif = Image.Exif()
    exif[306] = "2026:08:01 12:34:56"
    Image.new("RGB", (48, 32), "blue").save(paths[1], format="HEIF", exif=exif.tobytes())
    monkeypatch.setenv("PHOTO_MANAGER_VLM_API_KEY", "example-test-token")
    run_vlm_preflight("example-vision-model")
    with vlm_session("example-vision-model") as session:
        result = session.generate(VlmGenerateInput(
            prompt="Compare these photographs.", images=paths, max_tokens=90, temperature=0.2,
        ))
    assert result == "ok"
    request = endpoint["requests"][0]
    assert request["path"] == "/v1/chat/completions"
    assert request["auth"] == "Bearer example-test-token"
    body = request["body"]
    assert body["model"] == "example-vision-model"
    assert body["max_tokens"] == 90 and body["temperature"] == 0.2
    assert body["stream"] is False
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Compare these photographs."}
    sizes = []
    for part in content[1:]:
        header, encoded = part["image_url"]["url"].split(",", 1)
        assert header == "data:image/jpeg;base64"
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            sizes.append(image.size)
            assert not image.getexif()
    assert sizes == [(1024, 512), (48, 32)]
    with pytest.raises(RuntimeError, match="after unload"):
        session.generate(VlmGenerateInput(prompt="again", images=[]))


@pytest.mark.parametrize("response", [
    {"choices": [{"message": {"content": "answer"}}]},
    {"choices": [{"message": {"content": [{"type": "text", "text": "answer"}]}}]},
])
def test_remote_text_response_formats(endpoint, response):
    endpoint["response"] = response
    with vlm_session("vision") as session:
        assert session.generate(VlmGenerateInput(prompt="Question", images=[])) == "answer"
    assert endpoint["requests"][0]["auth"] is None


@pytest.mark.parametrize("status,response,expected", [
    (401, {"error": "example-test-token"}, "HTTP 401"),
    (302, {}, "HTTP 302"),
    (200, b"not json", "invalid JSON"),
    (200, {"choices": []}, "no assistant text"),
    (200, {"choices": [{"message": {"content": None}}]}, "no assistant text"),
])
def test_remote_failures_are_clear_and_do_not_expose_response_bodies(endpoint, status, response, expected):
    endpoint.update(status=status, response=response)
    with vlm_session("vision") as session:
        with pytest.raises(RemoteVlmError, match=expected) as failure:
            session.generate(VlmGenerateInput(prompt="Question", images=[]))
    assert "example-test-token" not in str(failure.value)
    assert len(endpoint["requests"]) == 1


@pytest.mark.parametrize("url", ["file:///tmp/server", "http://user:password@example.test/v1", "https://example.test/v1?key=value"])
def test_preflight_rejects_invalid_endpoint_without_echoing_it(monkeypatch, url):
    monkeypatch.setenv("PHOTO_MANAGER_VLM_BASE_URL", url)
    with pytest.raises(VLMResolutionError, match="Invalid remote VLM") as failure:
        run_vlm_preflight("vision")
    assert url not in str(failure.value)


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "invalid"])
def test_preflight_requires_finite_positive_timeout(endpoint, monkeypatch, timeout):
    monkeypatch.setenv("PHOTO_MANAGER_VLM_TIMEOUT", timeout)
    with pytest.raises(VLMResolutionError, match="Invalid remote VLM"):
        run_vlm_preflight("vision")


def test_no_endpoint_retains_local_backend(monkeypatch):
    monkeypatch.delenv("PHOTO_MANAGER_VLM_BASE_URL", raising=False)
    assert remote_config("local-model") is None


def test_remote_timeout(endpoint, monkeypatch):
    endpoint["delay"] = 0.2
    monkeypatch.setenv("PHOTO_MANAGER_VLM_TIMEOUT", "0.02")
    with vlm_session("vision") as session:
        with pytest.raises(RemoteVlmError, match="timed out"):
            session.generate(VlmGenerateInput(prompt="Question", images=[]))


def test_event_scoring_uses_remote_session(endpoint, tmp_path):
    from cull.stage3.event_scoring import rate_photo
    from cull.stage3.vlm_scoring import VlmRequest, VlmScoreCallInput
    from cull.stage3.prompt import PromptContext

    path = tmp_path / "sample.hif"
    Image.new("RGB", (48, 32), "green").save(path, format="HEIF")
    endpoint["response"] = {"choices": [{"message": {"content": json.dumps({
        "rating": 5, "reason": "Clear subject and balanced composition.",
    })}}]}
    with vlm_session("vision") as session:
        result = rate_photo(VlmScoreCallInput(
            request=VlmRequest(image_path=path, model="vision", context=PromptContext()),
            session=session,
        ))
    assert result.rating == 5 and result.is_keeper
    assert len(endpoint["requests"]) == 1
