from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app as app_module
from BookVoiceParser.book_voice_parser.audit import _call_llm
from BookVoiceParser.book_voice_parser.batch_llm_attributor import BatchConfig, BatchLLMAttributor
from BookVoiceParser.book_voice_parser.spc_ranker import OpenAICompatibleSPCRanker
from schemas import CheckFilesRequest, LLMConfig, ParseV2Request
from security import SecurityError, validate_remote_url


def test_request_limit_rejects_content_length_and_chunked_body():
    mini = FastAPI()
    mini.add_middleware(app_module.RequestSizeLimitMiddleware, max_bytes=8)

    @mini.post("/")
    async def read_body(request: Request):
        return {"size": len(await request.body())}

    with TestClient(mini) as client:
        assert client.post("/", content=b"123456789").status_code == 413
        assert client.post("/", content=iter([b"1234", b"56789"])).status_code == 413


def test_remote_url_policy_fails_closed_on_dns_error(monkeypatch):
    def fail_resolution(*_args, **_kwargs):
        raise OSError("dns unavailable")

    monkeypatch.setattr("security.socket.getaddrinfo", fail_resolution)
    with pytest.raises(SecurityError):
        validate_remote_url("https://unresolved.example/v1")


def test_all_book_voice_parser_clients_reject_metadata_endpoint():
    base_url = "http://169.254.169.254"
    with pytest.raises(SecurityError):
        BatchLLMAttributor(BatchConfig(base_url=base_url))._normalize_base_url()
    with pytest.raises(SecurityError):
        OpenAICompatibleSPCRanker(SimpleNamespace(base_url=base_url))._normalize_base_url()
    with pytest.raises(SecurityError):
        _call_llm("test", {"base_url": base_url, "model": "test"})


def test_parse_v2_schema_and_api_smoke(monkeypatch):
    req = ParseV2Request(text="Narrator said: \"hello.\"")
    assert req.dense_llm is None
    monkeypatch.setenv("AUDIOBOOKSTUDIO_API_TOKEN", "test-token")
    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/parse_v2",
            headers={"x-audiobookstudio-token": "test-token"},
            json={"text": "Narrator said: \"hello.\"", "role_hints": ["Narrator"]},
        )
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_schema_limits_reject_oversized_collections_and_urls():
    with pytest.raises(ValidationError):
        CheckFilesRequest(files=["x"] * 2001)
    with pytest.raises(ValidationError):
        LLMConfig(base_url="https://" + "a" * 2048, api_key="x", model="m")
