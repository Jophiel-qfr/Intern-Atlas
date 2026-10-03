from __future__ import annotations

import json
import re
from pathlib import Path

import fitz
import httpx
import pytest
from fastapi.testclient import TestClient

import intern_atlas.server as server_module
from intern_atlas.db import connect
from intern_atlas.lineage_llm import OpenAICompatibleLineageClient
from intern_atlas.server import create_app


def _make_app(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    papers_dir = data_dir / "papers"
    papers_dir.mkdir(parents=True)
    monkeypatch.setenv("INTERN_ATLAS_DATA_DIR", str(data_dir))
    database = tmp_path / "graph.db"
    connection = connect(database)
    connection.close()
    return create_app(database), papers_dir


def _write_pdf(path: Path, title: str) -> None:
    document = fitz.open()
    page = document.new_page()
    lines = [
        "Abstract",
        f"{title} studies wearable sensor human activity recognition.",
        "1. Introduction",
        "Activity recognition from inertial signals remains challenging.",
        "2. Methods",
        "The method uses convolutional feature extraction and recurrent temporal modeling.",
        "3. Experiments",
        "We evaluate the approach on wearable activity recognition datasets.",
        "4. Results",
        "The reported results compare recognition performance across subjects.",
        "Conclusion",
        "The method supports activity recognition from sensor data.",
        "References",
        "This text must not be included after the references heading.",
    ]
    for index, line in enumerate(lines):
        page.insert_text((45, 35 + index * 26), line, fontsize=10)
    metadata = dict(document.metadata)
    metadata["title"] = title
    document.set_metadata(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    document.close()


def _request(source: str = "source.pdf", target: str = "target.pdf") -> dict[str, str]:
    return {"source_filename": source, "target_filename": target}


def _clear_llm_environment(monkeypatch) -> None:
    for name in (
        "LLM_API_KEY",
        "S4S_LLM_API_KEY",
        "OPENAI_API_KEY",
        "LLM_BASE_URL",
        "S4S_LLM_BASE_URL",
        "OPENAI_BASE_URL",
        "LLM_MODEL",
        "S4S_LLM_MODEL",
        "OPENAI_MODEL",
        "S4S_LLM_MODELS",
        "LLM_MODELS",
    ):
        monkeypatch.delenv(name, raising=False)


def _write_pair(papers_dir: Path) -> None:
    _write_pdf(papers_dir / "source.pdf", "Source Method Paper")
    _write_pdf(papers_dir / "target.pdf", "Target Method Paper")


def test_preview_returns_only_selected_evidence_and_never_constructs_llm(
    tmp_path, monkeypatch
) -> None:
    app, papers_dir = _make_app(tmp_path, monkeypatch)
    _write_pair(papers_dir)

    def forbidden_client():
        pytest.fail("preview must never instantiate the LLM client")

    monkeypatch.setattr(server_module, "OpenAICompatibleLineageClient", forbidden_client)
    with TestClient(app) as client:
        response = client.post("/api/local/lineage/preview", json=_request())

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"]["paper_id"] == "source"
    assert payload["target"]["paper_title"] == "Target Method Paper"
    assert payload["selected_evidence_count"] == len(
        payload["source_evidence"] + payload["target_evidence"]
    )
    assert payload["source_evidence"] and payload["target_evidence"]
    assert all(item["evidence_id"].startswith("S") for item in payload["source_evidence"])
    assert all(item["evidence_id"].startswith("T") for item in payload["target_evidence"])
    assert payload["total_characters"] == sum(
        len(item["text"])
        for item in payload["source_evidence"] + payload["target_evidence"]
    )
    assert all("selection_score" not in item for item in payload["source_evidence"])
    assert "API Key" not in response.text


def test_preview_rejects_same_pdf_and_path_traversal_without_leaking_paths(
    tmp_path, monkeypatch
) -> None:
    app, papers_dir = _make_app(tmp_path, monkeypatch)
    _write_pair(papers_dir)
    _write_pdf(tmp_path / "outside.pdf", "Outside Paper")

    with TestClient(app) as client:
        same = client.post("/api/local/lineage/preview", json=_request("source.pdf", "source.pdf"))
        traversal = client.post(
            "/api/local/lineage/preview",
            json=_request("..\\outside.pdf", "target.pdf"),
        )
        missing = client.post(
            "/api/local/lineage/preview", json=_request("missing.pdf", "target.pdf")
        )

    assert same.status_code == 400
    assert same.json()["detail"] == "源论文和目标论文不能相同。"
    assert traversal.status_code == 400
    assert traversal.json()["detail"] == "PDF 文件名无效。"
    assert missing.status_code == 404
    assert str(tmp_path) not in same.text + traversal.text + missing.text


def test_analyze_uses_mock_transport_and_builds_evidence_backed_lineage(
    tmp_path, monkeypatch
) -> None:
    app, papers_dir = _make_app(tmp_path, monkeypatch)
    _write_pair(papers_dir)
    _clear_llm_environment(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://mock-llm.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-secret-that-must-not-return")
    monkeypatch.setenv("LLM_MODEL", "mock-lineage-model")
    observed: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        observed.append(body)
        assert request.headers["authorization"] == "Bearer test-secret-that-must-not-return"
        assert body["max_tokens"] == 4000
        user_text = body["messages"][1]["content"]
        ids = re.findall(r"\[([ST]\d+)\]", user_text)
        source_id = next(item for item in ids if item.startswith("S"))
        target_id = next(item for item in ids if item.startswith("T"))
        raw = {
            "analysis_status": "probable",
            "relation_type": "extends",
            "confidence": 0.72,
            "uncertainty": "Some comparison details remain incomplete.",
            "relation_evidence_ids": [source_id, target_id],
            "inherited_components": [],
            "changed_components": [],
            "added_components": [],
            "removed_components": [],
            "problem_addressed": None,
            "claimed_contribution": None,
            "experimental_evidence": [],
            "limitations": [],
            "method_changes": [],
        }
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]},
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        server_module,
        "OpenAICompatibleLineageClient",
        lambda: OpenAICompatibleLineageClient(
            transport=transport, sleep_fn=lambda _seconds: None
        ),
    )
    with TestClient(app) as client:
        response = client.post("/api/local/lineage/analyze", json=_request())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(observed) == 1
    assert payload["model"] == "mock-lineage-model"
    assert payload["analysis"]["analysis_status"] == "probable"
    assert payload["analysis"]["relation_type"] == "extends"
    assert {item["paper_id"] for item in payload["analysis"]["evidence"]} == {
        "source",
        "target",
    }
    assert all(item["quote"] for item in payload["analysis"]["evidence"])
    assert "test-secret-that-must-not-return" not in response.text


def test_analyze_without_api_key_returns_safe_configuration_error(
    tmp_path, monkeypatch
) -> None:
    app, papers_dir = _make_app(tmp_path, monkeypatch)
    _write_pair(papers_dir)
    _clear_llm_environment(monkeypatch)

    with TestClient(app) as client:
        response = client.post("/api/local/lineage/analyze", json=_request())

    assert response.status_code == 503
    assert "API Key" in response.json()["detail"]
    assert "None" not in response.text


@pytest.mark.parametrize(
    ("kind", "expected_status", "expected_calls"),
    [
        ("auth", 502, 1),
        ("rate", 429, 3),
        ("timeout", 504, 1),
        ("upstream", 502, 3),
        ("malformed", 502, 1),
        ("truncated", 502, 1),
    ],
)
def test_analyze_maps_llm_failures_to_safe_http_errors(
    tmp_path, monkeypatch, kind, expected_status, expected_calls
) -> None:
    app, papers_dir = _make_app(tmp_path, monkeypatch)
    _write_pair(papers_dir)
    _clear_llm_environment(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://mock-llm.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "never-return-this-secret")
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if kind == "auth":
            return httpx.Response(401, json={"message": "secret gateway details"})
        if kind == "rate":
            return httpx.Response(429, json={})
        if kind == "upstream":
            return httpx.Response(503, json={})
        if kind == "timeout":
            raise httpx.ReadTimeout("private transport detail")
        if kind == "malformed":
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "length", "message": {"content": "{"}}]},
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        server_module,
        "OpenAICompatibleLineageClient",
        lambda: OpenAICompatibleLineageClient(
            transport=transport, sleep_fn=lambda _seconds: None
        ),
    )
    with TestClient(app) as client:
        response = client.post("/api/local/lineage/analyze", json=_request())

    assert response.status_code == expected_status
    assert calls == expected_calls
    assert "never-return-this-secret" not in response.text
    assert "private transport detail" not in response.text
    assert "secret gateway details" not in response.text


def test_local_papers_page_has_separate_preview_and_confirmed_analysis_actions(
    tmp_path, monkeypatch
) -> None:
    app, _ = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        page = client.get("/local-papers")

    assert page.status_code == 200
    assert "方法演化分析" in page.text
    assert "预览分析证据（免费）" in page.text
    assert "调用 LLM 分析（可能产生 API 费用）" in page.text
    assert "本次操作将调用已配置的 LLM API，可能产生费用。是否继续？" in page.text
    assert "/api/local/lineage/preview" in page.text
    assert "/api/local/lineage/analyze" in page.text



def test_analyze_groups_only_final_verified_evidence_for_web_regions(tmp_path, monkeypatch) -> None:
    app, papers_dir = _make_app(tmp_path, monkeypatch)
    _write_pair(papers_dir)
    _clear_llm_environment(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://mock-llm.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "web-group-secret-must-not-return")
    monkeypatch.setenv("LLM_MODEL", "mock-lineage-model")
    _, _, context = server_module.build_local_lineage_context(papers_dir, "source.pdf", "target.pdf")
    source = next(item for item in context.selected_evidence if item.paper_role == "source" and item.section == "methods")
    target = next(item for item in context.selected_evidence if item.paper_role == "target" and item.section == "methods")
    result = next(item for item in context.selected_evidence if item.paper_role == "target" and item.section == "results")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        raw = {
            "analysis_status": "probable", "relation_type": "extends", "confidence": 0.75,
            "uncertainty": "The supplied excerpts leave some comparison details incomplete.",
            "relation_evidence_ids": [source.evidence_id, target.evidence_id, target.evidence_id],
            "inherited_components": [{"text": "CNN and recurrent feature processing", "evidence_ids": [source.evidence_id, target.evidence_id]}],
            "changed_components": [],
            "added_components": [{"text": "Model claim text must never become an evidence quote.", "evidence_ids": [target.evidence_id]}],
            "removed_components": [],
            "problem_addressed": {"text": "Temporal activity modeling", "evidence_ids": [target.evidence_id]},
            "claimed_contribution": {"text": "A recurrent method for activity recognition", "evidence_ids": [target.evidence_id]},
            "experimental_evidence": [{"text": "Recognition performance is compared across subjects", "evidence_ids": [result.evidence_id]}],
            "limitations": [{"text": "The context leaves comparison details incomplete", "evidence_ids": [target.evidence_id]}],
            "method_changes": [{
                "component": "temporal_modeling", "change_type": "inherited",
                "from_value": "recurrent layer", "to_value": "recurrent layer",
                "description": "Retains recurrent temporal processing.",
                "evidence_ids": [source.evidence_id, target.evidence_id], "confidence": 0.75,
            }],
        }
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]})

    monkeypatch.setattr(server_module, "OpenAICompatibleLineageClient", lambda: OpenAICompatibleLineageClient(
        transport=httpx.MockTransport(handler), sleep_fn=lambda _seconds: None,
    ))
    with TestClient(app) as client:
        response = client.post("/api/local/lineage/analyze", json=_request())
    assert response.status_code == 200, response.text
    assert len(calls) == 1 and calls[0]["max_tokens"] == 4000
    payload = response.json()
    groups = payload["evidence_groups"]
    assert set(groups) == {
        "relation", "inherited_components", "changed_components", "added_components",
        "removed_components", "problem_addressed", "claimed_contribution",
        "experimental_evidence", "limitations",
    }
    assert [item["quote"] for item in groups["relation"]] == [source.text, target.text]
    assert [item["quote"] for item in groups["inherited_components"]] == [source.text, target.text]
    for name in ("added_components", "problem_addressed", "claimed_contribution", "limitations"):
        assert [item["quote"] for item in groups[name]] == [target.text]
    assert [item["quote"] for item in groups["experimental_evidence"]] == [result.text]
    assert groups["changed_components"] == groups["removed_components"] == []
    final_evidence = payload["analysis"]["evidence"]
    for items in groups.values():
        for item in items:
            assert item in final_evidence
            assert item["paper_title"] in {context.source_paper_title, context.target_paper_title}
            assert item["source_kind"] == "pdf"
            assert item["section"] and item["location"]
            assert item["quote"] in {source.text, target.text, result.text}
    assert "limitation" in groups["limitations"][0]["supports"]
    assert groups["experimental_evidence"][0]["supports"] == "experimental_evidence"
    assert [item["quote"] for item in payload["analysis"]["method_changes"][0]["evidence"]] == [source.text, target.text]
    assert "web-group-secret-must-not-return" not in response.text
    assert str(tmp_path) not in response.text
