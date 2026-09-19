from urllib.parse import unquote

import httpx
import pytest
from fastapi.testclient import TestClient

from intern_atlas.db import connect
from intern_atlas.discovery import (
    CitationCandidate,
    DiscoveryService,
    candidate_level,
    identify_query,
    normalize_title,
)
from intern_atlas.integrations.semantic_scholar import (
    SemanticScholarClient,
    SemanticScholarError,
    SemanticScholarPaper,
    relation_fields,
)
from intern_atlas.server import create_app


TARGET = {
    "paperId": "target-paper",
    "title": "DeepConvLSTM for Wearable Activity Recognition",
    "abstract": "A wearable sensor model with LSTM representation learning.",
    "year": 2017,
    "authors": [{"name": "Researcher"}],
    "venue": "Sensors",
    "externalIds": {"DOI": "10.1000/target"},
    "citationCount": 100,
    "influentialCitationCount": 20,
    "publicationTypes": ["JournalArticle"],
    "url": "https://www.semanticscholar.org/paper/target-paper",
    "fieldsOfStudy": ["Computer Science"],
    "s2FieldsOfStudy": [{"category": "Computer Science"}],
}


def paper(paper_id, title, *, abstract=None, year=None):
    return {
        "paperId": paper_id,
        "title": title,
        "abstract": abstract,
        "year": year,
        "authors": None,
        "venue": "",
        "externalIds": None,
        "citationCount": None,
        "influentialCitationCount": None,
        "publicationTypes": None,
        "url": None,
        "fieldsOfStudy": None,
        "s2FieldsOfStudy": None,
    }


def mock_transport(calls):
    def handler(request):
        path = unquote(request.url.path)
        calls.append(request)
        assert request.headers.get("x-api-key") == "test-key"
        if path.endswith("/paper/search"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {**TARGET, "score": 0.98},
                        {**paper("other", "Background Activity Recognition"), "score": 0.2},
                        {"title": "Result without a paper id", "matchScore": 0.99},
                    ]
                },
            )
        if path.endswith("/references"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "citedPaper": {
                                **paper(
                                    "ref-method",
                                    "Wearable LSTM Activity Recognition",
                                    abstract="A methodology for wearable activity recognition.",
                                    year=2015,
                                ),
                            },
                            "contexts": ["We build on this method."],
                            "intents": ["methodology"],
                            "isInfluential": True,
                        },
                        {
                            "citedPaper": paper("ref-background", "Background on sensors"),
                            "intents": ["background"],
                            "isInfluential": False,
                        },
                    ]
                },
            )
        if path.endswith("/citations"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "citingPaper": paper(
                                "cite-method",
                                "Transformer Attention for Wearable Activity Recognition",
                                abstract="A later method using attention for activity recognition.",
                                year=2022,
                            ),
                            "contexts": ["We improve the sequence model."],
                            "intents": ["methodology"],
                            "isInfluential": True,
                        }
                    ]
                },
            )
        if "/paper/" in path:
            return httpx.Response(200, json=TARGET)
        return httpx.Response(404, json={"error": "missing"})

    return httpx.MockTransport(handler)


def make_service(tmp_path, calls):
    client = SemanticScholarClient(
        base_url="https://example.test/graph/v1",
        api_key="test-key",
        cache_dir=tmp_path,
        cache_ttl_seconds=3600,
        transport=mock_transport(calls),
    )
    return DiscoveryService(client), client


@pytest.mark.parametrize(
    ("query", "input_type", "identifier"),
    [
        ("10.1000/example", "doi", "DOI:10.1000/example"),
        ("https://doi.org/10.1000/example", "doi", "DOI:10.1000/example"),
        ("arXiv:2301.12345", "arxiv", "ARXIV:2301.12345"),
        ("2301.12345", "arxiv", "ARXIV:2301.12345"),
        ("abcdef0123456789abcdef0123456789abcdef01", "paper_id", "abcdef0123456789abcdef0123456789abcdef01"),
        ("A paper title", "title", "A paper title"),
    ],
)
def test_identify_query_formats(query, input_type, identifier):
    assert identify_query(query) == (input_type, identifier)


def test_discovery_reads_relations_ranks_candidates_and_hits_cache(tmp_path):
    calls = []
    service, client = make_service(tmp_path, calls)
    try:
        first = service.discover_lineage("DeepConvLSTM for Wearable Activity Recognition")
        assert first.target.paper_id == "target-paper"
        assert first.references[0].paper.paper_id == "ref-method"
        assert first.references[0].candidate_level == "method_candidate"
        assert "Influential citation" in first.references[0].reasons
        assert first.references[0].contexts == ["We build on this method."]
        assert first.references[1].candidate_level == "background_candidate"
        assert first.citations[0].direction == "citation"
        assert first.cached is False
        assert first.cache_status == {"resolution": False, "references": False, "citations": False}
        assert first.references[0].paper.match_score is None
        assert any("activity recognition" in reason for reason in first.references[0].reasons)
        assert [request.url.params.get("limit") for request in calls if request.url.path.endswith("/references")]
        relation_requests = [
            request
            for request in calls
            if request.url.path.endswith("/references") or request.url.path.endswith("/citations")
        ]
        assert relation_requests
        for request in relation_requests:
            fields = request.url.params["fields"]
            assert "citedPaper.title" not in fields
            assert "citingPaper.title" not in fields
            assert "title" in fields
            assert "abstract" in fields
            assert "authors" in fields
        assert all(match.paper_id for match in first.resolution.matches)

        call_count = len(calls)
        second = service.discover_lineage("DeepConvLSTM for Wearable Activity Recognition")
        assert second.cached is True
        assert second.cache_status == {"resolution": True, "references": True, "citations": True}
        assert len(calls) == call_count
        cache_files = list((tmp_path / "semantic_scholar").glob("*.json"))
        assert cache_files
        assert all("test-key" not in path.read_text(encoding="utf-8") for path in cache_files)
    finally:
        client.close()


def test_relation_fields_are_unprefixed_and_title_normalization_preserves_order():
    fields = relation_fields("citedPaper")
    assert "citedPaper.title" not in fields
    assert "citingPaper.title" not in fields
    assert "contexts" in fields
    assert "title" in fields
    assert "authors" in fields
    assert normalize_title("Wearable Activity Recognition") != normalize_title("Activity Recognition Wearable")


def test_evidence_poor_candidates_stay_uncertain():
    target = SemanticScholarPaper(paper_id="target", title="Wearable Activity Recognition")
    candidate = CitationCandidate(
        paper=SemanticScholarPaper(paper_id="candidate", title="An Unrelated Survey"),
        direction="reference",
    )
    assert candidate_level(0) == "uncertain"

    background = CitationCandidate(
        paper=SemanticScholarPaper(paper_id="background", title="An Unrelated Survey"),
        direction="reference",
        intents=["background"],
    )
    assert candidate_level(0, has_background_intent=True) == "background_candidate"


def test_title_search_requires_selection_for_non_exact_matches(tmp_path):
    def handler(request):
        path = unquote(request.url.path)
        if path.endswith("/paper/search"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {**paper("candidate-a", "Wearable Activity Recognition Methods"), "matchScore": 0.8},
                        {**paper("candidate-b", "Activity Recognition with Wearable Sensors"), "matchScore": 0.7},
                    ]
                },
            )
        if path.endswith("/references") or path.endswith("/citations"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404, json={"error": "missing"})

    client = SemanticScholarClient(
        base_url="https://example.test/graph/v1",
        cache_dir=tmp_path,
        cache_ttl_seconds=3600,
        transport=httpx.MockTransport(handler),
    )
    service = DiscoveryService(client)
    try:
        resolved = service.resolve("Wearable Activity Recognition")
        assert resolved.paper is None
        assert resolved.selection_required is True
        assert resolved.status == "selection_required"
        assert [paper.paper_id for paper in resolved.matches] == ["candidate-a", "candidate-b"]

        with pytest.raises(SemanticScholarError) as exc_info:
            service.discover_lineage("Wearable Activity Recognition")
        assert exc_info.value.code == "selection_required"

        result = service.discover_lineage(
            "Wearable Activity Recognition",
            paper_id="candidate-b",
        )
        assert result.target.paper_id == "candidate-b"
        assert result.resolution.selection_required is False
    finally:
        client.close()


def test_match_score_is_read_and_cache_key_separates_base_urls(tmp_path):
    paper_data = {"paperId": "p", "title": "Paper", "matchScore": 0.75, "score": 0.1}
    assert SemanticScholarPaper.from_api(paper_data).match_score == 0.75

    first = SemanticScholarClient(base_url="https://example.test/graph/v1", cache_dir=tmp_path)
    second = SemanticScholarClient(base_url="https://other.test/graph/v1", cache_dir=tmp_path)
    try:
        first_path = first._cache_path("/paper/search", {"query": "paper"})
        second_path = second._cache_path("/paper/search", {"query": "paper"})
        assert first_path != second_path
    finally:
        first.close()
        second.close()


def test_discovery_handles_missing_metadata(tmp_path):
    calls = []
    service, client = make_service(tmp_path, calls)
    try:
        result = service.discover_lineage("DeepConvLSTM for Wearable Activity Recognition")
        missing = result.references[1].paper
        assert missing.abstract == ""
        assert missing.authors == []
        assert missing.year is None
        assert missing.external_ids == {}
    finally:
        client.close()


@pytest.mark.parametrize(
    ("status", "code", "status_code"),
    [(404, "not_found", 404), (429, "rate_limited", 429), (500, "upstream_error", 502)],
)
def test_semantic_scholar_errors_are_structured(tmp_path, status, code, status_code, monkeypatch):
    calls = []
    waits = []
    monkeypatch.setattr("intern_atlas.integrations.semantic_scholar.time.sleep", waits.append)

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "test"})

    client = SemanticScholarClient(
        base_url="https://example.test/graph/v1",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(SemanticScholarError) as exc_info:
            client.get_paper("target-paper")
        assert exc_info.value.code == code
        assert exc_info.value.status_code == status_code
        assert len(calls) == (1 if status == 404 else 3)
        assert waits == ([] if status == 404 else [1.0, 2.0])
    finally:
        client.close()


def test_semantic_scholar_retries_with_retry_after_then_succeeds(tmp_path, monkeypatch):
    responses = [
        httpx.Response(429, headers={"Retry-After": "2.5"}, json={"error": "busy"}),
        httpx.Response(503, json={"error": "busy"}),
        httpx.Response(200, json=TARGET),
    ]
    waits = []
    monkeypatch.setattr("intern_atlas.integrations.semantic_scholar.time.sleep", waits.append)

    def handler(request):
        return responses.pop(0)

    client = SemanticScholarClient(
        base_url="https://example.test/graph/v1",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
    )
    try:
        result, cached = client.get_paper("target-paper")
        assert result.paper_id == "target-paper"
        assert cached is False
        assert waits == [2.5, 2.0]
    finally:
        client.close()


def test_semantic_scholar_retry_count_is_bounded(tmp_path, monkeypatch):
    calls = []
    waits = []
    monkeypatch.setattr("intern_atlas.integrations.semantic_scholar.time.sleep", waits.append)

    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"error": "still busy"})

    client = SemanticScholarClient(
        base_url="https://example.test/graph/v1",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(SemanticScholarError) as exc_info:
            client.get_paper("target-paper")
        assert exc_info.value.code == "upstream_error"
        assert len(calls) == 3
        assert waits == [1.0, 2.0]
    finally:
        client.close()


def test_discovery_api_uses_injected_client_and_bounds_limits(tmp_path):
    db_path = tmp_path / "graph.db"
    conn = connect(db_path)
    conn.close()
    calls = []
    service, client = make_service(tmp_path, calls)
    try:
        with TestClient(create_app(db_path, discovery_service=service)) as api:
            resolved = api.post("/api/v1/discovery/resolve", json={"query": "10.1000/target"})
            assert resolved.status_code == 200
            assert resolved.json()["paper"]["paper_id"] == "target-paper"

            lineage = api.post(
                "/api/v1/discovery/lineage",
                json={"query": "10.1000/target", "max_references": 1, "max_citations": 1},
            )
            assert lineage.status_code == 200
            assert len(lineage.json()["references"]) <= 1
            assert len(lineage.json()["citations"]) <= 1

            too_many = api.post(
                "/api/v1/discovery/lineage",
                json={"query": "10.1000/target", "max_references": 51},
            )
            assert too_many.status_code == 422
    finally:
        client.close()
