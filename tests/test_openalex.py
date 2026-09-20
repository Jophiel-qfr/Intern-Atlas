from urllib.parse import unquote

import httpx
import pytest
from fastapi.testclient import TestClient

from intern_atlas.db import connect
from intern_atlas.discovery import CitationCandidate, DiscoveryService, rank_candidates
from intern_atlas.discovery_models import DiscoveredPaper
from intern_atlas.integrations.openalex import (
    OpenAlexClient,
    OpenAlexError,
    identify_openalex_query,
    reconstruct_abstract,
)
from intern_atlas.server import create_app


TARGET_ID = "W2741809807"
REFERENCE_IDS = ["W1000000001", "W1000000002"]


def work(
    work_id: str,
    title: str,
    *,
    abstract: dict | None = None,
    referenced_works: list[str] | None = None,
    doi: str | None = "https://doi.org/10.1000/target",
    cited_by_count: int = 42,
) -> dict:
    payload = {
        "id": f"https://openalex.org/{work_id}",
        "display_name": title,
        "publication_year": 2020,
        "authorships": [{"author": {"display_name": "Researcher"}}],
        "doi": doi,
        "primary_location": {
            "landing_page_url": f"https://doi.org/10.1000/{work_id}",
            "source": {"display_name": "Sensors"},
        },
        "cited_by_count": cited_by_count,
        "type": "article",
        "abstract_inverted_index": abstract,
        "referenced_works": referenced_works or [],
    }
    return payload


def client_for(tmp_path, handler, *, api_key=None):
    return OpenAlexClient(
        base_url="https://example.test",
        api_key=api_key,
        cache_dir=tmp_path,
        cache_ttl_seconds=3600,
        transport=httpx.MockTransport(handler),
    )


def test_openalex_query_identification():
    assert identify_openalex_query("A paper title") == ("title", "A paper title")
    assert identify_openalex_query("10.1000/example") == (
        "doi",
        "https://doi.org/10.1000/example",
    )
    assert identify_openalex_query("https://doi.org/10.1000/example")[0] == "doi"
    assert identify_openalex_query(TARGET_ID) == ("openalex_id", TARGET_ID)
    assert identify_openalex_query(f"https://openalex.org/{TARGET_ID}") == (
        "openalex_id",
        TARGET_ID,
    )


def test_reconstruct_abstract_is_tolerant():
    assert reconstruct_abstract({"human": [0, 5], "activity": [1], "recognition": [2]}) == (
        "human activity recognition human"
    )
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({"broken": ["0"], 1: None}) == ""


def test_openalex_search_converts_metadata_without_api_key(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        assert request.url.path == "/works"
        assert request.url.params["search"] == "DeepConvLSTM"
        assert request.url.params["per-page"] == "5"
        assert "api_key" not in request.url.params
        return httpx.Response(
            200,
            json={
                "results": [
                    work(
                        TARGET_ID,
                        "DeepConvLSTM for Activity Recognition",
                        abstract={"wearable": [0], "activity": [1], "recognition": [2]},
                        referenced_works=[f"https://openalex.org/{REFERENCE_IDS[0]}"],
                    ),
                    {"display_name": "Missing ID"},
                ]
            },
        )

    client = client_for(tmp_path, handler)
    try:
        papers, cached = client.search_works("DeepConvLSTM", limit=5)
        assert cached is False
        assert len(papers) == 1
        paper = papers[0]
        assert paper.provider == "openalex"
        assert paper.provider_id == TARGET_ID
        assert paper.paper_id == TARGET_ID
        assert paper.abstract == "wearable activity recognition"
        assert paper.authors == ["Researcher"]
        assert paper.venue == "Sensors"
        assert paper.external_ids["DOI"] == "https://doi.org/10.1000/target"
        assert paper.citation_count == 42
        assert paper.referenced_work_ids == [REFERENCE_IDS[0]]
    finally:
        client.close()


def test_openalex_exact_and_ambiguous_title_resolution(tmp_path):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "results": [
                    work(TARGET_ID, "Exact Activity Recognition"),
                    work("W2741809808", "Activity Recognition Methods"),
                ]
            },
        )

    client = client_for(tmp_path, handler)
    service = DiscoveryService(openalex_client=client)
    try:
        exact = service.resolve("Exact Activity Recognition")
        assert exact.provider == "openalex"
        assert exact.paper.paper_id == TARGET_ID
        assert exact.selection_required is False

        ambiguous = service.resolve("Activity Recognition")
        assert ambiguous.paper is None
        assert ambiguous.selection_required is True
        assert ambiguous.status == "selection_required"
    finally:
        service.close()


@pytest.mark.parametrize("query", ["10.1000/example", TARGET_ID, f"https://openalex.org/{TARGET_ID}"])
def test_openalex_identifier_lookup_uses_work_endpoint(tmp_path, query):
    paths = []

    def handler(request):
        paths.append(unquote(request.url.path))
        return httpx.Response(200, json=work(TARGET_ID, "A Work"))

    client = client_for(tmp_path, handler)
    service = DiscoveryService(openalex_client=client)
    try:
        result = service.resolve(query)
        assert result.paper.paper_id == TARGET_ID
        assert result.input_type in {"doi", "openalex_id"}
        assert paths == [paths[0]]
        assert paths[0].startswith("/works/")
    finally:
        service.close()


def discovery_handler(calls):
    def handler(request):
        calls.append(request)
        params = request.url.params
        if request.url.path == "/works" and params.get("search"):
            return httpx.Response(
                200,
                json={
                    "results": [
                        work(
                            TARGET_ID,
                            "Wearable Sensor Activity Recognition",
                            abstract={"wearable": [0], "sensor": [1], "activity": [2]},
                            referenced_works=REFERENCE_IDS + [f"W{i:010d}" for i in range(3, 40)],
                        )
                    ]
                },
            )
        if params.get("filter", "").startswith("openalex:"):
            return httpx.Response(
                200,
                json={
                    "results": [
                        work(
                            REFERENCE_IDS[0],
                            "Sensor Fusion for Activity Recognition",
                            abstract={"sensor": [0], "fusion": [1], "activity": [2]},
                        ),
                        work(REFERENCE_IDS[1], "Unrelated Survey", abstract=None),
                    ]
                },
            )
        if params.get("filter") == f"cites:{TARGET_ID}":
            return httpx.Response(
                200,
                json={"results": [work("W1000000003", "Later Activity Recognition Work")]},
            )
        return httpx.Response(404, json={"error": "missing"})

    return handler


def test_openalex_lineage_batches_references_and_uses_cites_filter(tmp_path):
    calls = []
    client = client_for(tmp_path, discovery_handler(calls))
    service = DiscoveryService(openalex_client=client)
    try:
        result = service.discover_lineage("Wearable Sensor Activity Recognition")
        assert result.target.provider == "openalex"
        assert len(result.references) == 2
        assert len(result.citations) == 1
        assert result.references[0].candidate_level == "method_candidate"
        assert result.references[1].candidate_level == "uncertain"
        relation_calls = [request for request in calls if request.url.params.get("filter")]
        assert len(relation_calls) == 2
        reference_call = next(
            request for request in relation_calls if request.url.params["filter"].startswith("openalex:")
        )
        assert reference_call.url.params["per-page"] == "30"
        assert reference_call.url.params["filter"].count("|") == 29
        citation_call = next(request for request in relation_calls if request.url.params["filter"].startswith("cites:"))
        assert citation_call.url.params["filter"] == f"cites:{TARGET_ID}"
        assert len(calls) == 3

        call_count = len(calls)
        cached = service.discover_lineage("Wearable Sensor Activity Recognition")
        assert cached.cached is True
        assert len(calls) == call_count
    finally:
        service.close()


def ranking_paper(paper_id: str, title: str, abstract: str = "") -> DiscoveredPaper:
    return DiscoveredPaper(
        provider="openalex",
        provider_id=paper_id,
        title=title,
        abstract=abstract,
    )


def test_openalex_ranking_downweights_generic_method_words_and_reviews():
    target = ranking_paper(
        "target",
        "DeepConvLSTM for Human Activity Recognition Using Wearable Sensors",
        "A recurrent wearable sensor model for activity recognition.",
    )
    rows = [
        {
            "_paper": ranking_paper(
                "deeppose",
                "DeepPose: Human Pose Estimation via Deep Neural Networks",
                "Deep neural networks and convolutional representations.",
            )
        },
        {
            "_paper": ranking_paper(
                "survey",
                "A Survey and Review of Deep Learning for Human Activity Recognition",
            )
        },
        {
            "_paper": ranking_paper(
                "conv-lstm",
                "Convolutional, Long Short-Term Memory, Fully Connected Deep Neural Networks",
            )
        },
        {
            "_paper": ranking_paper(
                "cnn-har",
                "Convolutional Neural Networks for Human Activity Recognition using Mobile Sensors",
            )
        },
        {
            "_paper": ranking_paper(
                "ensemble-lstm",
                "Ensembles of Deep LSTM Learners for Activity Recognition using Wearables",
            )
        },
        {
            "_paper": ranking_paper(
                "generic",
                "Deep Neural Networks for Data Modeling",
                "A learning based model using data and networks.",
            )
        },
    ]

    ranked = rank_candidates(target, rows, direction="reference")
    by_id = {candidate.paper.paper_id: candidate for candidate in ranked}

    assert by_id["deeppose"].candidate_level != "method_candidate"
    assert "High title overlap" not in by_id["deeppose"].reasons
    assert by_id["survey"].candidate_level == "uncertain"
    assert "Review/survey paper" in by_id["survey"].reasons
    assert by_id["conv-lstm"].candidate_level == "method_candidate"
    assert by_id["cnn-har"].candidate_level == "method_candidate"
    assert by_id["ensemble-lstm"].candidate_level == "method_candidate"
    assert by_id["generic"].candidate_level == "uncertain"
    assert "Shares abstract terms" in by_id["generic"].reasons


def test_openalex_ranking_keeps_domain_only_and_dataset_candidates_uncertain():
    target = ranking_paper("target", "Human Activity Recognition with Wearable Sensors")
    rows = [
        {
            "_paper": ranking_paper(
                "domain-only",
                "Human Activity Recognition using Wearable Sensors",
            )
        },
        {
            "_paper": ranking_paper(
                "dataset",
                "A Benchmark Dataset for Human Activity Recognition",
            )
        },
        {
            "_paper": ranking_paper("conv-only", "Convolutional Image Classifier"),
        },
    ]

    ranked = rank_candidates(target, rows, direction="reference")
    by_id = {candidate.paper.paper_id: candidate for candidate in ranked}
    assert by_id["domain-only"].candidate_level == "uncertain"
    assert by_id["dataset"].candidate_level == "uncertain"
    assert by_id["conv-only"].candidate_level == "uncertain"
    assert "Dataset/benchmark paper" in by_id["dataset"].reasons


def test_openalex_reference_limit_is_capped_at_50_and_uses_one_request(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"results": []})

    client = client_for(tmp_path, handler)
    try:
        rows, cached = client.references(
            TARGET_ID,
            referenced_work_ids=[f"W{i:010d}" for i in range(60)],
            limit=100,
        )
        assert rows == []
        assert cached is False
        assert len(calls) == 1
        assert calls[0].url.params["per-page"] == "50"
        assert calls[0].url.params["filter"].count("|") == 49
    finally:
        client.close()


def test_openalex_cache_hit_base_url_isolation_and_api_key_exclusion(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"results": [work(TARGET_ID, "A Work")]})

    first = client_for(tmp_path, handler, api_key="secret-key")
    second = OpenAlexClient(
        base_url="https://other.example",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
    )
    try:
        first.search_works("A Work")
        first.search_works("A Work")
        cache_files = list((tmp_path / "openalex").glob("*.json"))
        assert len(cache_files) == 1
        assert all("secret-key" not in path.read_text(encoding="utf-8") for path in cache_files)
        assert first._cache_path("/works", {"search": "A Work"}) != second._cache_path(
            "/works", {"search": "A Work"}
        )
    finally:
        first.close()
        second.close()


def test_openalex_retries_429_and_5xx_without_infinite_retry(tmp_path, monkeypatch):
    responses = [
        httpx.Response(429, headers={"Retry-After": "0.5"}, json={"error": "busy"}),
        httpx.Response(503, json={"error": "busy"}),
        httpx.Response(200, json=work(TARGET_ID, "A Work")),
    ]
    waits = []
    monkeypatch.setattr("intern_atlas.integrations.openalex.time.sleep", waits.append)

    def handler(request):
        return responses.pop(0)

    client = client_for(tmp_path, handler)
    try:
        paper, cached = client.get_work(TARGET_ID)
        assert paper.paper_id == TARGET_ID
        assert cached is False
        assert waits == [0.5, 2.0]
    finally:
        client.close()


def test_openalex_retry_count_404_and_timeout(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr("intern_atlas.integrations.openalex.time.sleep", waits.append)
    calls = []

    def not_found(request):
        calls.append(request)
        return httpx.Response(404, json={"error": "missing"})

    client = client_for(tmp_path, not_found)
    try:
        with pytest.raises(OpenAlexError) as exc_info:
            client.get_work(TARGET_ID)
        assert exc_info.value.code == "not_found"
        assert len(calls) == 1
        assert waits == []
    finally:
        client.close()

    def timeout(request):
        raise httpx.ReadTimeout("timeout")

    client = client_for(tmp_path / "timeout", timeout)
    try:
        with pytest.raises(OpenAlexError) as exc_info:
            client.get_work(TARGET_ID)
        assert exc_info.value.code == "timeout"
        assert waits == []
    finally:
        client.close()


def test_openalex_provider_is_default_and_invalid_provider_is_4xx(tmp_path):
    calls = []
    client = client_for(tmp_path, discovery_handler(calls))
    service = DiscoveryService(openalex_client=client)
    db_path = tmp_path / "graph.db"
    connect(db_path).close()
    try:
        with TestClient(create_app(db_path, discovery_service=service)) as api:
            resolved = api.post("/api/v1/discovery/resolve", json={"query": "Wearable Sensor Activity Recognition"})
            assert resolved.status_code == 200
            assert resolved.json()["provider"] == "openalex"

            explicit = api.post(
                "/api/v1/discovery/resolve",
                json={"query": "Wearable Sensor Activity Recognition", "provider": "openalex"},
            )
            assert explicit.status_code == 200

            invalid = api.post(
                "/api/v1/discovery/resolve",
                json={"query": "A paper", "provider": "unknown"},
            )
            assert invalid.status_code == 422
            assert invalid.json()["detail"]["code"] == "invalid_provider"
    finally:
        service.close()
