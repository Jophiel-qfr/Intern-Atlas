import json

import httpx
import pytest

from intern_atlas.config import get_settings
from intern_atlas.evidence_input import LineageEvidenceInput, PaperEvidenceChunk, PaperEvidencePackage
from intern_atlas.lineage_context import build_lineage_llm_context
from intern_atlas.lineage_llm import (
    LineageLLMAuthenticationError,
    LineageLLMMalformedResponseError,
    LineageLLMRateLimitError,
    LineageLLMTimeoutError,
    LineageLLMUpstreamError,
    OpenAICompatibleLineageClient,
    build_lineage_messages,
)
from intern_atlas.lineage_response import (
    LineageResponseParseError,
    LineageResponseValidationError,
    build_method_lineage_analysis,
    parse_lineage_llm_response,
)


def _chunk(paper_id: str, text: str, role: str, page: int = 1, section: str = "methods"):
    return PaperEvidenceChunk(
        paper_id=paper_id,
        paper_title=f"{role.title()} HAR paper",
        text=text,
        section=section,
        page=page,
        location=f"PDF page {page}",
        source_kind="pdf",
        chunk_type="paragraph",
    )


def _context():
    source = PaperEvidencePackage(
        paper_id="source-paper",
        paper_title="DeepConvLSTM source paper",
        chunks=[
            _chunk("source-paper", "Source describes CNN feature extraction and LSTM temporal modeling.", "source", 1),
            _chunk("source-paper", "The original model uses a unidirectional LSTM.", "source", 2),
        ],
    )
    target = PaperEvidencePackage(
        paper_id="target-paper",
        paper_title="BiLSTM attention target paper",
        chunks=[
            _chunk("target-paper", "Target retains CNN and replaces LSTM with BiLSTM.", "target", 3),
            _chunk("target-paper", "The target adds an attention block.", "target", 4, "results"),
        ],
    )
    return build_lineage_llm_context(
        LineageEvidenceInput(source=source, target=target)
    )


def _response(**overrides):
    value = {
        "analysis_status": "confirmed",
        "relation_type": "improves",
        "confidence": 0.91,
        "uncertainty": None,
        "relation_evidence_ids": ["S001", "T001"],
        "inherited_components": [],
        "changed_components": [],
        "added_components": [],
        "removed_components": [],
        "problem_addressed": None,
        "claimed_contribution": None,
        "experimental_evidence": [],
        "limitations": [],
        "method_changes": [
            {
                "component": "temporal_modeling",
                "change_type": "replaced",
                "from_value": "LSTM",
                "to_value": "BiLSTM",
                "description": "The target uses bidirectional temporal modeling.",
                "evidence_ids": ["S002", "T001"],
                "confidence": 0.88,
            }
        ],
    }
    value.update(overrides)
    return value


def _http_json(payload, status_code=200):
    return httpx.Response(status_code, json=payload)


def _completion(content: str):
    return {"choices": [{"message": {"content": content}}]}


def test_valid_confirmed_response_rebuilds_evidence_from_context() -> None:
    context = _context()
    analysis = build_method_lineage_analysis(context, _response())

    assert analysis.analysis_status == "confirmed"
    assert analysis.source_paper_id == "source-paper"
    assert analysis.target_paper_id == "target-paper"
    assert analysis.method_changes[0].from_value == "LSTM"
    assert analysis.method_changes[0].to_value == "BiLSTM"
    original_evidence = analysis.method_changes[0].evidence[0]
    selected = context.get_evidence("S002")
    assert original_evidence.quote == selected.text
    assert original_evidence.paper_id == context.source_paper_id
    assert original_evidence.paper_title == context.source_paper_title
    assert original_evidence.section == selected.section
    assert original_evidence.location == selected.location
    assert original_evidence.source_kind == selected.source_kind
    assert original_evidence.location == "PDF page 2"
    assert analysis.evidence[0].quote == context.get_evidence("S001").text
    assert analysis.method_changes[0].evidence[0].paper_title == context.source_paper_title


def test_unknown_evidence_id_is_rejected() -> None:
    with pytest.raises(LineageResponseValidationError, match="T999"):
        parse_lineage_llm_response(
            json.dumps(_response(relation_evidence_ids=["S001", "T999"])), _context()
        )


def test_raw_schema_rejects_model_supplied_quote() -> None:
    raw = _response(quote="model invented text")

    with pytest.raises(LineageResponseParseError, match="quote"):
        parse_lineage_llm_response(json.dumps(raw), _context())


def test_citation_only_confirmed_relation_is_downgraded() -> None:
    source = PaperEvidencePackage(
        paper_id="source-paper",
        paper_title="Source",
        chunks=[
            PaperEvidenceChunk(
                paper_id="source-paper",
                paper_title="Source",
                text="Citation context from source.",
                section="unknown",
                source_kind="citation_context",
                chunk_type="citation_context",
            )
        ],
    )
    target = PaperEvidencePackage(
        paper_id="target-paper",
        paper_title="Target",
        chunks=[
            PaperEvidenceChunk(
                paper_id="target-paper",
                paper_title="Target",
                text="Citation context from target.",
                section="unknown",
                source_kind="citation_context",
                chunk_type="citation_context",
            )
        ],
    )
    context = build_lineage_llm_context(LineageEvidenceInput(source=source, target=target))

    analysis = build_method_lineage_analysis(
        context,
        _response(relation_type="extends", relation_evidence_ids=["S001", "T001"], method_changes=[]),
    )

    assert analysis.analysis_status == "uncertain"
    assert analysis.relation_type == "extends"
    assert "citation-context" in analysis.uncertainty
    assert all(item.evidence_type == "citation_context" for item in analysis.evidence)


def test_metadata_only_confirmed_relation_is_downgraded() -> None:
    context = _context()
    for item in context.selected_evidence:
        item.source_kind = "metadata"

    analysis = build_method_lineage_analysis(
        context,
        _response(method_changes=[]),
    )

    assert analysis.analysis_status == "uncertain"
    assert "metadata" in analysis.uncertainty
    assert all(item.source_kind == "metadata" for item in analysis.evidence)


def test_confirmed_method_change_requires_evidence_ids() -> None:
    raw = _response(
        method_changes=[
            {
                "component": "temporal_modeling",
                "change_type": "replaced",
                "from_value": "LSTM",
                "to_value": "BiLSTM",
                "description": "Replacement",
                "evidence_ids": [],
                "confidence": 0.9,
            }
        ]
    )

    with pytest.raises(LineageResponseParseError, match="at least one evidence_id"):
        parse_lineage_llm_response(json.dumps(raw), _context())


def test_confirmed_target_change_cannot_cite_only_source() -> None:
    raw = _response(
        method_changes=[
            {
                "component": "attention_mechanism",
                "change_type": "added",
                "from_value": None,
                "to_value": "attention",
                "description": "Adds attention.",
                "evidence_ids": ["S002"],
                "confidence": 0.9,
            }
        ]
    )

    with pytest.raises(LineageResponseValidationError, match="requires target evidence"):
        build_method_lineage_analysis(_context(), raw)


def test_target_only_replacement_is_downgraded_to_uncertain() -> None:
    raw = _response(
        method_changes=[
            {
                "component": "temporal_modeling",
                "change_type": "replaced",
                "from_value": "LSTM",
                "to_value": "BiLSTM",
                "description": "Replaces the temporal model.",
                "evidence_ids": ["T001"],
                "confidence": 0.9,
            }
        ]
    )

    analysis = build_method_lineage_analysis(_context(), raw)

    assert analysis.analysis_status == "uncertain"
    assert "lacks source evidence" in analysis.uncertainty


def test_insufficient_evidence_with_null_relation_is_a_valid_result() -> None:
    response = _response(
        analysis_status="insufficient_evidence",
        relation_type=None,
        confidence=0.2,
        uncertainty="Only abstracts are available.",
        relation_evidence_ids=[],
        method_changes=[],
    )

    analysis = build_method_lineage_analysis(_context(), response)

    assert analysis.analysis_status == "insufficient_evidence"
    assert analysis.relation_type is None
    assert analysis.evidence == []


def test_invalid_relation_type_is_rejected() -> None:
    with pytest.raises(LineageResponseParseError, match="relation_type is invalid"):
        parse_lineage_llm_response(json.dumps(_response(relation_type="cites")), _context())


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_out_of_range_confidence_is_rejected(confidence: float) -> None:
    with pytest.raises(LineageResponseParseError, match="between 0 and 1"):
        parse_lineage_llm_response(json.dumps(_response(confidence=confidence)), _context())


def test_fenced_json_is_accepted_but_broken_json_is_not_repaired() -> None:
    response = _response()
    content = "```json\n" + json.dumps(response) + "\n```"

    parsed = parse_lineage_llm_response(content, _context())
    assert parsed.relation_type == "improves"
    with pytest.raises(LineageResponseParseError, match="not valid JSON"):
        parse_lineage_llm_response("{analysis_status: confirmed}", _context())


def test_prompt_includes_evidence_locations_but_omits_selection_scores() -> None:
    messages = build_lineage_messages(_context())
    prompt = "\n".join(message["content"] for message in messages)

    assert "[S001]" in prompt and "[T001]" in prompt
    assert "paper_role: source" in prompt and "paper_role: target" in prompt
    assert "section: methods" in prompt
    assert "location: PDF page 1" in prompt
    assert "source_kind: pdf" in prompt
    assert "text:\n" in prompt
    assert "Source describes CNN feature extraction" in prompt
    assert "selection_score" not in prompt
    assert "selection_reasons" not in prompt


def test_repeated_evidence_ids_are_deduplicated_in_stable_context_order() -> None:
    raw = _response(
        relation_evidence_ids=["T001", "S001", "T001"],
        claimed_contribution={"text": "The target adds bidirectionality.", "evidence_ids": ["T001"]},
        method_changes=[],
    )

    analysis = build_method_lineage_analysis(_context(), raw)

    assert [item.paper_id for item in analysis.evidence] == ["source-paper", "target-paper"]
    assert len(analysis.evidence) == 2
    target_evidence = analysis.evidence[1]
    assert set(target_evidence.supports) == {"relation", "claimed_contribution"}


def test_auth_settings_support_requested_environment_names(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://mock.example/v1/")
    monkeypatch.setenv("LLM_API_KEY", "placeholder-key")
    monkeypatch.setenv("LLM_MODEL", "mock-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "17.5")

    settings = get_settings()

    assert settings.llm_base_url == "https://mock.example/v1"
    assert settings.llm_api_key == "placeholder-key"
    assert settings.llm_models[0] == "mock-model"
    assert settings.llm_timeout_seconds == 17.5


def test_http_401_is_clear_and_never_echoes_api_key() -> None:
    key = "dummy-not-a-real-key"
    transport = httpx.MockTransport(lambda request: _http_json({"error": "unauthorized"}, 401))
    client = OpenAICompatibleLineageClient(
        api_key=key,
        transport=transport,
        sleep_fn=lambda delay: None,
    )
    try:
        with pytest.raises(LineageLLMAuthenticationError) as error:
            client.complete(_context())
        assert "401" in str(error.value)
        assert key not in str(error.value)
    finally:
        client.close()


@pytest.mark.parametrize("first_status", [429, 503])
def test_http_429_and_5xx_retry_then_succeed(first_status: int) -> None:
    calls = []
    waits = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return _http_json({"error": "retry"}, first_status)
        return _http_json(_completion(json.dumps(_response())))

    client = OpenAICompatibleLineageClient(
        api_key="placeholder-key",
        transport=httpx.MockTransport(handler),
        sleep_fn=waits.append,
    )
    try:
        parsed = client.analyze(_context())
        assert parsed.analysis_status == "confirmed"
        assert len(calls) == 2
        assert waits == [1]
    finally:
        client.close()


def test_http_retry_is_bounded_to_two_retries() -> None:
    calls = []
    waits = []

    def handler(request):
        calls.append(request)
        return _http_json({"error": "unavailable"}, 503)

    client = OpenAICompatibleLineageClient(
        api_key="placeholder-key",
        transport=httpx.MockTransport(handler),
        sleep_fn=waits.append,
    )
    try:
        with pytest.raises(LineageLLMUpstreamError, match="after 2 retries"):
            client.complete(_context())
        assert len(calls) == 3
        assert waits == [1, 2]
    finally:
        client.close()


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
        {"broken": True},
    ],
)
def test_missing_or_malformed_http_content_is_reported(payload) -> None:
    client = OpenAICompatibleLineageClient(
        api_key="placeholder-key",
        transport=httpx.MockTransport(lambda request: _http_json(payload)),
        sleep_fn=lambda delay: None,
    )
    try:
        with pytest.raises(LineageLLMMalformedResponseError):
            client.complete(_context())
    finally:
        client.close()


def test_http_timeout_has_sanitized_error() -> None:
    def handler(request):
        raise httpx.ReadTimeout("timeout contains no credentials")

    client = OpenAICompatibleLineageClient(
        api_key="placeholder-key",
        transport=httpx.MockTransport(handler),
        sleep_fn=lambda delay: None,
    )
    try:
        with pytest.raises(LineageLLMTimeoutError, match="timed out"):
            client.complete(_context())
    finally:
        client.close()
