import json

import pytest

from intern_atlas.evidence_input import (
    LineageEvidenceInput,
    PaperEvidenceChunk,
    PaperEvidencePackage,
)
from intern_atlas.lineage_context import (
    EvidenceSelectionPolicy,
    LineageLLMContext,
    build_lineage_llm_context,
    validate_evidence_ids,
)


def _chunk(
    paper_id: str,
    text: str,
    section: str,
    *,
    source_kind: str = "pdf",
    chunk_type: str = "paragraph",
    page: int = 1,
    location: str | None = None,
) -> PaperEvidenceChunk:
    if chunk_type == "abstract":
        source_kind = "abstract"
    if chunk_type == "citation_context":
        source_kind = "citation_context"
    return PaperEvidenceChunk(
        paper_id=paper_id,
        paper_title=f"Title {paper_id}",
        text=text,
        section=section,
        page=page,
        location=location or f"PDF page {page}",
        source_kind=source_kind,
        chunk_type=chunk_type,
    )


def _input(
    source_chunks: list[PaperEvidenceChunk],
    target_chunks: list[PaperEvidenceChunk],
    *,
    source_title: str = "DeepConvLSTM wearable sensor recognition",
    target_title: str = "BiLSTM for wearable activity recognition",
) -> LineageEvidenceInput:
    source = PaperEvidencePackage(
        paper_id="source-1", paper_title=source_title, chunks=source_chunks
    )
    target = PaperEvidencePackage(
        paper_id="target-1", paper_title=target_title, chunks=target_chunks
    )
    return LineageEvidenceInput(source=source, target=target)


def test_methods_chunk_outranks_generic_unknown_chunk() -> None:
    evidence_input = _input(
        [
            _chunk("source-1", "Ordinary text without a section label.", "unknown"),
            _chunk("source-1", "The LSTM temporal model uses wearable sensors.", "methods"),
        ],
        [_chunk("target-1", "Target abstract.", "abstract", chunk_type="abstract")],
    )

    context = build_lineage_llm_context(
        evidence_input,
        policy=EvidenceSelectionPolicy(max_total_chars=1000, max_chars_per_paper=500),
    )

    selected_source = [item for item in context.selected_evidence if item.paper_role == "source"]
    method_chunk = next(item for item in selected_source if item.section == "methods")
    unknown_chunk = next(item for item in selected_source if item.section == "unknown")
    assert method_chunk.selection_score > unknown_chunk.selection_score
    assert [item.section for item in selected_source] == ["unknown", "methods"]


def test_coverage_selects_overview_method_and_experiment_evidence() -> None:
    evidence_input = _input(
        [
            _chunk("source-1", "Overview abstract.", "abstract", chunk_type="abstract"),
            _chunk("source-1", "Method architecture details.", "methods"),
            _chunk("source-1", "Experiment result details.", "experiments"),
            _chunk("source-1", "More method details.", "methods", page=2),
        ],
        [
            _chunk("target-1", "Target introduction.", "introduction"),
            _chunk("target-1", "Target method details.", "methods"),
            _chunk("target-1", "Target results.", "results"),
        ],
    )

    context = build_lineage_llm_context(
        evidence_input,
        policy=EvidenceSelectionPolicy(
            max_total_chars=4000,
            max_chars_per_paper=2000,
            max_chunks_per_paper=3,
        ),
    )

    for role in ("source", "target"):
        sections = {item.section for item in context.selected_evidence if item.paper_role == role}
        assert sections & {"abstract", "introduction", "related_work", "background"}
        assert "methods" in sections
        assert sections & {"experiments", "results"}


def test_source_and_target_have_independent_character_budgets() -> None:
    source_chunks = [
        _chunk("source-1", f"Source method paragraph {index}. " + "x" * 150, "methods", page=index)
        for index in range(1, 7)
    ]
    target_chunks = [
        _chunk("target-1", f"Target method paragraph {index}. " + "y" * 150, "methods", page=index)
        for index in range(1, 4)
    ]
    context = build_lineage_llm_context(
        _input(source_chunks, target_chunks),
        policy=EvidenceSelectionPolicy(
            max_total_chars=600,
            max_chars_per_paper=300,
            max_chunks_per_paper=10,
        ),
    )

    source_chars = sum(len(item.text) for item in context.selected_evidence if item.paper_role == "source")
    target_chars = sum(len(item.text) for item in context.selected_evidence if item.paper_role == "target")
    assert 0 < source_chars <= 300
    assert 0 < target_chars <= 300
    assert context.total_characters <= 600


def test_abstract_only_papers_produce_a_valid_context() -> None:
    context = build_lineage_llm_context(
        _input(
            [_chunk("source-1", "Source abstract only.", "abstract", chunk_type="abstract")],
            [_chunk("target-1", "Target abstract only.", "abstract", chunk_type="abstract")],
        )
    )

    assert len(context.selected_evidence) == 2
    assert {item.section for item in context.selected_evidence} == {"abstract"}
    assert context.total_characters == len("Source abstract only.") + len("Target abstract only.")


def test_citation_context_is_prioritized_without_creating_a_relation() -> None:
    context = build_lineage_llm_context(
        _input(
            [
                _chunk(
                    "source-1",
                    "The target cites the source in a method discussion.",
                    "unknown",
                    chunk_type="citation_context",
                ),
                _chunk("source-1", "Routine uncategorized text.", "unknown", page=2),
            ],
            [_chunk("target-1", "Target abstract.", "abstract", chunk_type="abstract")],
        ),
        policy=EvidenceSelectionPolicy(max_total_chars=1000, max_chars_per_paper=500),
    )

    citation = next(item for item in context.selected_evidence if item.source_kind == "citation_context")
    ordinary = next(
        item
        for item in context.selected_evidence
        if item.paper_role == "source" and item.source_kind != "citation_context"
    )
    assert citation.selection_score > ordinary.selection_score
    assert any("not a relation judgment" in reason for reason in citation.selection_reasons)
    assert "relation_type" not in context.to_dict()
    assert "method_changes" not in context.to_dict()


def test_generic_title_words_do_not_get_a_strong_lexical_boost() -> None:
    evidence_input = _input(
        [
            _chunk("source-1", "deep neural network learning model data", "methods"),
            _chunk(
                "source-1",
                "LSTM wearable sensor activity recognition with DeepConvLSTM.",
                "methods",
                page=2,
            ),
        ],
        [_chunk("target-1", "An abstract.", "abstract", chunk_type="abstract")],
    )

    context = build_lineage_llm_context(
        evidence_input,
        policy=EvidenceSelectionPolicy(max_total_chars=1000, max_chars_per_paper=500),
    )
    generic = next(item for item in context.selected_evidence if item.text.startswith("deep neural"))
    method = next(item for item in context.selected_evidence if "LSTM wearable" in item.text)

    assert generic.selection_score == 60.0
    assert not any("Strong method terms" in reason or "HAR domain terms" in reason for reason in generic.selection_reasons)
    assert method.selection_score > generic.selection_score
    assert any("Strong method terms" in reason for reason in method.selection_reasons)


def test_small_budget_skips_chunks_and_records_warnings() -> None:
    context = build_lineage_llm_context(
        _input(
            [_chunk("source-1", "s" * 80, "methods")],
            [_chunk("target-1", "t" * 80, "methods")],
        ),
        policy=EvidenceSelectionPolicy(max_total_chars=100, max_chars_per_paper=100),
    )

    assert context.selected_evidence == []
    assert context.warnings
    assert any("source: skipped 1 chunk" in warning for warning in context.warnings)
    assert any("target: skipped 1 chunk" in warning for warning in context.warnings)


def test_evidence_ids_and_selection_are_deterministic_and_ordered() -> None:
    evidence_input = _input(
        [
            _chunk("source-1", "Source unknown first.", "unknown", page=1),
            _chunk("source-1", "Source methods second.", "methods", page=2),
            _chunk("source-1", "Source results third.", "results", page=3),
        ],
        [
            _chunk("target-1", "Target unknown first.", "unknown", page=1),
            _chunk("target-1", "Target methods second.", "methods", page=2),
        ],
    )
    policy = EvidenceSelectionPolicy(max_total_chars=2000, max_chars_per_paper=1000)

    first = build_lineage_llm_context(evidence_input, policy=policy)
    second = build_lineage_llm_context(evidence_input, policy=policy)

    assert [item.to_dict() for item in first.selected_evidence] == [
        item.to_dict() for item in second.selected_evidence
    ]
    assert [item.evidence_id for item in first.selected_evidence] == [
        "S001",
        "S002",
        "S003",
        "T001",
        "T002",
    ]
    assert [item.text for item in first.selected_evidence[:3]] == [
        "Source unknown first.",
        "Source methods second.",
        "Source results third.",
    ]


def test_evidence_ids_are_unique_and_lookup_resolves_source_and_target() -> None:
    context = build_lineage_llm_context(
        _input(
            [_chunk("source-1", "Source method.", "methods")],
            [_chunk("target-1", "Target method.", "methods")],
        )
    )
    ids = [item.evidence_id for item in context.selected_evidence]

    assert ids == ["S001", "T001"]
    assert len(ids) == len(set(ids))
    assert context.get_evidence("T001").paper_role == "target"
    assert [item.evidence_id for item in validate_evidence_ids(context, ids)] == ids


def test_unknown_evidence_id_fails_validation() -> None:
    context = build_lineage_llm_context(
        _input(
            [_chunk("source-1", "Source abstract.", "abstract", chunk_type="abstract")],
            [_chunk("target-1", "Target abstract.", "abstract", chunk_type="abstract")],
        )
    )

    with pytest.raises(ValueError, match="unknown evidence_id"):
        context.validate_evidence_ids(["S001", "T999"])
    with pytest.raises(KeyError, match="unknown evidence_id"):
        context.get_evidence("X001")


def test_context_json_round_trip_preserves_ids_and_policy() -> None:
    original = build_lineage_llm_context(
        _input(
            [_chunk("source-1", "Source abstract.", "abstract", chunk_type="abstract")],
            [_chunk("target-1", "Target abstract.", "abstract", chunk_type="abstract")],
        ),
        policy=EvidenceSelectionPolicy(max_total_chars=1200, max_chars_per_paper=600),
    )

    restored = LineageLLMContext.from_mapping(
        json.loads(json.dumps(original.to_dict(), ensure_ascii=False))
    )

    assert restored.to_dict() == original.to_dict()
