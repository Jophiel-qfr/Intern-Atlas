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
    _score_cross_paper_pair,
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


SOURCE_OUTPUT_TEXT = (
    "The final LSTM hidden state at the last time step is passed to a softmax classifier."
)
TARGET_OUTPUT_TEXT = (
    "An attention-weighted embedding is constructed from LSTM hidden states before classification."
)


def _comparison_input() -> LineageEvidenceInput:
    # Distractors outrank the output paragraphs under the previous independent scoring.
    source = [
        _chunk("source-1", "Source overview.", "introduction"),
        _chunk("source-1", "Source evaluation results.", "results"),
    ]
    target = [
        _chunk("target-1", "Target overview.", "abstract", chunk_type="abstract"),
        _chunk("target-1", "Target experimental results.", "experiments"),
    ]
    for index in range(10):
        source.append(_chunk(
            "source-1", f"CNN transformer contrastive optimization variant {index}.",
            "methods", page=index + 2,
        ))
        target.append(_chunk(
            "target-1", f"GRU fusion pooling regularization variant {index}.",
            "methods", page=index + 2,
        ))
    source.append(_chunk("source-1", SOURCE_OUTPUT_TEXT, "methods", page=12))
    target.append(_chunk("target-1", TARGET_OUTPUT_TEXT, "methods", page=12))
    return _input(source, target, source_title="Source study", target_title="Target study")


def test_cross_paper_reservation_retains_comparable_output_despite_method_distractors() -> None:
    context = build_lineage_llm_context(
        _comparison_input(),
        policy=EvidenceSelectionPolicy(max_chunks_per_paper=4),
    )
    assert SOURCE_OUTPUT_TEXT in {item.text for item in context.selected_evidence}
    assert TARGET_OUTPUT_TEXT in {item.text for item in context.selected_evidence}
    assert len(context.selection_policy["cross_paper_pairs"]) == 1
    pair = context.selection_policy["cross_paper_pairs"][0]
    assert context.get_evidence(pair["source_evidence_id"]).text == SOURCE_OUTPUT_TEXT
    assert context.get_evidence(pair["target_evidence_id"]).text == TARGET_OUTPUT_TEXT
    assert "hidden state" in pair["cues"]
    for evidence_id in (pair["source_evidence_id"], pair["target_evidence_id"]):
        assert any(
            reason.startswith("Cross-paper comparison cue:")
            for reason in context.get_evidence(evidence_id).selection_reasons
        )


def test_generic_har_overlap_does_not_create_comparison_boost() -> None:
    source = _chunk(
        "source-1", "Human activity recognition uses wearable sensor data.", "methods"
    )
    target = _chunk(
        "target-1", "We evaluate human activity recognition using wearable sensor data.", "methods"
    )
    assert _score_cross_paper_pair(source, target) == (0.0, [])
    context = build_lineage_llm_context(_input([source], [target]))
    assert context.selection_policy["cross_paper_pairs"] == []
    assert not any(
        "Cross-paper" in reason
        for item in context.selected_evidence for reason in item.selection_reasons
    )


def test_shared_lstm_is_a_comparison_cue_without_relation_judgment() -> None:
    context = build_lineage_llm_context(_input(
        [_chunk("source-1", "LSTM recurrent layers model temporal dynamics.", "methods")],
        [_chunk("target-1", "Attention is applied over LSTM hidden states.", "methods")],
    ))
    assert len(context.selection_policy["cross_paper_pairs"]) == 1
    assert "lstm" in context.selection_policy["cross_paper_pairs"][0]["cues"]
    for item in context.selected_evidence:
        reasons = " ".join(item.selection_reasons).lower()
        assert "cross-paper comparison cue" in reasons
        assert all(word not in reasons for word in ("inherited", "modified", "replaced"))
    assert "relation_type" not in context.to_dict()
    assert "method_changes" not in context.to_dict()


def test_pair_reservation_preserves_overview_method_and_experimental_coverage() -> None:
    context = build_lineage_llm_context(
        _comparison_input(), policy=EvidenceSelectionPolicy(max_chunks_per_paper=4)
    )
    for role in ("source", "target"):
        chunks = [item for item in context.selected_evidence if item.paper_role == role]
        assert len(chunks) == 4
        assert any(item.section in {"abstract", "introduction"} for item in chunks)
        assert any(item.section == "methods" for item in chunks)
        assert any(item.section in {"experiments", "results"} for item in chunks)


def test_default_pair_selection_keeps_all_original_budgets() -> None:
    evidence_input = _comparison_input()
    for package in (evidence_input.source, evidence_input.target):
        package.chunks.extend(
            _chunk(package.paper_id, f"Additional method text {index}. " + "x" * 1500, "methods")
            for index in range(30)
        )
    context = build_lineage_llm_context(evidence_input)
    assert context.selection_policy["max_total_chars"] == 28000
    assert context.selection_policy["max_chars_per_paper"] == 14000
    assert context.selection_policy["max_chunks_per_paper"] == 20
    for role in ("source", "target"):
        chunks = [item for item in context.selected_evidence if item.paper_role == role]
        assert len(chunks) <= 20
        assert sum(len(item.text) for item in chunks) <= 14000
    assert context.total_characters <= 28000
    assert context.selection_policy["cross_paper_pairs"]


def test_paired_selection_and_ids_are_deterministic_in_document_order() -> None:
    evidence_input = _comparison_input()
    policy = EvidenceSelectionPolicy(max_chunks_per_paper=4)
    first = build_lineage_llm_context(evidence_input, policy=policy)
    second = build_lineage_llm_context(evidence_input, policy=policy)
    assert first.to_dict() == second.to_dict()
    assert [item.evidence_id for item in first.selected_evidence] == [
        "S001", "S002", "S003", "S004", "T001", "T002", "T003", "T004",
    ]
    for role, package in (("source", evidence_input.source), ("target", evidence_input.target)):
        positions = [
            next(index for index, chunk in enumerate(package.chunks) if chunk.text == item.text)
            for item in first.selected_evidence if item.paper_role == role
        ]
        assert positions == sorted(positions)
    restored = LineageLLMContext.from_mapping(json.loads(json.dumps(first.to_dict())))
    assert restored.to_dict() == first.to_dict()


def test_comparison_reservation_does_not_repeat_chunks_or_same_component_signature() -> None:
    source = [
        _chunk("source-1", f"LSTM hidden states encode temporal inputs in variant {index}.", "methods")
        for index in range(12)
    ]
    target = [
        _chunk("target-1", f"Attention weights LSTM hidden states for variant {index}.", "methods")
        for index in range(12)
    ]
    context = build_lineage_llm_context(_input(source, target))
    pairs = context.selection_policy["cross_paper_pairs"]
    assert len(pairs) == 1
    assert len({item.evidence_id for item in context.selected_evidence}) == len(context.selected_evidence)


def test_architecture_comparison_in_experiments_is_eligible_but_generic_evaluation_is_not() -> None:
    source = _chunk(
        "source-1", "The LSTM outputs class probability distribution at the last time step.",
        "experiments",
    )
    target = _chunk("target-1", TARGET_OUTPUT_TEXT, "methods")
    score, cues = _score_cross_paper_pair(source, target)
    assert score >= 5
    assert "output/representation" in cues
    generic = _chunk(
        "source-1", "The LSTM is evaluated on wearable activity recognition datasets.", "experiments"
    )
    assert _score_cross_paper_pair(generic, target) == (0.0, [])
    context = build_lineage_llm_context(_input([generic, source], [target]))
    pair = context.selection_policy["cross_paper_pairs"][0]
    assert context.get_evidence(pair["source_evidence_id"]).text == source.text


def test_pair_is_not_reserved_when_one_side_exceeds_its_budget() -> None:
    source = _chunk("source-1", SOURCE_OUTPUT_TEXT, "methods")
    target = _chunk("target-1", TARGET_OUTPUT_TEXT + " x" * 100, "methods")
    context = build_lineage_llm_context(
        _input([source], [target]),
        policy=EvidenceSelectionPolicy(max_total_chars=240, max_chars_per_paper=120),
    )
    assert context.selection_policy["cross_paper_pairs"] == []
    assert [item.text for item in context.selected_evidence] == [SOURCE_OUTPUT_TEXT]
    assert any("target: skipped" in warning for warning in context.warnings)
    assert not any("Cross-paper" in reason for reason in context.selected_evidence[0].selection_reasons)


def test_comparison_matches_pdf_ligatures_without_rewriting_evidence() -> None:
    source = _chunk("source-1", "The softmax classi\ufb01er transforms the final embedding.", "methods")
    target = _chunk("target-1", "The softmax classifier takes an attention embedding.", "methods")
    score, cues = _score_cross_paper_pair(source, target)
    assert score >= 5
    assert {"softmax", "classifier", "embedding"} <= set(cues)
    context = build_lineage_llm_context(_input([source], [target]))
    assert context.get_evidence("S001").text == source.text


def test_named_architecture_mention_is_only_an_extra_retrieval_cue() -> None:
    source = _chunk(
        "source-1", "TemporalNet computes LSTM hidden states for temporal classification.", "methods"
    )
    named_target = _chunk(
        "target-1", "TemporalNet attention weights LSTM hidden states for classification.", "methods"
    )
    other_target = _chunk(
        "target-1", "Attention weights LSTM hidden states for classification.", "methods"
    )
    named_score, named_cues = _score_cross_paper_pair(source, named_target)
    ordinary_score, _ = _score_cross_paper_pair(source, other_target)
    assert named_score > ordinary_score
    assert "temporalnet" in named_cues


@pytest.mark.parametrize("section", ["introduction", "abstract", "unknown"])
def test_comparison_reservation_is_limited_to_component_sections(section: str) -> None:
    source = _chunk("source-1", SOURCE_OUTPUT_TEXT, section)
    target = _chunk("target-1", TARGET_OUTPUT_TEXT, "methods")
    assert _score_cross_paper_pair(source, target) == (0.0, [])


def test_methods_pair_has_priority_over_equivalent_experiment_pair() -> None:
    source = _chunk("source-1", SOURCE_OUTPUT_TEXT, "methods")
    experiment = _chunk("source-1", SOURCE_OUTPUT_TEXT, "experiments")
    target = _chunk("target-1", TARGET_OUTPUT_TEXT, "methods")
    method_score, _ = _score_cross_paper_pair(source, target)
    experiment_score, _ = _score_cross_paper_pair(experiment, target)
    assert method_score > experiment_score >= 5


def test_comparison_slots_are_limited_and_cover_different_components() -> None:
    components = [
        "hidden states", "feature maps", "softmax classifier", "attention layers",
        "pooling", "fusion", "embedding", "temporal representation",
    ]
    source = [
        _chunk("source-1", f"The architecture computes {component} for sequence processing.", "methods")
        for component in components
    ]
    target = [
        _chunk("target-1", f"The approach computes {component} for sequence processing.", "methods")
        for component in components
    ]
    context = build_lineage_llm_context(_input(source, target))
    pairs = context.selection_policy["cross_paper_pairs"]
    assert len(pairs) == 6
    assert len({pair["source_evidence_id"] for pair in pairs}) == 6
    assert len({pair["target_evidence_id"] for pair in pairs}) == 6
    assert len({tuple(pair["cues"]) for pair in pairs}) == 6
