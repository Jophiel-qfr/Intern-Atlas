import json

import pytest

from intern_atlas.analysis import (
    EvidenceItem,
    HARAnalysis,
    MethodChange,
    MethodLineageAnalysis,
)


def _method_evidence(
    paper_id: str,
    quote: str,
    supports: str,
) -> EvidenceItem:
    return EvidenceItem(
        paper_id=paper_id,
        paper_title="Example HAR paper",
        section="Methods",
        location="p. 4",
        quote=quote,
        evidence_type="method_description",
        supports=supports,
        source_kind="fulltext",
    )


def test_method_lineage_can_represent_deepconvlstm_to_bilstm_changes() -> None:
    analysis = MethodLineageAnalysis(
        source_paper_id="deepconvlstm",
        source_paper_title="DeepConvLSTM",
        target_paper_id="bilstm-paper",
        target_paper_title="A hypothetical BiLSTM HAR paper",
        relation_type="improves",
        inherited_components=["feature_extraction"],
        changed_components=["temporal_modeling"],
        added_components=["attention_mechanism"],
        method_changes=[
            MethodChange(
                component="feature_extraction",
                change_type="inherited",
                from_value="CNN",
                to_value="CNN",
                evidence=[
                    _method_evidence(
                        "deepconvlstm",
                        "The convolutional layers extract local features.",
                        "inherited_components",
                    )
                ],
                confidence=0.92,
            ),
            MethodChange(
                component="temporal_modeling",
                change_type="replaced",
                from_value="LSTM",
                to_value="BiLSTM",
                evidence=[
                    _method_evidence(
                        "bilstm-paper",
                        "We replace the LSTM with a bidirectional LSTM.",
                        "changed_components",
                    )
                ],
                confidence=0.96,
            ),
            MethodChange(
                component="attention_mechanism",
                change_type="added",
                to_value="attention",
                evidence=[
                    _method_evidence(
                        "bilstm-paper",
                        "An attention layer is added after temporal encoding.",
                        "added_components",
                    )
                ],
                confidence=0.9,
            ),
        ],
        evidence=[
            _method_evidence(
                "bilstm-paper",
                "We replace the LSTM with a bidirectional LSTM.",
                "method_changes",
            )
        ],
        confidence=0.9,
        uncertainty=["The source paper's implementation details are incomplete."],
        analysis_status="probable",
        source_analysis=HARAnalysis(
            base_model="DeepConvLSTM",
            feature_extraction="CNN",
            temporal_modeling="LSTM",
        ),
        target_analysis=HARAnalysis(
            base_model="BiLSTM HAR model",
            feature_extraction="CNN",
            temporal_modeling="BiLSTM",
            attention_mechanism="attention",
        ),
    )

    assert analysis.method_changes[1].from_value == "LSTM"
    assert analysis.method_changes[1].to_value == "BiLSTM"
    assert analysis.inherited_components == ["feature_extraction"]
    assert analysis.added_components == ["attention_mechanism"]
    assert analysis.source_analysis.temporal_modeling == "LSTM"


def test_citation_only_evidence_can_remain_insufficient() -> None:
    citation = EvidenceItem(
        paper_id="target",
        paper_title="Target paper",
        quote="The target paper cites the source paper.",
        evidence_type="citation_context",
        source_kind="metadata",
        supports="citation only",
    )

    analysis = MethodLineageAnalysis(
        source_paper_id="source",
        source_paper_title="Source paper",
        target_paper_id="target",
        target_paper_title="Target paper",
        evidence=[citation],
        analysis_status="insufficient_evidence",
    )

    assert analysis.relation_type is None
    assert analysis.analysis_status == "insufficient_evidence"


def test_confirmed_analysis_requires_evidence() -> None:
    with pytest.raises(ValueError, match="requires at least one evidence"):
        MethodLineageAnalysis(
            source_paper_id="source",
            source_paper_title="Source paper",
            target_paper_id="target",
            target_paper_title="Target paper",
            relation_type="extends",
            analysis_status="confirmed",
        )


@pytest.mark.parametrize("analysis_status", ["confirmed", "probable"])
def test_confirmed_and_probable_analysis_require_a_relation(analysis_status: str) -> None:
    with pytest.raises(ValueError, match="requires a relation_type"):
        MethodLineageAnalysis(
            source_paper_id="source",
            source_paper_title="Source paper",
            target_paper_id="target",
            target_paper_title="Target paper",
            analysis_status=analysis_status,
        )


def test_citation_context_alone_cannot_confirm_method_lineage() -> None:
    citation = EvidenceItem(
        paper_id="target",
        paper_title="Target paper",
        quote="The target paper cites the source paper.",
        evidence_type="citation_context",
        source_kind="fulltext",
        supports="citation only",
    )

    with pytest.raises(ValueError, match="beyond citation or metadata"):
        MethodLineageAnalysis(
            source_paper_id="source",
            source_paper_title="Source paper",
            target_paper_id="target",
            target_paper_title="Target paper",
            relation_type="extends",
            evidence=[citation],
            analysis_status="confirmed",
        )


def test_metadata_alone_cannot_confirm_method_lineage() -> None:
    metadata = EvidenceItem(
        paper_id="target",
        paper_title="Target paper",
        quote="Target paper metadata",
        evidence_type="abstract_only",
        source_kind="metadata",
        supports="metadata only",
    )

    with pytest.raises(ValueError, match="beyond citation or metadata"):
        MethodLineageAnalysis(
            source_paper_id="source",
            source_paper_title="Source paper",
            target_paper_id="target",
            target_paper_title="Target paper",
            relation_type="extends",
            evidence=[metadata],
            analysis_status="confirmed",
        )


def test_confirmed_method_changes_require_their_own_evidence() -> None:
    lineage_evidence = _method_evidence(
        "target",
        "The target modifies the temporal block.",
        "method_changes",
    )

    with pytest.raises(ValueError, match="method changes require"):
        MethodLineageAnalysis(
            source_paper_id="source",
            source_paper_title="Source paper",
            target_paper_id="target",
            target_paper_title="Target paper",
            relation_type="improves",
            method_changes=[
                MethodChange(
                    component="temporal_modeling",
                    change_type="replaced",
                    from_value="LSTM",
                    to_value="BiLSTM",
                )
            ],
            evidence=[lineage_evidence],
            analysis_status="confirmed",
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_must_be_between_zero_and_one(confidence: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        MethodChange(
            component="temporal_modeling",
            change_type="modified",
            confidence=confidence,
        )


def test_invalid_relation_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="relation_type"):
        MethodLineageAnalysis(
            source_paper_id="source",
            source_paper_title="Source paper",
            target_paper_id="target",
            target_paper_title="Target paper",
            relation_type="cites",
        )


@pytest.mark.parametrize("evidence_type", ["direct_statement", "experiment_result"])
def test_direct_and_experiment_evidence_require_a_quote(evidence_type: str) -> None:
    with pytest.raises(ValueError, match="requires a non-empty quote"):
        EvidenceItem(
            paper_id="paper",
            paper_title="Paper",
            quote="",
            evidence_type=evidence_type,
        )


def test_lineage_json_round_trip_preserves_nested_analysis() -> None:
    original = MethodLineageAnalysis(
        source_paper_id="source",
        source_paper_title="Source paper",
        target_paper_id="target",
        target_paper_title="Target paper",
        relation_type="uses_component",
        inherited_components=["feature_extraction"],
        method_changes=[
            MethodChange(
                component="feature_extraction",
                change_type="inherited",
                from_value="CNN",
                to_value="CNN",
                description="The target retains the source feature extractor.",
                evidence=[
                    _method_evidence(
                        "target",
                        "The same convolutional feature extractor is used.",
                        "inherited_components",
                    )
                ],
                confidence=0.8,
            )
        ],
        evidence=[
            _method_evidence(
                "target",
                "The same convolutional feature extractor is used.",
                "method_changes",
            )
        ],
        confidence=0.8,
        analysis_status="confirmed",
        source_analysis=HARAnalysis(base_model="CNN-LSTM"),
        target_analysis=HARAnalysis(base_model="CNN-LSTM", attention_mechanism="attention"),
        extra={"review_note": "human reviewed"},
    )

    encoded = json.dumps(original.to_dict(), ensure_ascii=False)
    restored = MethodLineageAnalysis.from_mapping(json.loads(encoded))

    assert restored.to_dict() == original.to_dict()
