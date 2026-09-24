"""Extensible analysis schemas.

This module contains data contracts for future analysis workflows.  It does not
run an LLM or infer values from papers yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from .util import VALID_EDGE_TYPES


EVIDENCE_TYPES = frozenset(
    {
        "direct_statement",
        "method_description",
        "experiment_result",
        "citation_context",
        "abstract_only",
    }
)
EVIDENCE_SOURCE_KINDS = frozenset({"fulltext", "pdf", "abstract", "metadata"})
METHOD_CHANGE_TYPES = frozenset({"inherited", "modified", "replaced", "added", "removed"})
ANALYSIS_STATUSES = frozenset(
    {"confirmed", "probable", "uncertain", "insufficient_evidence"}
)
METHOD_LINEAGE_RELATIONS = frozenset(
    {
        relation
        for relation in (
            "extends",
            "improves",
            "replaces",
            "adapts",
            "combines",
            "uses_component",
        )
        if relation in VALID_EDGE_TYPES
    }
)


def _validate_choice(name: str, value: str, allowed: frozenset[str]) -> str:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}")
    return value


def _validate_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("confidence must be a number between 0 and 1")
    if not 0 <= value <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return float(value)


def _serialize_value(value: Any, *, include_none: bool) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict(include_none=include_none)
    if isinstance(value, list):
        return [_serialize_value(item, include_none=include_none) for item in value]
    if isinstance(value, tuple):
        return [_serialize_value(item, include_none=include_none) for item in value]
    if isinstance(value, dict):
        return {
            key: _serialize_value(item, include_none=include_none)
            for key, item in value.items()
        }
    return value


def _evidence_item(value: EvidenceItem | Mapping[str, Any]) -> EvidenceItem:
    if isinstance(value, EvidenceItem):
        return value
    if isinstance(value, Mapping):
        return EvidenceItem.from_mapping(value)
    raise TypeError("evidence items must be EvidenceItem instances or mappings")


@dataclass
class HARAnalysis:
    """Optional structured metadata for wearable-sensor HAR papers.

    The values are deliberately permissive: a paper may describe a dataset as
    text, a list, or a richer object.  Unknown fields are retained in ``extra``
    so adding a future analysis dimension does not require changing old callers.
    """

    task: Any = None
    sensor_modality: Any = None
    input_representation: Any = None
    base_model: Any = None
    feature_extraction: Any = None
    temporal_modeling: Any = None
    sensor_fusion: Any = None
    attention_mechanism: Any = None
    training_strategy: Any = None
    loss_function: Any = None
    dataset: Any = None
    inherited_components: list[str] | None = None
    changed_components: list[str] | None = None
    added_components: list[str] | None = None
    problem_addressed: Any = None
    experimental_evidence: Any = None
    limitations: Any = None
    performance_metrics: Any = None
    computational_cost: Any = None
    model_complexity: Any = None
    cross_subject_generalization: Any = None
    cross_dataset_generalization: Any = None
    real_time_capability: Any = None
    deployment_scenario: Any = None
    extra: dict[str, Any] | None = None

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "task",
        "sensor_modality",
        "input_representation",
        "base_model",
        "feature_extraction",
        "temporal_modeling",
        "sensor_fusion",
        "attention_mechanism",
        "training_strategy",
        "loss_function",
        "dataset",
        "inherited_components",
        "changed_components",
        "added_components",
        "problem_addressed",
        "experimental_evidence",
        "limitations",
        "performance_metrics",
        "computational_cost",
        "model_complexity",
        "cross_subject_generalization",
        "cross_dataset_generalization",
        "real_time_capability",
        "deployment_scenario",
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "HARAnalysis":
        known = {name: data.get(name) for name in cls.FIELD_NAMES if name in data}
        unknown = {key: value for key, value in data.items() if key not in cls.FIELD_NAMES and key != "extra"}
        supplied_extra = data.get("extra")
        if isinstance(supplied_extra, Mapping):
            unknown = {
                **{key: value for key, value in supplied_extra.items() if key not in cls.FIELD_NAMES},
                **unknown,
            }
        if unknown:
            known["extra"] = unknown
        return cls(**known)

    def to_dict(self, *, include_none: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in self.FIELD_NAMES:
            value = getattr(self, name)
            if include_none or value is not None:
                payload[name] = value
        if self.extra:
            payload.update(self.extra)
        return payload


@dataclass
class EvidenceItem:
    """A locatable piece of original paper text supporting an analysis claim."""

    paper_id: str
    paper_title: str
    quote: str
    section: str | None = None
    location: str | int | None = None
    evidence_type: str = "abstract_only"
    supports: str | list[str] | None = None
    source_kind: str = "metadata"
    confidence: float | None = None

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "paper_id",
        "paper_title",
        "section",
        "location",
        "quote",
        "evidence_type",
        "supports",
        "source_kind",
        "confidence",
    )

    def __post_init__(self) -> None:
        self.evidence_type = _validate_choice(
            "evidence_type", self.evidence_type, EVIDENCE_TYPES
        )
        self.source_kind = _validate_choice(
            "source_kind", self.source_kind, EVIDENCE_SOURCE_KINDS
        )
        self.confidence = _validate_confidence(self.confidence)
        if self.evidence_type in {"direct_statement", "experiment_result"}:
            if not isinstance(self.quote, str) or not self.quote.strip():
                raise ValueError(
                    f"{self.evidence_type} evidence requires a non-empty quote"
                )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvidenceItem":
        known = {name: data[name] for name in cls.FIELD_NAMES if name in data}
        return cls(**known)

    def to_dict(self, *, include_none: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in self.FIELD_NAMES:
            value = getattr(self, name)
            if include_none or value is not None:
                payload[name] = _serialize_value(value, include_none=include_none)
        return payload


@dataclass
class MethodChange:
    """A claimed change to one method component, backed by evidence."""

    component: str
    change_type: str
    from_value: Any = None
    to_value: Any = None
    description: str | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    confidence: float | None = None

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "component",
        "change_type",
        "from_value",
        "to_value",
        "description",
        "evidence",
        "confidence",
    )

    def __post_init__(self) -> None:
        self.change_type = _validate_choice(
            "change_type", self.change_type, METHOD_CHANGE_TYPES
        )
        self.evidence = [_evidence_item(item) for item in self.evidence]
        self.confidence = _validate_confidence(self.confidence)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MethodChange":
        known = {name: data[name] for name in cls.FIELD_NAMES if name in data}
        return cls(**known)

    def to_dict(self, *, include_none: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in self.FIELD_NAMES:
            value = getattr(self, name)
            if include_none or value is not None:
                payload[name] = _serialize_value(value, include_none=include_none)
        return payload


@dataclass
class MethodLineageAnalysis:
    """Evidence-backed relationship analysis between two papers.

    This schema stores claims but does not infer them.  In particular, a
    citation or citation context alone never changes ``analysis_status`` to
    ``confirmed``.
    """

    source_paper_id: str
    source_paper_title: str
    target_paper_id: str
    target_paper_title: str
    relation_type: str | None = None
    inherited_components: list[str] = field(default_factory=list)
    changed_components: list[str] = field(default_factory=list)
    added_components: list[str] = field(default_factory=list)
    removed_components: list[str] = field(default_factory=list)
    problem_addressed: Any = None
    claimed_contribution: Any = None
    experimental_evidence: Any = None
    limitations: Any = None
    method_changes: list[MethodChange] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    confidence: float | None = None
    uncertainty: Any = None
    analysis_status: str = "insufficient_evidence"
    source_analysis: HARAnalysis | None = None
    target_analysis: HARAnalysis | None = None
    extra: dict[str, Any] | None = None

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "source_paper_id",
        "source_paper_title",
        "target_paper_id",
        "target_paper_title",
        "relation_type",
        "inherited_components",
        "changed_components",
        "added_components",
        "removed_components",
        "problem_addressed",
        "claimed_contribution",
        "experimental_evidence",
        "limitations",
        "method_changes",
        "evidence",
        "confidence",
        "uncertainty",
        "analysis_status",
        "source_analysis",
        "target_analysis",
    )

    def __post_init__(self) -> None:
        self.analysis_status = _validate_choice(
            "analysis_status", self.analysis_status, ANALYSIS_STATUSES
        )
        if self.relation_type is not None:
            self.relation_type = _validate_choice(
                "relation_type", self.relation_type, METHOD_LINEAGE_RELATIONS
            )
        self.method_changes = [
            item
            if isinstance(item, MethodChange)
            else MethodChange.from_mapping(item)
            for item in self.method_changes
        ]
        self.evidence = [_evidence_item(item) for item in self.evidence]
        self.confidence = _validate_confidence(self.confidence)
        if isinstance(self.source_analysis, Mapping):
            self.source_analysis = HARAnalysis.from_mapping(self.source_analysis)
        if isinstance(self.target_analysis, Mapping):
            self.target_analysis = HARAnalysis.from_mapping(self.target_analysis)
        if self.analysis_status in {"confirmed", "probable"} and self.relation_type is None:
            raise ValueError(
                f"{self.analysis_status} analysis requires a relation_type"
            )
        if self.analysis_status == "confirmed":
            if not self.evidence:
                raise ValueError("confirmed analysis requires at least one evidence item")
            if not any(
                item.source_kind != "metadata"
                and item.evidence_type != "citation_context"
                for item in self.evidence
            ):
                raise ValueError(
                    "confirmed analysis requires method evidence beyond citation or metadata"
                )
            if any(not change.evidence for change in self.method_changes):
                raise ValueError(
                    "confirmed method changes require at least one evidence item each"
                )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MethodLineageAnalysis":
        known = {name: data[name] for name in cls.FIELD_NAMES if name in data}
        unknown = {
            key: value
            for key, value in data.items()
            if key not in cls.FIELD_NAMES and key != "extra"
        }
        supplied_extra = data.get("extra")
        if isinstance(supplied_extra, Mapping):
            unknown = {
                **{key: value for key, value in supplied_extra.items() if key not in cls.FIELD_NAMES},
                **unknown,
            }
        if unknown:
            known["extra"] = unknown
        return cls(**known)

    def to_dict(self, *, include_none: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in self.FIELD_NAMES:
            value = getattr(self, name)
            if include_none or value is not None:
                payload[name] = _serialize_value(value, include_none=include_none)
        if self.extra:
            payload.update(_serialize_value(self.extra, include_none=include_none))
        return payload
