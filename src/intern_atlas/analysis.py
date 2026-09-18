"""Extensible analysis schemas.

This module contains data contracts for future analysis workflows.  It does not
run an LLM or infer values from papers yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping


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
