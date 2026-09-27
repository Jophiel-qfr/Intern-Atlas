"""Strict raw response validation and evidence-backed lineage conversion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, ClassVar, Mapping

from .analysis import (
    ANALYSIS_STATUSES,
    METHOD_CHANGE_TYPES,
    METHOD_LINEAGE_RELATIONS,
    EvidenceItem,
    MethodChange,
    MethodLineageAnalysis,
)
from .lineage_context import LineageLLMContext, SelectedEvidenceChunk


class LineageResponseParseError(ValueError):
    """The model content is not valid JSON or does not match the raw schema."""


class LineageResponseValidationError(ValueError):
    """The valid JSON makes claims that fail local evidence constraints."""


def _strict_fields(
    data: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - set(data)
    extra = set(data) - required - optional
    if missing:
        raise LineageResponseParseError(
            "LLM response is missing required field(s): " + ", ".join(sorted(missing))
        )
    if extra:
        raise LineageResponseParseError(
            "LLM response contains unsupported field(s): " + ", ".join(sorted(extra))
        )


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise LineageResponseParseError(f"{field_name} must be an array of non-empty strings")
    return list(value)


def _confidence(value: Any, field_name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LineageResponseParseError(f"{field_name} must be a number between 0 and 1")
    result = float(value)
    if not isfinite(result) or not 0 <= result <= 1:
        raise LineageResponseParseError(f"{field_name} must be a number between 0 and 1")
    return result


@dataclass
class EvidenceBackedClaim:
    text: str
    evidence_ids: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise LineageResponseParseError("claim text must be a non-empty string")
        self.evidence_ids = _string_list(self.evidence_ids, "claim evidence_ids")
        if not self.evidence_ids:
            raise LineageResponseParseError("each claim must cite at least one evidence_id")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvidenceBackedClaim":
        if not isinstance(data, Mapping):
            raise LineageResponseParseError("evidence-backed claims must be JSON objects")
        _strict_fields(data, required={"text", "evidence_ids"})
        text = data["text"]
        if not isinstance(text, str) or not text.strip():
            raise LineageResponseParseError("claim text must be a non-empty string")
        evidence_ids = _string_list(data["evidence_ids"], "claim evidence_ids")
        if not evidence_ids:
            raise LineageResponseParseError("each claim must cite at least one evidence_id")
        return cls(text=text, evidence_ids=evidence_ids)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "evidence_ids": list(self.evidence_ids)}


@dataclass
class LLMMethodChange:
    component: str
    change_type: str
    from_value: str | None
    to_value: str | None
    description: str
    evidence_ids: list[str]
    confidence: float | None = None

    FIELDS: ClassVar[set[str]] = {
        "component",
        "change_type",
        "from_value",
        "to_value",
        "description",
        "evidence_ids",
        "confidence",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component.strip():
            raise LineageResponseParseError("method change component must be a non-empty string")
        if self.change_type not in METHOD_CHANGE_TYPES:
            raise LineageResponseParseError("method change has an invalid change_type")
        for name in ("from_value", "to_value"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise LineageResponseParseError(f"{name} must be a string or null")
        if not isinstance(self.description, str):
            raise LineageResponseParseError("method change description must be a string")
        self.evidence_ids = _string_list(self.evidence_ids, "method change evidence_ids")
        if not self.evidence_ids:
            raise LineageResponseParseError("each method change must cite at least one evidence_id")
        self.confidence = _confidence(
            self.confidence, "method change confidence", optional=True
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LLMMethodChange":
        if not isinstance(data, Mapping):
            raise LineageResponseParseError("method_changes entries must be JSON objects")
        _strict_fields(data, required=cls.FIELDS)
        component = data["component"]
        description = data["description"]
        if not isinstance(component, str) or not component.strip():
            raise LineageResponseParseError("method change component must be a non-empty string")
        change_type = data["change_type"]
        if not isinstance(change_type, str) or change_type not in METHOD_CHANGE_TYPES:
            raise LineageResponseParseError("method change has an invalid change_type")
        for name in ("from_value", "to_value"):
            if data[name] is not None and not isinstance(data[name], str):
                raise LineageResponseParseError(f"{name} must be a string or null")
        if not isinstance(description, str):
            raise LineageResponseParseError("method change description must be a string")
        evidence_ids = _string_list(data["evidence_ids"], "method change evidence_ids")
        if not evidence_ids:
            raise LineageResponseParseError("each method change must cite at least one evidence_id")
        return cls(
            component=component,
            change_type=change_type,
            from_value=data["from_value"],
            to_value=data["to_value"],
            description=description,
            evidence_ids=evidence_ids,
            confidence=_confidence(data["confidence"], "method change confidence", optional=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "change_type": self.change_type,
            "from_value": self.from_value,
            "to_value": self.to_value,
            "description": self.description,
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
        }


@dataclass
class LineageLLMResponse:
    """Model declarations plus references to evidence IDs, never evidence text."""

    analysis_status: str
    relation_type: str | None
    confidence: float
    uncertainty: str | None
    relation_evidence_ids: list[str]
    inherited_components: list[EvidenceBackedClaim] = field(default_factory=list)
    changed_components: list[EvidenceBackedClaim] = field(default_factory=list)
    added_components: list[EvidenceBackedClaim] = field(default_factory=list)
    removed_components: list[EvidenceBackedClaim] = field(default_factory=list)
    problem_addressed: EvidenceBackedClaim | None = None
    claimed_contribution: EvidenceBackedClaim | None = None
    experimental_evidence: list[EvidenceBackedClaim] = field(default_factory=list)
    limitations: list[EvidenceBackedClaim] = field(default_factory=list)
    method_changes: list[LLMMethodChange] = field(default_factory=list)

    REQUIRED_FIELDS: ClassVar[set[str]] = {
        "analysis_status",
        "relation_type",
        "confidence",
        "uncertainty",
        "relation_evidence_ids",
        "inherited_components",
        "changed_components",
        "added_components",
        "removed_components",
        "problem_addressed",
        "claimed_contribution",
        "experimental_evidence",
        "limitations",
        "method_changes",
    }

    def __post_init__(self) -> None:
        if self.analysis_status not in ANALYSIS_STATUSES:
            raise LineageResponseParseError("analysis_status is invalid")
        if self.relation_type is not None and self.relation_type not in METHOD_LINEAGE_RELATIONS:
            raise LineageResponseParseError("relation_type is invalid")
        if self.analysis_status == "insufficient_evidence" and self.relation_type is not None:
            raise LineageResponseParseError(
                "insufficient_evidence response must use relation_type null"
            )
        self.confidence = _confidence(self.confidence, "confidence")  # type: ignore[assignment]
        if self.uncertainty is not None and not isinstance(self.uncertainty, str):
            raise LineageResponseParseError("uncertainty must be a string or null")
        self.relation_evidence_ids = _string_list(
            self.relation_evidence_ids, "relation_evidence_ids"
        )
        for name in (
            "inherited_components",
            "changed_components",
            "added_components",
            "removed_components",
            "experimental_evidence",
            "limitations",
        ):
            values = getattr(self, name)
            if not isinstance(values, list):
                raise LineageResponseParseError(f"{name} must be an array")
            normalized_claims = []
            for item in values:
                if isinstance(item, EvidenceBackedClaim):
                    normalized_claims.append(item)
                elif isinstance(item, Mapping):
                    normalized_claims.append(EvidenceBackedClaim.from_mapping(item))
                else:
                    raise LineageResponseParseError(f"{name} entries must be claim objects")
            setattr(self, name, normalized_claims)
        for name in ("problem_addressed", "claimed_contribution"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, EvidenceBackedClaim):
                if isinstance(value, Mapping):
                    setattr(self, name, EvidenceBackedClaim.from_mapping(value))
                else:
                    raise LineageResponseParseError(f"{name} must be a claim object or null")
        normalized_changes = []
        for change in self.method_changes:
            if isinstance(change, LLMMethodChange):
                normalized_changes.append(change)
            elif isinstance(change, Mapping):
                normalized_changes.append(LLMMethodChange.from_mapping(change))
            else:
                raise LineageResponseParseError("method_changes entries must be objects")
        self.method_changes = normalized_changes
        if self.analysis_status in {"confirmed", "probable"}:
            if self.relation_type is None:
                raise LineageResponseParseError(
                    f"{self.analysis_status} response requires relation_type"
                )
            if not self.relation_evidence_ids:
                raise LineageResponseParseError(
                    f"{self.analysis_status} response requires relation_evidence_ids"
                )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LineageLLMResponse":
        if not isinstance(data, Mapping):
            raise LineageResponseParseError("LLM response must be a JSON object")
        _strict_fields(data, required=cls.REQUIRED_FIELDS)
        status = data["analysis_status"]
        relation = data["relation_type"]
        if not isinstance(status, str):
            raise LineageResponseParseError("analysis_status must be a string")
        if relation is not None and not isinstance(relation, str):
            raise LineageResponseParseError("relation_type must be a string or null")
        uncertainty = data["uncertainty"]
        if uncertainty is not None and not isinstance(uncertainty, str):
            raise LineageResponseParseError("uncertainty must be a string or null")

        def claims(name: str) -> list[EvidenceBackedClaim]:
            value = data[name]
            if not isinstance(value, list):
                raise LineageResponseParseError(f"{name} must be an array")
            return [EvidenceBackedClaim.from_mapping(item) for item in value]

        def optional_claim(name: str) -> EvidenceBackedClaim | None:
            value = data[name]
            if value is None:
                return None
            if not isinstance(value, Mapping):
                raise LineageResponseParseError(f"{name} must be a claim object or null")
            return EvidenceBackedClaim.from_mapping(value)

        raw_changes = data["method_changes"]
        if not isinstance(raw_changes, list):
            raise LineageResponseParseError("method_changes must be an array")
        raw_ids = _string_list(data["relation_evidence_ids"], "relation_evidence_ids")
        return cls(
            analysis_status=status,
            relation_type=relation,
            confidence=_confidence(data["confidence"], "confidence"),  # type: ignore[arg-type]
            uncertainty=uncertainty,
            relation_evidence_ids=raw_ids,
            inherited_components=claims("inherited_components"),
            changed_components=claims("changed_components"),
            added_components=claims("added_components"),
            removed_components=claims("removed_components"),
            problem_addressed=optional_claim("problem_addressed"),
            claimed_contribution=optional_claim("claimed_contribution"),
            experimental_evidence=claims("experimental_evidence"),
            limitations=claims("limitations"),
            method_changes=[LLMMethodChange.from_mapping(item) for item in raw_changes],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_status": self.analysis_status,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "relation_evidence_ids": list(self.relation_evidence_ids),
            "inherited_components": [claim.to_dict() for claim in self.inherited_components],
            "changed_components": [claim.to_dict() for claim in self.changed_components],
            "added_components": [claim.to_dict() for claim in self.added_components],
            "removed_components": [claim.to_dict() for claim in self.removed_components],
            "problem_addressed": self.problem_addressed.to_dict() if self.problem_addressed else None,
            "claimed_contribution": self.claimed_contribution.to_dict() if self.claimed_contribution else None,
            "experimental_evidence": [claim.to_dict() for claim in self.experimental_evidence],
            "limitations": [claim.to_dict() for claim in self.limitations],
            "method_changes": [change.to_dict() for change in self.method_changes],
        }


def _unwrap_json_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    first = lines[0].strip().casefold()
    if first not in {"```", "```json"} or len(lines) < 3 or lines[-1].strip() != "```":
        raise LineageResponseParseError("LLM response has a malformed JSON code fence")
    return "\n".join(lines[1:-1]).strip()


def parse_lineage_llm_response(
    content: str,
    context: LineageLLMContext,
) -> LineageLLMResponse:
    """Parse JSON, reject unknown fields, and verify every cited context ID."""

    try:
        payload = json.loads(_unwrap_json_fence(content))
    except LineageResponseParseError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        raise LineageResponseParseError("LLM response is not valid JSON.") from exc
    response = LineageLLMResponse.from_mapping(payload)
    for evidence_ids in _all_evidence_id_lists(response):
        try:
            context.validate_evidence_ids(evidence_ids)
        except (KeyError, ValueError) as exc:
            raise LineageResponseValidationError(
                f"LLM response cites an evidence_id outside this context: {exc}"
            ) from exc
    return response


def _all_evidence_id_lists(response: LineageLLMResponse) -> list[list[str]]:
    lists = [response.relation_evidence_ids]
    for name in (
        "inherited_components",
        "changed_components",
        "added_components",
        "removed_components",
        "experimental_evidence",
        "limitations",
    ):
        lists.extend(claim.evidence_ids for claim in getattr(response, name))
    for claim in (response.problem_addressed, response.claimed_contribution):
        if claim:
            lists.append(claim.evidence_ids)
    lists.extend(change.evidence_ids for change in response.method_changes)
    return lists


def _chunk_to_evidence_item(
    context: LineageLLMContext,
    evidence_id: str,
    supports: str,
) -> EvidenceItem:
    selected: SelectedEvidenceChunk = context.get_evidence(evidence_id)
    source_kind = selected.source_kind
    if source_kind == "citation_context" or selected.chunk_type == "citation_context":
        evidence_type = "citation_context"
        # EvidenceItem's source_kind tracks where text came from; citation
        # context is a citation-derived metadata source in the current contract.
        if source_kind == "citation_context":
            source_kind = "metadata"
    elif selected.chunk_type == "abstract" or source_kind == "abstract":
        evidence_type = "abstract_only"
    elif selected.section in {"experiments", "results"}:
        evidence_type = "experiment_result"
    else:
        evidence_type = "method_description"
    location: str | int | None = selected.location
    if location is None and selected.page is not None:
        location = f"PDF page {selected.page}"
    return EvidenceItem(
        paper_id=(
            context.source_paper_id if selected.paper_role == "source" else context.target_paper_id
        ),
        paper_title=(
            context.source_paper_title
            if selected.paper_role == "source"
            else context.target_paper_title
        ),
        section=selected.section,
        location=location,
        quote=selected.text,
        evidence_type=evidence_type,
        supports=supports,
        source_kind=source_kind,
    )


def _append_support(
    references: dict[str, list[str]], evidence_ids: list[str], support: str
) -> None:
    for evidence_id in evidence_ids:
        supports = references.setdefault(evidence_id, [])
        if support not in supports:
            supports.append(support)


def build_method_lineage_analysis(
    context: LineageLLMContext,
    llm_response: LineageLLMResponse | Mapping[str, Any] | str,
) -> MethodLineageAnalysis:
    """Rebuild final evidence only from context IDs; never trust model quotes."""

    if isinstance(llm_response, str):
        response = parse_lineage_llm_response(llm_response, context)
    elif isinstance(llm_response, LineageLLMResponse):
        response = llm_response
    elif isinstance(llm_response, Mapping):
        response = LineageLLMResponse.from_mapping(llm_response)
    else:
        raise TypeError("llm_response must be JSON text, a mapping, or LineageLLMResponse")

    for evidence_ids in _all_evidence_id_lists(response):
        try:
            context.validate_evidence_ids(evidence_ids)
        except (KeyError, ValueError) as exc:
            raise LineageResponseValidationError(
                f"LLM response cites an evidence_id outside this context: {exc}"
            ) from exc

    status = response.analysis_status
    uncertainty = response.uncertainty
    downgrade_reasons: list[str] = []
    relation_items = [context.get_evidence(item) for item in response.relation_evidence_ids]
    if status in {"confirmed", "probable"}:
        substantive_roles = {
            item.paper_role
            for item in relation_items
            if item.source_kind not in {"metadata", "citation_context"}
            and item.chunk_type != "citation_context"
        }
        relation_roles = {item.paper_role for item in relation_items}
        if substantive_roles != {"source", "target"}:
            downgrade_reasons.append(
                "Relation evidence lacks substantive non-citation text from both papers; citation-context or metadata alone is insufficient."
            )
        if relation_roles != {"source", "target"}:
            downgrade_reasons.append(
                "Relation evidence does not include both source and target papers."
            )

    prepared_changes: list[tuple[LLMMethodChange, list[SelectedEvidenceChunk]]] = []
    for change in response.method_changes:
        items = context.validate_evidence_ids(change.evidence_ids)
        roles = {item.paper_role for item in items}
        substantive_roles = {
            item.paper_role
            for item in items
            if item.source_kind not in {"metadata", "citation_context"}
            and item.chunk_type != "citation_context"
        }
        if status in {"confirmed", "probable"} and "target" not in substantive_roles:
            raise LineageResponseValidationError(
                f"{change.change_type} method change {change.component!r} requires target evidence with substantive method text."
            )
        if (
            status in {"confirmed", "probable"}
            and change.change_type in {"inherited", "modified", "replaced", "removed"}
            and "source" not in substantive_roles
        ):
            downgrade_reasons.append(
                f"Method change {change.component!r} lacks source evidence for comparison."
            )
        prepared_changes.append((change, items))

    target_facing_claims: list[tuple[str, EvidenceBackedClaim, bool]] = []
    for field_name in (
        "changed_components",
        "added_components",
        "removed_components",
        "experimental_evidence",
        "limitations",
    ):
        target_facing_claims.extend(
            (field_name, claim, False) for claim in getattr(response, field_name)
        )
    if response.claimed_contribution:
        target_facing_claims.append(
            ("claimed_contribution", response.claimed_contribution, False)
        )
    if response.problem_addressed:
        target_facing_claims.append(("problem_addressed", response.problem_addressed, False))
    target_facing_claims.extend(
        ("inherited_components", claim, True)
        for claim in response.inherited_components
    )
    if status in {"confirmed", "probable"}:
        for field_name, claim, needs_source in target_facing_claims:
            roles = {
                item.paper_role
                for item in context.validate_evidence_ids(claim.evidence_ids)
                if item.source_kind not in {"metadata", "citation_context"}
                and item.chunk_type != "citation_context"
            }
            if "target" not in roles:
                downgrade_reasons.append(
                    f"Claim {field_name!r} lacks substantive target evidence."
                )
            if needs_source and "source" not in roles:
                downgrade_reasons.append(
                    "An inherited-component claim lacks substantive source evidence."
                )

    if downgrade_reasons:
        status = "uncertain"
        note = " ".join(dict.fromkeys(downgrade_reasons))
        uncertainty = f"{uncertainty} {note}".strip() if uncertainty else note

    references: dict[str, list[str]] = {}
    _append_support(references, response.relation_evidence_ids, "relation")

    def register_claim(claim: EvidenceBackedClaim, support: str) -> str:
        _append_support(references, claim.evidence_ids, support)
        return claim.text

    inherited = [
        register_claim(claim, "inherited_components")
        for claim in response.inherited_components
    ]
    changed = [
        register_claim(claim, "changed_components")
        for claim in response.changed_components
    ]
    added = [
        register_claim(claim, "added_components")
        for claim in response.added_components
    ]
    removed = [
        register_claim(claim, "removed_components")
        for claim in response.removed_components
    ]
    problem = (
        register_claim(response.problem_addressed, "problem_addressed")
        if response.problem_addressed
        else None
    )
    contribution = (
        register_claim(response.claimed_contribution, "claimed_contribution")
        if response.claimed_contribution
        else None
    )
    experiments = [
        register_claim(claim, "experimental_evidence")
        for claim in response.experimental_evidence
    ]
    limitations = [
        register_claim(claim, "limitation") for claim in response.limitations
    ]

    method_changes: list[MethodChange] = []
    for change, _ in prepared_changes:
        support = f"method_change:{change.component}"
        _append_support(references, change.evidence_ids, support)
        method_changes.append(
            MethodChange(
                component=change.component,
                change_type=change.change_type,
                from_value=change.from_value,
                to_value=change.to_value,
                description=change.description,
                evidence=[
                    _chunk_to_evidence_item(context, evidence_id, support)
                    for evidence_id in dict.fromkeys(change.evidence_ids)
                ],
                confidence=change.confidence,
            )
        )

    all_evidence = [
        _chunk_to_evidence_item(
            context,
            selected.evidence_id,
            references[selected.evidence_id][0]
            if len(references[selected.evidence_id]) == 1
            else references[selected.evidence_id],
        )
        for selected in context.selected_evidence
        if selected.evidence_id in references
    ]
    return MethodLineageAnalysis(
        source_paper_id=context.source_paper_id,
        source_paper_title=context.source_paper_title,
        target_paper_id=context.target_paper_id,
        target_paper_title=context.target_paper_title,
        relation_type=response.relation_type,
        inherited_components=inherited,
        changed_components=changed,
        added_components=added,
        removed_components=removed,
        problem_addressed=problem,
        claimed_contribution=contribution,
        experimental_evidence=experiments,
        limitations=limitations,
        method_changes=method_changes,
        evidence=all_evidence,
        confidence=response.confidence,
        uncertainty=uncertainty,
        analysis_status=status,
    )
