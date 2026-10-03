"""Deterministic selection of raw evidence for a future model context.

Selection scores are reading-priority cues only. In particular, citation
context priority does not confirm method inheritance, and lexical overlap is
not a relation judgment. No model or network API is called here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, ClassVar, Mapping, Sequence
from unicodedata import normalize

from .discovery import (
    GENERIC_OVERLAP_TERMS,
    HAR_DOMAIN_PHRASES,
    HAR_DOMAIN_TOKENS,
    STRONG_METHOD_PHRASES,
    STRONG_METHOD_TOKENS,
    method_signals_in_text,
    normalized_text,
    token_set,
)
from .evidence_input import (
    LineageEvidenceInput,
    PaperEvidenceChunk,
    PaperEvidencePackage,
)


SECTION_PRIORITIES = {
    "methods": 60.0,
    "experiments": 48.0,
    "results": 48.0,
    "introduction": 42.0,
    "related_work": 42.0,
    "background": 38.0,
    "abstract": 35.0,
    "discussion": 22.0,
    "conclusion": 18.0,
    "unknown": 8.0,
}
OVERVIEW_SECTIONS = frozenset({"abstract", "introduction", "related_work", "background"})
METHOD_SECTIONS = frozenset({"methods"})
EXPERIMENT_SECTIONS = frozenset({"experiments", "results"})
_EVIDENCE_ID_RE = re.compile(r"^[ST]\d{3,}$")

# Retrieval cues only: these do not assert that a component was inherited or changed.
_MAX_CROSS_PAPER_PAIRS = 6
_MIN_CROSS_PAPER_SCORE = 5.0
_COMPARISON_STOP_TERMS = GENERIC_OVERLAP_TERMS | HAR_DOMAIN_TOKENS | {
    "human", "activity", "activities", "recognition", "sensor", "sensors", "har",
    "wearables", "this", "that", "these", "those", "from", "into", "are", "was",
    "were", "has", "have", "had", "can", "could", "will", "would", "should",
    "may", "not", "also", "which", "such", "than", "then", "each", "all",
    "our", "their", "other", "use", "uses", "used", "both", "only", "two",
    "one", "first", "last", "approach", "approaches", "method", "methods",
    "architecture", "architectures", "layer", "layers", "feature", "features",
    "representation", "representations", "input", "output", "time", "step", "steps",
    "after", "before", "best", "better", "impact", "results", "result", "evaluation",
    "evaluate", "evaluated", "experiment", "experiments", "experimental", "proposed",
    "present", "presented", "paper", "study", "studies", "more", "most", "same",
    "different", "similar", "between", "through", "new", "system", "systems",
    "performance", "framework", "frameworks", "task", "tasks", "application",
    "applications", "however", "while", "during", "when", "where", "over", "under",
    "without", "there", "therefore", "thus", "every", "any", "some",
}
_COMPONENT_PHRASES = {
    "hidden state": ("hidden state", "hidden states"),
    "softmax": ("softmax",),
    "classifier": ("classifier", "classifiers", "classification layer"),
    "embedding": ("embedding", "embeddings"),
    "feature map": ("feature map", "feature maps", "features maps"),
    "convolutional layer": ("convolutional layer", "convolutional layers", "convolution layers"),
    "recurrent layer": ("recurrent layer", "recurrent layers", "recurrent dense layers"),
    "lstm layer": ("lstm layer", "lstm layers"),
    "output layer": ("output layer", "output layers"),
    "pooling": ("pooling",),
    "fusion": ("fusion",),
    "attention layer": ("attention layer", "attention layers"),
    "temporal representation": ("temporal representation", "temporal representations"),
    "feature representation": ("feature representation", "feature representations"),
    "last time step": ("last time step", "last time steps"),
    "model output": ("model output", "output of the model", "class probability distribution"),
}
_OUTPUT_COMPONENTS = frozenset({
    "hidden state", "embedding", "output layer", "temporal representation",
    "feature representation", "last time step", "model output",
})


@dataclass(frozen=True)
class EvidenceSelectionPolicy:
    """Character and chunk-count budgets for each paper and their pair."""

    max_total_chars: int = 28000
    max_chars_per_paper: int = 14000
    max_chunks_per_paper: int = 20

    def __post_init__(self) -> None:
        if self.max_total_chars < 2:
            raise ValueError("max_total_chars must be at least 2")
        if self.max_chars_per_paper < 1:
            raise ValueError("max_chars_per_paper must be positive")
        if self.max_chunks_per_paper < 1:
            raise ValueError("max_chunks_per_paper must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_total_chars": self.max_total_chars,
            "max_chars_per_paper": self.max_chars_per_paper,
            "max_chunks_per_paper": self.max_chunks_per_paper,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvidenceSelectionPolicy":
        return cls(
            max_total_chars=int(data.get("max_total_chars", 28000)),
            max_chars_per_paper=int(data.get("max_chars_per_paper", 14000)),
            max_chunks_per_paper=int(data.get("max_chunks_per_paper", 20)),
        )


@dataclass
class SelectedEvidenceChunk:
    """A raw chunk selected for this context, with a stable context-local ID."""

    evidence_id: str
    paper_role: str
    paper_id: str
    paper_title: str
    text: str
    section: str
    page: int | None
    location: str | None
    source_kind: str
    chunk_type: str
    selection_score: float
    selection_reasons: list[str] = field(default_factory=list)

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "evidence_id",
        "paper_role",
        "paper_id",
        "paper_title",
        "text",
        "section",
        "page",
        "location",
        "source_kind",
        "chunk_type",
        "selection_score",
        "selection_reasons",
    )

    def __post_init__(self) -> None:
        if self.paper_role not in {"source", "target"}:
            raise ValueError("paper_role must be 'source' or 'target'")
        expected_prefix = "S" if self.paper_role == "source" else "T"
        if not _EVIDENCE_ID_RE.fullmatch(self.evidence_id):
            raise ValueError("evidence_id must use the S001 or T001 format")
        if not self.evidence_id.startswith(expected_prefix):
            raise ValueError("evidence_id prefix must match paper_role")
        if not isfinite(float(self.selection_score)):
            raise ValueError("selection_score must be a finite number")
        self.selection_score = round(float(self.selection_score), 3)
        if not all(isinstance(reason, str) for reason in self.selection_reasons):
            raise TypeError("selection_reasons must contain strings")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SelectedEvidenceChunk":
        return cls(**{name: data[name] for name in cls.FIELD_NAMES if name in data})

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.FIELD_NAMES}


@dataclass
class LineageLLMContext:
    """Bounded, ordered context that can be supplied to a future model."""

    source_paper_id: str
    source_paper_title: str
    target_paper_id: str
    target_paper_title: str
    selected_evidence: list[SelectedEvidenceChunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    selection_policy: dict[str, Any] = field(default_factory=dict)
    total_characters: int = 0

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "source_paper_id",
        "source_paper_title",
        "target_paper_id",
        "target_paper_title",
        "selected_evidence",
        "warnings",
        "selection_policy",
        "total_characters",
    )

    def __post_init__(self) -> None:
        self.selected_evidence = [
            item
            if isinstance(item, SelectedEvidenceChunk)
            else SelectedEvidenceChunk.from_mapping(item)
            for item in self.selected_evidence
        ]
        evidence_ids = [item.evidence_id for item in self.selected_evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_id values must be unique within the context")
        if not all(isinstance(warning, str) for warning in self.warnings):
            raise TypeError("warnings must contain strings")
        actual_characters = sum(len(item.text) for item in self.selected_evidence)
        if self.total_characters not in (0, actual_characters):
            raise ValueError("total_characters does not match selected evidence text")
        self.total_characters = actual_characters

    def get_evidence(self, evidence_id: str) -> SelectedEvidenceChunk:
        for item in self.selected_evidence:
            if item.evidence_id == evidence_id:
                return item
        raise KeyError(f"unknown evidence_id: {evidence_id}")

    def validate_evidence_ids(
        self, evidence_ids: Sequence[str]
    ) -> list[SelectedEvidenceChunk]:
        """Resolve model-returned IDs and reject any ID outside this context."""

        resolved: list[SelectedEvidenceChunk] = []
        for evidence_id in evidence_ids:
            if not isinstance(evidence_id, str):
                raise ValueError("evidence IDs must be strings")
            try:
                resolved.append(self.get_evidence(evidence_id))
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
        return resolved

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LineageLLMContext":
        return cls(**{name: data[name] for name in cls.FIELD_NAMES if name in data})

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_paper_id": self.source_paper_id,
            "source_paper_title": self.source_paper_title,
            "target_paper_id": self.target_paper_id,
            "target_paper_title": self.target_paper_title,
            "selected_evidence": [item.to_dict() for item in self.selected_evidence],
            "warnings": self.warnings,
            "selection_policy": self.selection_policy,
            "total_characters": self.total_characters,
        }


def _lexical_signals(text: str) -> tuple[set[str], set[str]]:
    tokens = token_set(text) - GENERIC_OVERLAP_TERMS
    normalized = normalized_text(text)
    method_signals = set(method_signals_in_text(text))
    domain_signals = tokens & HAR_DOMAIN_TOKENS
    domain_signals.update(phrase for phrase in HAR_DOMAIN_PHRASES if phrase in normalized)
    # Use existing discovery vocabularies; generic words such as "deep",
    # "neural", and "network" never become strong signals here.
    method_signals.intersection_update(
        STRONG_METHOD_TOKENS | STRONG_METHOD_PHRASES
    )
    return method_signals, domain_signals


def _pairwise_boost(
    chunk: PaperEvidenceChunk,
    role: str,
    source: PaperEvidencePackage,
    target: PaperEvidencePackage,
) -> tuple[float, list[str]]:
    text = chunk.text
    chunk_methods, chunk_domains = _lexical_signals(text)
    boost = 0.0
    reasons: list[str] = []

    if chunk_methods:
        hits = sorted(chunk_methods)
        boost += min(3, len(hits)) * 1.1
        reasons.append("Strong method terms: " + ", ".join(hits[:3]))
    if chunk_domains:
        hits = sorted(chunk_domains)
        boost += min(2, len(hits)) * 0.65
        reasons.append("HAR domain terms: " + ", ".join(hits[:2]))

    source_methods, _ = _lexical_signals(source.paper_title)
    target_methods, _ = _lexical_signals(target.paper_title)
    shared_title_methods = source_methods & target_methods & chunk_methods
    if shared_title_methods:
        boost += 1.25
        reasons.append(
            "Method term shared by both titles: "
            + ", ".join(sorted(shared_title_methods)[:2])
        )

    other_title = target.paper_title if role == "source" else source.paper_title
    normalized_chunk = normalized_text(text)
    normalized_other_title = normalized_text(other_title)
    other_title_tokens = token_set(other_title) - GENERIC_OVERLAP_TERMS
    if (
        other_title_tokens
        and normalized_other_title
        and normalized_other_title in normalized_chunk
    ):
        boost += 2.0
        reasons.append("Mentions paired paper title (lexical cue)")
    else:
        overlap = (token_set(text) - GENERIC_OVERLAP_TERMS) & other_title_tokens
        # Require multiple distinctive words unless the match is itself a
        # recognized method signal, avoiding boosts from one generic word.
        strong_overlap = overlap & (STRONG_METHOD_TOKENS | HAR_DOMAIN_TOKENS)
        if len(overlap) >= 2 or strong_overlap:
            boost += min(1.5, 0.5 * len(overlap) + 0.5 * bool(strong_overlap))
            reasons.append("Paired-title term overlap (lexical cue)")
    return boost, reasons


def _score_chunk(
    chunk: PaperEvidenceChunk,
    role: str,
    source: PaperEvidencePackage,
    target: PaperEvidencePackage,
) -> tuple[float, list[str]]:
    reasons = [f"Section priority: {chunk.section}"]
    if chunk.source_kind == "citation_context" or chunk.chunk_type == "citation_context":
        # Citation context is useful to inspect, but never confirms inheritance.
        score = 72.0
        reasons.append("Citation context prioritized; this is not a relation judgment")
    else:
        score = SECTION_PRIORITIES.get(chunk.section, SECTION_PRIORITIES["unknown"])
    lexical_boost, lexical_reasons = _pairwise_boost(chunk, role, source, target)
    score += lexical_boost
    reasons.extend(lexical_reasons)
    return round(score, 3), reasons


def _coverage_group(chunk: PaperEvidenceChunk) -> str | None:
    if chunk.section in OVERVIEW_SECTIONS:
        return "overview"
    if chunk.section in METHOD_SECTIONS:
        return "method"
    if chunk.section in EXPERIMENT_SECTIONS:
        return "evidence"
    return None


@dataclass(frozen=True)
class _ComparisonSignals:
    methods: frozenset[str]
    components: frozenset[str]
    tokens: frozenset[str]
    named_terms: frozenset[str]
    eligible: bool


def _comparison_signals(chunk: PaperEvidenceChunk) -> _ComparisonSignals:
    # Normalize extraction ligatures for matching only; evidence text is unchanged.
    text = normalize("NFKC", chunk.text)
    normalized = " " + normalized_text(text) + " "
    components = frozenset(
        cue for cue, aliases in _COMPONENT_PHRASES.items()
        if any(" " + alias + " " in normalized for alias in aliases)
    )
    words = token_set(text)
    tokens = words - _COMPARISON_STOP_TERMS
    methods = set(method_signals_in_text(text)) | (components & {"pooling", "fusion"})
    # Mixed-case identifiers such as named architectures can be extra retrieval cues.
    # Do not assume a mention of a named method proves any relationship.
    names = frozenset(
        word.lower() for word in re.findall(r"\b[A-Za-z][A-Za-z0-9]{3,}\b", text)
        if any(char.isupper() for char in word[1:])
        and any(char.islower() for char in word)
        and word.lower() not in _COMPARISON_STOP_TERMS
    )
    eligible = (
        chunk.section in METHOD_SECTIONS | EXPERIMENT_SECTIONS
        and len(words) >= 4
        and (chunk.section in METHOD_SECTIONS or bool(components))
    )
    return _ComparisonSignals(frozenset(methods), components, frozenset(tokens), names, eligible)


def _score_cross_paper_pair(
    source_chunk: PaperEvidenceChunk,
    target_chunk: PaperEvidenceChunk,
    *,
    source_signals: _ComparisonSignals | None = None,
    target_signals: _ComparisonSignals | None = None,
) -> tuple[float, list[str]]:
    """Score comparable component descriptions, never a method relationship."""

    source_signals = source_signals or _comparison_signals(source_chunk)
    target_signals = target_signals or _comparison_signals(target_chunk)
    # Exclude table labels and generic evaluation text without architecture cues.
    if not source_signals.eligible or not target_signals.eligible:
        return 0.0, []

    methods = source_signals.methods & target_signals.methods
    components = source_signals.components & target_signals.components
    lexical = source_signals.tokens & target_signals.tokens
    named = source_signals.named_terms & target_signals.tokens
    output_comparison = bool(
        source_signals.components & _OUTPUT_COMPONENTS
        and target_signals.components & _OUTPUT_COMPONENTS
        and not components & _OUTPUT_COMPONENTS
    )
    if not methods and not components and not output_comparison and len(lexical) < 3:
        return 0.0, []

    cues = set(methods) | set(components)
    score = 3.0 * min(3, len(methods)) + 4.0 * min(3, len(components))
    if output_comparison:
        score += 3.0
        cues.add("output/representation")
    if len(lexical) >= 2:
        score += 0.75 * min(4, len(lexical))
        if not cues:
            cues.update(sorted(lexical)[:4])
    if named:
        score += 1.5
        cues.update(named)
    # Methods are preferred, but experiments can contain essential output details.
    score += sum(chunk.section in METHOD_SECTIONS for chunk in (source_chunk, target_chunk))
    return round(score, 3), sorted(cues)


_RankedChunk = tuple[int, PaperEvidenceChunk, float, list[str]]


@dataclass
class _PaperSelection:
    ranked: list[_RankedChunk]
    char_budget: int
    max_chunks: int
    chosen: dict[int, _RankedChunk] = field(default_factory=dict)
    used_chars: int = 0

    def can_include(self, candidate: _RankedChunk) -> bool:
        index, chunk, _, _ = candidate
        return index in self.chosen or (
            len(self.chosen) < self.max_chunks
            and self.used_chars + len(chunk.text) <= self.char_budget
        )

    def include(self, candidate: _RankedChunk) -> None:
        if candidate[0] not in self.chosen:
            self.chosen[candidate[0]] = candidate
            self.used_chars += len(candidate[1].text)

    def reserve_coverage(self) -> None:
        for group in ("overview", "method", "evidence"):
            for candidate in self.ranked:
                if _coverage_group(candidate[1]) == group and self.can_include(candidate):
                    self.include(candidate)
                    break

    def mark_comparison(self, candidate: _RankedChunk, score: float, cues: list[str]) -> None:
        self.include(candidate)
        index, chunk, base_score, reasons = candidate
        self.chosen[index] = (
            index, chunk, round(base_score + score, 3),
            reasons + ["Cross-paper comparison cue: " + ", ".join(cues)],
        )

    def fill(self) -> None:
        for candidate in self.ranked:
            if self.can_include(candidate):
                self.include(candidate)

    def ordered(self) -> list[_RankedChunk]:
        return [self.chosen[index] for index in sorted(self.chosen)]


def _rank_for_paper(
    package: PaperEvidencePackage,
    role: str,
    other: PaperEvidencePackage,
    role_char_budget: int,
    max_chunks: int,
) -> _PaperSelection:
    ranked = []
    for index, chunk in enumerate(package.chunks):
        score, reasons = _score_chunk(
            chunk, role, package if role == "source" else other,
            other if role == "source" else package,
        )
        ranked.append((index, chunk, score, reasons))
    ranked.sort(key=lambda item: (-item[2], item[0]))
    return _PaperSelection(ranked, role_char_budget, max_chunks)


def _reserve_cross_paper_pairs(
    source: _PaperSelection, target: _PaperSelection,
) -> list[tuple[int, int, float, list[str]]]:
    candidates = []
    # Precompute signals once per chunk rather than tokenize inside every pair.
    source_signals = {item[0]: _comparison_signals(item[1]) for item in source.ranked}
    target_signals = {item[0]: _comparison_signals(item[1]) for item in target.ranked}
    source_candidates = [
        item for item in source.ranked
        if source_signals[item[0]].eligible and source.can_include(item)
    ]
    target_candidates = [
        item for item in target.ranked
        if target_signals[item[0]].eligible and target.can_include(item)
    ]
    for source_item in source_candidates:
        for target_item in target_candidates:
            score, cues = _score_cross_paper_pair(
                source_item[1], target_item[1],
                source_signals=source_signals[source_item[0]],
                target_signals=target_signals[target_item[0]],
            )
            if score >= _MIN_CROSS_PAPER_SCORE:
                candidates.append((score, source_item, target_item, cues))
    candidates.sort(key=lambda item: (-item[0], item[1][0], item[2][0]))

    pairs = []
    used_source: set[int] = set()
    used_target: set[int] = set()
    signatures: list[set[str]] = []
    for score, source_item, target_item, cues in candidates:
        if len(pairs) >= _MAX_CROSS_PAPER_PAIRS:
            break
        if source_item[0] in used_source or target_item[0] in used_target:
            continue
        signature = set(cues)
        if any(len(signature & prior) / len(signature | prior) >= 0.75 for prior in signatures):
            continue
        if not source.can_include(source_item) or not target.can_include(target_item):
            continue  # Reserve both sides together, without evicting coverage chunks.
        source.mark_comparison(source_item, score, cues)
        target.mark_comparison(target_item, score, cues)
        pairs.append((source_item[0], target_item[0], score, cues))
        used_source.add(source_item[0])
        used_target.add(target_item[0])
        signatures.append(signature)
    return pairs


def build_lineage_llm_context(
    evidence_input: LineageEvidenceInput,
    *,
    policy: EvidenceSelectionPolicy | None = None,
) -> LineageLLMContext:
    """Select balanced, deterministic evidence without making relation claims."""

    policy = policy or EvidenceSelectionPolicy()
    source = evidence_input.source
    target = evidence_input.target
    # Reserve separate, equal ceilings so a large source paper cannot consume
    # the target paper's allowance. Odd total budgets leave at most one char unused.
    role_char_budget = min(policy.max_chars_per_paper, policy.max_total_chars // 2)
    source_selection = _rank_for_paper(
        source, "source", target, role_char_budget, policy.max_chunks_per_paper
    )
    target_selection = _rank_for_paper(
        target, "target", source, role_char_budget, policy.max_chunks_per_paper
    )
    source_selection.reserve_coverage()
    target_selection.reserve_coverage()
    pairs = _reserve_cross_paper_pairs(source_selection, target_selection)
    source_selection.fill()
    target_selection.fill()
    source_selected = source_selection.ordered()
    target_selected = target_selection.ordered()

    warnings = []
    for role, package, selection in (
        ("source", source, source_selection), ("target", target, target_selection),
    ):
        skipped = len(package.chunks) - len(selection.chosen)
        if skipped:
            warnings.append(
                f"{role}: skipped {skipped} chunk(s) because of the per-paper/total "
                "character budget or per-paper chunk limit."
            )
    if not source.chunks:
        warnings.append("source: no evidence chunks are available.")
    if not target.chunks:
        warnings.append("target: no evidence chunks are available.")

    selected: list[SelectedEvidenceChunk] = []
    evidence_ids: dict[tuple[str, int], str] = {}
    for role, items in (("source", source_selected), ("target", target_selected)):
        prefix = "S" if role == "source" else "T"
        for ordinal, (index, chunk, score, reasons) in enumerate(items, start=1):
            evidence_ids[(role, index)] = f"{prefix}{ordinal:03d}"
            selected.append(
                SelectedEvidenceChunk(
                    evidence_id=f"{prefix}{ordinal:03d}",
                    paper_role=role,
                    paper_id=chunk.paper_id,
                    paper_title=chunk.paper_title,
                    text=chunk.text,
                    section=chunk.section,
                    page=chunk.page,
                    location=chunk.location,
                    source_kind=chunk.source_kind,
                    chunk_type=chunk.chunk_type,
                    selection_score=score,
                    selection_reasons=reasons,
                )
            )

    return LineageLLMContext(
        source_paper_id=source.paper_id,
        source_paper_title=source.paper_title,
        target_paper_id=target.paper_id,
        target_paper_title=target.paper_title,
        selected_evidence=selected,
        warnings=warnings,
        selection_policy={
            **policy.to_dict(),
            "effective_chars_per_paper": role_char_budget,
            "section_priorities": dict(SECTION_PRIORITIES),
            "lexical_scoring": "lightweight lexical cues only; not a relation judgment",
            "cross_paper_pair_limit": _MAX_CROSS_PAPER_PAIRS,
            "cross_paper_min_score": _MIN_CROSS_PAPER_SCORE,
            "cross_paper_pairs": [
                {
                    "source_evidence_id": evidence_ids[("source", source_index)],
                    "target_evidence_id": evidence_ids[("target", target_index)],
                    "score": score,
                    "cues": cues,
                }
                for source_index, target_index, score, cues in pairs
            ],
        },
        total_characters=sum(len(item.text) for item in selected),
    )


def validate_evidence_ids(
    context: LineageLLMContext,
    evidence_ids: Sequence[str],
) -> list[SelectedEvidenceChunk]:
    """Resolve model-returned IDs against the exact context that was supplied."""

    return context.validate_evidence_ids(evidence_ids)
