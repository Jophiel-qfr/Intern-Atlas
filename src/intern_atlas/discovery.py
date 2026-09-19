"""Semantic Scholar discovery and transparent candidate ranking.

The ranking here is only a reading-priority heuristic.  It must not be read as
evidence that one paper inherits, improves, or replaces another paper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .integrations.semantic_scholar import (
    SemanticScholarClient,
    SemanticScholarError,
    SemanticScholarPaper,
)


HAR_TOKEN_KEYWORDS = {
    "wearable",
    "accelerometer",
    "gyroscope",
    "imu",
    "inertial",
    "multimodal",
    "cnn",
    "lstm",
    "deepconvlstm",
    "transformer",
    "attention",
    "self-supervised",
}
HAR_PHRASES = {
    "human activity recognition",
    "activity recognition",
    "wearable sensor",
    "sensor fusion",
    "contrastive learning",
    "representation learning",
}
HAR_KEYWORDS = HAR_TOKEN_KEYWORDS | HAR_PHRASES
# Only these exact Semantic Scholar intent labels are strong enough to affect
# candidate classification.  Generic labels such as ``uses`` are evidence-poor
# and must remain neutral.
METHOD_INTENT_TERMS = {"methodology"}
BACKGROUND_INTENT_TERMS = {"background"}
TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{2,}", re.IGNORECASE)
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
ARXIV_RE = re.compile(r"^(?:arxiv\s*:\s*)?(\d{4}\.\d{4,5}(?:v\d+)?)$", re.IGNORECASE)
PAPER_ID_RE = re.compile(r"^(?:s2\s*:\s*|paperid\s*:\s*)?([0-9a-f]{32,})$", re.IGNORECASE)


@dataclass
class CitationCandidate:
    paper: SemanticScholarPaper
    direction: str
    contexts: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    is_influential: bool = False
    relevance_score: float = 0.0
    candidate_level: str = "uncertain"
    reasons: list[str] = field(default_factory=list)

    @classmethod
    def from_relation(cls, row: dict[str, Any], *, direction: str) -> "CitationCandidate | None":
        paper_key = str(row.get("_paper_key") or ("citedPaper" if direction == "reference" else "citingPaper"))
        paper_data = row.get(paper_key)
        if not isinstance(paper_data, dict):
            return None
        paper = SemanticScholarPaper.from_api(paper_data)
        if not paper.paper_id:
            return None
        return cls(
            paper=paper,
            direction=direction,
            contexts=as_string_list(row.get("contexts")),
            intents=as_string_list(row.get("intents")),
            is_influential=bool(row.get("isInfluential")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.paper.to_dict()
        data.update(
            {
                "direction": self.direction,
                "contexts": self.contexts,
                "intents": self.intents,
                "is_influential": self.is_influential,
                "relevance_score": self.relevance_score,
                "candidate_level": self.candidate_level,
                "reasons": self.reasons,
            }
        )
        return data


@dataclass
class ResolvedPaper:
    query: str
    input_type: str
    paper: SemanticScholarPaper | None
    matches: list[SemanticScholarPaper] = field(default_factory=list)
    cached: bool = False
    selection_required: bool = False
    status: str = "selected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "input_type": self.input_type,
            "paper": self.paper.to_dict() if self.paper else None,
            "matches": [paper.to_dict() for paper in self.matches],
            "cached": self.cached,
            "selection_required": self.selection_required,
            "status": self.status,
        }


@dataclass
class DiscoveryResult:
    query: str
    target: SemanticScholarPaper
    references: list[CitationCandidate]
    citations: list[CitationCandidate]
    resolution: ResolvedPaper
    cached: bool = False
    cache_status: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        method_candidates = sum(
            candidate.candidate_level == "method_candidate"
            for candidate in [*self.references, *self.citations]
        )
        return {
            "query": self.query,
            "target": self.target.to_dict(),
            "references": [candidate.to_dict() for candidate in self.references],
            "citations": [candidate.to_dict() for candidate in self.citations],
            "resolution": self.resolution.to_dict(),
            "cached": self.cached,
            "cache_status": self.cache_status,
            "summary": {
                "reference_count": len(self.references),
                "citation_count": len(self.citations),
                "method_candidate_count": method_candidates,
                "note": "候选排序基于 citation metadata，仅用于安排后续阅读，不代表方法继承关系。",
            },
        }


class DiscoveryService:
    """Application service coordinating target resolution and one-hop discovery."""

    def __init__(self, client: SemanticScholarClient | None = None) -> None:
        self.client = client or SemanticScholarClient()

    def close(self) -> None:
        self.client.close()

    def resolve(self, query: str) -> ResolvedPaper:
        original = query.strip()
        if not original:
            raise SemanticScholarError("请输入论文标题、DOI、arXiv ID 或 Semantic Scholar paperId。", code="empty_query", status_code=422)
        input_type, identifier = identify_query(original)
        if input_type != "title":
            paper, cached = self.client.get_paper(identifier)
            ensure_paper_url(paper)
            return ResolvedPaper(original, input_type, paper, [paper], cached)

        matches, cached = self.client.search_papers(original, limit=5)
        ranked = sorted(
            (paper for paper in matches if paper.paper_id),
            key=lambda paper: title_match_score(original, paper),
            reverse=True,
        )
        exact_matches = [paper for paper in ranked if normalize_title(original) == normalize_title(paper.title)]
        selected = exact_matches[0] if len(exact_matches) == 1 else None
        for paper in ranked:
            ensure_paper_url(paper)
        if selected is not None:
            status = "selected"
        elif ranked:
            status = "selection_required"
        else:
            status = "not_found"
        return ResolvedPaper(
            original,
            "title",
            selected,
            ranked,
            cached,
            selection_required=selected is None and bool(ranked),
            status=status,
        )

    def discover_lineage(
        self,
        query: str,
        *,
        max_references: int = 30,
        max_citations: int = 30,
        paper_id: str | None = None,
    ) -> DiscoveryResult:
        resolution = self.resolve(query)
        if paper_id:
            selected = next(
                (paper for paper in resolution.matches if paper.paper_id == paper_id.strip()),
                None,
            )
            if selected is None:
                raise SemanticScholarError(
                    "所选论文不在当前搜索结果中，请先重新搜索。",
                    code="invalid_selection",
                    status_code=422,
                )
            resolution.paper = selected
            resolution.selection_required = False
            resolution.status = "selected"
        if resolution.paper is None:
            if resolution.selection_required:
                raise SemanticScholarError(
                    "搜索返回了多个候选结果，请先选择一篇论文。",
                    code="selection_required",
                    status_code=409,
                )
            raise SemanticScholarError(
                "Semantic Scholar 没有找到匹配论文，请尝试更完整的标题或 DOI。",
                code="no_match",
                status_code=404,
            )
        target = resolution.paper
        reference_rows, references_cached = self.client.references(target.paper_id, limit=max_references)
        citation_rows, citations_cached = self.client.citations(target.paper_id, limit=max_citations)
        references = rank_candidates(target, reference_rows, direction="reference")
        citations = rank_candidates(target, citation_rows, direction="citation")
        return DiscoveryResult(
            query=query.strip(),
            target=target,
            references=references,
            citations=citations,
            resolution=resolution,
            cached=all((resolution.cached, references_cached, citations_cached)),
            cache_status={
                "resolution": resolution.cached,
                "references": references_cached,
                "citations": citations_cached,
            },
        )


def identify_query(query: str) -> tuple[str, str]:
    """Classify user input and return the Academic Graph identifier when known."""

    value = query.strip()
    doi_value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    doi_value = re.sub(r"^doi:\s*", "", doi_value, flags=re.IGNORECASE)
    if DOI_RE.fullmatch(doi_value):
        return "doi", f"DOI:{doi_value}"

    arxiv_match = ARXIV_RE.fullmatch(value)
    if arxiv_match:
        return "arxiv", f"ARXIV:{arxiv_match.group(1)}"

    paper_match = PAPER_ID_RE.fullmatch(value)
    if paper_match:
        return "paper_id", paper_match.group(1)
    return "title", value


def rank_candidates(
    target: SemanticScholarPaper,
    rows: list[dict[str, Any]],
    *,
    direction: str,
) -> list[CitationCandidate]:
    candidates: list[CitationCandidate] = []
    for row in rows:
        candidate = CitationCandidate.from_relation(row, direction=direction)
        if candidate is None:
            continue
        score, reasons = candidate_relevance(target, candidate)
        candidate.relevance_score = round(score, 2)
        candidate.reasons = reasons
        evidence = candidate_evidence(target, candidate)
        candidate.candidate_level = candidate_level(
            score,
            has_methodology_intent=evidence["methodology_intent"],
            has_background_intent=evidence["background_intent"],
            has_positive_evidence=evidence["positive_evidence"],
        )
        candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda item: (
            item.relevance_score,
            item.is_influential,
            item.paper.citation_count or 0,
        ),
        reverse=True,
    )


def candidate_relevance(target: SemanticScholarPaper, candidate: CitationCandidate) -> tuple[float, list[str]]:
    target_title_tokens = token_set(target.title)
    candidate_title_tokens = token_set(candidate.paper.title)
    target_text_tokens = token_set(f"{target.title} {target.abstract}")
    candidate_text_tokens = token_set(f"{candidate.paper.title} {candidate.paper.abstract}")
    title_overlap = target_title_tokens & candidate_title_tokens
    text_overlap = target_text_tokens & candidate_text_tokens
    evidence = candidate_evidence(target, candidate)
    shared_har = evidence["shared_har"]
    normalized_intents = {intent.lower().replace("_", " ") for intent in candidate.intents}

    score = 0.0
    reasons: list[str] = []
    method_intents = sorted(normalized_intents & METHOD_INTENT_TERMS)
    background_intents = sorted(normalized_intents & BACKGROUND_INTENT_TERMS)
    if method_intents:
        score += 3.0
        reasons.append("Semantic Scholar intent includes methodology")
    if background_intents and not method_intents:
        score -= 1.0
        reasons.append("Citation intent suggests background")
    if candidate.is_influential:
        score += 2.0
        reasons.append("Influential citation")
    if title_overlap:
        score += min(3, len(title_overlap)) * 0.8
        if len(title_overlap) >= 2:
            reasons.append("High title overlap")
    if shared_har:
        score += min(4, len(shared_har)) * 0.7
        reasons.append(f"Shares HAR keyword: {', '.join(shared_har[:3])}")
    elif text_overlap:
        score += min(8, len(text_overlap)) * 0.2
        reasons.append("Shares abstract terms")
    if not reasons:
        reasons.append("Limited citation metadata overlap")
    return score, reasons


def candidate_level(
    score: float,
    *,
    has_methodology_intent: bool = False,
    has_background_intent: bool = False,
    has_positive_evidence: bool = False,
) -> str:
    if has_methodology_intent or (has_positive_evidence and score >= 4.0):
        return "method_candidate"
    if has_background_intent and not has_positive_evidence:
        return "background_candidate"
    return "uncertain"


def title_match_score(query: str, paper: SemanticScholarPaper) -> float:
    query_tokens = token_set(query)
    title_tokens = token_set(paper.title)
    overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
    api_score = paper.match_score if paper.match_score is not None else 0.0
    exact_bonus = 1.0 if normalize_title(query) == normalize_title(paper.title) else 0.0
    return exact_bonus * 10.0 + overlap * 5.0 + api_score


def token_set(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "") if token.lower() not in {"the", "and", "for", "with", "using", "based"}}


def normalize_title(text: str) -> str:
    normalized = re.sub(r"[\W_]+", " ", (text or "").lower(), flags=re.UNICODE)
    return " ".join(normalized.split())


def normalized_text(text: str) -> str:
    return normalize_title(text)


def candidate_evidence(target: SemanticScholarPaper, candidate: CitationCandidate) -> dict[str, Any]:
    target_title_tokens = token_set(target.title)
    candidate_title_tokens = token_set(candidate.paper.title)
    target_text = f"{target.title} {target.abstract}"
    candidate_text = f"{candidate.paper.title} {candidate.paper.abstract}"
    target_text_tokens = token_set(target_text)
    candidate_text_tokens = token_set(candidate_text)
    title_overlap = target_title_tokens & candidate_title_tokens
    text_overlap = target_text_tokens & candidate_text_tokens
    target_normalized = normalized_text(target_text)
    candidate_normalized = normalized_text(candidate_text)
    shared_tokens = sorted((target_text_tokens & candidate_text_tokens) & HAR_TOKEN_KEYWORDS)
    shared_phrases = sorted(
        phrase
        for phrase in HAR_PHRASES
        if phrase in target_normalized and phrase in candidate_normalized
    )
    shared_har = shared_phrases + shared_tokens
    normalized_intents = {intent.lower().replace("_", " ").strip() for intent in candidate.intents}
    methodology_intent = bool(normalized_intents & METHOD_INTENT_TERMS)
    background_intent = bool(normalized_intents & BACKGROUND_INTENT_TERMS)
    positive_evidence = bool(
        methodology_intent
        or candidate.is_influential
        or title_overlap
        or text_overlap
        or shared_har
    )
    return {
        "title_overlap": title_overlap,
        "text_overlap": text_overlap,
        "shared_har": shared_har,
        "methodology_intent": methodology_intent,
        "background_intent": background_intent,
        "positive_evidence": positive_evidence,
    }


def as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def ensure_paper_url(paper: SemanticScholarPaper) -> None:
    if not paper.url and paper.paper_id:
        paper.url = f"https://www.semanticscholar.org/paper/{paper.paper_id}"
