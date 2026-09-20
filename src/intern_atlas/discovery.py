"""Semantic Scholar discovery and transparent candidate ranking.

The ranking here is only a reading-priority heuristic.  It must not be read as
evidence that one paper inherits, improves, or replaces another paper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .discovery_models import DiscoveredPaper
from .integrations.openalex import OpenAlexClient, OpenAlexError, identify_openalex_query
from .integrations.semantic_scholar import (
    SemanticScholarClient,
    SemanticScholarError,
    SemanticScholarPaper,
)


HAR_DOMAIN_TOKENS = {
    "wearable",
    "accelerometer",
    "gyroscope",
    "imu",
    "inertial",
    "multimodal",
}
HAR_DOMAIN_PHRASES = {
    "human activity recognition",
    "activity recognition",
    "wearable sensor",
}
STRONG_METHOD_TOKENS = {
    "cnn",
    "lstm",
    "deepconvlstm",
    "transformer",
    "attention",
    "gru",
    "convolutional",
    "recurrent",
    "self-supervised",
    "contrastive",
}
STRONG_METHOD_PHRASES = {
    "sensor fusion",
    "long short term memory",
    "contrastive learning",
}
HAR_CONTEXT_PHRASES = {"representation learning"}
HAR_TOKEN_KEYWORDS = HAR_DOMAIN_TOKENS | STRONG_METHOD_TOKENS
HAR_PHRASES = HAR_DOMAIN_PHRASES | STRONG_METHOD_PHRASES | HAR_CONTEXT_PHRASES
HAR_KEYWORDS = HAR_TOKEN_KEYWORDS | HAR_PHRASES
GENERIC_OVERLAP_TERMS = {
    "deep",
    "neural",
    "network",
    "networks",
    "learning",
    "model",
    "models",
    "data",
    "using",
    "based",
}
REVIEW_TITLE_PHRASES = {
    "review",
    "survey",
    "tutorial",
    "state of the art",
    "dataset",
    "benchmark",
}
METHOD_SIGNAL_FAMILIES = {
    "convolution": {"cnn", "convolutional"},
    "sequence": {"lstm", "long short term memory", "recurrent", "gru"},
    "attention": {"transformer", "attention"},
    "fusion": {"sensor fusion"},
    "self_supervised": {"self-supervised", "contrastive", "contrastive learning"},
    "deepconvlstm": {"deepconvlstm"},
}
# Only these exact Semantic Scholar intent labels are strong enough to affect
# candidate classification.  Generic labels such as ``uses`` are evidence-poor
# and must remain neutral.
METHOD_INTENT_TERMS = {"methodology"}
BACKGROUND_INTENT_TERMS = {"background"}
TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{2,}", re.IGNORECASE)
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
ARXIV_RE = re.compile(r"^(?:arxiv\s*:\s*)?(\d{4}\.\d{4,5}(?:v\d+)?)$", re.IGNORECASE)
PAPER_ID_RE = re.compile(r"^(?:s2\s*:\s*|paperid\s*:\s*)?([0-9a-f]{32,})$", re.IGNORECASE)


def semantic_paper_to_discovered(paper: SemanticScholarPaper) -> DiscoveredPaper:
    return DiscoveredPaper(
        provider="semantic_scholar",
        provider_id=paper.paper_id,
        title=paper.title,
        abstract=paper.abstract,
        year=paper.year,
        authors=paper.authors,
        venue=paper.venue,
        external_ids=paper.external_ids,
        citation_count=paper.citation_count,
        url=paper.url,
        publication_types=paper.publication_types,
        fields_of_study=paper.fields_of_study,
        s2_fields_of_study=paper.s2_fields_of_study,
        influential_citation_count=paper.influential_citation_count,
        match_score=paper.match_score,
    )


class SemanticScholarProvider:
    name = "semantic_scholar"

    def __init__(self, client: SemanticScholarClient | None = None) -> None:
        self.client = client or SemanticScholarClient()

    def close(self) -> None:
        self.client.close()

    def identify_query(self, query: str) -> tuple[str, str]:
        return identify_query(query)

    def get_paper(self, identifier: str) -> tuple[DiscoveredPaper, bool]:
        paper, cached = self.client.get_paper(identifier)
        return semantic_paper_to_discovered(paper), cached

    def search_papers(self, query: str, *, limit: int) -> tuple[list[DiscoveredPaper], bool]:
        papers, cached = self.client.search_papers(query, limit=limit)
        return [semantic_paper_to_discovered(paper) for paper in papers], cached

    def references(self, paper: DiscoveredPaper, *, limit: int) -> tuple[list[dict[str, Any]], bool]:
        rows, cached = self.client.references(paper.provider_id, limit=limit)
        return self._convert_relation_rows(rows, direction="reference"), cached

    def citations(self, paper: DiscoveredPaper, *, limit: int) -> tuple[list[dict[str, Any]], bool]:
        rows, cached = self.client.citations(paper.provider_id, limit=limit)
        return self._convert_relation_rows(rows, direction="citation"), cached

    @staticmethod
    def _convert_relation_rows(rows: list[dict[str, Any]], *, direction: str) -> list[dict[str, Any]]:
        paper_key = "citedPaper" if direction == "reference" else "citingPaper"
        converted: list[dict[str, Any]] = []
        for row in rows:
            paper_data = row.get(paper_key)
            if not isinstance(paper_data, dict):
                continue
            paper = SemanticScholarPaper.from_api(paper_data)
            if not paper.paper_id:
                continue
            normalized = dict(row)
            normalized["_paper"] = semantic_paper_to_discovered(paper)
            converted.append(normalized)
        return converted


class OpenAlexProvider:
    name = "openalex"

    def __init__(self, client: OpenAlexClient | None = None) -> None:
        self.client = client or OpenAlexClient()

    def close(self) -> None:
        self.client.close()

    def identify_query(self, query: str) -> tuple[str, str]:
        return identify_openalex_query(query)

    def get_paper(self, identifier: str) -> tuple[DiscoveredPaper, bool]:
        return self.client.get_work(identifier)

    def search_papers(self, query: str, *, limit: int) -> tuple[list[DiscoveredPaper], bool]:
        return self.client.search_works(query, limit=limit)

    def references(self, paper: DiscoveredPaper, *, limit: int) -> tuple[list[dict[str, Any]], bool]:
        return self.client.references(
            paper.provider_id,
            referenced_work_ids=paper.referenced_work_ids,
            limit=limit,
        )

    def citations(self, paper: DiscoveredPaper, *, limit: int) -> tuple[list[dict[str, Any]], bool]:
        return self.client.citations(paper.provider_id, limit=limit)


@dataclass
class CitationCandidate:
    paper: DiscoveredPaper
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
        normalized_paper = row.get("_paper")
        if isinstance(normalized_paper, DiscoveredPaper):
            paper = normalized_paper
            paper_data = None
        else:
            paper_data = row.get(paper_key)
            paper = None
        if not isinstance(paper_data, dict):
            if paper is None:
                return None
        else:
            paper = semantic_paper_to_discovered(SemanticScholarPaper.from_api(paper_data))
        if not paper.provider_id:
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
    paper: DiscoveredPaper | None
    matches: list[DiscoveredPaper] = field(default_factory=list)
    cached: bool = False
    selection_required: bool = False
    status: str = "selected"
    provider: str = "semantic_scholar"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "input_type": self.input_type,
            "paper": self.paper.to_dict() if self.paper else None,
            "matches": [paper.to_dict() for paper in self.matches],
            "cached": self.cached,
            "selection_required": self.selection_required,
            "status": self.status,
            "provider": self.provider,
        }


@dataclass
class DiscoveryResult:
    query: str
    target: DiscoveredPaper
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
            "provider": self.target.provider,
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

    def __init__(
        self,
        client: SemanticScholarClient | None = None,
        *,
        openalex_client: OpenAlexClient | None = None,
    ) -> None:
        if client is not None:
            # Preserve the existing injected-client behavior used by tests and
            # callers while the normal application defaults to OpenAlex.
            self.providers: dict[str, Any] = {"semantic_scholar": SemanticScholarProvider(client)}
            self.default_provider = "semantic_scholar"
        else:
            self.providers = {
                "openalex": OpenAlexProvider(openalex_client),
                "semantic_scholar": SemanticScholarProvider(),
            }
            self.default_provider = "openalex"

    def close(self) -> None:
        closed: set[int] = set()
        for provider in self.providers.values():
            identity = id(provider.client)
            if identity in closed:
                continue
            provider.close()
            closed.add(identity)

    def resolve(self, query: str, *, provider: str | None = None) -> ResolvedPaper:
        original = query.strip()
        if not original:
            raise SemanticScholarError("请输入论文标题或论文 ID。", code="empty_query", status_code=422)
        adapter = self._get_provider(provider)
        input_type, identifier = adapter.identify_query(original)
        if input_type != "title":
            paper, cached = adapter.get_paper(identifier)
            ensure_paper_url(paper)
            return ResolvedPaper(
                original,
                input_type,
                paper,
                [paper],
                cached,
                provider=adapter.name,
            )

        matches, cached = adapter.search_papers(original, limit=5)
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
            provider=adapter.name,
        )

    def discover_lineage(
        self,
        query: str,
        *,
        max_references: int = 30,
        max_citations: int = 30,
        paper_id: str | None = None,
        provider: str | None = None,
    ) -> DiscoveryResult:
        adapter = self._get_provider(provider)
        resolution = self.resolve(query, provider=adapter.name)
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
                "没有找到匹配论文，请尝试更完整的标题或论文 ID。",
                code="no_match",
                status_code=404,
            )
        target = resolution.paper
        reference_rows, references_cached = adapter.references(target, limit=max_references)
        citation_rows, citations_cached = adapter.citations(target, limit=max_citations)
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

    def _get_provider(self, provider: str | None) -> Any:
        name = (provider or self.default_provider).strip().lower()
        selected = self.providers.get(name)
        if selected is None:
            raise SemanticScholarError(
                f"不支持的数据来源：{provider}。可选值为 openalex 或 semantic_scholar。",
                code="invalid_provider",
                status_code=422,
            )
        return selected


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
    target: DiscoveredPaper,
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
            method_signal_family_count=evidence["method_signal_family_count"],
            has_har_domain=evidence["has_har_domain"],
            is_review_like=evidence["is_review_like"],
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


def candidate_relevance(target: DiscoveredPaper, candidate: CitationCandidate) -> tuple[float, list[str]]:
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
    meaningful_title_overlap = title_overlap - GENERIC_OVERLAP_TERMS
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
    if meaningful_title_overlap:
        score += min(3, len(meaningful_title_overlap)) * 0.8
        if len(meaningful_title_overlap) >= 2:
            reasons.append("High title overlap")
    if shared_har:
        score += min(4, len(shared_har)) * 0.4
        reasons.append(f"Shares HAR keyword: {', '.join(shared_har[:3])}")
    elif text_overlap:
        score += min(8, len(text_overlap)) * 0.2
        reasons.append("Shares abstract terms")
    if evidence["method_signal_family_count"]:
        score += min(3, evidence["method_signal_family_count"]) * 0.7
        reasons.append(
            "Strong method signals: "
            + ", ".join(evidence["method_signals"][:4])
        )
    if evidence["is_review_like"]:
        score -= 2.0
        reasons.extend(evidence["review_reasons"])
    if not reasons:
        reasons.append("Limited citation metadata overlap")
    return score, reasons


def candidate_level(
    score: float,
    *,
    has_methodology_intent: bool = False,
    has_background_intent: bool = False,
    has_positive_evidence: bool = False,
    method_signal_family_count: int = 0,
    has_har_domain: bool = False,
    is_review_like: bool = False,
) -> str:
    if is_review_like:
        return "uncertain"
    if has_methodology_intent:
        return "method_candidate"
    if method_signal_family_count >= 2 or (method_signal_family_count >= 1 and has_har_domain):
        return "method_candidate"
    if has_background_intent and not has_positive_evidence:
        return "background_candidate"
    return "uncertain"


def title_match_score(query: str, paper: DiscoveredPaper) -> float:
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


def candidate_evidence(target: DiscoveredPaper, candidate: CitationCandidate) -> dict[str, Any]:
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
    shared_domain_tokens = sorted((target_text_tokens & candidate_text_tokens) & HAR_DOMAIN_TOKENS)
    shared_method_tokens = sorted((target_text_tokens & candidate_text_tokens) & STRONG_METHOD_TOKENS)
    shared_domain_phrases = sorted(
        phrase
        for phrase in HAR_DOMAIN_PHRASES
        if phrase in target_normalized and phrase in candidate_normalized
    )
    shared_method_phrases = sorted(
        phrase
        for phrase in STRONG_METHOD_PHRASES
        if phrase in target_normalized and phrase in candidate_normalized
    )
    shared_context_phrases = sorted(
        phrase
        for phrase in HAR_CONTEXT_PHRASES
        if phrase in target_normalized and phrase in candidate_normalized
    )
    shared_har = (
        shared_domain_phrases
        + shared_method_phrases
        + shared_context_phrases
        + shared_domain_tokens
        + shared_method_tokens
    )
    method_signals = method_signals_in_text(candidate_text)
    method_signal_families = {
        family
        for family, signals in METHOD_SIGNAL_FAMILIES.items()
        if signals & set(method_signals)
    }
    review_reasons = review_reasons_for_title(candidate.paper.title)
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
        "has_har_domain": bool(shared_domain_tokens or shared_domain_phrases),
        "method_signals": method_signals,
        "method_signal_family_count": len(method_signal_families),
        "is_review_like": bool(review_reasons),
        "review_reasons": review_reasons,
        "methodology_intent": methodology_intent,
        "background_intent": background_intent,
        "positive_evidence": positive_evidence,
    }


def method_signals_in_text(text: str) -> list[str]:
    tokens = token_set(text)
    normalized = normalized_text(text)
    signals = {
        signal
        for signal in STRONG_METHOD_TOKENS
        if signal in tokens
    }
    signals.update(
        phrase
        for phrase in STRONG_METHOD_PHRASES
        if phrase in normalized
    )
    return sorted(signals)


def review_reasons_for_title(title: str) -> list[str]:
    normalized = normalized_text(title)
    reasons: list[str] = []
    if any(
        phrase in normalized
        for phrase in {"review", "survey", "tutorial", "state of the art"}
    ):
        reasons.append("Review/survey paper")
    if any(phrase in normalized for phrase in {"dataset", "benchmark"}):
        reasons.append("Dataset/benchmark paper")
    return reasons


def as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def ensure_paper_url(paper: DiscoveredPaper) -> None:
    if not paper.url and paper.paper_id:
        if paper.provider == "openalex":
            paper.url = f"https://openalex.org/{paper.paper_id}"
        else:
            paper.url = f"https://www.semanticscholar.org/paper/{paper.paper_id}"
