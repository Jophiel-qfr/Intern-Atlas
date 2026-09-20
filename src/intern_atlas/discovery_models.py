"""Provider-neutral paper data used by the discovery workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiscoveredPaper:
    """Small common representation shared by discovery providers."""

    provider: str
    provider_id: str
    title: str = ""
    abstract: str = ""
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    external_ids: dict[str, str] = field(default_factory=dict)
    citation_count: int | None = None
    url: str = ""
    publication_type: str = ""
    publication_types: list[str] = field(default_factory=list)
    fields_of_study: list[str] = field(default_factory=list)
    s2_fields_of_study: list[dict[str, Any]] = field(default_factory=list)
    influential_citation_count: int | None = None
    match_score: float | None = None
    referenced_work_ids: list[str] = field(default_factory=list)

    @property
    def paper_id(self) -> str:
        """Backward-compatible alias used by the existing discovery UI/API."""

        return self.provider_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "provider_id": self.provider_id,
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "authors": self.authors,
            "venue": self.venue,
            "external_ids": self.external_ids,
            "citation_count": self.citation_count,
            "url": self.url,
            "publication_type": self.publication_type,
            "publication_types": self.publication_types,
            "fields_of_study": self.fields_of_study,
            "s2_fields_of_study": self.s2_fields_of_study,
            "influential_citation_count": self.influential_citation_count,
            "match_score": self.match_score,
        }
