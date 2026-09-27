"""Raw, locatable paper text prepared for later analysis.

``PaperEvidenceChunk`` stores text that has not yet been selected to support a
claim. ``analysis.EvidenceItem`` is the later-stage record for text selected to
support a specific conclusion. This module does not infer paper relationships.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Mapping

from .discovery_models import DiscoveredPaper


MAX_CHUNK_CHARS = 4000
PAPER_SECTIONS = frozenset(
    {
        "abstract",
        "introduction",
        "related_work",
        "background",
        "methods",
        "experiments",
        "results",
        "discussion",
        "conclusion",
        "unknown",
    }
)
SOURCE_KINDS = frozenset({"pdf", "abstract", "citation_context"})
CHUNK_TYPES = frozenset({"abstract", "paragraph", "citation_context"})


class EvidenceExtractionError(RuntimeError):
    """Raised when a local PDF cannot provide trustworthy extracted text."""


def _validate_choice(name: str, value: str, allowed: frozenset[str]) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(allowed))}")
    return value


def _split_long_text(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split oversized text near whitespace without shortening its content."""

    remaining = text.strip()
    chunks: list[str] = []
    while len(remaining) > limit:
        split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at < limit // 2:
            forward_space = remaining.find(" ", limit)
            split_at = forward_space if 0 <= forward_space <= limit else limit
        if split_at < 0:
            split_at = limit
        piece = remaining[:split_at].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


@dataclass
class PaperEvidenceChunk:
    """An original text fragment with its paper, source, and location."""

    paper_id: str
    paper_title: str
    text: str
    section: str = "unknown"
    page: int | None = None
    location: str | None = None
    source_kind: str = "abstract"
    chunk_type: str = "abstract"
    metadata: dict[str, Any] = field(default_factory=dict)

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "paper_id",
        "paper_title",
        "text",
        "section",
        "page",
        "location",
        "source_kind",
        "chunk_type",
        "metadata",
    )

    def __post_init__(self) -> None:
        self.section = _validate_choice("section", self.section, PAPER_SECTIONS)
        self.source_kind = _validate_choice("source_kind", self.source_kind, SOURCE_KINDS)
        self.chunk_type = _validate_choice("chunk_type", self.chunk_type, CHUNK_TYPES)
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("evidence chunk text must not be empty")
        if len(self.text) > MAX_CHUNK_CHARS:
            raise ValueError(f"evidence chunk text cannot exceed {MAX_CHUNK_CHARS} characters")
        if self.page is not None and (not isinstance(self.page, int) or self.page < 1):
            raise ValueError("page must be a positive one-based page number or None")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PaperEvidenceChunk":
        return cls(**{name: data[name] for name in cls.FIELD_NAMES if name in data})

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.FIELD_NAMES}


@dataclass
class PaperEvidencePackage:
    """A paper's raw evidence chunks and their source records."""

    paper_id: str
    paper_title: str
    chunks: list[PaperEvidenceChunk] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "paper_id",
        "paper_title",
        "chunks",
        "sources",
        "warnings",
    )

    def __post_init__(self) -> None:
        normalized: list[PaperEvidenceChunk] = []
        for chunk in self.chunks:
            if isinstance(chunk, PaperEvidenceChunk):
                normalized.append(chunk)
            elif isinstance(chunk, Mapping):
                normalized.append(PaperEvidenceChunk.from_mapping(chunk))
            else:
                raise TypeError("chunks must contain PaperEvidenceChunk instances or mappings")
        self.chunks = _deduplicate_chunks(normalized)
        if not all(isinstance(source, dict) for source in self.sources):
            raise TypeError("sources must be a list of dictionaries")
        if not all(isinstance(warning, str) for warning in self.warnings):
            raise TypeError("warnings must be a list of strings")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PaperEvidencePackage":
        return cls(**{name: data[name] for name in cls.FIELD_NAMES if name in data})

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "sources": self.sources,
            "warnings": self.warnings,
        }


@dataclass
class LineageEvidenceInput:
    """The raw evidence packages for two papers, without a relation claim."""

    source: PaperEvidencePackage
    target: PaperEvidencePackage

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LineageEvidenceInput":
        source = data["source"]
        target = data["target"]
        return cls(
            source=source
            if isinstance(source, PaperEvidencePackage)
            else PaperEvidencePackage.from_mapping(source),
            target=target
            if isinstance(target, PaperEvidencePackage)
            else PaperEvidencePackage.from_mapping(target),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source.to_dict(), "target": self.target.to_dict()}


def discovered_paper_to_evidence_chunks(
    paper: DiscoveredPaper,
) -> list[PaperEvidenceChunk]:
    """Convert the provider-neutral discovered abstract into raw evidence."""

    abstract = paper.abstract.strip()
    if not abstract:
        return []
    chunks = _split_long_text(abstract)
    return [
        PaperEvidenceChunk(
            paper_id=paper.paper_id,
            paper_title=paper.title,
            text=text,
            section="abstract",
            location="Abstract",
            source_kind="abstract",
            chunk_type="abstract",
            metadata={"provider": paper.provider, "provider_id": paper.provider_id},
        )
        for text in chunks
    ]


def discovered_paper_to_evidence_package(
    paper: DiscoveredPaper,
) -> PaperEvidencePackage:
    chunks = discovered_paper_to_evidence_chunks(paper)
    return PaperEvidencePackage(
        paper_id=paper.paper_id,
        paper_title=paper.title,
        chunks=chunks,
        sources=(
            [{"source_kind": "abstract", "provider": paper.provider, "provider_id": paper.provider_id}]
            if chunks
            else []
        ),
        warnings=[] if chunks else ["No abstract is available for this paper."],
    )


_SECTION_HEADINGS = {
    "abstract": "abstract",
    "introduction": "introduction",
    "state of the art": "related_work",
    "related work": "related_work",
    "literature review": "related_work",
    "background": "background",
    "method": "methods",
    "methods": "methods",
    "methodology": "methods",
    "approach": "methods",
    "model": "methods",
    "architecture": "methods",
    "network architecture": "methods",
    "proposed architecture": "methods",
    "proposed method": "methods",
    "proposed model": "methods",
    "framework": "methods",
    "experiments": "experiments",
    "experimental setup": "experiments",
    "evaluation": "experiments",
    "results": "results",
    "results and discussion": "results",
    "experimental results": "results",
    "evaluation results": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
}
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[.)])?\s+(.+?)\s*$")


def _heading_info(line: str, current_section: str) -> tuple[str, str] | None:
    value = line.strip().strip("# ")
    numbered = _NUMBERED_HEADING_RE.match(value)
    number = numbered.group(1) if numbered else ""
    title = numbered.group(2) if numbered else value
    normalized = re.sub(r"[\s:.-]+$", "", title.casefold()).strip()
    section = _SECTION_HEADINGS.get(normalized)
    # These generic labels commonly occur in tables. Accept them only when
    # explicitly numbered; plural Methods and Methodology remain unambiguous.
    if not numbered and normalized in {"method", "model"}:
        section = None
    if section:
        return section, title
    # A numbered subsection such as "3.1 Network Architecture" updates the
    # location while inheriting the current top-level section.
    if numbered and "." in number and current_section != "unknown":
        return current_section, title
    return None


def _clean_pdf_lines(lines: list[str]) -> str:
    # Join visual line wraps while retaining all lexical characters, including
    # hyphens, formulas, digits, model names, and citation markers.
    return re.sub(r"[\t ]+", " ", " ".join(line.strip() for line in lines)).strip()


def _is_references_heading(line: str) -> bool:
    """Recognize only a standalone, optional top-level numbered bibliography heading."""

    value = line.strip().strip("# ")
    numbered = _NUMBERED_HEADING_RE.match(value)
    if numbered and "." in numbered.group(1):
        return False
    title = numbered.group(2) if numbered else value
    normalized = re.sub(r"[\s:.-]+$", "", title.casefold()).strip()
    return normalized in {"references", "bibliography"}


def _pdf_paragraphs(raw_text: str) -> list[tuple[str, bool]]:
    """Return (text, is_heading) entries, merging wrapped lines in paragraphs."""

    entries: list[tuple[str, bool]] = []
    pending: list[str] = []

    def flush() -> None:
        text = _clean_pdf_lines(pending)
        if text:
            entries.append((text, False))
        pending.clear()

    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if _is_references_heading(stripped):
            flush()
            entries.append((stripped, True))
            continue
        inline_abstract = re.match(r"^abstract\s*:\s*(.*)$", stripped, re.IGNORECASE)
        if inline_abstract:
            flush()
            entries.append(("Abstract", True))
            abstract_text = inline_abstract.group(1).strip()
            if abstract_text:
                pending.append(abstract_text)
            continue
        heading = _heading_info(stripped, "methods")
        # Parse clear heading lines independently from wrapped body text.
        if heading:
            flush()
            entries.append((stripped, True))
        else:
            pending.append(stripped)
    flush()
    return entries


def pdf_to_evidence_chunks(
    path: str | Path,
    *,
    paper_id: str,
    paper_title: str | None = None,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> list[PaperEvidenceChunk]:
    """Extract selectable PDF text as page-aware paragraph chunks using PyMuPDF."""

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise EvidenceExtractionError(f"PDF file does not exist or is not a file: {pdf_path}")
    if not 1 <= max_chunk_chars <= MAX_CHUNK_CHARS:
        raise ValueError(f"max_chunk_chars must be between 1 and {MAX_CHUNK_CHARS}")
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - import depends on installed wheel
        raise EvidenceExtractionError("PyMuPDF is required to extract local PDF text.") from exc

    chunks: list[PaperEvidenceChunk] = []
    section = "unknown"
    active_heading: str | None = None
    extracted_any_text = False
    references_started = False
    try:
        with fitz.open(pdf_path) as document:
            title = (
                paper_title
                or (document.metadata or {}).get("title")
                or pdf_path.stem
            ).strip()
            for page_index, page in enumerate(document, start=1):
                blocks = page.get_text("blocks", sort=True)
                for block in blocks:
                    if len(block) > 6 and block[6] != 0:
                        continue
                    raw_text = str(block[4] or "")
                    if raw_text.strip():
                        extracted_any_text = True
                    for text, is_heading in _pdf_paragraphs(raw_text):
                        if _is_references_heading(text):
                            references_started = True
                            break
                        if is_heading:
                            heading = _heading_info(text, section)
                            if heading:
                                section, active_heading = heading
                                continue
                        location = f"PDF page {page_index}"
                        if active_heading:
                            location += f" · Section {active_heading}"
                        for part_index, part in enumerate(
                            _split_long_text(text, max_chunk_chars), start=1
                        ):
                            metadata: dict[str, Any] = {
                                "source_filename": pdf_path.name,
                            }
                            if active_heading:
                                metadata["section_heading"] = active_heading
                            if part_index > 1:
                                metadata["paragraph_part"] = part_index
                            chunks.append(
                                PaperEvidenceChunk(
                                    paper_id=paper_id,
                                    paper_title=title,
                                    text=part,
                                    section=section,
                                    page=page_index,
                                    location=location,
                                    source_kind="pdf",
                                    chunk_type="paragraph",
                                    metadata=metadata,
                                )
                            )
                    if references_started:
                        break
                if references_started:
                    break
    except EvidenceExtractionError:
        raise
    except Exception as exc:
        raise EvidenceExtractionError(
            f"Could not open or extract text from PDF '{pdf_path.name}': {exc}"
        ) from exc

    if not extracted_any_text or not chunks:
        raise EvidenceExtractionError(
            f"PDF '{pdf_path.name}' contains no extractable text; scanned PDFs require OCR, which is unsupported."
        )
    return chunks


def pdf_to_evidence_package(
    path: str | Path,
    *,
    paper_id: str,
    paper_title: str | None = None,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> PaperEvidencePackage:
    pdf_path = Path(path)
    chunks = pdf_to_evidence_chunks(
        pdf_path,
        paper_id=paper_id,
        paper_title=paper_title,
        max_chunk_chars=max_chunk_chars,
    )
    title = chunks[0].paper_title
    return PaperEvidencePackage(
        paper_id=paper_id,
        paper_title=title,
        chunks=chunks,
        sources=[{"source_kind": "pdf", "filename": pdf_path.name}],
    )


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _deduplicate_chunks(chunks: list[PaperEvidenceChunk]) -> list[PaperEvidenceChunk]:
    """Collapse only whitespace/case-normalized exact duplicates, retaining provenance."""

    unique: list[PaperEvidenceChunk] = []
    by_text: dict[str, PaperEvidenceChunk] = {}
    for original in chunks:
        chunk = replace(original, metadata=dict(original.metadata))
        key = _normalized_text(chunk.text)
        prior = by_text.get(key)
        if prior is None:
            by_text[key] = chunk
            unique.append(chunk)
            continue
        duplicate_sources = prior.metadata.setdefault(
            "duplicate_sources", [prior.source_kind]
        )
        if chunk.source_kind not in duplicate_sources:
            duplicate_sources.append(chunk.source_kind)
        duplicate_locations = prior.metadata.setdefault(
            "duplicate_locations", [prior.location]
        )
        if chunk.location not in duplicate_locations:
            duplicate_locations.append(chunk.location)
    return unique


def merge_paper_evidence_packages(
    *packages: PaperEvidencePackage,
) -> PaperEvidencePackage:
    """Combine sources for one paper, deduplicating only normalized exact text."""

    if not packages:
        raise ValueError("at least one evidence package is required")
    first = packages[0]
    for package in packages[1:]:
        if package.paper_id != first.paper_id:
            raise ValueError("cannot merge evidence packages for different paper IDs")
        if package.paper_title and first.paper_title and package.paper_title != first.paper_title:
            raise ValueError("cannot merge evidence packages with conflicting paper titles")
    return PaperEvidencePackage(
        paper_id=first.paper_id,
        paper_title=first.paper_title or next(
            (package.paper_title for package in packages if package.paper_title), ""
        ),
        chunks=[chunk for package in packages for chunk in package.chunks],
        sources=[source for package in packages for source in package.sources],
        warnings=[warning for package in packages for warning in package.warnings],
    )


def prepare_lineage_evidence_input(
    source: PaperEvidencePackage,
    target: PaperEvidencePackage,
) -> LineageEvidenceInput:
    """Pair raw source and target evidence without creating a relationship claim."""

    return LineageEvidenceInput(source=source, target=target)
