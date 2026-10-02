"""Safe listing and evidence extraction for user-managed local paper PDFs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evidence_input import (
    PaperEvidencePackage,
    pdf_to_evidence_package,
    prepare_lineage_evidence_input,
)
from .lineage_context import LineageLLMContext, build_lineage_llm_context


class InvalidLocalPaperName(ValueError):
    """Raised when a requested filename is not a safe PDF basename."""


class LocalPaperNotFound(FileNotFoundError):
    """Raised when a requested PDF is missing from the configured papers folder."""


class SameLocalPaperSelection(ValueError):
    """Raised when source and target resolve to the same local PDF."""


def list_local_pdfs(papers_dir: str | Path) -> list[dict[str, Any]]:
    """List direct child PDF files only; missing directories are simply empty."""

    root = Path(papers_dir)
    if not root.is_dir():
        return []
    resolved_root = root.resolve()
    papers: list[dict[str, Any]] = []
    for path in root.iterdir():
        if path.suffix.casefold() != ".pdf":
            continue
        try:
            resolved_path = path.resolve(strict=True)
            resolved_path.relative_to(resolved_root)
            if not resolved_path.is_file():
                continue
            stat = resolved_path.stat()
        except (OSError, ValueError):
            # Ignore broken links and links that point outside papers_dir.
            continue
        papers.append(
            {
                "filename": path.name,
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds"),
            }
        )
    return sorted(papers, key=lambda item: item["filename"].casefold())


def resolve_local_pdf(papers_dir: str | Path, filename: str) -> Path:
    """Resolve one PDF basename and ensure the resolved file remains in papers_dir."""

    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or Path(filename).name != filename
        or Path(filename).suffix.casefold() != ".pdf"
    ):
        raise InvalidLocalPaperName("A PDF filename must be a single .pdf filename.")

    root = Path(papers_dir).resolve()
    try:
        resolved_path = (root / filename).resolve(strict=True)
    except FileNotFoundError as exc:
        raise LocalPaperNotFound(filename) from exc
    except (OSError, ValueError) as exc:
        raise InvalidLocalPaperName("The requested PDF filename is invalid.") from exc
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise InvalidLocalPaperName("The requested PDF is outside the papers folder.") from exc
    if not resolved_path.is_file() or resolved_path.suffix.casefold() != ".pdf":
        raise LocalPaperNotFound(filename)
    return resolved_path


def _reliable_title(title: str, stem: str) -> str:
    candidate = title.strip()
    folded = candidate.casefold()
    if (
        not candidate
        or len(candidate) > 300
        or folded in {"untitled", "unknown", "document", "microsoft word"}
        or folded.startswith(("microsoft word -", "untitled document"))
        or "/" in candidate
        or "\\" in candidate
    ):
        return stem
    return candidate


def _load_local_pdf_package(pdf_path: str | Path) -> PaperEvidencePackage:
    path = Path(pdf_path)
    package = pdf_to_evidence_package(path, paper_id=path.stem)
    paper_title = _reliable_title(package.paper_title, path.stem)
    package.paper_title = paper_title
    for chunk in package.chunks:
        chunk.paper_title = paper_title
    return package


def build_local_lineage_context(
    papers_dir: str | Path,
    source_filename: str,
    target_filename: str,
) -> tuple[dict[str, str], dict[str, str], LineageLLMContext]:
    """Build the existing bounded lineage context from two safe local PDFs."""

    if source_filename.casefold() == target_filename.casefold():
        raise SameLocalPaperSelection("Source and target papers must be different.")

    source_path = resolve_local_pdf(papers_dir, source_filename)
    target_path = resolve_local_pdf(papers_dir, target_filename)
    try:
        if source_path.samefile(target_path):
            raise SameLocalPaperSelection(
                "Source and target papers must be different."
            )
    except OSError:
        if source_path == target_path:
            raise SameLocalPaperSelection(
                "Source and target papers must be different."
            )

    source_package = _load_local_pdf_package(source_path)
    target_package = _load_local_pdf_package(target_path)
    evidence_input = prepare_lineage_evidence_input(source_package, target_package)
    context = build_lineage_llm_context(evidence_input)
    source = {
        "filename": source_path.name,
        "paper_id": source_package.paper_id,
        "paper_title": source_package.paper_title,
    }
    target = {
        "filename": target_path.name,
        "paper_id": target_package.paper_id,
        "paper_title": target_package.paper_title,
    }
    return source, target, context


def build_local_pdf_evidence(pdf_path: str | Path) -> dict[str, Any]:
    """Use the shared evidence parser and return a compact browser-facing payload."""

    path = Path(pdf_path)
    package: PaperEvidencePackage = _load_local_pdf_package(path)
    paper_title = package.paper_title
    section_counts = dict(Counter(chunk.section for chunk in package.chunks))
    return {
        "paper_id": package.paper_id,
        "paper_title": paper_title,
        "chunk_count": len(package.chunks),
        "section_counts": section_counts,
        "chunks": [
            {
                "section": chunk.section,
                "page": chunk.page,
                "location": chunk.location,
                "section_heading": chunk.metadata.get("section_heading"),
                "text": chunk.text,
            }
            for chunk in package.chunks
        ],
    }
