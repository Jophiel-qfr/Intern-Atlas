"""Safe listing and evidence extraction for user-managed local paper PDFs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evidence_input import PaperEvidencePackage, pdf_to_evidence_package


class InvalidLocalPaperName(ValueError):
    """Raised when a requested filename is not a safe PDF basename."""


class LocalPaperNotFound(FileNotFoundError):
    """Raised when a requested PDF is missing from the configured papers folder."""


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


def build_local_pdf_evidence(pdf_path: str | Path) -> dict[str, Any]:
    """Use the shared evidence parser and return a compact browser-facing payload."""

    path = Path(pdf_path)
    paper_id = path.stem
    package: PaperEvidencePackage = pdf_to_evidence_package(
        path,
        paper_id=paper_id,
    )
    paper_title = _reliable_title(package.paper_title, path.stem)
    for chunk in package.chunks:
        chunk.paper_title = paper_title
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
