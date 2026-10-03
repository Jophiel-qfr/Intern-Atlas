"""Local, versioned analysis records and Markdown exports; no network calls.

Only the validated Web result is persisted. Provider responses, settings,
request headers and PDF paths are never part of this storage contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .analysis import EvidenceItem, HARAnalysis, MethodChange, MethodLineageAnalysis


_ID_RE = re.compile(r"^\d{8}T\d{6}(?:\d{6})?Z_[a-f0-9]{12}$", re.ASCII)
_RESULT_FIELDS = (
    "source", "target", "model", "analysis", "evidence_groups", "selected_evidence", "warnings",
)
_GROUP_NAMES = (
    "relation", "inherited_components", "changed_components", "added_components",
    "removed_components", "problem_addressed", "claimed_contribution", "experimental_evidence", "limitations",
)
_EVIDENCE_FIELDS = EvidenceItem.FIELD_NAMES
_SELECTED_FIELDS = (
    "evidence_id", "paper_role", "paper_id", "paper_title", "section", "page", "location",
    "source_kind", "chunk_type", "text",
)
_ANALYSIS_FIELDS = MethodLineageAnalysis.FIELD_NAMES
_CHANGE_FIELDS = MethodChange.FIELD_NAMES


class LocalAnalysisStoreError(Exception):
    """Safe storage error, with no system path or provider details."""


class InvalidAnalysisId(LocalAnalysisStoreError):
    pass


class SavedAnalysisNotFound(LocalAnalysisStoreError):
    pass


class CorruptedSavedAnalysis(LocalAnalysisStoreError):
    pass


def _validate_id(analysis_id: str) -> None:
    if not isinstance(analysis_id, str) or not _ID_RE.fullmatch(analysis_id):
        raise InvalidAnalysisId("Invalid saved analysis ID.")


def _record_path(directory: str | Path, analysis_id: str) -> Path:
    _validate_id(analysis_id)
    base = Path(directory).resolve()
    path = base / f"{analysis_id}.json"
    if path.resolve().parent != base:
        raise InvalidAnalysisId("Saved analysis must stay in its configured directory.")
    return path


def _pick(mapping: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        raise CorruptedSavedAnalysis("Saved analysis contains an invalid object.")
    return {name: mapping[name] for name in names if name in mapping}


def _evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise CorruptedSavedAnalysis("Saved evidence must be a list.")
    return [_pick(item, _EVIDENCE_FIELDS) for item in items]


def _safe_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project known Web fields, excluding incidental request/settings metadata."""
    try:
        result = _pick(payload, _RESULT_FIELDS)
        for role in ("source", "target"):
            paper = _pick(result[role], ("paper_id", "paper_title", "filename"))
            if not all(isinstance(paper.get(name), str) for name in ("paper_id", "paper_title", "filename")):
                raise ValueError("Invalid paper identity")
            # Store only basenames, never absolute paths or traversal strings.
            if not paper["filename"] or any(separator in paper["filename"] for separator in ("/", "\\", ":")) or paper["filename"] in {".", ".."}:
                raise ValueError("Invalid paper filename")
            result[role] = paper
        analysis = _pick(result["analysis"], _ANALYSIS_FIELDS)
        if not all(name in analysis for name in ("source_paper_id", "target_paper_id", "analysis_status")):
            raise ValueError("Missing final analysis fields")
        analysis["evidence"] = _evidence(analysis.get("evidence", []))
        changes = []
        for item in analysis.get("method_changes", []):
            change = _pick(item, _CHANGE_FIELDS)
            change["evidence"] = _evidence(change.get("evidence", []))
            changes.append(change)
        analysis["method_changes"] = changes
        # The current analyze flow does not embed per-paper HARAnalysis objects.
        # Keep any future documented dimensions, without their unrestricted extras.
        if any(analysis.get(name) is not None for name in ("source_analysis", "target_analysis")):
            for name in ("source_analysis", "target_analysis"):
                if analysis.get(name) is not None:
                    analysis[name] = _pick(analysis[name], HARAnalysis.FIELD_NAMES)
        MethodLineageAnalysis.from_mapping(analysis)  # Reuse the existing contract; do not infer anything.
        result["analysis"] = analysis
        groups = result["evidence_groups"]
        if not isinstance(groups, Mapping):
            raise ValueError("Invalid evidence groups")
        result["evidence_groups"] = {name: _evidence(groups.get(name, [])) for name in _GROUP_NAMES}
        if not isinstance(result["selected_evidence"], list):
            raise ValueError("Invalid selected evidence")
        result["selected_evidence"] = [_pick(item, _SELECTED_FIELDS) for item in result["selected_evidence"]]
        if not isinstance(result["model"], str) or not isinstance(result["warnings"], list):
            raise ValueError("Invalid result metadata")
        if not all(isinstance(warning, str) for warning in result["warnings"]):
            raise ValueError("Invalid warnings")
        # Copy JSON values so later mutations of the live response cannot change a record.
        return json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except (KeyError, TypeError, ValueError) as exc:
        raise CorruptedSavedAnalysis("Saved analysis has an invalid result structure.") from exc


def _created_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("Missing timezone")
        return parsed.astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CorruptedSavedAnalysis("Saved analysis has an invalid creation time.") from exc


def save_analysis(directory: str | Path, payload: Mapping[str, Any]) -> dict[str, str]:
    """Persist a completed validated result as a new version, atomically."""
    result = _safe_result(payload)
    now = datetime.now(timezone.utc)
    created_at = now.isoformat(timespec="microseconds").replace("+00:00", "Z")
    digest_input = [result["source"]["filename"], result["target"]["filename"], created_at, result["model"], uuid4().hex]
    digest = hashlib.sha256(json.dumps(digest_input, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    analysis_id = f"{now:%Y%m%dT%H%M%S%fZ}_{digest}"
    record = {"schema_version": 1, "analysis_id": analysis_id, "created_at": created_at, **result}
    path = _record_path(directory, analysis_id)
    temporary = path.with_suffix(".tmp")
    temporary_created = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            temporary_created = True
            json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        if path.exists():
            raise FileExistsError("A record with this ID already exists")
        # Readers see a complete record; interrupted temporary files are not listed.
        os.replace(temporary, path)
    except (OSError, ValueError, TypeError) as exc:
        raise LocalAnalysisStoreError("Could not save the completed analysis locally.") from exc
    finally:
        # This is only the temporary file created for this new record, never a history record.
        try:
            if temporary_created:
                temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return {"analysis_id": analysis_id, "created_at": created_at}


def load_analysis(directory: str | Path, analysis_id: str) -> dict[str, Any]:
    """Load a safe record independently of PDFs, preview or an LLM client."""
    path = _record_path(directory, analysis_id)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SavedAnalysisNotFound("Saved analysis not found.") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CorruptedSavedAnalysis("Saved analysis JSON is corrupted.") from exc
    except OSError as exc:
        raise LocalAnalysisStoreError("Could not read saved analysis.") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or raw.get("analysis_id") != analysis_id:
        raise CorruptedSavedAnalysis("Saved analysis has an unsupported or invalid record format.")
    _created_at(raw.get("created_at"))
    return {"schema_version": 1, "analysis_id": analysis_id, "created_at": raw["created_at"], **_safe_result(raw)}


def list_analyses(directory: str | Path) -> list[dict[str, Any]]:
    """Return lightweight summaries; a bad record never breaks the whole list."""
    base = Path(directory)
    if not base.exists():
        return []
    summaries = []
    try:
        paths = list(base.glob("*.json"))
    except OSError as exc:
        raise LocalAnalysisStoreError("Could not read local analysis history.") from exc
    for path in paths:
        try:
            record = load_analysis(base, path.stem)
        except LocalAnalysisStoreError:
            continue
        analysis = record["analysis"]
        summaries.append({
            "analysis_id": record["analysis_id"], "created_at": record["created_at"],
            "source_title": record["source"]["paper_title"], "target_title": record["target"]["paper_title"],
            "source_filename": record["source"]["filename"], "target_filename": record["target"]["filename"],
            "relation_type": analysis.get("relation_type"), "analysis_status": analysis["analysis_status"],
            "confidence": analysis.get("confidence"), "model": record["model"],
        })
    summaries.sort(key=lambda item: (_created_at(item["created_at"]), item["analysis_id"]), reverse=True)
    return summaries


def render_analysis_markdown(saved_payload: Mapping[str, Any]) -> str:
    """Export final claims and cited evidence, never the unused preview chunks."""
    analysis = saved_payload["analysis"]
    lines = ["# Method Lineage Analysis", "", "## Papers", "", "Source:", saved_payload["source"]["paper_title"], "", "Target:", saved_payload["target"]["paper_title"], "", "## Relation", ""]
    for label, value in (
        ("Type", analysis.get("relation_type")), ("Status", analysis.get("analysis_status")),
        ("Confidence", analysis.get("confidence")), ("Model", saved_payload.get("model")),
        ("Created", saved_payload.get("created_at")), ("Analysis ID", saved_payload.get("analysis_id")),
    ):
        lines.append(f"- {label}: {value if value is not None else '—'}")
    if analysis.get("uncertainty"):
        lines.extend(["", "Uncertainty:", str(analysis["uncertainty"])])
    for name, title in (
        ("inherited_components", "Inherited Components"), ("changed_components", "Changed Components"),
        ("added_components", "Added Components"), ("removed_components", "Removed Components"),
        ("problem_addressed", "Problem Addressed"), ("claimed_contribution", "Claimed Contribution"),
        ("experimental_evidence", "Experimental Evidence"), ("limitations", "Limitations"),
    ):
        value = analysis.get(name)
        values = value if isinstance(value, list) else ([value] if value else [])
        lines.extend(["", f"## {title}", ""])
        if values:
            lines.extend(f"- {item}" for item in values)
        else:
            lines.append("None supported by current evidence.")
    lines.extend(["", "## Method Changes", ""])
    changes = analysis.get("method_changes", [])
    if not changes:
        lines.append("None supported by current evidence.")

    def append_evidence(item: Mapping[str, Any]) -> None:
        lines.extend(["", f"- Paper: {item.get('paper_title') or item.get('paper_id')}", f"- Location: {item.get('location') or 'Unavailable'}", f"- Section: {item.get('section') or 'unknown'}", f"- Evidence type: {item.get('evidence_type') or 'unknown'}", f"- Source kind: {item.get('source_kind') or 'unknown'}", ""])
        # Quote content and line breaks are retained, with Markdown blockquote markers.
        lines.extend("> " + line for line in str(item.get("quote") or "").split("\n"))

    for change in changes:
        lines.extend([f"### {change['component']} — {change['change_type']}", ""])
        for label, key in (("From", "from_value"), ("To", "to_value"), ("Description", "description"), ("Confidence", "confidence")):
            value = change.get(key)
            lines.extend([f"{label}:", str(value) if value is not None else "—", ""])
        lines.append("Evidence:")
        for item in change.get("evidence", []):
            append_evidence(item)
    lines.extend(["", "## Cited Evidence", ""])
    # Final analysis.evidence already has stable order and deduplicated context IDs.
    for item in analysis.get("evidence", []):
        append_evidence(item)
    return "\n".join(lines).rstrip() + "\n"
