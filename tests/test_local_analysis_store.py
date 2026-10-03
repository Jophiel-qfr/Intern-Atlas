from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import intern_atlas.local_analysis_store as store
from intern_atlas.analysis import EvidenceItem, MethodChange, MethodLineageAnalysis


SOURCE_QUOTE = "CNN layers extract features before the final LSTM representation."
TARGET_QUOTE = "Attention weights the LSTM hidden states before classification."
UNUSED_QUOTE = "UNUSED PREVIEW TEXT MUST NOT BE EXPORTED TO MARKDOWN."


def _payload() -> dict:
    source_evidence = EvidenceItem(
        paper_id="source", paper_title="Source CNN-LSTM paper", quote=SOURCE_QUOTE,
        section="methods", location="PDF page 4 · Section Architecture", source_kind="pdf",
        evidence_type="method_description", supports=["relation", "inherited_components"],
    )
    target_evidence = EvidenceItem(
        paper_id="target", paper_title="Target Attention paper", quote=TARGET_QUOTE,
        section="methods", location="PDF page 3 · Section Attention", source_kind="pdf",
        evidence_type="method_description", supports=["relation", "added_components", "method_change:attention_mechanism"],
    )
    analysis = MethodLineageAnalysis(
        source_paper_id="source", source_paper_title=source_evidence.paper_title,
        target_paper_id="target", target_paper_title=target_evidence.paper_title,
        analysis_status="probable", relation_type="extends", confidence=0.85,
        inherited_components=["CNN and LSTM architecture"], added_components=["Attention layer"],
        method_changes=[MethodChange(
            component="attention_mechanism", change_type="added", to_value="attention layer",
            description="Weights recurrent hidden states.", confidence=0.85, evidence=[target_evidence],
        )], evidence=[source_evidence, target_evidence],
    ).to_dict()
    groups = {name: [] for name in store._GROUP_NAMES}
    groups["relation"] = [source_evidence.to_dict(), target_evidence.to_dict()]
    groups["inherited_components"] = [source_evidence.to_dict()]
    groups["added_components"] = [target_evidence.to_dict()]
    selected = [
        {"evidence_id": "S001", "paper_role": "source", "paper_id": "source", "paper_title": source_evidence.paper_title,
         "section": "methods", "page": 4, "location": source_evidence.location, "source_kind": "pdf", "chunk_type": "paragraph", "text": SOURCE_QUOTE},
        {"evidence_id": "T001", "paper_role": "target", "paper_id": "target", "paper_title": target_evidence.paper_title,
         "section": "methods", "page": 3, "location": target_evidence.location, "source_kind": "pdf", "chunk_type": "paragraph", "text": TARGET_QUOTE},
        {"evidence_id": "T002", "paper_role": "target", "paper_id": "target", "paper_title": target_evidence.paper_title,
         "section": "abstract", "page": 1, "location": "PDF page 1", "source_kind": "pdf", "chunk_type": "abstract", "text": UNUSED_QUOTE},
    ]
    return {
        "source": {"paper_id": "source", "paper_title": source_evidence.paper_title, "filename": "source.pdf"},
        "target": {"paper_id": "target", "paper_title": target_evidence.paper_title, "filename": "target.pdf"},
        "model": "mock-model", "analysis": analysis, "evidence_groups": groups, "selected_evidence": selected, "warnings": [],
    }


def test_store_round_trip_creates_utf8_record_and_preserves_result(tmp_path) -> None:
    directory = tmp_path / "data" / "analyses"
    payload = _payload()
    payload["warnings"] = ["仅有少量证据。"]
    saved = store.save_analysis(directory, payload)
    path = directory / f"{saved['analysis_id']}.json"
    record = store.load_analysis(directory, saved["analysis_id"])
    assert record == {"schema_version": 1, **saved, **payload}
    assert "仅有少量证据。" in path.read_text(encoding="utf-8")
    assert list(directory.glob("*.tmp")) == []
    assert record["analysis"]["method_changes"][0]["evidence"][0]["quote"] == TARGET_QUOTE


def test_same_paper_pair_can_have_distinct_versions_at_same_timestamp(tmp_path, monkeypatch) -> None:
    class FrozenTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 3, 10, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(store, "datetime", FrozenTime)
    first = store.save_analysis(tmp_path, _payload())
    second = store.save_analysis(tmp_path, _payload())
    assert first["analysis_id"] != second["analysis_id"]
    assert first["created_at"] == second["created_at"]
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert store.load_analysis(tmp_path, first["analysis_id"])["analysis"]["relation_type"] == "extends"


def test_history_list_is_lightweight_sorted_newest_first(tmp_path, monkeypatch) -> None:
    moments = iter([datetime(2026, 10, 3, 10, tzinfo=timezone.utc), datetime(2026, 10, 3, 14, tzinfo=timezone.utc)])
    class FakeTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(moments)
    monkeypatch.setattr(store, "datetime", FakeTime)
    first = store.save_analysis(tmp_path, _payload())
    second = store.save_analysis(tmp_path, _payload())
    history = store.list_analyses(tmp_path)
    assert [item["analysis_id"] for item in history] == [second["analysis_id"], first["analysis_id"]]
    assert all("selected_evidence" not in item and "analysis" not in item and "evidence_groups" not in item for item in history)
    assert history[0]["source_title"] == "Source CNN-LSTM paper"


def test_missing_directory_is_empty_and_is_not_created_by_reading(tmp_path) -> None:
    directory = tmp_path / "missing"
    assert store.list_analyses(directory) == []
    assert not directory.exists()


@pytest.mark.parametrize("analysis_id", ["../x", "..\\x", "x/y", "x\\y", "C:\\secret", "x.json", "", "20261003T103000Z_abcdef123456\n"])
def test_invalid_analysis_ids_are_rejected_before_file_read(tmp_path, analysis_id) -> None:
    with pytest.raises(store.InvalidAnalysisId):
        store.load_analysis(tmp_path, analysis_id)


def test_store_cannot_follow_a_record_path_outside_analyses(tmp_path, monkeypatch) -> None:
    analysis_id = "20261003T103000Z_abcdef123456"
    original = Path.resolve
    outside = tmp_path / "outside.json"
    directory = tmp_path / "analyses"
    def redirected(path, *args, **kwargs):
        return outside if path.name == analysis_id + ".json" else original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", redirected)
    with pytest.raises(store.InvalidAnalysisId):
        store.load_analysis(directory, analysis_id)


def test_corrupt_and_unsupported_records_do_not_break_listing(tmp_path) -> None:
    saved = store.save_analysis(tmp_path, _payload())
    (tmp_path / "20261003T103000Z_111111111111.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "20261003T103000Z_222222222222.json").write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    (tmp_path / "unrelated.json").write_text("{}", encoding="utf-8")
    assert [item["analysis_id"] for item in store.list_analyses(tmp_path)] == [saved["analysis_id"]]
    with pytest.raises(store.CorruptedSavedAnalysis):
        store.load_analysis(tmp_path, "20261003T103000Z_111111111111")


def test_failed_disk_write_does_not_leave_a_success_record_or_expose_system_error(tmp_path, monkeypatch) -> None:
    def fail_replace(*args):
        raise OSError("private absolute path and disk error")
    monkeypatch.setattr(store.os, "replace", fail_replace)
    with pytest.raises(store.LocalAnalysisStoreError) as caught:
        store.save_analysis(tmp_path, _payload())
    assert "private" not in str(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_storage_allowlist_excludes_keys_headers_raw_responses_and_paths(tmp_path) -> None:
    payload = _payload()
    payload.update({"api_key": "test-store-secret", "Authorization": "Bearer test-store-secret", "raw_llm_response": "test-store-secret", "reasoning_content": "test-store-secret"})
    payload["source"]["pdf_path"] = str(tmp_path / "source.pdf")
    payload["analysis"]["raw_response"] = "test-store-secret"
    payload["selected_evidence"][0]["request_payload"] = "test-store-secret"
    saved = store.save_analysis(tmp_path, payload)
    record = store.load_analysis(tmp_path, saved["analysis_id"])
    exported = json.dumps(record, ensure_ascii=False) + store.render_analysis_markdown(record)
    assert "test-store-secret" not in exported
    assert "Authorization" not in exported
    assert "raw_llm_response" not in exported and "reasoning_content" not in exported
    assert str(tmp_path) not in exported


def test_markdown_exports_final_claims_changes_and_only_cited_quotes(tmp_path) -> None:
    saved = store.save_analysis(tmp_path, _payload())
    markdown = store.render_analysis_markdown(store.load_analysis(tmp_path, saved["analysis_id"]))
    for expected in ("Source CNN-LSTM paper", "Target Attention paper", "extends", "probable", "0.85", "mock-model", "attention_mechanism — added", "PDF page 3", SOURCE_QUOTE, TARGET_QUOTE):
        assert expected in markdown
    assert UNUSED_QUOTE not in markdown
    assert "Selected Evidence" not in markdown
