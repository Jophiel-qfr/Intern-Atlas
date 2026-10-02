from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from intern_atlas.db import connect
from intern_atlas.local_papers import (
    InvalidLocalPaperName,
    build_local_pdf_evidence,
    resolve_local_pdf,
)
from intern_atlas.server import create_app
from intern_atlas.ui import get_index_html
import intern_atlas.local_papers as local_papers_module


def _make_app(tmp_path: Path, monkeypatch, *, create_papers: bool = True):
    data_dir = tmp_path / "data"
    papers_dir = data_dir / "papers"
    if create_papers:
        papers_dir.mkdir(parents=True)
    monkeypatch.setenv("INTERN_ATLAS_DATA_DIR", str(data_dir))
    db_path = tmp_path / "graph.db"
    connection = connect(db_path)
    connection.close()
    return create_app(db_path), papers_dir


def _write_pdf(path: Path, *, title: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    lines = [
        "Abstract",
        "We introduce DeepConvLSTM for wearable activity recognition.",
        "1. Introduction",
        "Wearable activity recognition is challenging.",
        "2. Related Work",
        "Earlier methods use convolutional recurrent networks.",
        "3. Method",
        "DeepConvLSTM applies convolutional feature extraction and LSTM temporal modeling.",
        "4. Experiments",
        "We evaluate the model on PAMAP2.",
        "5. Results",
        "The model reaches 92.5 percent accuracy.",
        "Conclusion",
        "The approach supports wearable sensor recognition.",
    ]
    for index, text in enumerate(lines):
        page.insert_text((48, 40 + index * 30), text, fontsize=10)
    if title:
        metadata = dict(document.metadata)
        metadata["title"] = title
        document.set_metadata(metadata)
    document.save(path)
    document.close()


def test_local_paper_list_only_returns_direct_pdf_files(tmp_path, monkeypatch) -> None:
    app, papers_dir = _make_app(tmp_path, monkeypatch)
    _write_pdf(papers_dir / "2016_DeepConvLSTM.pdf")
    (papers_dir / "notes.txt").write_text("not a paper", encoding="utf-8")
    (papers_dir / "nested").mkdir()
    _write_pdf(papers_dir / "nested" / "nested.pdf")

    with TestClient(app) as client:
        response = client.get("/api/local/papers")

    assert response.status_code == 200
    payload = response.json()
    assert [item["filename"] for item in payload] == ["2016_DeepConvLSTM.pdf"]
    assert set(payload[0]) == {"filename", "size_bytes", "modified_time"}
    assert payload[0]["size_bytes"] > 0
    assert payload[0]["modified_time"]


def test_missing_papers_directory_returns_empty_list(tmp_path, monkeypatch) -> None:
    app, papers_dir = _make_app(tmp_path, monkeypatch, create_papers=False)

    with TestClient(app) as client:
        response = client.get("/api/local/papers")

    assert response.status_code == 200
    assert response.json() == []
    assert not papers_dir.exists()


def test_pdf_path_traversal_is_rejected(tmp_path) -> None:
    papers_dir = tmp_path / "data" / "papers"
    papers_dir.mkdir(parents=True)
    outside = tmp_path / "outside.pdf"
    _write_pdf(outside)

    with pytest.raises(InvalidLocalPaperName):
        resolve_local_pdf(papers_dir, "../outside.pdf")


def test_pdf_evidence_api_rejects_encoded_path_traversal(tmp_path, monkeypatch) -> None:
    app, _ = _make_app(tmp_path, monkeypatch)
    _write_pdf(tmp_path / "outside.pdf")

    with TestClient(app) as client:
        response = client.get("/api/local/papers/..%5Coutside.pdf/evidence")

    assert response.status_code == 400
    assert response.json()["detail"] == "PDF 文件名无效。"
    assert str(tmp_path) not in response.text


def test_missing_pdf_returns_not_found_without_local_path(tmp_path, monkeypatch) -> None:
    app, _ = _make_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.get("/api/local/papers/missing.pdf/evidence")

    assert response.status_code == 404
    assert response.json()["detail"] == "未找到该本地 PDF 文件。"
    assert str(tmp_path) not in response.text


def test_evidence_api_uses_shared_pdf_package_and_counts_sections(
    tmp_path, monkeypatch
) -> None:
    app, papers_dir = _make_app(tmp_path, monkeypatch)
    _write_pdf(
        papers_dir / "2016_DeepConvLSTM.pdf",
        title="DeepConvLSTM: A CNN-LSTM Architecture",
    )
    calls = []
    original = local_papers_module.pdf_to_evidence_package

    def track_package(path, **kwargs):
        calls.append((Path(path).name, kwargs))
        return original(path, **kwargs)

    monkeypatch.setattr(local_papers_module, "pdf_to_evidence_package", track_package)

    with TestClient(app) as client:
        response = client.get("/api/local/papers/2016_DeepConvLSTM.pdf/evidence")

    assert response.status_code == 200
    payload = response.json()
    assert calls == [("2016_DeepConvLSTM.pdf", {"paper_id": "2016_DeepConvLSTM"})]
    assert payload["paper_id"] == "2016_DeepConvLSTM"
    assert payload["paper_title"] == "DeepConvLSTM: A CNN-LSTM Architecture"
    assert payload["section_counts"] == {
        "abstract": 1,
        "introduction": 1,
        "related_work": 1,
        "methods": 1,
        "experiments": 1,
        "results": 1,
        "conclusion": 1,
    }
    assert payload["chunk_count"] == len(payload["chunks"])
    method = next(chunk for chunk in payload["chunks"] if chunk["section"] == "methods")
    assert method["page"] == 1
    assert "DeepConvLSTM applies" in method["text"]


def test_unreliable_pdf_metadata_title_falls_back_to_stem(tmp_path) -> None:
    pdf_path = tmp_path / "2016_DeepConvLSTM.pdf"
    _write_pdf(pdf_path, title="Microsoft Word - paper.docx")

    result = build_local_pdf_evidence(pdf_path)

    assert result["paper_title"] == "2016_DeepConvLSTM"
    assert all(chunk["text"] for chunk in result["chunks"])


@pytest.mark.parametrize(
    ("filename", "contents", "expected_detail"),
    [
        (
            "broken.pdf",
            b"not a PDF",
            "无法解析该 PDF，请确认文件完整且格式有效。",
        ),
        (
            "scanned.pdf",
            None,
            "PDF 中没有可提取的文字；扫描版 PDF 暂不支持 OCR。",
        ),
    ],
)
def test_pdf_extraction_errors_are_chinese_and_do_not_leak_paths(
    tmp_path, monkeypatch, filename, contents, expected_detail
) -> None:
    app, papers_dir = _make_app(tmp_path, monkeypatch)
    path = papers_dir / filename
    if contents is None:
        document = fitz.open()
        document.new_page()
        document.save(path)
        document.close()
    else:
        path.write_bytes(contents)

    with TestClient(app) as client:
        response = client.get(f"/api/local/papers/{filename}/evidence")

    assert response.status_code == 422
    assert response.json()["detail"] == expected_detail
    assert str(papers_dir) not in response.text


def test_homepage_links_to_chinese_local_papers_page(tmp_path, monkeypatch) -> None:
    app, _ = _make_app(tmp_path, monkeypatch)
    html = get_index_html("zh-CN")

    assert 'href="/local-papers"' in html
    assert "本地论文" in html
    with TestClient(app) as client:
        page = client.get("/local-papers")

    assert page.status_code == 200
    assert "章节概览" in page.text
    assert "证据片段" in page.text
    assert "/api/local/papers" in page.text
