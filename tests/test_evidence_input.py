import json

import fitz
import pytest

from intern_atlas.discovery_models import DiscoveredPaper
from intern_atlas.evidence_input import (
    EvidenceExtractionError,
    LineageEvidenceInput,
    PaperEvidenceChunk,
    PaperEvidencePackage,
    discovered_paper_to_evidence_chunks,
    discovered_paper_to_evidence_package,
    merge_paper_evidence_packages,
    pdf_to_evidence_chunks,
    pdf_to_evidence_package,
    prepare_lineage_evidence_input,
)


@pytest.mark.parametrize("provider", ["openalex", "semantic_scholar"])
def test_discovered_paper_abstract_becomes_provider_neutral_chunk(provider: str) -> None:
    paper = DiscoveredPaper(
        provider=provider,
        provider_id="W123",
        title="DeepConvLSTM",
        abstract="We introduce DeepConvLSTM for wearable activity recognition.",
    )

    chunks = discovered_paper_to_evidence_chunks(paper)

    assert len(chunks) == 1
    assert chunks[0].paper_id == "W123"
    assert chunks[0].paper_title == "DeepConvLSTM"
    assert chunks[0].text == paper.abstract
    assert chunks[0].section == "abstract"
    assert chunks[0].page is None
    assert chunks[0].location == "Abstract"
    assert chunks[0].source_kind == "abstract"
    assert chunks[0].chunk_type == "abstract"


@pytest.mark.parametrize("provider", ["openalex", "semantic_scholar"])
def test_missing_discovered_abstract_returns_empty_chunks(provider: str) -> None:
    paper = DiscoveredPaper(provider=provider, provider_id="paper-1", title="No abstract")

    assert discovered_paper_to_evidence_chunks(paper) == []
    package = discovered_paper_to_evidence_package(paper)
    assert package.chunks == []
    assert package.warnings


def _make_pdf(path) -> None:
    document = fitz.open()
    first = document.new_page()
    first.insert_text((48, 45), "Preface text with no recognized section.", fontsize=11)
    first.insert_text((48, 85), "Abstract", fontsize=12)
    first.insert_textbox(
        fitz.Rect(48, 95, 540, 130),
        "We present DeepConvLSTM using dataset PAMAP2 and report 92.5% accuracy [12].",
        fontsize=11,
    )
    first.insert_text((48, 160), "1. Introduction", fontsize=12)
    first.insert_text((48, 180), "Wearable sensor recognition remains challenging.", fontsize=11)
    first.insert_text((48, 225), "3. Method", fontsize=12)
    first.insert_text((48, 245), "3.1 Network Architecture", fontsize=11)
    first.insert_textbox(
        fitz.Rect(48, 255, 300, 310),
        "The DeepConvLSTM model reaches 92.5% accuracy [12] with an LSTM-based temporal encoder.",
        fontsize=11,
    )

    second = document.new_page()
    second.insert_text((48, 55), "4. Experiments", fontsize=12)
    second.insert_text((48, 75), "We evaluate on PAMAP2 and Opportunity.", fontsize=11)
    second.insert_text((48, 120), "Conclusion", fontsize=12)
    second.insert_text((48, 140), "The model preserves the reported 92.5% result.", fontsize=11)
    document.save(path)
    document.close()


def test_pdf_chunks_keep_pages_sections_and_section_inheritance(tmp_path) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    _make_pdf(pdf_path)

    chunks = pdf_to_evidence_chunks(
        pdf_path,
        paper_id="paper-1",
        paper_title="Fixture HAR paper",
    )

    def contains(fragment: str):
        return next(chunk for chunk in chunks if fragment in chunk.text)

    preface = contains("Preface text")
    abstract = contains("We present DeepConvLSTM")
    introduction = contains("Wearable sensor recognition")
    method = contains("The DeepConvLSTM model")
    experiment = contains("We evaluate on PAMAP2")
    conclusion = contains("The model preserves")

    assert (preface.section, preface.page) == ("unknown", 1)
    assert (abstract.section, abstract.page) == ("abstract", 1)
    assert (introduction.section, introduction.page) == ("introduction", 1)
    assert method.section == "methods"
    assert method.page == 1
    assert "Network Architecture" in method.location
    assert (experiment.section, experiment.page) == ("experiments", 2)
    assert (conclusion.section, conclusion.page) == ("conclusion", 2)
    assert all(chunk.source_kind == "pdf" for chunk in chunks)
    assert all(chunk.chunk_type == "paragraph" for chunk in chunks)


def test_pdf_minimal_line_cleanup_preserves_terms_numbers_and_citations(tmp_path) -> None:
    pdf_path = tmp_path / "wrapped.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((48, 45), "3. Method", fontsize=12)
    page.insert_textbox(
        fitz.Rect(48, 55, 235, 140),
        "DeepConvLSTM uses PAMAP2 dataset. Accuracy is 92.5% [12]. The model includes LSTM and CNN modules.",
        fontsize=11,
    )
    document.save(pdf_path)
    document.close()

    chunks = pdf_to_evidence_chunks(pdf_path, paper_id="wrapped-paper")
    text = " ".join(chunk.text for chunk in chunks)

    assert "DeepConvLSTM uses PAMAP2 dataset." in text
    assert "92.5% [12]" in text
    assert "LSTM and CNN" in text
    assert not any("\n" in chunk.text for chunk in chunks)


def test_unknown_section_is_retained_as_unknown(tmp_path) -> None:
    pdf_path = tmp_path / "unknown.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((48, 50), "Ablation Study", fontsize=12)
    page.insert_text((48, 72), "We compare the model variants.", fontsize=11)
    document.save(pdf_path)
    document.close()

    chunks = pdf_to_evidence_chunks(pdf_path, paper_id="unknown-paper")

    assert chunks
    assert all(chunk.section == "unknown" for chunk in chunks)
    assert any("Ablation Study" in chunk.text for chunk in chunks)


def test_pdf_chunks_are_bounded_and_preserve_the_full_text(tmp_path) -> None:
    pdf_path = tmp_path / "long.pdf"
    original = "word " * 1800
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(fitz.Rect(20, 20, 570, 800), original, fontsize=8)
    document.save(pdf_path)
    document.close()

    chunks = pdf_to_evidence_chunks(
        pdf_path,
        paper_id="long-paper",
        max_chunk_chars=700,
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 700 for chunk in chunks)
    assert sum(len(chunk.text.split()) for chunk in chunks) == 1800


def test_paper_evidence_package_json_round_trip() -> None:
    original = discovered_paper_to_evidence_package(
        DiscoveredPaper(
            provider="semantic_scholar",
            provider_id="s2-1",
            title="A HAR paper",
            abstract="Abstract evidence remains in its original language.",
        )
    )

    payload = json.loads(json.dumps(original.to_dict(), ensure_ascii=False))
    restored = PaperEvidencePackage.from_mapping(payload)

    assert restored.to_dict() == original.to_dict()


def test_normalized_duplicate_chunks_keep_both_source_provenances() -> None:
    abstract_package = PaperEvidencePackage(
        paper_id="paper-1",
        paper_title="Paper",
        chunks=[
            PaperEvidenceChunk(
                paper_id="paper-1",
                paper_title="Paper",
                text="DeepConvLSTM uses PAMAP2.",
                section="abstract",
                location="Abstract",
                source_kind="abstract",
                chunk_type="abstract",
            )
        ],
        sources=[{"source_kind": "abstract", "provider": "openalex"}],
    )
    pdf_package = PaperEvidencePackage(
        paper_id="paper-1",
        paper_title="Paper",
        chunks=[
            PaperEvidenceChunk(
                paper_id="paper-1",
                paper_title="Paper",
                text="  DEEPCONVLSTM uses\nPAMAP2. ",
                section="abstract",
                page=1,
                location="PDF page 1",
                source_kind="pdf",
                chunk_type="paragraph",
            )
        ],
        sources=[{"source_kind": "pdf", "filename": "paper.pdf"}],
    )

    merged = merge_paper_evidence_packages(abstract_package, pdf_package)

    assert len(merged.chunks) == 1
    assert merged.chunks[0].source_kind == "abstract"
    assert merged.chunks[0].metadata["duplicate_sources"] == ["abstract", "pdf"]
    assert merged.chunks[0].metadata["duplicate_locations"] == [
        "Abstract",
        "PDF page 1",
    ]
    assert len(merged.sources) == 2


def test_missing_and_invalid_pdf_raise_clear_extraction_errors(tmp_path) -> None:
    missing = tmp_path / "missing.pdf"
    with pytest.raises(EvidenceExtractionError, match="does not exist"):
        pdf_to_evidence_chunks(missing, paper_id="paper")

    invalid = tmp_path / "invalid.pdf"
    invalid.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(EvidenceExtractionError, match="Could not open or extract"):
        pdf_to_evidence_chunks(invalid, paper_id="paper")


def test_pdf_package_and_two_paper_input_are_json_ready(tmp_path) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    _make_pdf(pdf_path)
    source = pdf_to_evidence_package(pdf_path, paper_id="source", paper_title="Source")
    target = discovered_paper_to_evidence_package(
        DiscoveredPaper(
            provider="openalex",
            provider_id="target",
            title="Target",
            abstract="Target abstract text.",
        )
    )

    combined = prepare_lineage_evidence_input(source, target)
    restored = LineageEvidenceInput.from_mapping(
        json.loads(json.dumps(combined.to_dict(), ensure_ascii=False))
    )

    assert restored.to_dict() == combined.to_dict()
    assert set(restored.to_dict()) == {"source", "target"}
    assert all("relation_type" not in chunk.to_dict() for chunk in source.chunks)
