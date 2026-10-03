# Paper evidence input

`intern_atlas.evidence_input` converts existing `DiscoveredPaper` abstracts and
selectable text from local PDFs into page-aware `PaperEvidenceChunk` records.
Chunks retain the original text, paper identity, source kind, section, and
location. Long paragraphs are split at whitespace at a bounded character size.
Abstract and PDF packages can be combined; only whitespace-normalized exact
duplicates are collapsed, with duplicate source and location details retained.

`PaperEvidenceChunk` is raw text that has not yet been selected to support a
claim. `analysis.EvidenceItem` is for text selected later to support a specific
analysis conclusion. `LineageEvidenceInput` only pairs the raw packages for two
papers; it does not create a relation or method change.

Currently supported:

- Abstracts already present in provider-neutral `DiscoveredPaper` metadata.
- Selectable text from local PDFs through the project's existing PyMuPDF
  dependency, with one-based page numbers and lightweight English section
  heading recognition.
- JSON-ready packages and source/target evidence input.

Not currently supported:

- Automatic full-text download or source retrieval.
- OCR for scanned PDFs.
- LLM analysis or method-lineage judgments within this input module; the separate
  application analysis layer is described in [LLM_LINEAGE_ANALYSIS.md](LLM_LINEAGE_ANALYSIS.md).
- Embeddings or a vector database.
