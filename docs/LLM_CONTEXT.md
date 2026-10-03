# Lineage evidence context

The evidence pipeline keeps three stages distinct:

- `PaperEvidenceChunk` is raw, locatable text from an abstract or local PDF.
- `SelectedEvidenceChunk` is a raw chunk chosen for a bounded model context. It
  adds a context-local `evidence_id`, paper role, selection score, and reasons.
- `analysis.EvidenceItem` is text selected after analysis to support a specific
  conclusion.

`build_lineage_llm_context` deterministically selects evidence for both papers.
It first tries to cover overview text (abstract/introduction/related work),
methods, and experiments/results for each paper, reserves up to six pairs of
lexically comparable source/target method or experiment paragraphs within the
same budgets, then fills remaining budgets by score. The default maximum is 28,000 characters total, 14,000 per paper, and
20 chunks per paper. Output is ordered by source then target and by each
package's document order. IDs such as `S001` and `T001` are stable for the same
input and policy and can be resolved with `get_evidence` or validated with
`validate_evidence_ids`.

Scores use section priority, citation context, and limited lexical signals.
They describe selection priority only; lexical overlap does not establish a
method relationship, and citation-context priority does not confirm method
inheritance. The context contains no relation type or method changes.

This context-selection module makes no LLM/API calls or method-relationship
judgments, and performs no token counting, embedding, or vector storage. The
application's separate online analysis layer is documented in
[LLM_LINEAGE_ANALYSIS.md](LLM_LINEAGE_ANALYSIS.md). Character limits are a lightweight
proxy for context size, and section coverage depends on available chunks and
the configured budgets.
