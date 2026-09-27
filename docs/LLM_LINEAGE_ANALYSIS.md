# Evidence-constrained lineage analysis

The analysis path keeps source text, model selection, model claims, and final
evidence records separate:

1. `PaperEvidenceChunk` holds raw, locatable paper text.
2. `SelectedEvidenceChunk` is sent in a bounded `LineageLLMContext` with a
   context-local ID such as `S001` or `T001`.
3. `LineageLLMResponse` accepts structured claims and evidence IDs only. The
   parser rejects unknown fields, including model-supplied `quote` fields, and
   verifies every ID against the exact context.
4. Conversion resolves IDs back to selected chunks and creates `EvidenceItem`
   records. `EvidenceItem.quote`, paper identity, section, page, location, and
   source kind come from the context, never from model output.
5. The existing `MethodLineageAnalysis` performs its local validation on the
   converted result.

The prompt requires JSON only, forbids external knowledge and invented IDs,
distinguishes inherited/modified/replaced/added/removed changes, and defines the
allowed relation values. Citation context by itself cannot establish method
inheritance. Confirmed claims based only on citation context or metadata are
downgraded to `uncertain`; missing source comparison evidence for a replacement
or modification also downgrades the result. A target change with no target
evidence is rejected.

`insufficient_evidence` with a null relation is a normal successful result.
There is no automatic relation selection when evidence is insufficient. The
OpenAI-compatible client uses `httpx`, reads `LLM_BASE_URL`, `LLM_API_KEY`,
`LLM_MODEL`, and `LLM_TIMEOUT_SECONDS` (with existing project aliases), and
does not cache results. Tests use mock transports only; this code does not
require a real key or call an API during tests.
