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
   records. `EvidenceItem.quote`, paper identity, section, location, and source
   kind come from the context, never from model output. Final `EvidenceItem` has
   no separate page field: PDF page numbers are retained in `location`, with a
   page fallback when location is absent; raw/selected chunks retain `page`.
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

Optional `LLM_THINKING_MODE`, `LLM_REASONING_EFFORT`, and `LLM_JSON_MODE` control
provider-neutral request parameters only when configured. The Web analysis
uses `max_tokens=4000`; output compression is requested by the prompt, and
`finish_reason=length` is rejected before parsing. The client does not cache
provider output. The server separately persists validated results to the
configured `data_dir/analyses` for local history and JSON/Markdown export;
opening those saved results does not call the client or re-extract PDFs.
