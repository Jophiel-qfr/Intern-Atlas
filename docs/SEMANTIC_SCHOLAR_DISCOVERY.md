# Semantic Scholar 论文发现

本阶段提供轻量的一层论文发现流程：

```text
标题 / DOI / arXiv / Semantic Scholar paperId
  -> 目标论文识别
  -> 最多 30 条 references + 最多 30 条 citations
  -> citation metadata 启发式排序
  -> 用户选择候选，导出 JSON
```

接口：

- `POST /api/v1/discovery/resolve`：识别目标论文，并返回匹配结果。
- `POST /api/v1/discovery/lineage`：返回目标论文、一层前置/后续候选和排序理由。

候选标签只有：`method_candidate`、`uncertain`、`background_candidate`。这些标签用于安排阅读顺序，不代表 `extends`、`improves` 或其他方法关系。

Semantic Scholar client 位于 `src/intern_atlas/integrations/semantic_scholar.py`，发现编排和 ranking 位于 `src/intern_atlas/discovery.py`。响应缓存位于配置的 `cache_dir/semantic_scholar/`，默认 TTL 为 7 天。请求可以使用 `SEMANTIC_SCHOLAR_API_KEY`，client 会发送 `x-api-key` header。

测试使用 `httpx.MockTransport`，不会访问真实 Semantic Scholar，也不会下载 PDF、embedding 或模型。
