# OpenAlex 论文发现

论文发现接口默认使用 OpenAlex，也可以在请求中指定 `semantic_scholar`：

```json
{
  "query": "DeepConvLSTM: A Deep Convolutional LSTM Network for Activity Recognition Using Wearables",
  "provider": "openalex"
}
```

OpenAlex 支持标题、DOI、OpenAlex Work ID 和完整 OpenAlex URL。基础查询不需要 API key；如需配置，可使用 `OPENALEX_API_KEY`。缓存写入配置的 `cache_dir/openalex/`，不会保存 API key。

标题搜索最多返回 5 个候选。只有规范化标题明确一致时才自动选择，否则由用户选择目标论文。References 使用 `referenced_works` 的一次批量 Work 查询，Citations 使用 `filter=cites:<Work ID>`，两者都只获取第一层，默认各 30 条、最多 50 条。

OpenAlex 和 Semantic Scholar 的结果都会转换为共用的 `DiscoveredPaper`，因此 HAR 候选排序不依赖具体数据源。排序标签只表示阅读优先级，不表示方法继承关系。
