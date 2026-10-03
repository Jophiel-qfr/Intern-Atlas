# HAR Method Atlas 当前开发状态

当前开发分支：`feature/method-lineage-analysis`。

## 已完成阶段

- 第一阶段：集中配置、中文文案、HARAnalysis 和代码/运行数据分离。
- 第二阶段：OpenAlex / Semantic Scholar 一层论文发现与保守候选排序。
- 3.1–3.4：方法继承数据契约、原始证据输入、有限上下文、证据约束 LLM 客户端与本地验证。
- 3.5：已有真实 smoke test 的兼容性修复；本次收口不重复真实调用。
- 4.1–4.4：本地 PDF、免费预览、跨论文证据配对、方法演化网页与按需展开证据。
- 4.5：成功结果自动保存、多版本历史和 JSON / Markdown 导出。
- 5.1：功能一致性检查、使用文档、安全检查和离线回归；不添加新算法。

## 当前稳定功能

| 能力 | 实际实现与边界 |
| --- | --- |
| 论文发现 | 首页及 `/api/v1/discovery/*`；标题、DOI、来源 ID；歧义标题需选择；只找一层 references/citations，默认各 30 条，最多各 50 条 |
| 候选排序 | `method_candidate` / `uncertain` / `background_candidate`，通用词降权，Review/Survey 等不直接升级为方法候选；不是继承判断 |
| 原始证据 | 统一 abstract 和本地可选文本 PDF；PaperEvidenceChunk / PaperEvidencePackage 可 JSON 往返；保留页码、章节和来源，References/Bibliography 标题之后停止提取 |
| 证据选择 | 先兼顾概述、方法、实验，再保留跨论文可比较段落；确定性 lexical 规则；每篇最多 20 块/14,000 字符，总计 28,000 字符；Sxxx/Txxx 按最终文档顺序编号 |
| 在线分析接口 | httpx 的 OpenAI-compatible client；可选 thinking、reasoning_effort、JSON mode；输出压缩要求；网页仍为 max_tokens=4000；仅 429/5xx 最多重试 2 次；安全错误及截断诊断 |
| 本地约束 | 模型只引用证据 ID，quote 由程序从原始上下文重建；非法 ID 拒绝；citation/metadata 不能单独确认继承；缺少必要比较证据可降为 uncertain；支持 insufficient_evidence |
| 分析内容 | inherited/changed/added/removed、problem addressed、claimed contribution、experimental evidence、limitations、method_changes；单篇 HARAnalysis 可选，不要求先完成它才能比较两篇 |
| 网页 | `/local-papers`；免费 preview 和覆盖概览；详细输入默认折叠；费用确认；结论优先与证据按需展开 |
| 保存与导出 | 配置数据目录下 `analyses/`；成功校验后保存；失败不伪装成功历史；多版本、轻量列表、复用结果展示、UTF-8 JSON、只含最终引用证据的 Markdown；读取历史不调用模型或 PDF 解析 |

数据默认位于仓库外。未新增基础设施、数据库迁移或第三方依赖；原 Intern Atlas 图构建与 API 保留。

## 文档一致性说明

- 未发现当前指南承诺了代码完全不存在的功能。旧证据文档里的“无 LLM/关系判断”描述的是证据输入/选择模块，完整应用已通过独立 lineage LLM 层提供在线分析；已明确模块边界。
- 原 README 的“at most 30”应理解为默认请求规模；实际接口允许每侧最多 50 条，此处和用户指南给出准确限制。保留原 README 内容，本轮仅增加 HAR 入口。
- 最终 EvidenceItem 没有单独的 `page` 字段；PDF 页码保留在 `location` 中，原始/选中证据块仍有 `page`，已修正文档措辞。
- `serve` 要求已有本项目数据库；用户指南包含不调用 LLM 的首次初始化方法。

## 收口发现与最小修复

畸形 LLM 响应中 `choices[0]` 为字符串、数组或 null 时，原先会抛出未处理的 AttributeError。已补充格式校验并归入现有安全 malformed-response 错误，增加三种 mock 回归用例；不改 prompt、selector、schema 或正常请求流程。

## 已知限制

1. 无 OCR，扫描 PDF 需要用户先获得可提取文本。
2. section detection 仍是轻量英文标题规则；排版、双栏、表格和非标准标题可能影响顺序或分类。
3. evidence retrieval 是 lexical / deterministic，不是 semantic embeddings；预算可能遗漏重要段落，分数不是继承置信度。
4. LLM 结果质量仍受模型 / Gateway 影响；兼容选项取决于服务支持，输出压缩是 prompt 要求，不是 schema 的硬长度约束。
5. 目前主要针对英文论文 PDF，正文 quote 不翻译。
6. 暂无批量自动分析或自动全文下载。
7. 暂无从 saved analyses 生成的完整可视化 lineage graph；原 Intern Atlas 图页面仍保留。
8. saved history 尚无版本 diff，也没有历史删除管理界面。
9. PDF metadata 可能不可靠，已有 filename stem fallback，但不是完美标题识别。
10. local evidence validation 检查结构、来源和证据角色，不能证明所有模型结论的语义正确，也不提供经校准的置信概率。

## 回归与合并条件

完整回归包含发现客户端与候选排序、PDF/章节识别、证据预算与配对、schema/JSON、mock LLM payload/错误、本地验证、Web API、历史与导出、费用保护。测试不依赖真实 Key 或 Gateway。

```powershell
python -m pytest -q
```

5.1 收口回归：**215 passed，79 warnings**。warnings 为现有 FastAPI/Starlette/httpx 兼容层与 PyMuPDF 弃用提示，无测试失败。

本次仅检查本地 Git 状态：`har-method-atlas` 为当前分支的祖先，无目标分支分叉。本轮未联网确认远端是否有更新，未执行提交、推送或合并。

合并前应满足：全部测试通过；无密钥或运行数据进入待提交文件；只提交审阅过的代码与文档；检查目标 `har-method-atlas` 的最新状态。已有弃用 warning 与测试失败分别处理。本阶段不 commit、push 或执行合并。

## 下一阶段可选项

- 以人工核验的论文对评估结论质量和证据覆盖。
- 历史版本差异、标注与管理。
- 从已保存分析生成方法演化图。
- Zotero 导入、辅助阅读或人工确认流程。

这些是候选方向，当前未实现；批量在线分析应另行明确费用和范围。

使用方法见 [中文用户指南](HAR_METHOD_ATLAS_USER_GUIDE.md)。
