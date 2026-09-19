# 当前架构

Intern Atlas 仍然是一个轻量 Python 包。输入、分析、存储和 HTTP 层保持现有边界，新增能力通过薄模块接入。

## 核心模块

- `io.py`：读取 TXT、JSON、JSONL、CSV 和 PDF，统一生成 `PaperRecord`。
- `builder.py`：执行启发式或 LLM 方法抽取，并复用现有候选关系逻辑。
- `models.py`：论文、方法提及和关系边的数据容器。
- `util.py`：方法关系枚举、ID 和文本规范化；现有关系值继续在这里维护。
- `db.py`：SQLite schema、图写入、JSON 导出和查询辅助。
- `server.py`：FastAPI 本地 API、版本化证据接口和远程 API 代理。
- `ui.py`：无需 Node.js 的嵌入式网页；固定文案由 `i18n.py` 渲染。
- `llm.py`：OpenAI-compatible chat client。API key 只从环境变量读取。
- `remote.py`：托管 Intern Atlas API client。
- `config.py`：集中解析数据目录、数据库路径、API key、远程地址、CORS 和界面语言。
- `analysis.py`：HAR 分析 schema。当前只保存结构，不执行批量 LLM 分析。
- `integrations/semantic_scholar.py`：Semantic Scholar Academic Graph API client、轻量 JSON 缓存和 API 错误处理。
- `discovery.py`：目标论文识别、一层 references/citations 编排和透明候选排序；不判断方法继承关系。

## 数据与配置

默认运行数据写到仓库旁的 `data/`、`cache/` 和 `logs/`，不写入 `app`。可用以下环境变量覆盖：

`INTERN_ATLAS_DATA_DIR`、`INTERN_ATLAS_CACHE_DIR`、`INTERN_ATLAS_LOG_DIR`、`INTERN_ATLAS_DB_PATH`、`INTERN_ATLAS_UI_LANGUAGE`。

Semantic Scholar 使用 `SEMANTIC_SCHOLAR_BASE_URL`、可选的
`SEMANTIC_SCHOLAR_API_KEY` 和 `SEMANTIC_SCHOLAR_CACHE_TTL_SECONDS`。缓存写入
`cache/semantic_scholar/`，只保存 JSON API 响应，不保存 API key、PDF 或 embedding。

LLM 和托管 API 继续使用现有的 `S4S_LLM_*`、`OPENAI_*`、`INTERN_ATLAS_REMOTE_*` 变量。CLI 的 `--out` 仍可覆盖默认数据库路径。

## 扩展规则

- 新的论文分析放在 `analysis.py` 或其后续独立模块中，不把 HAR 逻辑塞入 `builder.py` 的通用关系流程。
- 新的 Semantic Scholar、Zotero 等来源放在 `integrations/` 边界内，并转换成现有 `PaperRecord`；不要复制一套数据库关系系统。
- Semantic Scholar discovery 只获取目标论文的一层 references 和 citations，默认各 30 条；候选排序是阅读优先级提示，不是方法继承结论。
- 新业务流程放到 service 层前，先确认是否能复用 `builder.py`、`db.py` 和现有 API。
- UI 新文案先加入 `i18n.py`；论文标题、作者、模型名、数据集名、缩写和引用原文保留原文。
- 新字段应允许缺失，并尽量通过 schema 的可选字段或 `extra` 扩展，避免修改所有旧数据。
