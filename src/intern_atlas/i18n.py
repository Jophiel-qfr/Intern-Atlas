"""Small dependency-free UI language layer.

The page remains English-keyed internally so IDs, API names, model names, and
relationship values stay stable.  User-facing fixed labels are translated at
render time and can be switched back to English with ``INTERN_ATLAS_UI_LANGUAGE``.
"""

from __future__ import annotations

import re

from .config import get_settings


UI_TRANSLATIONS = {
    "zh-CN": {
        "Evidence layer for research agents": "面向科研智能体的证据层",
        "Papers": "论文",
        "Methods": "方法",
        "Edges": "关系边",
        "Research query": "研究问题",
        "Retrieval mode": "检索模式",
        "Light": "轻量",
        "Balanced": "平衡",
        "Deep": "深入",
        "Data source": "数据来源",
        "Local graph": "本地图谱",
        "Hosted API": "托管 API",
        "Hosted base URL": "托管服务地址",
        "Hosted API key": "托管 API Key",
        "Check hosted API": "检查托管 API",
        "Year from": "起始年份",
        "Year to": "结束年份",
        "Edge type": "关系类型",
        "Any edge": "任意关系",
        "Method filter": "方法筛选",
        "Depth": "图深度",
        "Include prompt-ready context": "包含可直接提供给模型的上下文",
        "Run evidence search": "开始证据检索",
        "Reset": "重置",
        "Copy context": "复制上下文",
        "API docs": "API 文档",
        "Evidence Workspace": "证据工作区",
        "Build a query-specific evidence pack, inspect method evolution, and export data for downstream LLM or agent workflows.": "构建与问题相关的证据包，查看方法演化，并导出供下游 LLM 或智能体使用的数据。",
        "Evidence papers": "证据论文",
        "Method edges": "方法关系边",
        "Bottlenecks": "瓶颈",
        "Mechanisms": "机制",
        "Mode applied": "当前模式",
        "Method Evolution Graph": "方法演化图",
        "Open neighborhood": "打开邻域",
        "Run an evidence search to draw a graph.": "开始证据检索后显示图谱。",
        "Searching evidence...": "正在检索证据……",
        "No paper selected": "未选择论文",
        "Select a node or paper row to inspect a local neighborhood.": "选择节点或论文行以查看局部邻域。",
        "Evidence Papers": "证据论文",
        "No papers loaded yet.": "尚未加载论文。",
        "Downloads": "下载",
        "Export the current evidence view": "导出当前证据视图",
        "Evidence JSON": "证据 JSON",
        "Papers CSV": "论文 CSV",
        "Edges CSV": "关系边 CSV",
        "Context MD": "上下文 MD",
        "Timeline": "时间线",
        "Timeline appears after search.": "检索后显示时间线。",
        "Bottlenecks and Mechanisms": "瓶颈与机制",
        "No bottlenecks or mechanisms loaded.": "尚未加载瓶颈或机制。",
        "Method Edges": "方法关系边",
        "No edges loaded yet.": "尚未加载关系边。",
        "Paper Discovery": "论文发现",
        "Find one paper and inspect limited first-level citation candidates": "识别一篇论文，并查看有限的一层引用候选",
        "Paper title / DOI / arXiv / Semantic Scholar ID": "论文标题 / DOI / arXiv / Semantic Scholar ID",
        "Find paper": "查找论文",
        "Find references and citations": "查找前置与后续论文",
        "Enter a title, DOI, arXiv ID, or Semantic Scholar paperId.": "请输入论文标题、DOI、arXiv ID 或 Semantic Scholar paperId。",
        "References": "前置论文",
        "Citations": "后续论文",
        "Resolve a paper to see references.": "识别论文后查看前置论文。",
        "Resolve a paper to see citations.": "识别论文后查看后续论文。",
        "Export selected candidates": "导出所选候选",
        "Method candidate": "方法相关候选",
        "Uncertain": "待判断",
        "Background candidate": "偏背景引用",
        "View abstract": "查看摘要",
        "Selected paper candidate": "候选识别结果",
        "Review the matches before continuing.": "请先检查匹配结果，再继续查找引用关系。",
        "Review the matches and select one paper before continuing.": "请检查匹配结果并选择一篇论文，然后继续查找引用关系。",
        "Paper selected. You can now load references and citations.": "已选择论文，现在可以查找前置与后续论文。",
        "match:": "匹配度：",
        "Review matches and select a paper before loading lineage.": "请检查匹配结果并选择一篇论文，然后加载方法演化关系。",
        "No matching paper found.": "没有找到匹配论文。",
        "Loading paper metadata...": "正在加载论文信息……",
        "Loading references and citations...": "正在加载前置与后续论文……",
        "Discovery failed": "论文发现失败",
        "Selected candidates exported": "已导出所选候选",
        "Paper metadata loaded from cache.": "论文信息已从缓存加载。",
        "Paper metadata loaded.": "论文信息已加载。",
        "References and citations loaded from cache.": "前置与后续论文已从缓存加载。",
        "References and citations loaded.": "前置与后续论文已加载。",
        "No candidates returned.": "没有返回候选论文。",
        "Multiple matches returned; review the candidate metadata.": "返回了多个匹配结果，请检查候选论文信息。",
        "Semantic Scholar": "Semantic Scholar",
        "citations:": "引用次数：",
    },
    "en-US": {},
}


def localize_ui_html(template: str, language: str | None = None) -> str:
    language = language or get_settings().ui_language
    translations = UI_TRANSLATIONS.get(language, UI_TRANSLATIONS["en-US"])
    html = template.replace('<html lang="en">', f'<html lang="{language}">')
    for source, target in sorted(translations.items(), key=lambda item: len(item[0]), reverse=True):
        # Match visible words and phrases only.  A plain replacement would turn
        # identifiers such as ``downloadPapersBtn`` into invalid HTML/JavaScript.
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(source)}(?![A-Za-z0-9_])"
        html = re.sub(pattern, target, html)
    return html
