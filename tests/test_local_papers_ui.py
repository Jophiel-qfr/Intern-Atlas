from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest

from intern_atlas.ui import get_local_papers_html


class _PageElements(HTMLParser):
    def __init__(self, html: str) -> None:
        super().__init__()
        self.elements: dict[str, tuple[str, dict[str, str | None]]] = {}
        self.order: list[str] = []
        self.feed(html)

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if element_id := attributes.get("id"):
            self.elements[element_id] = (tag, attributes)
            self.order.append(element_id)


def _local_script(html: str) -> str:
    return re.search(r"<script>(.*?)</script>", html, re.DOTALL).group(1)


@pytest.mark.parametrize("language,label", [("zh-CN", "展开全部分析证据"), ("en-US", "Expand all analysis evidence")])
def test_preview_details_default_closed_and_result_precedes_input(language, label) -> None:
    html = get_local_papers_html(language)
    page = _PageElements(html)
    tag, attrs = page.elements["lineagePreviewDetails"]
    assert tag == "details" and "open" not in attrs
    assert label in html
    assert page.order.index("lineageResultPanel") < page.order.index("lineagePreviewPanel")
    assert "hidden" in page.elements["lineageResultPanel"][1]
    assert "hidden" in page.elements["lineagePreviewPanel"][1]
    assert 'id="lineagePreviewCount"' in html


def test_preview_summary_has_counts_characters_and_all_coverage_categories() -> None:
    html = get_local_papers_html("zh-CN")
    assert "分析前检查（免费）" in html
    assert "预计输入正文字符：" in html
    assert "源论文证据覆盖：" in html and "目标论文证据覆盖：" in html
    assert "概述" in html and "实验/结果" in html and "缺失" in html
    script = _local_script(html)
    assert "data.source_evidence.length" in script and "data.target_evidence.length" in script
    assert "data.total_characters" in script
    for section in ("abstract", "introduction", "related_work", "background", "methods", "experiments", "results"):
        assert f"'{section}'" in script
    assert ".coverage-tag.missing" in html  # Localization must preserve CSS hooks.


def test_evidence_expansion_is_local_and_method_changes_use_their_own_evidence() -> None:
    script = _local_script(get_local_papers_html("zh-CN"))
    toggle = script.split("$('lineagePreviewDetails').addEventListener('toggle',", 1)[1].split("$('previewLineageBtn').addEventListener", 1)[0]
    assert "currentPreview.source_evidence" in toggle
    assert "currentPreview.target_evidence" in toggle
    assert not any(word in toggle for word in ("fetch(", "postLineage(", "requestJson("))
    assert "renderSupportingEvidence(change.evidence || [])" in script
    assert "groups.inherited_components" in script and "groups.added_components" in script
    assert "item.quote" in script and "item.evidence_type" in script
    assert "selection_score" not in script and "selection_reasons" not in script
    assert '<details class="evidence-details" open' not in script


def test_selection_changes_clear_old_preview_analysis_and_require_new_preview() -> None:
    script = _local_script(get_local_papers_html("en-US"))
    changed = script.split("function onPaperPairChanged() {", 1)[1].split("$('sourcePaperSelect').addEventListener", 1)[0]
    assert "lastPreviewKey = ''" in changed and "currentPreview = null" in changed
    assert "$('lineagePreviewEvidence').innerHTML = ''" in changed
    assert "$('lineageResult').innerHTML = ''" in changed
    assert "$('lineagePreviewPanel').hidden = true" in changed
    assert "$('lineageResultPanel').hidden = true" in changed
    assert "pairRevision += 1" in changed
    assert not any(word in changed for word in ("fetch(", "postLineage(", "requestJson("))
    assert "requestedRevision !== pairRevision" in script
    assert "window.confirm(" in script
    assert script.count("postLineage('/api/local/lineage/analyze')") == 1


def test_local_papers_ui_keeps_narrow_layout_and_original_evidence_escaping() -> None:
    html = get_local_papers_html("zh-CN")
    assert "@media (max-width: 620px)" in html
    assert "grid-template-columns: 1fr" in html and "overflow-wrap: anywhere" in html
    assert "escapeHtml(item.quote || '')" in html
    assert "escapeHtml(item.text)" in html
    assert "不确定性说明：" in html
    assert "查看为什么没有达到已确认" in html
    assert "未发现有充分证据支持的内容。" in html
