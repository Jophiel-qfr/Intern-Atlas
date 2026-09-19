"""Small client for the Semantic Scholar Academic Graph API.

This integration only retrieves paper metadata and one level of references or
citations.  It never downloads PDFs, embeddings, or a complete citation graph.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from ..config import get_settings


PAPER_FIELDS = ",".join(
    (
        "paperId",
        "title",
        "abstract",
        "year",
        "authors",
        "venue",
        "externalIds",
        "citationCount",
        "influentialCitationCount",
        "publicationTypes",
        "url",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
    )
)
MAX_RETRIES = 2
BASE_RETRY_DELAY_SECONDS = 1.0
MAX_RETRY_AFTER_SECONDS = 30.0


def relation_fields(paper_key: str) -> str:
    # The relation endpoint wraps each paper in ``citedPaper`` or
    # ``citingPaper`` itself.  Its ``fields`` parameter uses the paper fields
    # without that response-wrapper prefix.
    del paper_key
    return f"contexts,intents,isInfluential,{PAPER_FIELDS}"


def retry_delay_seconds(response: httpx.Response, retry_index: int) -> float:
    """Return a bounded Retry-After delay or the local exponential fallback."""

    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay = retry_at.timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                delay = None
        if delay is not None:
            return max(0.0, min(delay, MAX_RETRY_AFTER_SECONDS))
    return min(BASE_RETRY_DELAY_SECONDS * (2**retry_index), MAX_RETRY_AFTER_SECONDS)


class SemanticScholarError(RuntimeError):
    """A safe, user-facing error from the Semantic Scholar integration."""

    def __init__(self, message: str, *, code: str = "semantic_scholar_error", status_code: int = 502):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class CachedResponse:
    payload: dict[str, Any]
    cached: bool


@dataclass
class SemanticScholarPaper:
    paper_id: str
    title: str = ""
    abstract: str = ""
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    external_ids: dict[str, str] = field(default_factory=dict)
    citation_count: int | None = None
    influential_citation_count: int | None = None
    publication_types: list[str] = field(default_factory=list)
    url: str = ""
    fields_of_study: list[str] = field(default_factory=list)
    s2_fields_of_study: list[dict[str, Any]] = field(default_factory=list)
    match_score: float | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any] | None) -> "SemanticScholarPaper":
        data = data or {}
        authors = data.get("authors") or []
        author_names: list[str] = []
        for author in authors if isinstance(authors, list) else []:
            if isinstance(author, dict):
                name = str(author.get("name") or "").strip()
            else:
                name = str(author or "").strip()
            if name:
                author_names.append(name)

        external_ids = data.get("externalIds") or {}
        if not isinstance(external_ids, dict):
            external_ids = {}
        normalized_external_ids = {
            str(key): str(value)
            for key, value in external_ids.items()
            if value is not None and str(value).strip()
        }

        def optional_int(value: Any) -> int | None:
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        def optional_float(value: Any) -> float | None:
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        fields = data.get("fieldsOfStudy") or []
        s2_fields = data.get("s2FieldsOfStudy") or []
        return cls(
            paper_id=str(data.get("paperId") or "").strip(),
            title=str(data.get("title") or "").strip(),
            abstract=str(data.get("abstract") or "").strip(),
            year=optional_int(data.get("year")),
            authors=author_names,
            venue=str(data.get("venue") or "").strip(),
            external_ids=normalized_external_ids,
            citation_count=optional_int(data.get("citationCount")),
            influential_citation_count=optional_int(data.get("influentialCitationCount")),
            publication_types=[str(item) for item in fields_or_empty(data.get("publicationTypes"))],
            url=str(data.get("url") or "").strip(),
            fields_of_study=[str(item) for item in fields_or_empty(fields)],
            s2_fields_of_study=[item for item in s2_fields if isinstance(item, dict)],
            match_score=optional_float(
                data.get("matchScore") if data.get("matchScore") is not None else data.get("score")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "authors": self.authors,
            "venue": self.venue,
            "external_ids": self.external_ids,
            "citation_count": self.citation_count,
            "influential_citation_count": self.influential_citation_count,
            "publication_types": self.publication_types,
            "url": self.url,
            "fields_of_study": self.fields_of_study,
            "s2_fields_of_study": self.s2_fields_of_study,
            "match_score": self.match_score,
        }


def fields_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


class SemanticScholarClient:
    """Bounded Academic Graph API client with JSON response caching."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        cache_ttl_seconds: int | None = None,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.semantic_scholar_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.semantic_scholar_api_key
        self.cache_dir = Path(cache_dir or settings.cache_dir) / "semantic_scholar"
        self.cache_ttl_seconds = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else settings.semantic_scholar_cache_ttl_seconds
        )
        self._client = httpx.Client(timeout=timeout_seconds, transport=transport)

    def close(self) -> None:
        self._client.close()

    def get_paper(self, identifier: str) -> tuple[SemanticScholarPaper, bool]:
        encoded = quote(identifier.strip(), safe="")
        result = self._request_json(f"/paper/{encoded}", {"fields": PAPER_FIELDS})
        paper = SemanticScholarPaper.from_api(result.payload)
        if not paper.paper_id:
            raise SemanticScholarError("Semantic Scholar returned a paper without paperId.", code="invalid_response")
        return paper, result.cached

    def search_papers(self, query: str, *, limit: int = 5) -> tuple[list[SemanticScholarPaper], bool]:
        result = self._request_json(
            "/paper/search",
            {"query": query, "limit": max(1, min(limit, 50)), "fields": PAPER_FIELDS},
        )
        rows = result.payload.get("data") or []
        if not isinstance(rows, list):
            raise SemanticScholarError("Semantic Scholar returned an invalid search response.", code="invalid_response")
        papers = [SemanticScholarPaper.from_api(row) for row in rows if isinstance(row, dict)]
        return [paper for paper in papers if paper.paper_id], result.cached

    def references(self, paper_id: str, *, limit: int = 30) -> tuple[list[dict[str, Any]], bool]:
        return self._relation_page(paper_id, "references", "citedPaper", limit=limit)

    def citations(self, paper_id: str, *, limit: int = 30) -> tuple[list[dict[str, Any]], bool]:
        return self._relation_page(paper_id, "citations", "citingPaper", limit=limit)

    def _relation_page(
        self,
        paper_id: str,
        relation: str,
        paper_key: str,
        *,
        limit: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        encoded = quote(paper_id.strip(), safe="")
        result = self._request_json(
            f"/paper/{encoded}/{relation}",
            {"limit": max(1, min(limit, 50)), "fields": relation_fields(paper_key)},
        )
        rows = result.payload.get("data") or []
        if not isinstance(rows, list):
            raise SemanticScholarError(
                f"Semantic Scholar returned an invalid {relation} response.",
                code="invalid_response",
            )
        safe_limit = max(1, min(limit, 50))
        return [
            {**row, "_paper_key": paper_key}
            for row in rows
            if isinstance(row, dict) and isinstance(row.get(paper_key), dict)
        ][:safe_limit], result.cached

    def _request_json(self, path: str, params: dict[str, Any]) -> CachedResponse:
        cache_path = self._cache_path(path, params)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return CachedResponse(cached, True)

        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        for retry_index in range(MAX_RETRIES + 1):
            try:
                response = self._client.get(self.base_url + path, params=params, headers=headers)
            except httpx.TimeoutException as exc:
                raise SemanticScholarError(
                    "Semantic Scholar 请求超时，请稍后重试。",
                    code="timeout",
                    status_code=504,
                ) from exc
            except httpx.RequestError as exc:
                raise SemanticScholarError(
                    "无法连接 Semantic Scholar，请检查网络连接。",
                    code="network_error",
                    status_code=502,
                ) from exc

            retryable = response.status_code == 429 or 500 <= response.status_code < 600
            if retryable and retry_index < MAX_RETRIES:
                time.sleep(retry_delay_seconds(response, retry_index))
                continue
            break

        if response.status_code == 404:
            raise SemanticScholarError(
                "Semantic Scholar 未找到对应论文。",
                code="not_found",
                status_code=404,
            )
        if response.status_code == 429:
            raise SemanticScholarError(
                "Semantic Scholar 请求受限，请稍后重试或配置 SEMANTIC_SCHOLAR_API_KEY。",
                code="rate_limited",
                status_code=429,
            )
        if response.status_code >= 500:
            raise SemanticScholarError(
                "Semantic Scholar 服务暂时不可用，请稍后重试。",
                code="upstream_error",
                status_code=502,
            )
        if response.status_code >= 400:
            raise SemanticScholarError(
                f"Semantic Scholar 请求失败（HTTP {response.status_code}）。",
                code="request_error",
                status_code=502,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise SemanticScholarError(
                "Semantic Scholar 返回了无法解析的 JSON。",
                code="invalid_response",
                status_code=502,
            ) from exc
        if not isinstance(payload, dict):
            raise SemanticScholarError("Semantic Scholar 返回格式无效。", code="invalid_response")
        self._write_cache(cache_path, payload)
        return CachedResponse(payload, False)

    def _cache_path(self, path: str, params: dict[str, Any]) -> Path:
        cache_key = json.dumps(
            {
                "base_url": self.base_url.rstrip("/").lower(),
                "path": path,
                "params": params,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, path: Path) -> dict[str, Any] | None:
        if self.cache_ttl_seconds <= 0 or not path.exists():
            return None
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            cached_at = float(cached.get("cached_at", 0))
            payload = cached.get("payload")
            if time.time() - cached_at > self.cache_ttl_seconds or not isinstance(payload, dict):
                return None
            return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _write_cache(self, path: Path, payload: dict[str, Any]) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"cached_at": time.time(), "payload": payload}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            # A read-only cache must not make discovery fail.
            return
