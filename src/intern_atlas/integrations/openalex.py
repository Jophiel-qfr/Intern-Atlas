"""Small OpenAlex Works API client for one-hop paper discovery."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from ..config import get_settings
from ..discovery_models import DiscoveredPaper
from .semantic_scholar import MAX_RETRIES, SemanticScholarError, retry_delay_seconds


WORK_SELECT = ",".join(
    (
        "id",
        "display_name",
        "publication_year",
        "authorships",
        "doi",
        "primary_location",
        "cited_by_count",
        "type",
        "abstract_inverted_index",
        "referenced_works",
    )
)
RELATION_SELECT = ",".join(
    field for field in WORK_SELECT.split(",") if field != "referenced_works"
)
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
WORK_ID_RE = re.compile(r"^W\d+$", re.IGNORECASE)


class OpenAlexError(SemanticScholarError):
    """Structured, user-facing OpenAlex error."""

    def __init__(self, message: str, *, code: str = "openalex_error", status_code: int = 502):
        super().__init__(message, code=code, status_code=status_code)


def reconstruct_abstract(value: Any) -> str:
    """Rebuild an OpenAlex abstract inverted index without NLP dependencies."""

    if not isinstance(value, dict):
        return ""
    words: list[tuple[int, str]] = []
    try:
        for word, positions in value.items():
            if not isinstance(word, str) or not isinstance(positions, list):
                continue
            for position in positions:
                if isinstance(position, int) and position >= 0:
                    words.append((position, word))
    except (AttributeError, TypeError, ValueError):
        return ""
    words.sort(key=lambda item: item[0])
    return " ".join(word for _, word in words)


def identify_openalex_query(query: str) -> tuple[str, str]:
    """Classify a title, DOI, OpenAlex ID, or OpenAlex URL."""

    value = query.strip()
    doi_value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    doi_value = re.sub(r"^doi:\s*", "", doi_value, flags=re.IGNORECASE)
    if DOI_RE.fullmatch(doi_value):
        return "doi", f"https://doi.org/{doi_value}"

    url_match = re.fullmatch(r"https?://openalex\.org/(W\d+)/?", value, flags=re.IGNORECASE)
    if url_match:
        return "openalex_id", url_match.group(1).upper()
    if WORK_ID_RE.fullmatch(value):
        return "openalex_id", value.upper()
    return "title", value


def normalize_work_id(value: str) -> str:
    value = value.strip().rstrip("/")
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    return value.upper() if WORK_ID_RE.fullmatch(value) else ""


class OpenAlexClient:
    """Bounded OpenAlex client with JSON caching and no-key operation."""

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
        self.base_url = (base_url or settings.openalex_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.openalex_api_key
        self.cache_dir = Path(cache_dir or settings.cache_dir) / "openalex"
        self.cache_ttl_seconds = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else settings.openalex_cache_ttl_seconds
        )
        self._client = httpx.Client(timeout=timeout_seconds, transport=transport)

    def close(self) -> None:
        self._client.close()

    def get_work(self, identifier: str) -> tuple[DiscoveredPaper, bool]:
        encoded = quote(identifier.strip(), safe="")
        result = self._request_json(f"/works/{encoded}", {"select": WORK_SELECT})
        paper = work_from_api(result.payload)
        if not paper.provider_id:
            raise OpenAlexError("OpenAlex 返回的 Work 缺少 id。", code="invalid_response")
        return paper, result.cached

    def search_works(self, query: str, *, limit: int = 5) -> tuple[list[DiscoveredPaper], bool]:
        result = self._request_json(
            "/works",
            {"search": query, "per-page": max(1, min(limit, 50)), "select": WORK_SELECT},
        )
        rows = result.payload.get("results") or []
        if not isinstance(rows, list):
            raise OpenAlexError("OpenAlex 返回了无效的搜索结果。", code="invalid_response")
        papers = [work_from_api(row) for row in rows if isinstance(row, dict)]
        return [paper for paper in papers if paper.provider_id], result.cached

    def references(
        self,
        work_id: str,
        *,
        referenced_work_ids: list[str] | None = None,
        limit: int = 30,
    ) -> tuple[list[dict[str, Any]], bool]:
        safe_limit = max(1, min(limit, 50))
        ids = [normalize_work_id(value) for value in (referenced_work_ids or [])]
        ids = [value for value in ids if value]
        if referenced_work_ids is None:
            target, target_cached = self.get_work(work_id)
            ids = target.referenced_work_ids
        else:
            target_cached = True
        ids = ids[:safe_limit]
        if not ids:
            return [], target_cached
        rows, metadata_cached = self._works_by_ids(ids, limit=safe_limit)
        return [
            {"_paper": paper, "_paper_key": "openalexWork"}
            for paper in rows
            if paper.provider_id
        ], target_cached and metadata_cached

    def citations(self, work_id: str, *, limit: int = 30) -> tuple[list[dict[str, Any]], bool]:
        normalized_id = normalize_work_id(work_id)
        if not normalized_id:
            raise OpenAlexError("OpenAlex Work ID 无效。", code="invalid_identifier", status_code=422)
        safe_limit = max(1, min(limit, 50))
        result = self._request_json(
            "/works",
            {
                "filter": f"cites:{normalized_id}",
                "per-page": safe_limit,
                "select": RELATION_SELECT,
            },
        )
        rows = result.payload.get("results") or []
        if not isinstance(rows, list):
            raise OpenAlexError("OpenAlex 返回了无效的 citations 结果。", code="invalid_response")
        papers = [work_from_api(row) for row in rows if isinstance(row, dict)]
        return [
            {"_paper": paper, "_paper_key": "openalexWork"}
            for paper in papers
            if paper.provider_id
        ][:safe_limit], result.cached

    def _works_by_ids(self, work_ids: list[str], *, limit: int) -> tuple[list[DiscoveredPaper], bool]:
        result = self._request_json(
            "/works",
            {
                "filter": "openalex:" + "|".join(work_ids),
                "per-page": max(1, min(limit, 50)),
                "select": RELATION_SELECT,
            },
        )
        rows = result.payload.get("results") or []
        if not isinstance(rows, list):
            raise OpenAlexError("OpenAlex 返回了无效的 references 结果。", code="invalid_response")
        return [work_from_api(row) for row in rows if isinstance(row, dict)], result.cached

    def _request_json(self, path: str, params: dict[str, Any]) -> "CachedResponse":
        cache_path = self._cache_path(path, params)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return CachedResponse(cached, True)

        request_params = dict(params)
        if self.api_key:
            request_params["api_key"] = self.api_key
        headers = {"Accept": "application/json"}
        for retry_index in range(MAX_RETRIES + 1):
            try:
                response = self._client.get(
                    self.base_url + path,
                    params=request_params,
                    headers=headers,
                )
            except httpx.TimeoutException as exc:
                raise OpenAlexError(
                    "OpenAlex 请求超时，请稍后重试。", code="timeout", status_code=504
                ) from exc
            except httpx.RequestError as exc:
                raise OpenAlexError(
                    "无法连接 OpenAlex，请检查网络连接。", code="network_error", status_code=502
                ) from exc

            retryable = response.status_code == 429 or 500 <= response.status_code < 600
            if retryable and retry_index < MAX_RETRIES:
                time.sleep(retry_delay_seconds(response, retry_index))
                continue
            break

        if response.status_code == 404:
            raise OpenAlexError("OpenAlex 未找到对应 Work。", code="not_found", status_code=404)
        if response.status_code == 429:
            raise OpenAlexError("OpenAlex 请求受限，请稍后重试。", code="rate_limited", status_code=429)
        if response.status_code >= 500:
            raise OpenAlexError("OpenAlex 服务暂时不可用，请稍后重试。", code="upstream_error", status_code=502)
        if response.status_code >= 400:
            raise OpenAlexError(
                f"OpenAlex 请求失败（HTTP {response.status_code}）。",
                code="request_error",
                status_code=502,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenAlexError("OpenAlex 返回了无法解析的 JSON。", code="invalid_response") from exc
        if not isinstance(payload, dict):
            raise OpenAlexError("OpenAlex 返回格式无效。", code="invalid_response")
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
            return


class CachedResponse:
    def __init__(self, payload: dict[str, Any], cached: bool):
        self.payload = payload
        self.cached = cached


def work_from_api(data: dict[str, Any] | None) -> DiscoveredPaper:
    data = data or {}
    work_id = normalize_work_id(str(data.get("id") or ""))
    authors: list[str] = []
    authorships = data.get("authorships") or []
    if isinstance(authorships, list):
        for authorship in authorships:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author") or {}
            if isinstance(author, dict) and str(author.get("display_name") or "").strip():
                authors.append(str(author["display_name"]).strip())

    primary_location = data.get("primary_location") or {}
    if not isinstance(primary_location, dict):
        primary_location = {}
    source = primary_location.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    doi = str(data.get("doi") or "").strip()
    external_ids = {"DOI": doi} if doi else {}
    url = str(primary_location.get("landing_page_url") or data.get("id") or "").strip()
    referenced_works = data.get("referenced_works") or []
    referenced_work_ids = [
        normalized
        for value in referenced_works
        if (normalized := normalize_work_id(str(value)))
    ] if isinstance(referenced_works, list) else []

    def optional_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return DiscoveredPaper(
        provider="openalex",
        provider_id=work_id,
        title=str(data.get("display_name") or data.get("title") or "").strip(),
        abstract=reconstruct_abstract(data.get("abstract_inverted_index")),
        year=optional_int(data.get("publication_year")),
        authors=authors,
        venue=str(source.get("display_name") or "").strip(),
        external_ids=external_ids,
        citation_count=optional_int(data.get("cited_by_count")),
        url=url,
        publication_type=str(data.get("type") or "").strip(),
        referenced_work_ids=referenced_work_ids,
    )
