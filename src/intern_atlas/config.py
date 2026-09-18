"""Centralized runtime configuration for the local builder and API.

The package deliberately reads configuration from environment variables instead
of embedding machine-specific paths.  Defaults place generated data beside the
``app`` repository when the project is run from this checkout.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_REMOTE_BASE_URL = "https://intern-atlas.opendatalab.org.cn/api"


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _split_models(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        decoded = json.loads(value)
        if isinstance(decoded, list):
            models = tuple(str(item).strip() for item in decoded if str(item).strip())
            if models:
                return models
    except json.JSONDecodeError:
        pass
    return tuple(item.strip() for item in value.split(",") if item.strip())


def project_root(start: str | Path | None = None) -> Path:
    """Return the repository root without depending on the current directory.

    In the source checkout, ``__file__`` resolves to
    ``app/src/intern_atlas/config.py``.  Starting there keeps defaults stable
    when a CLI command is launched from a different working directory.  The
    explicit ``start`` argument remains available for tests and callers that
    need to resolve another checkout.
    """

    candidate = (
        Path(start).resolve()
        if start is not None
        else Path(__file__).resolve().parents[2]
    )
    for path in (candidate, *candidate.parents):
        if (path / ".git").exists():
            return path
    return candidate


@dataclass(frozen=True)
class Settings:
    """Resolved application settings.

    Values are intentionally plain strings and paths so callers can use this
    object without introducing a settings framework or another dependency.
    """

    data_dir: Path
    cache_dir: Path
    log_dir: Path
    database_path: Path
    llm_base_url: str
    llm_api_key: str
    llm_models: tuple[str, ...]
    remote_base_url: str
    remote_api_key: str
    cors_origins: tuple[str, ...]
    ui_language: str


def get_settings(start: str | Path | None = None) -> Settings:
    root = project_root(start)
    sibling_root = root.parent if root.name.lower() == "app" else root
    data_dir = Path(_first_env("INTERN_ATLAS_DATA_DIR", default=str(sibling_root / "data"))).expanduser()
    cache_dir = Path(_first_env("INTERN_ATLAS_CACHE_DIR", default=str(sibling_root / "cache"))).expanduser()
    log_dir = Path(_first_env("INTERN_ATLAS_LOG_DIR", default=str(sibling_root / "logs"))).expanduser()
    database_path = Path(
        _first_env("INTERN_ATLAS_DB_PATH", default=str(data_dir / "local_method_graph.db"))
    ).expanduser()
    models = _split_models(
        _first_env("S4S_LLM_MODELS", "S4S_LLM_MODEL", "OPENAI_MODEL", default=DEFAULT_LLM_MODEL)
    ) or (DEFAULT_LLM_MODEL,)
    cors_origins = tuple(
        item.strip()
        for item in os.environ.get("INTERN_ATLAS_CORS_ORIGINS", "").split(",")
        if item.strip()
    )
    return Settings(
        data_dir=data_dir,
        cache_dir=cache_dir,
        log_dir=log_dir,
        database_path=database_path,
        llm_base_url=_first_env("S4S_LLM_BASE_URL", "OPENAI_BASE_URL", default=DEFAULT_LLM_BASE_URL).rstrip("/"),
        llm_api_key=_first_env("S4S_LLM_API_KEY", "OPENAI_API_KEY"),
        llm_models=models,
        remote_base_url=_first_env("INTERN_ATLAS_REMOTE_BASE_URL", default=DEFAULT_REMOTE_BASE_URL),
        remote_api_key=_first_env("INTERN_ATLAS_API_KEY", "INTERN_ATLAS_REMOTE_API_KEY"),
        cors_origins=cors_origins,
        ui_language=_first_env("INTERN_ATLAS_UI_LANGUAGE", default="zh-CN"),
    )
