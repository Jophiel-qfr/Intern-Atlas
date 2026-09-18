from pathlib import Path

from intern_atlas.analysis import HARAnalysis
from intern_atlas.config import get_settings
from intern_atlas.ui import get_index_html


def test_har_analysis_schema_keeps_optional_and_unknown_fields():
    analysis = HARAnalysis.from_mapping(
        {
            "task": "human activity recognition",
            "sensor_modality": ["accelerometer", "gyroscope"],
            "base_model": "DeepConvLSTM",
            "future_dimension": {"status": "unreviewed"},
        }
    )

    payload = analysis.to_dict()
    assert payload["task"] == "human activity recognition"
    assert payload["sensor_modality"] == ["accelerometer", "gyroscope"]
    assert payload["future_dimension"] == {"status": "unreviewed"}
    assert "loss_function" not in payload


def test_settings_default_runtime_data_is_outside_app(tmp_path, monkeypatch):
    for name in (
        "INTERN_ATLAS_DATA_DIR",
        "INTERN_ATLAS_CACHE_DIR",
        "INTERN_ATLAS_LOG_DIR",
        "INTERN_ATLAS_DB_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = get_settings(tmp_path)

    assert settings.database_path == tmp_path / "data" / "local_method_graph.db"
    assert settings.cache_dir == tmp_path / "cache"
    assert settings.log_dir == tmp_path / "logs"


def test_settings_default_paths_follow_source_checkout(monkeypatch, tmp_path):
    for name in (
        "INTERN_ATLAS_DATA_DIR",
        "INTERN_ATLAS_CACHE_DIR",
        "INTERN_ATLAS_LOG_DIR",
        "INTERN_ATLAS_DB_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

    app_root = Path(__file__).resolve().parents[1]
    project_root = app_root.parent
    settings = get_settings()

    assert settings.data_dir == project_root / "data"
    assert settings.cache_dir == project_root / "cache"
    assert settings.log_dir == project_root / "logs"
    assert settings.database_path == project_root / "data" / "local_method_graph.db"


def test_settings_paths_respect_environment_overrides(monkeypatch, tmp_path):
    data_dir = tmp_path / "custom-data"
    cache_dir = tmp_path / "custom-cache"
    log_dir = tmp_path / "custom-logs"
    database_path = tmp_path / "custom.db"
    monkeypatch.setenv("INTERN_ATLAS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("INTERN_ATLAS_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("INTERN_ATLAS_LOG_DIR", str(log_dir))
    monkeypatch.setenv("INTERN_ATLAS_DB_PATH", str(database_path))

    settings = get_settings()

    assert settings.data_dir == data_dir
    assert settings.cache_dir == cache_dir
    assert settings.log_dir == log_dir
    assert settings.database_path == database_path


def test_ui_defaults_to_chinese_fixed_labels():
    html = get_index_html("zh-CN")

    assert '<html lang="zh-CN">' in html
    assert "证据工作区" in html
    assert 'id="methodFilter"' in html
    assert 'id="downloadPapersBtn"' in html
    assert 'id="downloadEdgesBtn"' in html
    assert "downloadPapersCsv" in html
