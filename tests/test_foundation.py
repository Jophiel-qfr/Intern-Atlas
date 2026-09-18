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


def test_ui_defaults_to_chinese_fixed_labels():
    html = get_index_html("zh-CN")

    assert '<html lang="zh-CN">' in html
    assert "证据工作区" in html
    assert 'id="methodFilter"' in html
    assert 'id="downloadPapersBtn"' in html
    assert 'id="downloadEdgesBtn"' in html
    assert "downloadPapersCsv" in html
