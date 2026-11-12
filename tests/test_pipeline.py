import json

import pandas as pd

from tailslip import cli, pipeline, scoring


def test_quick_pipeline_writes_every_artifact(raw_dir, tmp_path):
    out, assets = tmp_path / "out", tmp_path / "assets"
    m = pipeline.run(raw_dir, out, assets, quick=True, log=lambda *_: None)
    for f in ("metrics.json", "tailslip_model.joblib", "test_replay.parquet"):
        assert (out / f).exists()
    for chart in (
        "lead_time",
        "delay_heatmap",
        "model_comparison",
        "calibration",
        "chain_effect",
        "economics",
        "quantile_coverage",
    ):
        assert (assets / f"{chart}.png").stat().st_size > 2000
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["selection"]["shipped"] in saved["models"]
    assert saved["streaming_check"]["max_abs_difference"] == 0
    assert saved["data"]["dropped_without_tail_number"] > 0
    assert 0.5 < saved["shipped_test"]["auc"] <= 1
    assert scoring.load_bundle(out / "tailslip_model.joblib")["features"]
    replay = pd.read_parquet(out / "test_replay.parquet")
    assert len(replay) == m["data"]["split_rows"]["test"]


def test_cli_show_prints_headline(raw_dir, tmp_path, capsys):
    out = tmp_path / "o"
    pipeline.run(raw_dir, out, tmp_path / "a", quick=True, log=lambda *_: None)
    assert cli.main(["show", "--out", str(out)]) == 0
    assert "shipped" in capsys.readouterr().out


def test_app_builds_from_saved_outputs(raw_dir, tmp_path):
    out = tmp_path / "o2"
    pipeline.run(raw_dir, out, tmp_path / "a2", quick=True, log=lambda *_: None)
    from tailslip import app

    demo = app.build_app(out / "tailslip_model.joblib", out / "test_replay.parquet")
    assert demo is not None
