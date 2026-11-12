import numpy as np
import pandas as pd
import pytest

from tailslip import evaluate as E


def test_ece_zero_for_perfect_bins():
    y = np.array([0] * 45 + [1] * 5 + [0] * 5 + [1] * 45)
    p = np.repeat([0.1, 0.9], 50)
    assert E.ece(y, p, bins=2) == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_overconfidence():
    y = np.array([0, 1] * 50)
    assert E.ece(y, np.full(100, 0.9), bins=1) == pytest.approx(0.4)


def test_score_block_keys_and_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.6, 0.01, 0.99)
    s = E.score_block(y, p)
    assert 0.5 < s["auc"] <= 1 and set(s) >= {"auc", "log_loss", "brier", "ece"}


def test_roc_auc_single_class_is_nan():
    assert np.isnan(E.roc_auc(np.ones(5), np.arange(5)))


def test_day_blocks_keep_days_together():
    dates = pd.Series(pd.to_datetime(["2013-01-01"] * 3 + ["2013-01-02"] * 2 + ["2013-01-03"]))
    blocks = E.DayBlocks.from_dates(dates)
    assert sorted(len(g) for g in blocks.groups) == [1, 2, 3]
    s = blocks.sample(np.random.default_rng(0))
    assert set(s) <= set(range(6))


def test_bootstrap_interval_brackets_point_estimate():
    rng = np.random.default_rng(1)
    dates = pd.Series(np.repeat(pd.date_range("2013-01-01", periods=40), 25))
    y = rng.integers(0, 2, 1000)
    p = np.clip(0.5 + (y - 0.5) * 0.4 + rng.normal(0, 0.3, 1000), 0.01, 0.99)
    blocks = E.DayBlocks.from_dates(dates)
    lo, hi = E.bootstrap_ci(blocks, lambda i: E.roc_auc(y[i], p[i]), reps=100)
    assert lo < E.roc_auc(y, p) < hi


def test_pinball_matches_hand_calculation():
    assert E.pinball(np.array([10.0, 0.0]), np.array([5.0, 5.0]), 0.8) == pytest.approx((0.8 * 5 + 0.2 * 5) / 2)


def test_alert_value_arithmetic():
    y = np.array([1, 1, 0, 0])
    flag = np.array([1, 0, 1, 0])
    assert E.alert_value(y, flag, value=60, cost=25) == pytest.approx((60 - 25) / 4)
    assert E.alert_value(y, np.zeros(4), 60, 25) == 0.0


def test_best_threshold_prefers_break_even_for_calibrated_scores():
    rng = np.random.default_rng(2)
    p = rng.random(60000)
    y = (rng.random(60000) < p).astype(int)
    thr, val, _, _ = E.best_threshold(y, p)
    assert abs(thr - 25 / 85) < 0.06 and val > 0


def test_alert_table_counts():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.7])
    t = E.alert_table(y, p, 0.5)
    assert t["alert_rate"] == pytest.approx(0.6) and t["precision"] == pytest.approx(2 / 3)
    assert t["recall"] == pytest.approx(2 / 3)
