import numpy as np
import pandas as pd
import pytest

from tailslip import explain, models, scoring
from tailslip import features as F


@pytest.fixture(scope="module")
def bundle(flights, tables):
    X = F.build_features(flights, tables, 2.0)
    y = flights["delayed"].to_numpy()
    tr = (flights["month"] <= 9).to_numpy()
    model = models.make_boosting(F.ALL_FEATURES, max_iter=40).fit(X[tr], y[tr])
    d = flights["dep_delay"].to_numpy()
    ok = tr & ~np.isnan(d)
    qf = models.make_quantile_forest(F.ALL_FEATURES).fit(X[ok], d[ok])
    return (
        dict(
            model=model,
            calibrator=None,
            threshold=0.3,
            quantile_model=qf,
            quantiles=(0.5, 0.8, 0.9),
            reference=explain.reference_row(X[tr]),
            horizon=2.0,
            features=F.ALL_FEATURES,
            shipped="boosting, full",
        ),
        X,
        flights,
    )


def test_score_frame_columns_and_ranges(bundle):
    b, X, _ = bundle
    s = scoring.score_frame(b, X.head(50))
    assert list(s.columns) == ["p_delay", "alert", "delay_p50_min", "delay_p80_min", "delay_p90_min"]
    assert s["p_delay"].between(0, 1).all()
    assert (s["alert"] == (s["p_delay"] >= 0.3)).all()


def test_leave_by_wraps_midnight_and_floors_negative_delay():
    assert scoring.leave_by(23, 50, 25) == "00:15"
    assert scoring.leave_by(9, 5, -12) == "09:05"


def test_what_if_override_raises_probability(bundle):
    b, X, _ = bundle
    row = X.iloc[[200]].copy()
    row["prev_dep_delay"] = 0.0
    late = scoring.apply_overrides(row, {"prev_dep_delay": 120.0, "recent_mean_delay": None})
    assert late["prev_dep_delay"].iloc[0] == 120.0
    assert scoring.delay_probability(b, late)[0] >= scoring.delay_probability(b, row)[0] - 0.05
    with pytest.raises(KeyError):
        scoring.apply_overrides(row, {"nope": 1.0})


def test_summarise_returns_reasons_in_plain_text(bundle):
    b, X, f = bundle
    i = 300
    out = scoring.summarise(b, X.iloc[[i]], int(f["hour"].iloc[i]), int(f["minute"].iloc[i]))
    assert 0 <= out["p_delay"] <= 1 and len(out["reasons"]) == 3
    assert all(isinstance(r["text"], str) and r["text"] for r in out["reasons"])
    assert ":" in out["p80_departs_by"]


def test_group_effects_zero_when_row_equals_reference(bundle):
    b, X, _ = bundle
    ref = b["reference"]
    row = pd.DataFrame([ref.to_dict()])
    for c in F.ALL_FEATURES:
        if c not in F.CATEGORICAL:
            row[c] = row[c].astype(float)
    eff = explain.group_effects(lambda r: scoring.delay_probability(b, r), row, ref)
    assert all(abs(v) < 1e-9 for v in eff.values())


def test_describe_covers_all_groups(bundle):
    b, X, _ = bundle
    row = X.iloc[200]
    for g in F.GROUPS:
        assert isinstance(explain.describe(g, row), str)
