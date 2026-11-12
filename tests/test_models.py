import numpy as np
import pytest

from tailslip import features as F
from tailslip import models


@pytest.fixture(scope="module")
def split(flights, tables):
    X = F.build_features(flights, tables, 2.0)
    y = flights["delayed"].to_numpy()
    m = (flights["month"] <= 8).to_numpy()
    return X, y, m


def test_slot_rate_recovers_group_means(split):
    X, y, m = split
    sr = models.SlotRate().fit(X[m], y[m])
    p = sr.predict_proba(X[~m])[:, 1]
    assert p.shape == ((~m).sum(),) and ((p >= 0) & (p <= 1)).all()
    assert sr.prior_ == pytest.approx(y[m].mean())


def test_rule_flags_only_late_aircraft(split):
    X, y, m = split
    r = models.RuleLate().fit(X[m], y[m])
    p = r.predict_proba(X[~m])[:, 1]
    late = ((X[~m]["prev_dep_delay"].fillna(0) >= 15) | (X[~m]["prev_overdue_min"] >= 15)).to_numpy()
    assert set(np.round(p[late], 6)) <= {round(r.p_late_, 6)}
    assert set(np.round(p[~late], 6)) <= {round(r.p_other_, 6)}


def test_boosting_pipeline_handles_unseen_category_and_missing(split):
    X, y, m = split
    pipe = models.make_boosting(F.ALL_FEATURES, max_iter=30).fit(X[m], y[m])
    Z = X[~m].head(20).copy()
    Z["dest"] = "ZZZ"
    Z["prev_dep_delay"] = np.nan
    p = pipe.predict_proba(Z)[:, 1]
    assert np.isfinite(p).all()


def test_logistic_pipeline_fits_and_beats_base_rate(split):
    from sklearn.metrics import roc_auc_score

    X, y, m = split
    lr = models.make_logistic(F.ALL_FEATURES).fit(X[m], y[m])
    assert roc_auc_score(y[~m], lr.predict_proba(X[~m])[:, 1]) > 0.55


def test_preprocessing_is_inside_the_pipeline(split):
    X, y, m = split
    pipe = models.make_boosting(F.ALL_FEATURES, max_iter=10)
    assert [n for n, _ in pipe.steps] == ["pre", "model"]
    lr = models.make_logistic(F.ALL_FEATURES)
    assert [n for n, _ in lr.steps] == ["pre", "model"]


def test_quantiles_are_ordered_and_roughly_covered(split, flights):
    X, y, m = split
    d = flights["dep_delay"].to_numpy()
    ok = ~np.isnan(d)
    qf = models.make_quantile_forest(F.ALL_FEATURES).fit(X[m & ok], d[m & ok])
    q = models.predict_quantiles(qf, X[~m & ok], (0.5, 0.8, 0.9))
    assert (q[:, 0] <= q[:, 1] + 1e-9).all() and (q[:, 1] <= q[:, 2] + 1e-9).all()
    cover = (d[~m & ok] <= q[:, 1]).mean()
    assert 0.65 < cover < 0.95
