"""Estimators and pipelines. All preprocessing lives inside the fitted pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from tailslip import features as F


class SlotRate(ClassifierMixin, BaseEstimator):
    """Smoothed historical delay rate for each carrier, airport and scheduled hour."""

    def __init__(self, keys=("carrier", "origin", "sched_hour"), prior_weight: float = 30.0):
        self.keys = keys
        self.prior_weight = prior_weight

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.prior_ = float(y.mean())
        g = y.groupby([X[k] for k in self.keys]).agg(["sum", "count"])
        self.table_ = (g["sum"] + self.prior_weight * self.prior_) / (g["count"] + self.prior_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(X[list(self.keys)])
        p = self.table_.reindex(idx).to_numpy()
        p = np.where(np.isnan(p), self.prior_, p)
        return np.column_stack([1 - p, p])


class RuleLate(ClassifierMixin, BaseEstimator):
    """The rule of thumb: flag a flight when its aircraft's last leg left 15 or more late."""

    def fit(self, X, y):
        y = np.asarray(y)
        known = X["prev_dep_delay"].notna() | (X["prev_overdue_min"] > 0)
        late = (X["prev_dep_delay"].fillna(0) >= 15) | (X["prev_overdue_min"] >= 15)
        self.p_late_ = float(y[late.to_numpy()].mean()) if late.any() else float(y.mean())
        self.p_other_ = float(y[(~late).to_numpy()].mean())
        self.known_ = float(known.mean())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        late = (X["prev_dep_delay"].fillna(0) >= 15) | (X["prev_overdue_min"] >= 15)
        p = np.where(late, self.p_late_, self.p_other_)
        return np.column_stack([1 - p, p])


def _split_columns(cols: list[str]) -> tuple[list[str], list[str]]:
    cat = [c for c in F.CATEGORICAL if c in cols]
    num = [c for c in cols if c not in cat]
    return cat, num


def make_boosting(cols: list[str], **params) -> Pipeline:
    cat, num = _split_columns(cols)
    pre = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cat),
            ("num", "passthrough", num),
        ]
    )
    defaults = dict(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )
    defaults.update(params)
    clf = HistGradientBoostingClassifier(categorical_features=list(range(len(cat))) if cat else None, **defaults)
    return Pipeline([("pre", pre), ("model", clf)])


def make_logistic(cols: list[str], C: float = 0.5) -> Pipeline:
    cat, num = _split_columns(cols)
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100), cat),
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                num,
            ),
        ]
    )
    return Pipeline([("pre", pre), ("model", LogisticRegression(C=C, max_iter=400))])


def make_base_rate() -> DummyClassifier:
    return DummyClassifier(strategy="prior")


def make_quantile_forest(cols: list[str], quantiles=(0.5, 0.8, 0.9), seed: int = 0) -> Pipeline:
    cat, num = _split_columns(cols)
    pre = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
            ("num", SimpleImputer(strategy="median", keep_empty_features=True), num),
        ]
    )
    forest = RandomForestQuantileRegressor(
        n_estimators=120,
        min_samples_leaf=60,
        max_features=0.4,
        default_quantiles=list(quantiles),
        n_jobs=-1,
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("model", forest)])


def predict_quantiles(pipe: Pipeline, X: pd.DataFrame, quantiles) -> np.ndarray:
    Z = pipe.named_steps["pre"].transform(X)
    return np.atleast_2d(pipe.named_steps["model"].predict(Z, quantiles=list(quantiles)))
