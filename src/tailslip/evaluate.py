"""Metrics, day-block bootstrap, calibration and the alert economics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

# Stated assumptions for the alert economics. These are inputs, not findings.
VALUE_PER_CAUGHT_DELAY = 60.0  # dollars saved when ops hear about a real delay early
COST_PER_FALSE_ALERT = 25.0  # dollars of wasted rebooking and comms work per false alarm


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p)
    y, p = np.asarray(y)[order], np.asarray(p)[order]
    total = 0.0
    for chunk in np.array_split(np.arange(len(p)), bins):
        if len(chunk):
            total += len(chunk) / len(p) * abs(y[chunk].mean() - p[chunk].mean())
    return float(total)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def score_block(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": brier(y, p),
        "ece": ece(y, p),
        "mean_pred": float(np.mean(p)),
        "base_rate": float(np.mean(y)),
    }


@dataclass
class DayBlocks:
    """Index lists per calendar day, so resampling keeps same-day flights together."""

    groups: list[np.ndarray]

    @classmethod
    def from_dates(cls, dates: pd.Series) -> DayBlocks:
        codes, _ = pd.factorize(dates)
        order = np.argsort(codes, kind="mergesort")
        bounds = np.flatnonzero(np.diff(codes[order])) + 1
        return cls(list(np.split(order, bounds)))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        pick = rng.integers(0, len(self.groups), len(self.groups))
        return np.concatenate([self.groups[i] for i in pick])


def bootstrap_ci(blocks: DayBlocks, fn, reps: int = 300, seed: int = 0, level: float = 0.95):
    rng = np.random.default_rng(seed)
    vals = np.array([fn(blocks.sample(rng)) for _ in range(reps)])
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def paired_logloss_se(blocks: DayBlocks, y, p_a, p_b, reps: int = 300, seed: int = 0) -> float:
    """Standard error of (log loss of A minus log loss of B) under a day-block bootstrap."""
    y = np.asarray(y)
    la = -(y * np.log(np.clip(p_a, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p_a, 1e-6, 1)))
    lb = -(y * np.log(np.clip(p_b, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p_b, 1e-6, 1)))
    diff = la - lb
    rng = np.random.default_rng(seed)
    draws = [diff[blocks.sample(rng)].mean() for _ in range(reps)]
    return float(np.std(draws, ddof=1))


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def alert_value(
    y: np.ndarray, flag: np.ndarray, value: float = VALUE_PER_CAUGHT_DELAY, cost: float = COST_PER_FALSE_ALERT
) -> float:
    """Net dollars per flight from alerting on ``flag``."""
    y, flag = np.asarray(y).astype(bool), np.asarray(flag).astype(bool)
    return float((value * (flag & y).sum() - cost * (flag & ~y).sum()) / len(y))


def best_threshold(y: np.ndarray, p: np.ndarray, grid: np.ndarray | None = None):
    grid = np.linspace(0.05, 0.9, 86) if grid is None else grid
    vals = np.array([alert_value(y, p >= g) for g in grid])
    i = int(np.argmax(vals))
    return float(grid[i]), float(vals[i]), grid, vals


def alert_table(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    y = np.asarray(y).astype(bool)
    flag = np.asarray(p) >= thr
    tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
    return {
        "threshold": float(thr),
        "alert_rate": float(flag.mean()),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(int(y.sum()), 1),
        "net_per_flight": alert_value(y, flag),
    }


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))
