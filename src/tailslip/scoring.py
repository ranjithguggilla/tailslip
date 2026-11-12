"""Score flights from a saved bundle. Kept free of UI code so it can be tested directly."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tailslip import explain, models
from tailslip import features as F


def load_bundle(path: str | Path = "outputs/tailslip_model.joblib") -> dict:
    return joblib.load(path)


def delay_probability(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    p = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
    cal = bundle.get("calibrator")
    return cal.predict(p) if cal is not None else p


def score_frame(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Probability of a late departure, an alert flag and the 50th, 80th and 90th percentile delay."""
    X = X[bundle["features"]]
    p = delay_probability(bundle, X)
    q = models.predict_quantiles(bundle["quantile_model"], X, bundle["quantiles"])
    out = pd.DataFrame({"p_delay": p, "alert": p >= bundle["threshold"]}, index=X.index)
    for j, tau in enumerate(bundle["quantiles"]):
        out[f"delay_p{int(round(tau * 100))}_min"] = q[:, j]
    return out


def apply_overrides(row: pd.DataFrame, overrides: dict[str, float | None]) -> pd.DataFrame:
    """What-if edits, for example a later previous-leg delay. ``None`` leaves a field alone."""
    out = row.copy()
    for key, value in overrides.items():
        if value is not None:
            if key not in out.columns:
                raise KeyError(key)
            out[key] = value
    return out


def explain_flight(bundle: dict, row: pd.DataFrame, top: int = 3) -> list[dict]:
    return explain.reasons(lambda r: delay_probability(bundle, r), row[F.ALL_FEATURES], bundle["reference"], top)


def leave_by(sched_local_hour: int, sched_local_minute: int, delay_minutes: float) -> str:
    """Clock time for a delay quantile, as HH:MM on a 24 hour clock."""
    total = int(round(sched_local_hour * 60 + sched_local_minute + max(delay_minutes, 0.0)))
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"


def summarise(bundle: dict, row: pd.DataFrame, sched_hour: int, sched_minute: int) -> dict:
    scored = score_frame(bundle, row).iloc[0]
    return {
        "p_delay": float(scored["p_delay"]),
        "alert": bool(scored["alert"]),
        "p50_departs_by": leave_by(sched_hour, sched_minute, scored["delay_p50_min"]),
        "p80_departs_by": leave_by(sched_hour, sched_minute, scored["delay_p80_min"]),
        "p90_departs_by": leave_by(sched_hour, sched_minute, scored["delay_p90_min"]),
        "p80_delay_min": float(scored["delay_p80_min"]),
        "reasons": explain_flight(bundle, row),
    }
