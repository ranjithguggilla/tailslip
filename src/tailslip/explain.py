"""Plain-language reasons for a single flight's delay probability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tailslip.features import ALL_FEATURES, CATEGORICAL, GROUPS


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def reference_row(train: pd.DataFrame) -> pd.Series:
    """A typical row: medians for numbers, the most common value for categories."""
    ref = {}
    for c in ALL_FEATURES:
        if c in CATEGORICAL:
            ref[c] = train[c].mode().iloc[0]
        else:
            ref[c] = float(train[c].median())
    return pd.Series(ref)


def group_effects(predict, row: pd.DataFrame, ref: pd.Series) -> dict[str, float]:
    """Change in log-odds when one block of features is reset to the typical row."""
    base = float(logit(predict(row)[0]))
    effects = {}
    for name, cols in GROUPS.items():
        alt = row.copy()
        for c in cols:
            alt[c] = ref[c]
        effects[name] = base - float(logit(predict(alt)[0]))
    return effects


def describe(group: str, row: pd.Series) -> str:
    g = row
    if group == "aircraft's previous leg":
        if not g.get("has_prev", 0):
            return "no earlier flight for this aircraft is on record"
        if g["prev_overdue_min"] > 0:
            return f"the aircraft's previous flight is still on the ground, {g['prev_overdue_min']:.0f} min past its slot"
        if pd.notna(g["prev_dep_delay"]):
            landed = ""
            if pd.notna(g["prev_arr_delay"]):
                landed = f" and landed {g['prev_arr_delay']:.0f} min vs schedule"
            return f"the aircraft's previous flight left {g['prev_dep_delay']:.0f} min vs schedule{landed}"
        return "the aircraft's previous flight is not yet in the picture"
    if group == "airport congestion":
        m = g["recent_mean_delay"]
        recent = "no recent departures" if pd.isna(m) else f"recent departures are running {m:.0f} min off schedule"
        q = int(g["queue_ahead"])
        return f"{recent}, with {q} flight{'' if q == 1 else 's'} queued ahead"
    if group == "weather at the airport":
        return f"visibility {g['visib']:.1f} mi, wind {g['wind_speed']:.0f} mph, rain over 3 h {g['precip_3h']:.2f} in"
    if group == "schedule slot":
        return f"a {g['sched_hour']:.0f}:00 slot with {g['sched_load']:.0f} departures within the hour"
    if group == "route and carrier":
        return f"{g['carrier']} from {g['origin']} to {g['dest']}"
    return "this aircraft type"


def reasons(predict, row: pd.DataFrame, ref: pd.Series, top: int = 3) -> list[dict]:
    eff = group_effects(predict, row, ref)
    ranked = sorted(eff.items(), key=lambda kv: -abs(kv[1]))[:top]
    out = []
    for name, val in ranked:
        direction = "raises" if val > 0 else "lowers"
        out.append(
            {
                "group": name,
                "effect": float(val),
                "text": f"{describe(name, row.iloc[0])} ({direction} the odds of a delay)",
            }
        )
    return out
