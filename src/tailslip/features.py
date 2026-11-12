"""Leak-safe feature building at a fixed decision horizon.

The idea: pick a clock time T = scheduled departure minus ``horizon`` hours and let every
feature use only what a planner could have known at T. A flight that has already
departed by T is known to have departed, a flight that has not is known to be overdue by
at least ``T - scheduled``, and so on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("2013-01-01")
# Airborne time in the data excludes taxiing, so I add an allowance to estimate touchdown.
TAXI_IN_MINUTES = 20.0

CATEGORICAL = ["carrier", "origin", "dest", "manufacturer", "plane_type"]
SCHEDULE = ["sched_hour", "sched_minute", "dow", "weekend", "distance", "sched_load"]
PLANE = ["seats", "plane_age", "engines"]
PREV_LEG = [
    "has_prev",
    "rotation_gap_h",
    "prev_dep_delay",
    "prev_arr_delay",
    "prev_overdue_min",
    "prev_cancelled",
    "prev_distance",
]
AIRPORT = [
    "recent_n",
    "recent_mean_delay",
    "recent_late_share",
    "recent3h_mean_delay",
    "queue_ahead",
    "backlog",
]
WEATHER = [
    "temp",
    "dewp",
    "humid",
    "wind_dir",
    "wind_speed",
    "wind_gust",
    "precip",
    "pressure",
    "visib",
    "precip_3h",
    "visib_min_3h",
]

GROUPS = {
    "aircraft's previous leg": PREV_LEG,
    "airport congestion": AIRPORT,
    "weather at the airport": WEATHER,
    "schedule slot": ["sched_hour", "sched_minute", "dow", "weekend", "sched_load"],
    "route and carrier": ["carrier", "origin", "dest", "distance"],
    "aircraft type": CATEGORICAL[3:] + PLANE,
}
NUMERIC = SCHEDULE + PLANE + PREV_LEG + AIRPORT + WEATHER
ALL_FEATURES = CATEGORICAL + NUMERIC


def minutes(ts: pd.Series) -> np.ndarray:
    return ((ts - EPOCH) / pd.Timedelta(minutes=1)).to_numpy(dtype="float64")


@dataclass
class OriginIndex:
    """Sorted schedule and departure arrays for one airport (cancelled flights left out)."""

    sched: np.ndarray
    actual: np.ndarray
    cum_delay: np.ndarray
    cum_late: np.ndarray

    @classmethod
    def build(cls, sched: np.ndarray, actual: np.ndarray, delay: np.ndarray, late: np.ndarray):
        order = np.argsort(actual, kind="mergesort")
        return cls(
            sched=np.sort(sched),
            actual=actual[order],
            cum_delay=np.concatenate([[0.0], np.cumsum(delay[order])]),
            cum_late=np.concatenate([[0.0], np.cumsum(late[order])]),
        )

    def window(self, t: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hi = np.searchsorted(self.actual, t, side="right")
        lo = np.searchsorted(self.actual, t - width, side="right")
        n = (hi - lo).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, (self.cum_delay[hi] - self.cum_delay[lo]) / n, np.nan)
            late = np.where(n > 0, (self.cum_late[hi] - self.cum_late[lo]) / n, np.nan)
        return n, mean, late


def build_origin_indexes(df: pd.DataFrame) -> dict[str, OriginIndex]:
    ok = df[~df["cancelled"]]
    out = {}
    for origin, g in ok.groupby("origin"):
        out[origin] = OriginIndex.build(
            minutes(g["sched_utc"]),
            minutes(g["actual_utc"]),
            g["dep_delay"].to_numpy(dtype="float64"),
            (g["dep_delay"] >= 15).to_numpy(dtype="float64"),
        )
    return out


def airport_features(df: pd.DataFrame, indexes: dict[str, OriginIndex], horizon: float):
    n = len(df)
    sched = minutes(df["sched_utc"])
    t = sched - horizon * 60.0
    cols = {k: np.full(n, np.nan) for k in AIRPORT + ["sched_load"]}
    for origin, idx in indexes.items():
        m = (df["origin"] == origin).to_numpy()
        if not m.any():
            continue
        tt, ss = t[m], sched[m]
        rn, rmean, rlate = idx.window(tt, 60.0)
        _, r3mean, _ = idx.window(tt, 180.0)
        gone = np.searchsorted(idx.actual, tt, side="right")
        before_me = np.searchsorted(idx.sched, ss, side="left")
        by_now = np.searchsorted(idx.sched, tt, side="right")
        load = np.searchsorted(idx.sched, ss + 30.0, side="right") - np.searchsorted(idx.sched, ss - 30.0, side="right")
        cols["recent_n"][m] = rn
        cols["recent_mean_delay"][m] = rmean
        cols["recent_late_share"][m] = rlate
        cols["recent3h_mean_delay"][m] = r3mean
        cols["queue_ahead"][m] = before_me - gone
        cols["backlog"][m] = by_now - gone
        cols["sched_load"][m] = load
    return pd.DataFrame(cols, index=df.index)


def prev_leg_features(df: pd.DataFrame, horizon: float) -> pd.DataFrame:
    """What is known at T about the previous departure of the same tail number."""
    order = df.sort_values(["tailnum", "sched_utc"], kind="mergesort")
    same = (order["tailnum"] == order["tailnum"].shift(1)) & (order["tailnum"] != "UNKNOWN")
    shifted = order[["sched_utc", "actual_utc", "dep_delay", "arr_delay", "air_time", "distance", "cancelled"]].shift(1)
    sched = minutes(order["sched_utc"])
    t = sched - horizon * 60.0
    p_sched = minutes(shifted["sched_utc"])
    p_actual = minutes(shifted["actual_utc"])
    p_air = shifted["air_time"].to_numpy(dtype="float64")
    has = same.to_numpy()
    p_canc = shifted["cancelled"].fillna(False).to_numpy(dtype=bool)

    left = has & ~p_canc & (p_actual <= t)
    p_land = p_actual + np.nan_to_num(p_air, nan=0.0) + TAXI_IN_MINUTES
    landed = left & (p_land <= t) & ~np.isnan(p_air)
    overdue = has & ~p_canc & (p_sched <= t) & ~(p_actual <= t)
    out = pd.DataFrame(index=order.index)
    out["has_prev"] = has.astype(float)
    out["rotation_gap_h"] = np.where(has, np.minimum((sched - p_sched) / 60.0, 72.0), np.nan)
    out["prev_dep_delay"] = np.where(left, shifted["dep_delay"].to_numpy(dtype="float64"), np.nan)
    out["prev_arr_delay"] = np.where(landed, shifted["arr_delay"].to_numpy(dtype="float64"), np.nan)
    out["prev_overdue_min"] = np.where(overdue, t - p_sched, 0.0)
    out["prev_cancelled"] = np.where(has & (p_sched <= t), p_canc.astype(float), np.nan)
    out["prev_distance"] = np.where(has, shifted["distance"].to_numpy(dtype="float64"), np.nan)
    return out.loc[df.index]


def weather_table(weather: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Hourly weather per airport on a gap-free index, observations lagged by one hour."""
    w = weather.copy()
    w["hour"] = pd.to_datetime(w["time_hour"], utc=True).dt.tz_localize(None)
    out = {}
    for origin, g in w.groupby("origin"):
        g = g.drop_duplicates("hour").set_index("hour").sort_index()
        full = pd.date_range(g.index.min(), g.index.max() + pd.Timedelta(days=1), freq="h")
        g = g.reindex(full).ffill(limit=12)
        keep = g[["temp", "dewp", "humid", "wind_dir", "wind_speed", "wind_gust", "precip", "pressure", "visib"]].copy()
        keep["precip_3h"] = keep["precip"].rolling(3, min_periods=1).sum()
        keep["visib_min_3h"] = keep["visib"].rolling(3, min_periods=1).min()
        out[origin] = keep
    return out


def weather_features(df: pd.DataFrame, wx: dict[str, pd.DataFrame], horizon: float, lag_hours: float = 1.0) -> pd.DataFrame:
    """Latest observation an hour or more before T; ``lag_hours=-horizon`` gives hindsight."""
    out = pd.DataFrame(np.nan, index=df.index, columns=WEATHER)
    t = df["sched_utc"] - pd.Timedelta(hours=horizon)
    key = (t - pd.Timedelta(hours=lag_hours)).dt.floor("h")
    for origin, table in wx.items():
        m = (df["origin"] == origin).to_numpy()
        if not m.any():
            continue
        vals = table.reindex(key[m]).to_numpy()
        out.loc[m, WEATHER] = vals
    return out


def plane_features(df: pd.DataFrame, planes: pd.DataFrame) -> pd.DataFrame:
    p = planes.rename(columns={"year": "built", "type": "plane_type"})
    merged = df[["tailnum"]].merge(
        p[["tailnum", "built", "plane_type", "manufacturer", "seats", "engines"]],
        on="tailnum",
        how="left",
    )
    merged.index = df.index
    merged["plane_age"] = 2013 - merged["built"]
    for c in ("plane_type", "manufacturer"):
        merged[c] = merged[c].fillna("unknown").astype(str)
    return merged[["plane_type", "manufacturer", "seats", "engines", "plane_age"]]


def schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["carrier"] = df["carrier"].astype(str)
    out["origin"] = df["origin"].astype(str)
    out["dest"] = df["dest"].astype(str)
    out["sched_hour"] = df["hour"].astype(float)
    out["sched_minute"] = df["minute"].astype(float)
    out["dow"] = df["sched_date"].dt.dayofweek.astype(float)
    out["weekend"] = (out["dow"] >= 5).astype(float)
    out["distance"] = df["distance"].astype(float)
    return out


def build_features(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    horizon: float,
    indexes: dict[str, OriginIndex] | None = None,
    wx: dict[str, pd.DataFrame] | None = None,
    hindsight_weather: bool = False,
) -> pd.DataFrame:
    indexes = indexes or build_origin_indexes(df)
    wx = wx or weather_table(tables["weather"])
    parts = [
        schedule_features(df),
        plane_features(df, tables["planes"]),
        prev_leg_features(df, horizon),
        airport_features(df, indexes, horizon),
        weather_features(df, wx, horizon, lag_hours=-horizon if hindsight_weather else 1.0),
    ]
    feats = pd.concat(parts, axis=1)
    # the airport block already holds sched_load; keep one copy of each column
    feats = feats.loc[:, ~feats.columns.duplicated()]
    feats = feats[ALL_FEATURES].copy()
    # a few wind readings in the raw table are instrument glitches (over 1,000 mph)
    feats["wind_speed"] = feats["wind_speed"].clip(upper=70.0)
    return feats
