"""Download, load and join the nycflights13 tables."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PYPI_JSON = "https://pypi.org/pypi/nycflights13/0.0.3/json"
SDIST_SHA256 = "d9ef2f5cf1bebca7e30b4daf69dcd7a8fd71f25b7196f5dc489879ad7e3e8a37"
TABLES = ("flights.csv.zip", "weather.csv", "planes.csv", "airports.csv", "airlines.csv")

# A flight that has not left within this many minutes of its schedule counts as delayed.
DELAY_MINUTES = 15


def fetch(raw_dir: str | Path = "data/raw") -> Path:
    """Fetch the source tarball from PyPI, verify its hash and unpack the csv tables."""
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    if all((raw / t).exists() for t in TABLES):
        return raw
    with urllib.request.urlopen(PYPI_JSON, timeout=60) as resp:
        meta = json.load(resp)
    url = next(u["url"] for u in meta["urls"] if u["packagetype"] == "sdist")
    with urllib.request.urlopen(url, timeout=300) as resp:
        blob = resp.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != SDIST_SHA256:
        raise RuntimeError(f"unexpected archive hash {digest}")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if "/data/" in member.name and name in TABLES:
                data = tar.extractfile(member)
                if data is not None:
                    (raw / name).write_bytes(data.read())
    return raw


def load_tables(raw_dir: str | Path = "data/raw") -> dict[str, pd.DataFrame]:
    raw = Path(raw_dir)
    with zipfile.ZipFile(raw / "flights.csv.zip") as zf:
        with zf.open(zf.namelist()[0]) as fh:
            flights = pd.read_csv(fh)
    return {
        "flights": flights,
        "weather": pd.read_csv(raw / "weather.csv"),
        "planes": pd.read_csv(raw / "planes.csv"),
        "airports": pd.read_csv(raw / "airports.csv"),
        "airlines": pd.read_csv(raw / "airlines.csv"),
    }


def prepare_flights(flights: pd.DataFrame) -> pd.DataFrame:
    """Add UTC schedule and actual departure times plus the label.

    A cancelled flight has no departure time. I count it as delayed, because from the
    passenger's side it did not leave within fifteen minutes of schedule either.
    """
    df = flights.copy()
    df["tailnum"] = df["tailnum"].fillna("UNKNOWN")
    hour_utc = pd.to_datetime(df["time_hour"], utc=True).dt.tz_localize(None)
    df["sched_utc"] = hour_utc + pd.to_timedelta(df["minute"], unit="m")
    df["cancelled"] = df["dep_delay"].isna()
    df["actual_utc"] = df["sched_utc"] + pd.to_timedelta(df["dep_delay"], unit="m")
    df["sched_date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    df["delayed"] = (df["cancelled"] | (df["dep_delay"] >= DELAY_MINUTES)).astype(int)
    df = df.sort_values(["sched_utc", "origin", "flight"], kind="mergesort").reset_index(drop=True)
    df["flight_id"] = np.arange(len(df))
    return df


def split_by_month(df: pd.DataFrame) -> pd.Series:
    """Forward split: January to August train, September and October validate, rest test."""
    month = df["month"]
    out = pd.Series("train", index=df.index)
    out[month.isin([9, 10])] = "validation"
    out[month.isin([11, 12])] = "test"
    return out


def drop_unassigned(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove flights with no tail number.

    In this table every one of them is a cancellation, so the missing tail number gives
    the answer away. A planner would not know it at decision time.
    """
    keep = df["tailnum"] != "UNKNOWN"
    return df[keep].reset_index(drop=True), int((~keep).sum())
