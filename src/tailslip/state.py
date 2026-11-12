"""Streaming view of an airport, for live scoring and for checking the batch features.

The batch code in ``features`` answers "what did the airport look like at time T" with
prefix sums over sorted arrays. This class answers the same question one flight at a
time with sorted containers, the way a live service would. A test pins the two together
so a model trained on the batch features sees the same numbers when it is served.
"""

from __future__ import annotations

from sortedcontainers import SortedKeyList, SortedList


class AirportBook:
    """Departures at one airport, kept ordered by schedule and by actual departure."""

    def __init__(self) -> None:
        self._sched = SortedList()
        self._actual: SortedKeyList = SortedKeyList(key=lambda rec: rec[0])

    def add(self, sched: float, actual: float, delay: float) -> None:
        """Record a flight that left at ``actual`` (minutes) after being due at ``sched``."""
        self._sched.add(sched)
        self._actual.add((actual, delay))

    def __len__(self) -> int:
        return len(self._sched)

    def _window(self, t: float, width: float) -> tuple[int, float, float]:
        recs = list(self._actual.irange_key(t - width, t, inclusive=(False, True)))
        n = len(recs)
        if n == 0:
            return 0, float("nan"), float("nan")
        mean = sum(d for _, d in recs) / n
        late = sum(1 for _, d in recs if d >= 15) / n
        return n, mean, late

    def snapshot(self, t: float, sched: float) -> dict[str, float]:
        """Airport features for a flight due at ``sched``, decided at clock time ``t``."""
        n, mean, late = self._window(t, 60.0)
        _, mean3, _ = self._window(t, 180.0)
        gone = self._actual.bisect_key_right(t)
        return {
            "recent_n": float(n),
            "recent_mean_delay": mean,
            "recent_late_share": late,
            "recent3h_mean_delay": mean3,
            "queue_ahead": float(self._sched.bisect_left(sched) - gone),
            "backlog": float(self._sched.bisect_right(t) - gone),
            "sched_load": float(self._sched.bisect_right(sched + 30.0) - self._sched.bisect_right(sched - 30.0)),
        }
