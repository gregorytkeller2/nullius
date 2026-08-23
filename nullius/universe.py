"""Point-in-time universes.

Survivorship bias is the one defect no amount of out-of-sample testing detects:
split a biased universe in half and you get two biased halves. The only fix is
knowing who was actually in the index on each date, including the companies that
have since vanished.

Ships historical S&P 500 membership, 1996 to present — 1,206 tickers, of which
703 are no longer in the index.

    from nullius.universe import SP500
    mask = SP500().mask(prices.index, prices.columns)
    Study(..., mask=mask)
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"


class PointInTimeUniverse:
    """Membership snapshots stamped on change dates.

    A date between snapshots inherits the most recent one, which is what an
    investor would actually have known.
    """

    def __init__(self, snapshots: pd.Series, name: str = "universe"):
        self.snapshots = snapshots.sort_index()
        self.name = name

    @property
    def tickers(self) -> list[str]:
        out: set[str] = set()
        for s in self.snapshots:
            out |= set(s)
        return sorted(t for t in out if t)

    def members_on(self, date) -> frozenset:
        i = self.snapshots.index.searchsorted(pd.Timestamp(date), side="right") - 1
        return self.snapshots.iloc[i] if i >= 0 else frozenset()

    def mask(self, dates, columns) -> pd.DataFrame:
        dates, cols = pd.DatetimeIndex(dates), pd.Index(columns)
        idx = self.snapshots.index.searchsorted(dates, side="right") - 1
        out = np.zeros((len(dates), len(cols)), dtype=bool)
        colpos = {c: i for i, c in enumerate(cols)}
        cache: dict[int, np.ndarray] = {}
        for row, k in enumerate(idx):
            if k < 0:
                continue
            if k not in cache:
                v = np.zeros(len(cols), dtype=bool)
                for t in self.snapshots.iloc[k]:
                    j = colpos.get(t)
                    if j is not None:
                        v[j] = True
                cache[k] = v
            out[row] = cache[k]
        return pd.DataFrame(out, index=dates, columns=cols)

    def coverage(self, columns) -> tuple[int, int]:
        """How many historical members the price data actually covers.

        Free vendors routinely drop delisted tickers. The gap falls on the
        SHORT leg, where the disappearances would have been, so it understates
        short-side profit rather than flattering it — but it should be reported,
        not assumed away.
        """
        have = set(columns) & set(self.tickers)
        return len(have), len(self.tickers)


@lru_cache(maxsize=4)
def _load(filename: str, name: str) -> PointInTimeUniverse:
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The bundled membership file ships with the "
            "package; reinstall, or build your own PointInTimeUniverse from a "
            "date -> constituents series.")
    df = pd.read_csv(path)
    return PointInTimeUniverse(
        pd.Series([frozenset(t.split(",")) for t in df["tickers"]],
                  index=pd.DatetimeIndex(pd.to_datetime(df["date"]))),
        name=name)


def SP500() -> PointInTimeUniverse:
    """Historical S&P 500 membership, 1996-present."""
    return _load("sp500_membership.csv.gz", "S&P 500")
