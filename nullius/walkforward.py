"""Walk-forward validation, expressed as an attack.

The question every other attack in this package dances around: does the edge
survive on data the researcher had not seen? Split the sample, run the same
unchanged parameters on each half, and compare the *edge* — strategy minus
benchmark — rather than the raw return. A strategy earning 8% in train and 12%
in test has not improved if the market went from 6% to 18%.

Nothing is re-optimised between windows. That is deliberate. A textbook
walk-forward re-fits parameters on each train window; doing so here would mean
the harness performs the tuning you are trying to detect.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from . import engine as E
from .report import Finding

ANN = 252

VERDICTS = {
    "PRESERVED": "Same sign out of sample, at least half the magnitude retained. "
                 "The only outcome consistent with a real edge — necessary, not "
                 "sufficient.",
    "DEGRADED": "Same sign, less than half the magnitude survived. Consistent "
                "with a weak real effect and equally with in-sample luck decaying.",
    "SIGN_FLIP": "Beat the benchmark in sample, lost to it out of sample. The "
                 "signature of overfitting.",
    "NO_TRAIN_EDGE": "No edge even in sample, so the test window tests nothing.",
}
_ORDER = ["SIGN_FLIP", "NO_TRAIN_EDGE", "DEGRADED", "PRESERVED"]


@dataclass(frozen=True)
class Window:
    label: str
    train: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]

    def describe(self) -> str:
        return (f"{self.label}: train {self.train[0]:%Y-%m}→{self.train[1]:%Y-%m}  "
                f"test {self.test[0]:%Y-%m}→{self.test[1]:%Y-%m}")


def make_windows(index: pd.DatetimeIndex,
                 mode: Literal["single", "anchored", "rolling"] = "single",
                 train_years: float = 5.0, test_years: float = 3.0,
                 step_years: Optional[float] = None,
                 split: Optional[str] = None) -> list[Window]:
    start, end = index[0], index[-1]
    yrs = lambda t, n: t + pd.DateOffset(days=int(round(n * 365.25)))

    if mode == "single":
        cut = pd.Timestamp(split) if split else yrs(start, train_years)
        if not (start < cut < end):
            cut = start + (end - start) / 2
        return [Window("split-1", (start, cut), (cut, end))]

    step = step_years or test_years
    out, i, tr_start = [], 1, start
    cursor = yrs(start, train_years)
    while yrs(cursor, test_years) <= end:
        te_end = yrs(cursor, test_years)
        out.append(Window(f"w{i}", (tr_start, cursor), (cursor, te_end)))
        cursor = yrs(cursor, step)
        if mode == "rolling":
            tr_start = yrs(tr_start, step)
        i += 1
    return out


def _benchmark(prices: pd.DataFrame, spec: E.BacktestSpec,
               kind: str) -> pd.Series:
    """Cash for a market-neutral book, equal-weight universe for a long-only one.

    Pairing a zero-beta strategy against an equity index would inject the
    market's variance into a difference that never contained any, and would
    judge persistence by whether the market rallied.
    """
    if kind == "cash" or (kind == "auto" and spec.long_short):
        return pd.Series(0.0, index=prices.index)
    return prices.pct_change().mean(axis=1).fillna(0.0)


def classify(is_edge: float, oos_edge: float, retention: float = 0.5) -> str:
    """The verdict for one window, from its two edge numbers alone.

    Kept as a pure function so the classification can be tested exhaustively
    without generating a market that happens to behave the right way — the
    branch that matters (an edge that inverts) is otherwise surprisingly hard
    to produce synthetically, because a trend-follower simply latches onto
    whatever the new trend is.
    """
    if not np.isfinite(is_edge) or is_edge <= 0:
        return "NO_TRAIN_EDGE"
    if not np.isfinite(oos_edge) or oos_edge < 0:
        return "SIGN_FLIP"
    return "PRESERVED" if oos_edge >= retention * is_edge else "DEGRADED"


def _edge(r: pd.Series, b: pd.Series) -> float:
    common = r.index.intersection(b.index)
    return E.cagr(r.loc[common]) - E.cagr(b.loc[common])


def walk_forward(prices, score, spec, mask=None, mode="single",
                 train_years: float = 5.0, test_years: float = 3.0,
                 step_years: Optional[float] = None, split: Optional[str] = None,
                 benchmark: str = "auto", min_days: int = 120) -> Finding:
    res = E.run(prices, score, spec, mask)
    r = res.returns
    if len(r) < min_days * 2:
        return Finding("Walk-forward", False,
                       f"only {len(r)} trading days — too short to split",
                       note="A split needs enough data on both sides to measure "
                            "anything at all.", value=np.nan)
    bench = _benchmark(prices, spec, benchmark)
    windows = make_windows(r.index, mode, train_years, test_years, step_years, split)

    rows, verdicts = [], []
    for w in windows:
        tr = r.loc[w.train[0]:w.train[1]]
        te = r.loc[w.test[0]:w.test[1]]
        if len(tr) < min_days or len(te) < min_days:
            continue
        e_tr = _edge(tr, bench)
        e_te = _edge(te, bench)
        v = classify(e_tr, e_te)
        d = (te - bench.reindex(te.index).fillna(0.0)).astype(float)
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std(ddof=1) > 0 else 0.0
        verdicts.append(v)
        rows.append({"window": w.label,
                     "train": f"{w.train[0]:%Y-%m}→{w.train[1]:%Y-%m}",
                     "test": f"{w.test[0]:%Y-%m}→{w.test[1]:%Y-%m}",
                     "is_edge": e_tr, "oos_edge": e_te,
                     "retained": (e_te / e_tr) if e_tr > 0 else np.nan,
                     "oos_t": t, "verdict": v})

    if not rows:
        return Finding("Walk-forward", False, "no window had enough data on both sides",
                       value=np.nan)

    df = pd.DataFrame(rows).set_index("window")
    # The overall verdict is the WORST window, not the average. A strategy that
    # works in three windows and inverts in the fourth is not a 75% strategy;
    # it is one whose edge you cannot predict.
    overall = min(verdicts, key=lambda v: _ORDER.index(v))
    sig = int((df.oos_t.abs() >= 2).sum())
    return Finding(
        name="Walk-forward", passed=(overall == "PRESERVED"),
        detail=f"{overall} across {len(df)} window(s); "
               f"median retention {df.retained.median():.0%}; "
               f"{sig}/{len(df)} with |t| ≥ 2 out of sample",
        note=VERDICTS[overall] + " Verdict is the worst window, not the average.",
        value=float(df.oos_edge.median()), table=df,
    )
