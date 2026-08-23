"""A minimal, honest backtest engine.

Deliberately small. The point of this package is the attacks, not the
simulator — but the attacks are only as trustworthy as the thing they attack,
so the few properties that matter are enforced here rather than assumed:

  * signals are stamped at the close of the rebalance date and traded at the
    close `execution_lag` days later;
  * weights drift with prices between rebalances, so turnover is measured
    against the real book rather than a stale target;
  * costs are charged on realised turnover the day the trade prints;
  * the return series starts when the book first holds something, so a
    strategy is never credited with flat days while its benchmark was invested.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

ANN = 252


@dataclass
class BacktestSpec:
    quantile: float = 0.10
    long_short: bool = True
    weighting: str = "equal"        # "equal" | "inverse_vol"
    max_weight: float = 0.10
    rebalance: str = "ME"
    execution_lag: int = 1
    cost_bps: float = 5.0
    vol_window: int = 63


@dataclass
class BacktestResult:
    returns: pd.Series
    turnover: pd.Series
    n_positions: pd.Series
    weights: pd.DataFrame

    @property
    def years(self) -> float:
        return len(self.returns) / ANN

    @property
    def annual_turnover(self) -> float:
        return float(self.turnover.sum() / self.years) if self.years else np.nan


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "ME") -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.resample(freq).last().dropna().values)


def _weights(names, vol_row, spec: BacktestSpec, sign: float) -> pd.Series:
    if len(names) == 0:
        return pd.Series(dtype=float)
    if spec.weighting == "inverse_vol":
        v = vol_row.reindex(names).replace(0.0, np.nan)
        v = v.fillna(v.median() if np.isfinite(v.median()) else 1.0)
        w = 1.0 / v
    else:
        w = pd.Series(1.0, index=pd.Index(names))
    w = w / w.sum()
    for _ in range(50):                       # water-fill the cap, no renormalise
        over = w > spec.max_weight
        if not over.any():
            break
        excess = float((w[over] - spec.max_weight).sum())
        w[over] = spec.max_weight
        room = ~over
        if not room.any() or float(w[room].sum()) <= 0:
            break
        w[room] += excess * w[room] / float(w[room].sum())
    return w.clip(upper=spec.max_weight) * sign


def run(prices: pd.DataFrame, score: pd.DataFrame, spec: BacktestSpec,
        mask: pd.DataFrame | None = None) -> BacktestResult:
    prices = prices.sort_index()
    rets = prices.pct_change()
    vol = rets.rolling(spec.vol_window,
                       min_periods=int(spec.vol_window * 0.7)).std() * np.sqrt(ANN)
    idx = prices.index
    pos = {d: i for i, d in enumerate(idx)}
    exec_map = {}
    for d in rebalance_dates(idx, spec.rebalance):
        j = pos[d] + spec.execution_lag
        if j < len(idx):
            exec_map[idx[j]] = d

    cols = prices.columns
    w = pd.Series(0.0, index=cols)
    tc = spec.cost_bps / 1e4
    out_r, out_t, out_n, wrec = [], [], [], {}
    first_live = None

    for i, d in enumerate(idx):
        if i == 0 or first_live is None:
            day = 0.0
        else:
            r = rets.loc[d].reindex(cols).fillna(0.0)
            day = float((w * r).sum())
            if abs(1.0 + day) > 1e-12:
                w = w * (1.0 + r) / (1.0 + day)

        turn = 0.0
        if d in exec_map:
            sd = exec_map[d]
            s = score.loc[sd] if sd in score.index else pd.Series(dtype=float)
            if mask is not None and sd in mask.index:
                s = s[mask.loc[sd].reindex(s.index).fillna(False)]
            s = s.dropna()
            tgt = pd.Series(0.0, index=cols)
            if len(s) >= 20:
                n = max(1, int(round(len(s) * spec.quantile)))
                asc = s.sort_values()
                longs = asc.index[-n:]
                wl = _weights(longs, vol.loc[sd], spec, +1.0)
                tgt = tgt.add(wl.reindex(cols).fillna(0.0), fill_value=0.0)
                if spec.long_short:
                    shorts = asc.index[:n].difference(longs)
                    ws = _weights(shorts, vol.loc[sd], spec, -1.0)
                    tgt = tgt.add(ws.reindex(cols).fillna(0.0), fill_value=0.0)
            turn = float((tgt - w).abs().sum())
            day -= turn * tc
            w = tgt
            if first_live is None and (tgt != 0).any():
                first_live = d
            wrec[d] = tgt[tgt != 0]

        out_r.append(day); out_t.append(turn)
        out_n.append(int((w.abs() > 1e-9).sum()))

    live = idx >= first_live if first_live is not None else np.ones(len(idx), bool)
    return BacktestResult(
        returns=pd.Series(out_r, index=idx).loc[live],
        turnover=pd.Series(out_t, index=idx).loc[live],
        n_positions=pd.Series(out_n, index=idx).loc[live],
        weights=pd.DataFrame(wrec).T.fillna(0.0),
    )


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation without scipy.

    `Series.corr(method="spearman")` delegates to scipy, which would put a
    third dependency in a package whose whole pitch is that it has two. Pearson
    on the ranks is the same number.
    """
    if len(a) < 3:
        return float("nan")
    return float(a.rank().corr(b.rank()))


# ------------------------------------------------------------------ metrics
def sharpe(r: pd.Series) -> float:
    s = r.std(ddof=1)
    return float(r.mean() / s * np.sqrt(ANN)) if s > 0 else np.nan


def cagr(r: pd.Series) -> float:
    if len(r) == 0:
        return np.nan
    total = float((1 + r).prod())
    return total ** (ANN / len(r)) - 1 if total > 0 else -1.0


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def rank_ic(score: pd.DataFrame, prices: pd.DataFrame, dates, horizon: int,
            mask: pd.DataFrame | None = None, min_names: int = 20) -> pd.Series:
    fwd = prices.shift(-horizon) / prices - 1.0
    out = {}
    for d in dates:
        if d not in score.index or d not in fwd.index:
            continue
        s, f = score.loc[d], fwd.loc[d]
        if mask is not None and d in mask.index:
            m = mask.loc[d].reindex(s.index).fillna(False)
            s, f = s[m], f[m]
        df = pd.concat([s, f], axis=1).dropna()
        if len(df) >= min_names:
            out[d] = spearman(df.iloc[:, 0], df.iloc[:, 1])
    return pd.Series(out, dtype=float)


def decile_returns(score: pd.DataFrame, prices: pd.DataFrame, dates,
                   horizon: int, n_q: int = 10,
                   mask: pd.DataFrame | None = None) -> pd.DataFrame:
    fwd = prices.shift(-horizon) / prices - 1.0
    rows = {}
    for d in dates:
        if d not in score.index or d not in fwd.index:
            continue
        s, f = score.loc[d], fwd.loc[d]
        if mask is not None and d in mask.index:
            m = mask.loc[d].reindex(s.index).fillna(False)
            s, f = s[m], f[m]
        df = pd.concat([s.rename("s"), f.rename("f")], axis=1).dropna()
        if len(df) < n_q * 2:
            continue
        try:
            q = pd.qcut(df["s"].rank(method="first"), n_q, labels=False)
        except ValueError:
            continue
        rows[d] = df.groupby(q)["f"].mean()
    return pd.DataFrame(rows).T
