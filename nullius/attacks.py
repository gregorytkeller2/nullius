"""The attacks.

Each one answers a question of the form "if this result were fake, how would it
be fake?" and returns a Finding that either survives or does not. They are
written to be run against any signal, not any particular strategy.
"""
from __future__ import annotations
from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd

from . import engine as E
from .report import Finding

ANN = 252


# ------------------------------------------------------------ 1. look-ahead
def lookahead_canary(prices, score, spec, mask=None, shift_days: int = 42) -> Finding:
    """Feed the engine a signal that knows the future and check it notices.

    A correctly wired engine is enormously sensitive to real foreknowledge. If
    a future-shifted signal is not dramatically more profitable than the honest
    one, the honest one is already peeking — the engine cannot tell them apart
    because it is leaking either way.
    """
    base = E.sharpe(E.run(prices, score, spec, mask).returns)
    cheat = E.sharpe(E.run(prices, score.shift(-shift_days), spec, mask).returns)
    ratio = cheat - base
    ok = np.isfinite(cheat) and np.isfinite(base) and ratio > 0.5
    return Finding(
        name="Look-ahead canary", passed=bool(ok),
        detail=f"future-shifted Sharpe {cheat:.2f} vs honest {base:.2f}",
        note=("A signal with 6 weeks of foreknowledge should be wildly better. "
              "If it is not, the honest run is already using future data."),
        value=float(ratio),
    )


# ------------------------------------------------------------ 2. permutation
def permutation_null(prices, score, spec, mask=None, n: int = 200, seed: int = 0,
                     max_p: float = 0.05, zero_cost: bool = True) -> Finding:
    """Reshuffle the signal across names, keep everything else identical.

    This is what "no edge, same plumbing" looks like. Run at zero cost by
    default: with costs included, a slow signal beats a shuffled one partly by
    trading less, which is a genuine advantage but is not forecasting power.
    """
    s = replace(spec, cost_bps=0.0) if zero_cost else spec
    real = E.sharpe(E.run(prices, score, s, mask).returns)
    vals = score.to_numpy()
    null = []
    for k in range(n):
        rng = np.random.default_rng(seed + k)
        shuffled = vals.copy()
        for i in range(shuffled.shape[0]):
            row = shuffled[i]
            ok = np.flatnonzero(np.isfinite(row))
            if ok.size > 1:
                row[ok] = row[rng.permutation(ok)]
        sc = pd.DataFrame(shuffled, index=score.index, columns=score.columns)
        v = E.sharpe(E.run(prices, sc, s, mask).returns)
        if np.isfinite(v):
            null.append(v)
    null = np.array(null)
    p = float((null >= real).mean()) if null.size else np.nan
    return Finding(
        name="Permutation null", passed=bool(np.isfinite(p) and p <= max_p),
        detail=f"p = {p:.3f} ({n} shuffles, {'zero-cost' if zero_cost else 'with costs'}), "
               f"real {real:.2f} vs null {null.mean():+.2f} ± {null.std(ddof=1):.2f}",
        note="Costs are switched off so this measures forecasting power rather "
             "than the turnover advantage of a slow signal.",
        value=p,
    )


# ------------------------------------------------------------ 3. breadth
def breadth_curve(prices, signal_fn: Callable, spec, sizes=(20, 50, 100, 250, 500),
                  draws: int = 8, seed: int = 0, mask=None) -> Finding:
    """Re-run on random sub-universes.

    Information ratio scales with the square root of the number of independent
    bets, so a real but small edge looks like nothing on a narrow universe. This
    measures the curve instead of assuming it — which turns a null result on 20
    names from evidence into an artefact.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        if n > prices.shape[1]:
            continue
        for k in range(draws):
            cols = list(rng.choice(prices.columns, size=n, replace=False))
            sub = prices[cols]
            try:
                sc = signal_fn(sub)
                m = mask[cols] if mask is not None else None
                rows.append({"n": n,
                             "sharpe": E.sharpe(E.run(sub, sc, spec, m).returns)})
            except Exception:
                continue
    df = pd.DataFrame(rows)
    if df.empty:
        return Finding("Breadth", True, "not enough names to vary", value=np.nan)
    med = df.groupby("n")["sharpe"].median()
    stable = med.index[med.index >= 100]
    ok = bool(len(stable) and med.loc[stable].min() > 0)
    return Finding(
        name="Breadth", passed=ok,
        detail=" | ".join(f"{int(k)}n: {v:+.2f}" for k, v in med.items()),
        note="Median Sharpe by universe size. If the result only appears at "
             "large N it is real but breadth-hungry; if it appears at small N "
             "and vanishes at large N, it is noise.",
        value=float(med.iloc[-1]),
        table=med.rename("median_sharpe").to_frame(),
    )


# ------------------------------------------------------------ 4. plateau
def parameter_plateau(prices, signal_fn: Callable, spec, grid: dict,
                      mask=None, min_positive: float = 0.80) -> Finding:
    """Vary the knobs and see whether the result is a plateau or a needle.

    A parameter surface where most settings work is the single most useful sign
    that a backtest measures something real. One that only works at the reported
    values is a fitted curve.
    """
    import itertools
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        kw = dict(zip(keys, combo))
        sp = replace(spec, **{k: v for k, v in kw.items() if hasattr(spec, k)})
        try:
            sc = signal_fn(prices, **{k: v for k, v in kw.items()
                                      if not hasattr(spec, k)})
            rows.append({**kw, "sharpe": E.sharpe(E.run(prices, sc, sp, mask).returns)})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return Finding("Parameter plateau", False, "no configurations ran", value=np.nan)
    frac = float((df.sharpe > 0).mean())
    return Finding(
        name="Parameter plateau", passed=frac >= min_positive,
        detail=f"{len(df)} configs, {frac:.0%} positive, median {df.sharpe.median():.2f}, "
               f"5th pct {df.sharpe.quantile(0.05):.2f}",
        note="A plateau is credible; a peak is fitted.",
        value=frac, table=df,
    )


# ------------------------------------------------------------ 5. costs
def cost_curve(prices, score, spec, mask=None,
               levels=(0, 5, 10, 20, 30, 50, 75, 100),
               min_breakeven: float = 20.0) -> Finding:
    rows = []
    for c in levels:
        r = E.run(prices, score, replace(spec, cost_bps=float(c)), mask).returns
        rows.append({"bps": c, "sharpe": E.sharpe(r), "cagr": E.cagr(r)})
    df = pd.DataFrame(rows)
    base = E.run(prices, score, spec, mask)
    turn = base.annual_turnover
    gross = float(df.loc[df.bps == min(levels), "cagr"].iloc[0])

    # Analytic break-even: annual cost drag is turnover x bps/1e4, so the cost
    # level that eats the gross return is gross / turnover. Computed directly
    # rather than read off the curve, because a strong signal may never cross
    # zero inside the sampled range -- which is the BEST case, not a failure,
    # and interpolating a crossing that isn't there returned NaN.
    be = (gross / turn * 1e4) if (np.isfinite(gross) and turn > 0) else np.nan
    unbounded = be > max(levels)
    shown = f">{max(levels)}" if unbounded else f"{be:.1f}"
    return Finding(
        name="Cost sensitivity",
        passed=bool(np.isfinite(be) and be >= min_breakeven),
        detail=f"break-even {shown} bps/side at {turn:.1f}x turnover "
               f"(gross {gross:+.1%}/yr)",
        note="An edge that only exists at zero friction is not an edge.",
        value=float(be), table=df,
    )


# ------------------------------------------------------------ 6. bootstrap
def block_bootstrap(returns: pd.Series, n: int = 2000, block: int = 21,
                    seed: int = 0) -> Finding:
    """Stationary block bootstrap: how much of this is sampling noise?"""
    rng = np.random.default_rng(seed)
    x = returns.dropna().to_numpy()
    T = len(x)
    if T < block * 4:
        return Finding("Bootstrap", False, "series too short to resample", value=np.nan)
    nb = int(np.ceil(T / block))
    sh = []
    for _ in range(n):
        starts = rng.integers(0, T - block, size=nb)
        path = np.concatenate([x[s:s + block] for s in starts])[:T]
        sd = path.std(ddof=1)
        sh.append(path.mean() / sd * np.sqrt(ANN) if sd > 0 else np.nan)
    sh = np.array([v for v in sh if np.isfinite(v)])
    lo, med, hi = np.quantile(sh, [0.05, 0.5, 0.95])
    return Finding(
        name="Bootstrap", passed=bool(lo > 0),
        detail=f"Sharpe 5th/50th/95th = {lo:.2f} / {med:.2f} / {hi:.2f}",
        note="If the 5th percentile is below zero, the sample cannot rule out "
             "no edge, however good the point estimate looks.",
        value=float(lo),
    )


# ------------------------------------------------------------ 7. monotonicity
def decile_monotonicity(prices, score, dates, horizon: int, mask=None,
                        require: bool = True) -> Finding:
    q = E.decile_returns(score, prices, dates, horizon, 10, mask)
    if q.empty:
        return Finding("Decile monotonicity", False, "no periods scored", value=np.nan)
    means = q.mean()
    rho = E.spearman(pd.Series(means.to_numpy()),
                     pd.Series(np.arange(len(means), dtype=float)))
    spread = float(means.iloc[-1] - means.iloc[0])
    return Finding(
        name="Decile monotonicity", passed=(not require) or rho > 0.6,
        detail=f"rank corr {rho:+.2f}, top-minus-bottom {spread*100:+.2f}%/period",
        note="A signal that only works in the extreme tails is usually a few "
             "outliers wearing a factor's clothes.",
        value=rho, table=means.rename("mean_fwd_return").to_frame(),
    )
