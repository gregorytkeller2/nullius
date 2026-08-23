"""The Study: point it at a signal, get a verdict."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from . import attacks as A
from . import engine as E
from . import walkforward as W
from .prereg import Preregistration
from .report import Report


class Study:
    """A falsification study of one signal on one universe.

    `signal` is any callable taking a price frame and returning a frame of
    cross-sectional scores with the same shape. It must be causal — the value
    on date t may use prices up to and including t and nothing later. The
    look-ahead canary is there to check you got that right.
    """

    def __init__(self, name: str, prices: pd.DataFrame, signal: Callable,
                 prereg: str | Path, spec: E.BacktestSpec | None = None,
                 mask: pd.DataFrame | None = None,
                 param_grid: dict | None = None,
                 horizon: int = 21, lock_dir: str | Path | None = None,
                 walk_forward: dict | bool | None = True):
        self.name = name
        self.prices = prices.sort_index()
        self.signal = signal
        self.spec = spec or E.BacktestSpec()
        self.mask = mask
        self.param_grid = param_grid
        self.horizon = horizon
        if walk_forward is True:
            walk_forward = {"mode": "single"}
        self.walk_forward = walk_forward or None
        self.prereg = Preregistration.load(prereg, lock_dir)
        self.score = signal(self.prices)
        if self.score.shape != self.prices.shape:
            self.score = self.score.reindex(index=self.prices.index,
                                            columns=self.prices.columns)

    # ------------------------------------------------------------------
    def run(self, permutations: int = 200, verbose: bool = True) -> Report:
        k = self.prereg.get
        res = E.run(self.prices, self.score, self.spec, self.mask)
        r = res.returns
        dates = E.rebalance_dates(self.prices.index, self.spec.rebalance)

        sh, cg, dd = E.sharpe(r), E.cagr(r), E.max_drawdown(r)
        headline = {"Sharpe": f"{sh:.2f}", "CAGR": f"{cg:+.1%}",
                    "MaxDD": f"{dd:+.1%}", "turnover": f"{res.annual_turnover:.1f}x",
                    "years": f"{res.years:.1f}"}

        findings, warns = [], []

        def step(msg, fn):
            if verbose:
                print(f"  [nullius] {msg}", flush=True)
            findings.append(fn())

        step("look-ahead canary",
             lambda: A.lookahead_canary(self.prices, self.score, self.spec, self.mask))
        step("decile monotonicity",
             lambda: A.decile_monotonicity(
                 self.prices, self.score, dates, self.horizon, self.mask,
                 require=bool(k("require_monotonic_deciles", True))))
        step("cost sensitivity",
             lambda: A.cost_curve(self.prices, self.score, self.spec, self.mask,
                                  min_breakeven=float(k("min_breakeven_bps", 20))))
        step("block bootstrap", lambda: A.block_bootstrap(r))
        step(f"permutation null ({permutations} shuffles)",
             lambda: A.permutation_null(self.prices, self.score, self.spec,
                                        self.mask, n=permutations,
                                        max_p=float(k("max_permutation_p", 0.05))))
        step("breadth curve",
             lambda: A.breadth_curve(self.prices, self.signal, self.spec,
                                     mask=self.mask))
        if self.walk_forward:
            step(f"walk-forward ({self.walk_forward.get('mode', 'single')})",
                 lambda: W.walk_forward(self.prices, self.score, self.spec,
                                        self.mask, **self.walk_forward))
        if self.param_grid:
            step("parameter plateau",
                 lambda: A.parameter_plateau(self.prices, self.signal, self.spec,
                                             self.param_grid, self.mask))

        # criteria that read straight off the headline
        ms = k("min_sharpe")
        if ms is not None:
            findings.append(A.Finding(
                "Minimum Sharpe", bool(sh >= float(ms)),
                f"{sh:.2f} vs pre-set {float(ms):.2f}", value=sh))
        mt = k("max_turnover")
        if mt is not None:
            findings.append(A.Finding(
                "Turnover ceiling", bool(res.annual_turnover <= float(mt)),
                f"{res.annual_turnover:.1f}x vs pre-set {float(mt):.0f}x",
                value=res.annual_turnover))
        my = k("min_years")
        if my is not None and res.years < float(my):
            warns.append(f"sample is {res.years:.1f} years, below the "
                         f"{float(my):.0f} you pre-registered as adequate")
        mit = k("min_ic_tstat")
        if mit is not None:
            ic = E.rank_ic(self.score, self.prices, dates, self.horizon, self.mask)
            t = float(ic.mean() / ic.std(ddof=1) * np.sqrt(ic.notna().sum())) \
                if ic.std(ddof=1) > 0 else np.nan
            findings.append(A.Finding(
                "Signal IC t-stat", bool(np.isfinite(t) and abs(t) >= float(mit)),
                f"IC {ic.mean():+.4f}, t = {t:+.2f} vs pre-set {float(mit):.1f}",
                note="Rank correlation uses the whole cross-section, so it is a "
                     "far more powerful test than the Sharpe of one portfolio.",
                value=t))

        return Report(name=self.name, hypothesis=self.prereg.hypothesis,
                      prereg_digest=self.prereg.lock.digest,
                      prereg_status=self.prereg.lock.status,
                      prereg_diff=self.prereg.lock.diff,
                      headline=headline, findings=findings, warnings=warns)
