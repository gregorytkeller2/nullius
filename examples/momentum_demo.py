"""nullius on a real universe: 1,280 US equities, 2018-2023.

Cross-sectional momentum — the effect this universe is known to contain.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/root/quantstrat")

import numpy as np, pandas as pd
from nullius import Study
from nullius.engine import BacktestSpec


def load_prices():
    import skfolio.datasets as sk
    px = sk.load_nasdaq_dataset()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    px = px.loc[:, px.median() >= 5.0]
    return px.loc[:, px.pct_change().abs().max() < 3.0]


def momentum(px, lookback=252, skip=21):
    """Risk-adjusted momentum: trailing return over trailing volatility."""
    past = px.shift(skip)
    raw = past / past.shift(lookback) - 1.0
    vol = px.pct_change().rolling(126, min_periods=100).std() * np.sqrt(252)
    adj = raw / vol.replace(0, np.nan)
    z = adj.sub(adj.mean(axis=1), axis=0).div(adj.std(axis=1), axis=0)
    return z.clip(-3, 3)


if __name__ == "__main__":
    px = load_prices()
    print(f"universe: {px.shape[1]} names, {px.shape[0]} days, "
          f"{px.index[0].date()} -> {px.index[-1].date()}\n")
    study = Study(
        name="Cross-sectional momentum — NASDAQ 2018-2023",
        prices=px, signal=momentum,
        prereg=Path(__file__).parent / "criteria.yaml",
        spec=BacktestSpec(quantile=0.10, long_short=True,
                          weighting="inverse_vol", max_weight=0.05, cost_bps=7.5),
        param_grid={"lookback": [126, 252], "skip": [0, 21], "quantile": [0.1, 0.2]},
    )
    report = study.run(permutations=80)
    print(report)
    out = Path(__file__).parent / "report.html"
    report.to_html(out)
    print(f"\n[nullius] html report -> {out}")
