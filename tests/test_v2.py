"""Tests for walk-forward, point-in-time universes, and the CLI."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nullius import Study, Preregistration, SP500, PointInTimeUniverse
from nullius import engine as E
from nullius import walkforward as W
from nullius.cli import main as cli_main

N_DAYS, N_NAMES = 1800, 200


def panel(seed=0, strength=0.0012, flip_at=None):
    """Optionally flip the drift halfway, so the edge genuinely reverses."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=N_DAYS)
    base = np.where(np.arange(N_NAMES) % 2 == 0, strength, -strength)
    d = np.tile(base, (N_DAYS, 1))
    if flip_at:
        d[flip_at:] *= -1
    r = rng.normal(0, 0.013, (N_DAYS, N_NAMES)) + d
    return pd.DataFrame(50 * np.exp(np.cumsum(r, axis=0)), index=idx,
                        columns=[f"S{i:03d}" for i in range(N_NAMES)])


def momentum(px, lookback=252, skip=21):
    past = px.shift(skip)
    raw = past / past.shift(lookback) - 1.0
    return raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1), axis=0).clip(-3, 3)


def noise(px, seed=5):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=px.shape), index=px.index, columns=px.columns)


@pytest.fixture(scope="module")
def px():
    return panel()


@pytest.fixture
def prereg(tmp_path):
    p = tmp_path / "criteria.yaml"
    p.write_text(Preregistration.template(), encoding="utf-8")
    return p


# ------------------------------------------------------------- windows
def test_single_split_is_disjoint(px):
    w = W.make_windows(px.index, "single")[0]
    assert w.train[1] == w.test[0] and w.train[0] < w.train[1] < w.test[1]


def test_anchored_windows_share_a_start(px):
    ws = W.make_windows(px.index, "anchored", train_years=2, test_years=1)
    assert len(ws) >= 2
    assert all(w.train[0] == ws[0].train[0] for w in ws)


def test_rolling_windows_move_their_start(px):
    ws = W.make_windows(px.index, "rolling", train_years=2, test_years=1, step_years=1)
    assert len(ws) >= 2 and ws[1].train[0] > ws[0].train[0]


def test_no_window_leaks_test_into_train(px):
    for mode in ("single", "anchored", "rolling"):
        for w in W.make_windows(px.index, mode, train_years=2, test_years=1):
            assert w.test[0] >= w.train[1], f"{mode} {w.label} overlaps"


# ------------------------------------------------------------- verdicts
def test_persistent_edge_is_preserved(px):
    f = W.walk_forward(px, momentum(px), E.BacktestSpec(), mode="single")
    assert f.table.verdict.iloc[0] in ("PRESERVED", "DEGRADED")
    assert f.table.is_edge.iloc[0] > 0


@pytest.mark.parametrize("is_edge,oos_edge,expected", [
    (0.10, 0.08, "PRESERVED"),      # kept most of it
    (0.10, 0.05, "PRESERVED"),      # exactly half survives
    (0.10, 0.02, "DEGRADED"),       # same sign, most of it gone
    (0.10, -0.03, "SIGN_FLIP"),     # inverted — the overfitting signature
    (-0.02, 0.09, "NO_TRAIN_EDGE"),  # no edge in sample, so nothing was tested
    (0.0, 0.09, "NO_TRAIN_EDGE"),
    (float("nan"), 0.05, "NO_TRAIN_EDGE"),
    (0.10, float("nan"), "SIGN_FLIP"),
])
def test_verdict_classification(is_edge, oos_edge, expected):
    """Exhaustive test of the branch logic.

    Done on the numbers rather than on generated prices: a trend-follower
    latches onto whatever trend exists, so simply reversing a synthetic drift
    produces a NEW trend it follows happily — which tests the panel, not the
    verdict.
    """
    assert W.classify(is_edge, oos_edge) == expected


def test_reverting_market_is_caught():
    """A market where past winners become future losers must not read as real."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2012-01-02", periods=N_DAYS)
    strength, look = 0.0016, 252
    lr = np.zeros((N_DAYS, N_NAMES))
    base = np.where(np.arange(N_NAMES) % 2 == 0, strength, -strength)
    half = N_DAYS // 2
    for t in range(N_DAYS):
        if t < half:
            drift = base                       # persistent: momentum works
        else:                                  # reverting: winners turn around
            window = lr[max(0, t - look):t].sum(axis=0)
            drift = -strength * np.sign(window)
        lr[t] = rng.normal(0, 0.011, N_NAMES) + drift
    p = pd.DataFrame(50 * np.exp(np.cumsum(lr, axis=0)), index=idx,
                     columns=[f"S{i:03d}" for i in range(N_NAMES)])
    f = W.walk_forward(p, momentum(p), E.BacktestSpec(), mode="single",
                       split=str(idx[half].date()))
    assert f.table.verdict.iloc[0] in ("SIGN_FLIP", "NO_TRAIN_EDGE")
    assert not f.passed


def test_no_edge_reports_no_train_edge(px):
    f = W.walk_forward(px, noise(px), E.BacktestSpec(), mode="single")
    assert f.table.verdict.iloc[0] in ("NO_TRAIN_EDGE", "SIGN_FLIP")
    assert not f.passed


def test_overall_verdict_is_the_worst_window(px):
    f = W.walk_forward(px, momentum(px), E.BacktestSpec(), mode="rolling",
                       train_years=2, test_years=1, step_years=1)
    worst = min(f.table.verdict, key=lambda v: W._ORDER.index(v))
    assert worst in f.detail


def test_short_sample_refuses_to_split(px):
    f = W.walk_forward(px.iloc[:300], momentum(px.iloc[:300]), E.BacktestSpec())
    assert not f.passed and "too short" in f.detail


def test_market_neutral_book_benchmarks_against_cash(px):
    b = W._benchmark(px, E.BacktestSpec(long_short=True), "auto")
    assert (b == 0).all()
    b2 = W._benchmark(px, E.BacktestSpec(long_short=False), "auto")
    assert not (b2 == 0).all()


# ------------------------------------------------------------- universe
def test_sp500_membership_loads():
    u = SP500()
    assert len(u.tickers) > 1000
    assert len(u.members_on("2010-06-30")) > 450


def test_departed_members_are_present():
    u = SP500()
    gone = set(u.tickers) - set(u.members_on("2026-01-01"))
    assert len(gone) > 500, "the whole point is the names that left"


def test_mask_is_point_in_time():
    u = SP500()
    dates = pd.bdate_range("2005-01-03", periods=40)
    cols = u.tickers[:300]
    m = u.mask(dates, cols)
    assert m.shape == (40, 300) and m.any().any()
    for d in dates[:5]:
        assert set(m.columns[m.loc[d]]) <= set(u.members_on(d))


def test_mask_excludes_unknown_tickers():
    u = SP500()
    m = u.mask(pd.bdate_range("2010-01-04", periods=5), ["NOT_A_TICKER"])
    assert not m.to_numpy().any()


def test_coverage_reports_the_gap():
    u = SP500()
    have, want = u.coverage(u.tickers[:400])
    assert have == 400 and want == len(u.tickers)


def test_mask_restricts_the_backtest(px):
    """A mask that admits nothing must produce no positions."""
    empty = pd.DataFrame(False, index=px.index, columns=px.columns)
    res = E.run(px, momentum(px), E.BacktestSpec(), empty)
    assert res.n_positions.max() == 0


# ------------------------------------------------------------- CLI
def test_cli_init_writes_scaffold(tmp_path):
    assert cli_main(["init", str(tmp_path)]) == 0
    assert (tmp_path / "criteria.yaml").exists()
    assert (tmp_path / "study.py").exists()


def test_cli_init_does_not_clobber(tmp_path):
    cli_main(["init", str(tmp_path)])
    (tmp_path / "criteria.yaml").write_text("hypothesis: mine\n")
    cli_main(["init", str(tmp_path)])
    assert "mine" in (tmp_path / "criteria.yaml").read_text()


def test_cli_attacks_lists_them(capsys):
    assert cli_main(["attacks"]) == 0
    assert "Look-ahead canary" in capsys.readouterr().out


def test_cli_run_end_to_end(tmp_path, px):
    px.to_pickle(tmp_path / "px.pkl")
    (tmp_path / "criteria.yaml").write_text(Preregistration.template())
    (tmp_path / "s.py").write_text(f'''
import pandas as pd, numpy as np, sys
sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
from nullius import Study
from nullius.engine import BacktestSpec
def signal(px):
    past = px.shift(21); raw = past / past.shift(252) - 1.0
    return raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1), axis=0).clip(-3, 3)
def build():
    px = pd.read_pickle({str(tmp_path / "px.pkl")!r})
    return Study("cli test", px, signal, {str(tmp_path / "criteria.yaml")!r},
                 spec=BacktestSpec(), walk_forward=False)
''')
    rc = cli_main(["run", str(tmp_path / "s.py"), "--permutations", "10", "--quiet"])
    assert rc in (0, 1)
    assert (tmp_path / "s.report.html").exists()


def test_cli_run_rejects_a_bad_file(tmp_path, capsys):
    (tmp_path / "empty.py").write_text("x = 1\n")
    assert cli_main(["run", str(tmp_path / "empty.py")]) == 2
    assert "must define" in capsys.readouterr().out
