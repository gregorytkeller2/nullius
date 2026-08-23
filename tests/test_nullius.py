"""The tool has one job: kill fakes and spare real signals.

Everything here runs on synthetic panels with a known answer, so a failure
means the harness is wrong rather than the market being unkind.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nullius import Study, Preregistration
from nullius import engine as E
from nullius import attacks as A
from nullius.prereg import canonical_hash, _mini_yaml

N_DAYS, N_NAMES = 1500, 220


def planted(seed=0, strength=0.0012):
    """Half the names drift up, half down, fixed for the whole sample."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=N_DAYS)
    drift = np.where(np.arange(N_NAMES) % 2 == 0, strength, -strength)
    r = rng.normal(0, 0.013, (N_DAYS, N_NAMES)) + drift
    return pd.DataFrame(50 * np.exp(np.cumsum(r, axis=0)), index=idx,
                        columns=[f"S{i:03d}" for i in range(N_NAMES)])


def momentum(px, lookback=252, skip=21):
    past = px.shift(skip)
    raw = past / past.shift(lookback) - 1.0
    z = raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1), axis=0)
    return z.clip(-3, 3)


def noise_signal(px, seed=99):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=px.shape), index=px.index, columns=px.columns)


def leaky_signal(px):
    """Deliberately cheats: ranks on returns 21 days in the FUTURE."""
    fwd = px.shift(-21) / px - 1.0
    return fwd.sub(fwd.mean(axis=1), axis=0).div(fwd.std(axis=1), axis=0)


@pytest.fixture(scope="module")
def px():
    return planted()


@pytest.fixture
def prereg(tmp_path):
    p = tmp_path / "criteria.yaml"
    p.write_text(Preregistration.template(), encoding="utf-8")
    return p


# ---------------------------------------------------------------- prereg lock
def test_lock_is_written_and_sealed(prereg):
    pr = Preregistration.load(prereg)
    assert pr.lock.status == "SEALED"
    assert (prereg.parent / "criteria.lock.json").exists()


def test_editing_criteria_after_the_lock_is_flagged(prereg):
    Preregistration.load(prereg)
    prereg.write_text(prereg.read_text().replace("min_sharpe: 0.50",
                                                 "min_sharpe: 0.10"))
    pr2 = Preregistration.load(prereg)
    assert pr2.lock.status == "AMENDED"
    assert any("min_sharpe" in d for d in pr2.lock.diff)


def test_amendment_history_is_appended_not_overwritten(prereg):
    Preregistration.load(prereg)
    for v in ("0.20", "0.30"):
        prereg.write_text(prereg.read_text().replace("min_sharpe: 0.50",
                                                     f"min_sharpe: {v}"))
        Preregistration.load(prereg)
        prereg.write_text(prereg.read_text().replace(f"min_sharpe: {v}",
                                                     "min_sharpe: 0.50"))
    hist = json.loads((prereg.parent / "criteria.lock.json").read_text())
    assert len(hist.get("amendments", [])) >= 2


def test_hash_ignores_key_order():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_missing_prereg_refuses_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="BEFORE running"):
        Preregistration.load(tmp_path / "nope.yaml")


def test_template_parses_without_pyyaml():
    d = _mini_yaml(Preregistration.template())
    assert d["kill_criteria"]["min_sharpe"] == 0.50
    assert d["kill_criteria"]["require_monotonic_deciles"] is True


# ---------------------------------------------------------------- the attacks
def test_canary_catches_a_leaky_signal(px):
    """The headline test. A signal that uses future data must be caught."""
    spec = E.BacktestSpec()
    honest = A.lookahead_canary(px, momentum(px), spec)
    leaky = A.lookahead_canary(px, leaky_signal(px), spec)
    assert honest.passed, "an honest signal should pass the canary"
    assert not leaky.passed, "a leaky signal must FAIL the canary"


def test_permutation_kills_noise(px):
    f = A.permutation_null(px, noise_signal(px), E.BacktestSpec(), n=40)
    assert not f.passed


def test_permutation_spares_a_real_signal(px):
    f = A.permutation_null(px, momentum(px), E.BacktestSpec(), n=40)
    assert f.passed, f"real signal wrongly killed: {f.detail}"


def test_decile_monotonicity_separates_signal_from_noise(px):
    dates = E.rebalance_dates(px.index)
    good = A.decile_monotonicity(px, momentum(px), dates, 21)
    bad = A.decile_monotonicity(px, noise_signal(px), dates, 21)
    assert good.passed and not bad.passed


def test_cost_curve_finds_a_breakeven(px):
    f = A.cost_curve(px, momentum(px), E.BacktestSpec())
    assert np.isfinite(f.value) and f.value > 0 and f.passed


def test_cost_curve_kills_a_high_turnover_signal(px):
    """A signal with no edge still trades constantly — break-even must be tiny."""
    f = A.cost_curve(px, noise_signal(px), E.BacktestSpec(), min_breakeven=20)
    assert not f.passed


def test_bootstrap_flags_a_short_sample():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 40))
    assert not A.block_bootstrap(r).passed


def test_breadth_curve_rises_with_universe_size(px):
    f = A.breadth_curve(px, momentum, E.BacktestSpec(),
                        sizes=(20, 200), draws=3)
    assert f.table is not None and len(f.table) == 2


# ---------------------------------------------------------------- engine
def test_engine_charges_costs(px):
    from dataclasses import replace
    spec = E.BacktestSpec()
    free = E.cagr(E.run(px, momentum(px), replace(spec, cost_bps=0)).returns)
    dear = E.cagr(E.run(px, momentum(px), replace(spec, cost_bps=100)).returns)
    assert free > dear


def test_engine_respects_the_weight_cap(px):
    spec = E.BacktestSpec(max_weight=0.05)
    res = E.run(px, momentum(px), spec)
    assert res.weights.abs().max().max() <= 0.05 + 1e-9


def test_engine_starts_when_the_book_does(px):
    res = E.run(px, momentum(px), E.BacktestSpec())
    assert res.returns.index[0] > px.index[250]


# ---------------------------------------------------------------- end to end
def test_study_kills_pure_noise(px, prereg):
    rep = Study("noise", px, noise_signal, prereg).run(permutations=30, verbose=False)
    assert not rep.survived
    assert "KILLED" in rep.verdict


def test_study_spares_a_planted_signal(px, prereg):
    rep = Study("planted", px, momentum, prereg).run(permutations=30, verbose=False)
    failed = [f.name for f in rep.findings if not f.passed]
    assert failed == [], f"real signal failed: {failed}"


def test_study_reports_amendment_in_the_verdict(px, prereg):
    Study("first", px, momentum, prereg).run(permutations=10, verbose=False)
    prereg.write_text(prereg.read_text().replace("min_sharpe: 0.50",
                                                 "min_sharpe: 0.01"))
    rep = Study("second", px, momentum, prereg).run(permutations=10, verbose=False)
    assert rep.prereg_status == "AMENDED"
    assert "amended" in rep.verdict.lower()
