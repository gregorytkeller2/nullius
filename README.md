# nullius

**A falsification harness for trading strategies.**

*Nullius in verba* — take nobody's word for it. Including your own.

Backtesting libraries help you produce a result. This one helps you destroy it.

```python
from nullius import Study

study = Study(name="my idea", prices=px, signal=my_signal,
              prereg="criteria.yaml")
print(study.run())
```

```
  pre-registration: SEALED [345b1d1c90e28960]
  Sharpe 0.23   CAGR +2.5%   MaxDD -39.2%   turnover 18.7x   years 4.2

  ATTACKS
  PASS  Look-ahead canary        future-shifted Sharpe 6.21 vs honest 0.23
  FAIL  Decile monotonicity      rank corr +0.59, top-minus-bottom +0.48%/period
  PASS  Cost sensitivity         break-even 21.3 bps/side at 18.7x turnover
  FAIL  Bootstrap                Sharpe 5th/50th/95th = -0.52 / 0.25 / 1.04
  FAIL  Permutation null         p = 0.225 (80 shuffles, zero-cost)
  PASS  Breadth                  20n:+0.27 | 100n:+0.22 | 250n:+0.11 | 500n:+0.19
  PASS  Parameter plateau        8 configs, 100% positive, median 0.27
  FAIL  Minimum Sharpe           0.23 vs pre-set 0.40
  FAIL  Signal IC t-stat         IC +0.0367, t = +1.82 vs pre-set 2.0

  VERDICT: KILLED  (5/10 attacks survived)
```

---

## Why

The bottleneck in retail quant research is not running backtests. It is that a
backtest will happily tell you what you want to hear, and the standard tools
have no opinion about that. Every library will compute your Sharpe ratio. None
of them will ask whether your signal is reading tomorrow's prices, whether a
shuffled version of it does just as well, or whether you moved your success
threshold after seeing the answer.

This package assumes the researcher is the adversary. That is not cynicism — it
is the position every one of us is in, because the person running the test wants
it to succeed.

## The pre-registration lock

The feature the rest of the package exists to support.

You write your kill criteria to a file *before* the first run. `nullius` hashes
it and keeps the hash. Change the criteria afterwards and every subsequent
report is stamped `AMENDED` and prints exactly what moved:

```
  pre-registration: AMENDED [f1c0a2...]
  !! criteria changed after the lock was written:
       kill_criteria.min_sharpe: 0.5 -> 0.3
```

Amending is allowed. Amending *silently* is not. The original lock is never
overwritten — amendments are appended, so the history of what you promised stays
visible.

The failure mode this prevents is not fraud, it is drift. You set a threshold,
miss it by 0.03, and a perfectly reasonable argument arrives for why the
threshold was always too strict. The argument is reasonable. It is also how
overfitted strategies get believed.

## The attacks

| Attack | The question it asks |
|---|---|
| **Look-ahead canary** | Feeds the engine a future-shifted signal. If that isn't dramatically better than the honest one, the honest one is already leaking. |
| **Permutation null** | Reshuffles the signal across names, keeps everything else identical. This is what "no edge, same plumbing" scores. Run at zero cost, so it measures forecasting power rather than the turnover advantage of a slow signal. |
| **Breadth curve** | Re-runs on random sub-universes. Information ratio scales with √breadth, so a null result on a narrow universe is an artefact, not evidence — this measures the curve instead of assuming it. |
| **Parameter plateau** | Sweeps the knobs. A plateau is credible; a peak is fitted. |
| **Cost sensitivity** | Computes the break-even cost. An edge that exists only at zero friction is not an edge. |
| **Block bootstrap** | Resamples 21-day blocks so volatility clustering survives. If the 5th percentile Sharpe is below zero, the sample cannot rule out nothing. |
| **Decile monotonicity** | Checks the response is ordered across the whole cross-section, not just the tails. |
| **Signal IC t-stat** | Rank correlation across every name and period — far more powerful than the Sharpe of one portfolio, which throws most of the information away. |
| **Walk-forward** | Same parameters on train and test — nothing is re-optimised, because doing so would mean the harness performs the tuning you are trying to detect. Compares the *edge* over the benchmark, not raw return. Five verdicts; the overall one is the **worst** window, not the average. |

## Point-in-time universes

Survivorship bias is the one defect no out-of-sample test can find — split a
biased universe in half and you get two biased halves. `nullius` ships
historical S&P 500 membership, 1996 to present: 1,206 tickers, of which **703
are no longer in the index**.

```python
from nullius import SP500

u = SP500()
mask = u.mask(prices.index, prices.columns)     # point-in-time eligibility
have, want = u.coverage(prices.columns)         # and report the gap honestly
Study(..., mask=mask)
```

The mask is ANDed with everything else, never ORed, so it can only ever
restrict the universe. Free price vendors drop many delisted tickers; the
missing names fall on the short leg, where the disappearances would have been,
so the residual bias understates short-side profit rather than flattering it.
`coverage()` exists so you report that rather than assume it away.

## Command line

```bash
nullius init                # scaffold criteria.yaml and study.py
nullius run study.py        # run it, print the verdict, write the HTML report
nullius attacks             # list what will be thrown at your signal
```

`nullius run` exits 0 if the study survived and 1 if it was killed, so it drops
straight into CI — a strategy that stops surviving becomes a failing build.

## Install and use

```bash
pip install pandas numpy        # that is the whole dependency list
python -m pytest tests -q       # 49 tests, no network needed
python examples/momentum_demo.py
```

Your signal is any callable `prices -> scores`, same shape, causal:

```python
def momentum(px, lookback=252, skip=21):
    past = px.shift(skip)
    raw = past / past.shift(lookback) - 1.0
    return raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1), axis=0)
```

`Preregistration.template()` prints a criteria file to start from.

## What it will not do

It will not tell you a strategy is good. Surviving every attack means the result
is not obviously an artefact — which is a much weaker claim, and the strongest
one available. Nothing here substitutes for out-of-sample time.

It also cannot see survivorship bias. Splitting a biased universe in half gives
two biased halves; only a point-in-time universe fixes that, and that is a data
problem rather than a harness problem.

## Provenance

Every attack here was written to kill a specific strategy that its author
believed in, and most of them succeeded. The look-ahead canary, the breadth
experiment, and the pre-registration lock each exist because a result that
looked real turned out not to be — and the specific way it failed is documented
in the test suite.

The demo above is the honest illustration: the same momentum effect scores
t = 1.82 with a single-lookback signal and a generic engine, and t = 2.45 with
multi-horizon scoring and better portfolio construction. The harness does not
tell you your idea is bad. It tells you how much of your result was construction.

MIT licensed. Not investment advice.
