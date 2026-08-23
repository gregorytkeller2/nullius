"""Command line interface.

    nullius init                 scaffold criteria.yaml and study.py
    nullius run study.py         run the study, print the verdict, write HTML
    nullius attacks              list what will be thrown at your signal
"""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path

STUDY_TEMPLATE = '''"""A nullius study. Run it with:  nullius run study.py"""
import numpy as np
import pandas as pd

from nullius import Study
from nullius.engine import BacktestSpec


def load_prices() -> pd.DataFrame:
    """Return a (dates x tickers) frame of ADJUSTED closes.

    Replace this with your own loader. Anything that returns adjusted closes
    indexed by date works.
    """
    raise NotImplementedError("point load_prices() at your data")


def signal(px: pd.DataFrame) -> pd.DataFrame:
    """prices -> cross-sectional scores, same shape.

    Must be causal: the value on date t may use prices up to and including t
    and nothing later. The look-ahead canary checks you meant it.
    """
    past = px.shift(21)
    raw = past / past.shift(252) - 1.0
    z = raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1), axis=0)
    return z.clip(-3, 3)


def build() -> Study:
    px = load_prices()
    return Study(
        name="my idea",
        prices=px,
        signal=signal,
        prereg="criteria.yaml",
        spec=BacktestSpec(quantile=0.10, long_short=True, cost_bps=7.5),
        # mask=SP500().mask(px.index, px.columns),   # point-in-time universe
        # param_grid={"lookback": [126, 252], "quantile": [0.1, 0.2]},
    )
'''


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def cmd_init(args) -> int:
    from .prereg import Preregistration
    d = Path(args.directory)
    d.mkdir(parents=True, exist_ok=True)
    crit, study = d / "criteria.yaml", d / "study.py"
    for p, text in ((crit, Preregistration.template()), (study, STUDY_TEMPLATE)):
        if p.exists() and not args.force:
            print(f"  exists, left alone: {p}")
            continue
        p.write_text(text, encoding="utf-8")
        print(f"  wrote {p}")
    print("\nEdit criteria.yaml FIRST — before you look at any result. That is "
          "the whole point.\nThen: nullius run study.py")
    return 0


def cmd_run(args) -> int:
    path = Path(args.study)
    if not path.exists():
        print(f"no such file: {path}")
        return 2
    mod = _load_module(path)
    study = getattr(mod, "study", None)
    if study is None:
        builder = getattr(mod, "build", None)
        if builder is None:
            print(f"{path} must define `study` or `build()` returning a Study")
            return 2
        study = builder()

    report = study.run(permutations=args.permutations, verbose=not args.quiet)
    print()
    print(report)

    out = Path(args.out) if args.out else path.with_suffix(".report.html")
    report.to_html(out)
    print(f"\n[nullius] html report -> {out}")
    return 0 if report.survived else 1


def cmd_attacks(args) -> int:
    rows = [
        ("Look-ahead canary", "feeds the engine a future-shifted signal; if that "
         "is not far better, the honest run is already leaking"),
        ("Permutation null", "reshuffles the signal across names, everything else "
         "identical; run at zero cost so it measures forecasting power"),
        ("Walk-forward", "same parameters on train and test; compares the EDGE, "
         "verdict is the worst window"),
        ("Breadth curve", "re-runs on random sub-universes; a null on 20 names "
         "is an artefact, not evidence"),
        ("Parameter plateau", "sweeps the knobs; a plateau is credible, a peak "
         "is fitted"),
        ("Cost sensitivity", "break-even cost; an edge that needs zero friction "
         "is not an edge"),
        ("Block bootstrap", "21-day blocks; if the 5th percentile Sharpe is "
         "below zero the sample cannot rule out nothing"),
        ("Decile monotonicity", "ordered response across the whole cross-section, "
         "not just the tails"),
        ("Signal IC t-stat", "rank correlation over every name and period — far "
         "more powerful than one portfolio's Sharpe"),
    ]
    print("\n  nullius attacks\n")
    for n, d in rows:
        print(f"  {n:<22} {d}")
    print()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="nullius",
        description="A falsification harness for trading strategies. "
                    "Take nobody's word for it, including your own.")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="scaffold criteria.yaml and study.py")
    i.add_argument("directory", nargs="?", default=".")
    i.add_argument("--force", action="store_true", help="overwrite existing files")
    i.set_defaults(func=cmd_init)

    r = sub.add_parser("run", help="run a study and write its report")
    r.add_argument("study")
    r.add_argument("--permutations", type=int, default=200)
    r.add_argument("--out", default=None, help="HTML output path")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("attacks", help="list the attacks")
    a.set_defaults(func=cmd_attacks)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
