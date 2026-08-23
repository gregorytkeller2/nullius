"""nullius — a falsification harness for trading strategies.

*Nullius in verba* — take nobody's word for it. Including your own.

Backtesting libraries help you produce a result. This one helps you destroy it.
Point it at any signal and it runs the attacks that separate a real edge from a
plausible-looking artefact: a look-ahead canary, a permutation null, a breadth
experiment, a parameter plateau check, a cost curve, and a block bootstrap.

The design assumption is that the researcher is the adversary. Kill criteria are
written to a file and hashed before the first run; editing them afterwards
stamps every subsequent report AMENDED and prints the diff.

    from nullius import Study

    study = Study(name="my idea", prices=px, signal=my_signal,
                  prereg="criteria.yaml")
    report = study.run()
    print(report)          # SURVIVED / KILLED, and exactly why
"""
from .prereg import Preregistration, PreregLock
from .study import Study
from .report import Report, Finding
from .universe import PointInTimeUniverse, SP500
from .walkforward import walk_forward, make_windows

__version__ = "0.1.0"
__all__ = ["Study", "Preregistration", "PreregLock", "Report", "Finding",
           "PointInTimeUniverse", "SP500", "walk_forward", "make_windows"]
