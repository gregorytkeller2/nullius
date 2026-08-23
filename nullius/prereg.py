"""Pre-registration: kill criteria that cannot be quietly rewritten.

The failure mode this exists to prevent is not dishonesty, it is drift. You set
a threshold, you miss it by 0.03, and a perfectly reasonable argument arrives
for why 2.8 was always too strict. The argument is reasonable. It is also how
every overfitted strategy in history got published.

So the criteria go in a file, the file gets hashed, and the hash goes in the
report. Change the criteria after a run and every later report says AMENDED and
shows what moved. Nothing is forbidden -- you can always amend -- but you cannot
do it silently, and the reader always knows.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except ImportError:
            return _mini_yaml(text)
    return json.loads(text)


def _mini_yaml(text: str) -> dict:
    """Two-level YAML subset, so PyYAML stays optional.

    Handles `key: value` and one level of indented block. Anything more
    elaborate should install PyYAML; this exists so the package has no hard
    dependency beyond pandas and numpy.
    """
    out: dict[str, Any] = {}
    cur: dict | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indented = line[:1].isspace()
        key, _, val = line.strip().partition(":")
        key, val = key.strip(), val.strip()
        if indented and cur is not None:
            cur[key] = _coerce(val)
        else:
            if val == "":
                cur = {}
                out[key] = cur
            else:
                out[key] = _coerce(val)
                cur = None
    return out


def _coerce(v: str):
    if v == "":
        return None
    low = v.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v.strip("'\"")


def canonical_hash(d: dict) -> str:
    """Stable hash of the criteria, insensitive to key order and formatting."""
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class PreregLock:
    """The record of what the criteria were when they were first committed."""
    digest: str
    criteria: dict
    amended: bool = False
    diff: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "AMENDED" if self.amended else "SEALED"


@dataclass
class Preregistration:
    """Kill criteria loaded from disk, with tamper detection."""

    path: Path
    criteria: dict
    lock: PreregLock

    @classmethod
    def load(cls, path, lock_dir=None) -> "Preregistration":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no pre-registration at {path}. Write your kill criteria "
                "BEFORE running the study — that is the entire point. "
                "`Preregistration.template()` prints a starting file."
            )
        crit = _load(path)
        digest = canonical_hash(crit)
        lock_path = Path(lock_dir or path.parent) / (path.stem + ".lock.json")

        amended, diff = False, []
        if lock_path.exists():
            prev = json.loads(lock_path.read_text(encoding="utf-8"))
            if prev.get("digest") != digest:
                amended = True
                diff = _diff(prev.get("criteria", {}), crit)
                # The original is never overwritten. An amendment is appended,
                # so the history of what you promised stays visible.
                hist = prev.setdefault("amendments", [])
                hist.append({"digest": digest, "criteria": crit, "diff": diff})
                lock_path.write_text(json.dumps(prev, indent=2), encoding="utf-8")
        else:
            lock_path.write_text(
                json.dumps({"digest": digest, "criteria": crit}, indent=2),
                encoding="utf-8")

        return cls(path=path, criteria=crit,
                   lock=PreregLock(digest, crit, amended, diff))

    # ---------------------------------------------------------------- helpers
    def get(self, key, default=None):
        return self.criteria.get("kill_criteria", {}).get(key, default)

    @property
    def hypothesis(self) -> str:
        return str(self.criteria.get("hypothesis", "(none stated)"))

    @staticmethod
    def template() -> str:
        return TEMPLATE


def _diff(old: dict, new: dict) -> list[str]:
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict) or isinstance(b, dict):
            for kk in sorted(set(a or {}) | set(b or {})):
                x, y = (a or {}).get(kk), (b or {}).get(kk)
                if x != y:
                    out.append(f"{k}.{kk}: {x!r} -> {y!r}")
        elif a != b:
            out.append(f"{k}: {a!r} -> {b!r}")
    return out


TEMPLATE = """\
# Written BEFORE the first run. Hashed on load; edits are recorded, not hidden.
hypothesis: >
  State what you believe and why, in one sentence, before you look.

kill_criteria:
  min_sharpe: 0.50           # below this, dead
  min_breakeven_bps: 20      # cost headroom per side
  max_permutation_p: 0.05    # vs a matched no-information null
  min_ic_tstat: 2.0          # on the primary forecast horizon
  require_monotonic_deciles: true
  max_turnover: 30           # times per year
  min_years: 5

notes: >
  Anything you want the future reader (usually you) to know about why these
  numbers and not others.
"""
