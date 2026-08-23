"""Findings and the report they add up to."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Finding:
    name: str
    passed: bool
    detail: str = ""
    note: str = ""
    value: float = float("nan")
    table: Optional[pd.DataFrame] = None

    def line(self) -> str:
        return f"  {'PASS' if self.passed else 'FAIL'}  {self.name:<24} {self.detail}"


@dataclass
class Report:
    name: str
    hypothesis: str
    prereg_digest: str
    prereg_status: str
    prereg_diff: list = field(default_factory=list)
    headline: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def survived(self) -> bool:
        return all(f.passed for f in self.findings) and not self.prereg_diff

    @property
    def verdict(self) -> str:
        if self.prereg_status == "AMENDED":
            return "KILLED (criteria amended after results)" if not all(
                f.passed for f in self.findings) else "SURVIVED — but criteria were amended"
        return "SURVIVED" if self.survived else "KILLED"

    def __str__(self) -> str:
        w = 76
        out = ["=" * w, f"  {self.name}", "=" * w,
               f"  hypothesis: {self.hypothesis}",
               f"  pre-registration: {self.prereg_status} [{self.prereg_digest}]"]
        if self.prereg_diff:
            out.append("  !! criteria changed after the lock was written:")
            out += [f"       {d}" for d in self.prereg_diff]
        out.append("")
        if self.headline:
            out.append("  " + "   ".join(f"{k} {v}" for k, v in self.headline.items()))
            out.append("")
        out.append("  ATTACKS")
        passed = sum(f.passed for f in self.findings)
        for f in self.findings:
            out.append(f.line())
            if not f.passed and f.note:
                out.append(f"        -> {f.note}")
        for wmsg in self.warnings:
            out.append(f"  WARN  {wmsg}")
        out += ["", "=" * w,
                f"  VERDICT: {self.verdict}  ({passed}/{len(self.findings)} attacks survived)",
                "=" * w]
        return "\n".join(out)

    def to_html(self, path=None, title: str | None = None) -> str:
        """Render a standalone HTML report; write it if `path` is given."""
        from .htmlreport import to_html as _render
        doc = _render(self, title)
        if path is not None:
            from pathlib import Path
            Path(path).write_text(doc, encoding="utf-8")
        return doc

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"attack": f.name, "passed": f.passed,
                              "detail": f.detail, "value": f.value}
                             for f in self.findings])
