"""Standalone HTML rendering of a Report.

Self-contained: no external stylesheet, no script, no fonts that must load. A
report you cannot open on someone else's machine is not a report.
"""
from __future__ import annotations
import html as _h
from datetime import datetime, timezone

import pandas as pd

CSS = """
:root{color-scheme:light;--bg:#eceaf1;--card:#fff;--card2:#f5f4f9;--ink:#1a1922;
--ink2:#434150;--muted:#6b6879;--rule:#d5d3de;--rule2:#e7e5ee;--accent:#1c6b67;
--pass:#1c6b67;--pass-bg:#dbeceb;--fail:#8c2f39;--fail-bg:#f2e2e4;
--warn:#9c5f16;--warn-bg:#f6ead9}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
--bg:#121118;--card:#1a1922;--card2:#211f2b;--ink:#ecebf2;--ink2:#c3c0d0;
--muted:#918ea2;--rule:#2f2d3a;--rule2:#242230;--accent:#5cb5af;
--pass:#5cb5af;--pass-bg:#12302e;--fail:#d8767f;--fail-bg:#331a1e;
--warn:#d9a052;--warn-bg:#302315}}
:root[data-theme=dark]{color-scheme:dark;--bg:#121118;--card:#1a1922;--card2:#211f2b;
--ink:#ecebf2;--ink2:#c3c0d0;--muted:#918ea2;--rule:#2f2d3a;--rule2:#242230;
--accent:#5cb5af;--pass:#5cb5af;--pass-bg:#12302e;--fail:#d8767f;--fail-bg:#331a1e;
--warn:#d9a052;--warn-bg:#302315}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 ui-sans-serif,
system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
.w{max-width:860px;margin:0 auto;padding:52px 24px 90px}
.eyebrow{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;
letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin:0 0 14px}
h1{font-size:2rem;line-height:1.15;letter-spacing:-.02em;margin:0 0 10px;font-weight:600}
.hyp{color:var(--ink2);max-width:64ch;margin:0 0 22px}
.seal{display:inline-flex;align-items:center;gap:9px;font-family:ui-monospace,monospace;
font-size:11.5px;padding:6px 12px;border-radius:20px;background:var(--pass-bg);
color:var(--pass);font-weight:600;letter-spacing:.04em}
.seal.amended{background:var(--warn-bg);color:var(--warn)}
.diff{margin:14px 0 0;padding:13px 16px;border-left:3px solid var(--warn);
background:var(--warn-bg);border-radius:0 6px 6px 0;font-family:ui-monospace,monospace;
font-size:12.5px;color:var(--ink2)}
.verdict{margin:30px 0;padding:26px 28px;border-radius:11px;background:var(--card);
border:1px solid var(--rule);box-shadow:0 1px 2px rgba(0,0,0,.05),0 10px 26px -16px rgba(0,0,0,.3)}
.vbig{font-size:2.4rem;font-weight:700;letter-spacing:-.03em;line-height:1.05;margin:0}
.vbig.k{color:var(--fail)}.vbig.s{color:var(--pass)}
.vsub{color:var(--muted);font-size:14px;margin:7px 0 0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(105px,1fr));
gap:2px;margin:22px 0 0;border-top:1px solid var(--rule2);padding-top:18px}
.st b{display:block;font-family:ui-monospace,monospace;font-size:1.25rem;font-weight:600;
letter-spacing:-.02em}
.st span{display:block;font-family:ui-monospace,monospace;font-size:9.5px;
letter-spacing:.11em;text-transform:uppercase;color:var(--muted);margin-top:3px}
h2{font-size:1.12rem;margin:38px 0 14px;font-weight:600}
.f{background:var(--card);border:1px solid var(--rule);border-radius:9px;
padding:16px 19px;margin-bottom:10px}
.fh{display:flex;align-items:baseline;gap:11px;flex-wrap:wrap}
.badge{font-family:ui-monospace,monospace;font-size:10px;font-weight:700;letter-spacing:.1em;
padding:3px 9px;border-radius:5px}
.badge.p{background:var(--pass-bg);color:var(--pass)}
.badge.f{background:var(--fail-bg);color:var(--fail)}
.fn{font-weight:600;font-size:15px}
.fd{font-family:ui-monospace,monospace;font-size:12.5px;color:var(--ink2);
margin:9px 0 0;word-break:break-word}
.note{font-size:13.5px;color:var(--muted);margin:9px 0 0;padding-left:13px;
border-left:2px solid var(--rule)}
.warn{background:var(--warn-bg);color:var(--warn);border-radius:8px;padding:12px 16px;
font-size:14px;margin-bottom:10px}
table{border-collapse:collapse;width:100%;font-family:ui-monospace,monospace;
font-size:12px;margin-top:11px}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--rule2)}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted);font-size:10px;letter-spacing:.08em;text-transform:uppercase}
details{margin-top:9px}summary{cursor:pointer;font-size:12.5px;color:var(--accent)}
.tw{overflow-x:auto}
footer{margin-top:44px;padding-top:20px;border-top:1px solid var(--rule);
font-size:12.5px;color:var(--muted)}
@media(max-width:640px){.w{padding:32px 16px 60px}.vbig{font-size:1.8rem}}
"""


def _tbl(df: pd.DataFrame, limit: int = 14) -> str:
    if df is None or len(df) == 0:
        return ""
    d = df.head(limit)
    head = "".join(f"<th>{_h.escape(str(c))}</th>" for c in d.columns)
    body = ""
    for i, row in d.iterrows():
        cells = "".join(
            f"<td>{v:.3f}</td>" if isinstance(v, float) else f"<td>{_h.escape(str(v))}</td>"
            for v in row)
        body += f"<tr><td>{_h.escape(str(i))}</td>{cells}</tr>"
    more = f"<p class='note'>{len(df)-limit} more rows</p>" if len(df) > limit else ""
    return (f"<details><summary>detail ({len(df)} rows)</summary><div class='tw'>"
            f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody>"
            f"</table></div>{more}</details>")


def to_html(rep, title: str | None = None) -> str:
    e = _h.escape
    passed = sum(f.passed for f in rep.findings)
    killed = "KILLED" in rep.verdict
    amended = rep.prereg_status == "AMENDED"

    stats = "".join(f"<div class='st'><b>{e(str(v))}</b><span>{e(k)}</span></div>"
                    for k, v in rep.headline.items())
    diff = ""
    if rep.prereg_diff:
        rows = "<br>".join(e(d) for d in rep.prereg_diff)
        diff = ("<div class='diff'><strong>Criteria changed after the lock was "
                f"written:</strong><br>{rows}</div>")

    findings = ""
    for f in rep.findings:
        cls = "p" if f.passed else "f"
        note = f"<p class='note'>{e(f.note)}</p>" if (f.note and not f.passed) else ""
        findings += (
            f"<div class='f'><div class='fh'><span class='badge {cls}'>"
            f"{'PASS' if f.passed else 'FAIL'}</span>"
            f"<span class='fn'>{e(f.name)}</span></div>"
            f"<p class='fd'>{e(f.detail)}</p>{note}{_tbl(f.table)}</div>")

    warns = "".join(f"<div class='warn'>{e(w)}</div>" for w in rep.warnings)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title or rep.name)} — nullius</title><style>{CSS}</style></head><body>
<div class="w">
  <p class="eyebrow">nullius · falsification report</p>
  <h1>{e(rep.name)}</h1>
  <p class="hyp">{e(rep.hypothesis)}</p>
  <span class="seal{' amended' if amended else ''}">
    {'⚠ PRE-REGISTRATION AMENDED' if amended else '🔒 PRE-REGISTRATION SEALED'}
    &nbsp;{e(rep.prereg_digest)}</span>
  {diff}
  <div class="verdict">
    <p class="vbig {'k' if killed else 's'}">{e(rep.verdict)}</p>
    <p class="vsub">{passed} of {len(rep.findings)} attacks survived</p>
    <div class="stats">{stats}</div>
  </div>
  {warns}
  <h2>Attacks</h2>
  {findings}
  <footer>Generated {stamp} by nullius 0.1.0. Surviving every attack means the
  result is not obviously an artefact — a much weaker claim than "this works",
  and the strongest one available. Not investment advice.</footer>
</div></body></html>"""
