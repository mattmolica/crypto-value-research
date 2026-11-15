"""Write the tracking results as a Markdown page GitHub shows in the repo.

Reads the `tracking.json` that `track` writes and produces `results/README.md`:
a scorecard of every saved score run, the latest run's groups, and every
token's return since it was first scored. GitHub renders it as a page, so the
results are readable in the repo even without a website.

Usage:
    python -m crypto_value_research.report --tracking site/data/tracking.json --out results/README.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def pct(x, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:+.{digits}%}"


def md_table(headers: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(headers) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def group_rows(groups: list[dict]) -> list[list]:
    return [[g["group"], g["priced"], pct(g["avg_return"]), pct(g["median_return"]),
             pct(g["vs_basket"]), "–" if g["beat_basket"] is None else f"{g['beat_basket']:.0%}"]
            for g in groups]


def build(report: list[dict], page_url: str | None = None) -> str:
    if not report:
        return "# Track record\n\nNo score snapshots tracked yet.\n"
    latest, first = report[-1], report[0]
    out = ["# Value score track record", ""]
    out.append(f"Updated automatically. Prices as of **{latest['as_of'][:16].replace('T', ' ')} UTC**, "
               f"{len(report)} saved score run{'s' if len(report) != 1 else ''}.")
    if page_url:
        out.append(f"Charts: [{page_url}]({page_url})")
    out += ["", "Returns are price only, measured from the moment each run scored the tokens. "
            "\"vs basket\" is a group's average return minus the average of every scored token. "
            "Short periods are noise; look for patterns that hold across many runs and months.", ""]

    out += ["## Every score run so far", ""]
    rows = []
    for r in report:
        L = {g["group"]: g for g in r.get("labels", [])}
        b = r["benchmarks"]
        rows.append([r["scored_at"][:10], f"{r['days']:.0f}",
                     pct(L.get("Value candidate", {}).get("avg_return")),
                     pct(L.get("Avoid", {}).get("avg_return")),
                     pct(b.get("All tokens (equal weight)")), pct(b.get("BTC")), pct(b.get("ETH")),
                     "–" if r["rank_correlation"] is None else f"{r['rank_correlation']:+.2f}"])
    out.append(md_table(["Scored", "Days", "Value candidates", "Avoid", "All tokens", "BTC", "ETH",
                         "Score vs return"], rows))
    out += ["", "\"Score vs return\" is the rank correlation between value score and later return: "
            "+1 means higher scores always did better, 0 means no link.", ""]

    out += [f"## Oldest run ({first['scored_at'][:10]}, {first['days']:.0f} days) by label", ""]
    out.append(md_table(["Label", "Tokens", "Avg", "Median", "vs basket", "Beat basket"],
                        group_rows(first.get("labels", []))))
    out += ["", "### By value score fifth", ""]
    out.append(md_table(["Score fifth", "Tokens", "Avg", "Median", "vs basket", "Beat basket"],
                        group_rows(first.get("buckets", []))))

    now = {t["symbol"]: t for t in latest.get("tokens_detail", [])}
    tokens = sorted(first.get("tokens_detail", []),
                    key=lambda t: -1e9 if t["return"] is None else t["return"], reverse=True)
    out += ["", f"## Every token since {first['scored_at'][:10]}", ""]
    rows = []
    for t in tokens:
        score = "–" if t.get("value_score") is None else f"{t['value_score']:.0f}"
        rows.append([t["symbol"], t["name"], t["verdict"], score,
                     now.get(t["symbol"], {}).get("verdict", "–"), pct(t["return"])])
    out.append(md_table(["Token", "Name", "Label then", "Score then", "Label now", "Return"], rows, "lllrlr"))
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracking", default="data/tracking/tracking.json")
    p.add_argument("--out", default="results/README.md")
    p.add_argument("--page-url", help="link to the charts page, shown at the top")
    args = p.parse_args(argv)
    report = json.loads(Path(args.tracking).read_text())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(build(report, args.page_url))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
