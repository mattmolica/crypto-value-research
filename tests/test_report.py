from crypto_value_research import report


def entry(day, days, vc, rc=0.3):
    labels = [{"group": "Value candidate", "priced": 2, "avg_return": vc, "median_return": vc,
               "vs_basket": 0.01, "beat_basket": 0.5},
              {"group": "Avoid", "priced": 1, "avg_return": -0.1, "median_return": -0.1,
               "vs_basket": -0.02, "beat_basket": 0.0}]
    return {
        "scored_at": f"{day}T18:55:00Z", "as_of": "2026-11-20T12:00:00Z", "days": days,
        "benchmarks": {"BTC": 0.02, "ETH": None, "All tokens (equal weight)": -0.01},
        "rank_correlation": rc, "labels": labels, "buckets": labels,
        "tokens_detail": [
            {"slug": "a", "symbol": "AAA", "name": "A|A", "verdict": "Value candidate", "value_score": 81.2, "return": 0.25},
            {"slug": "b", "symbol": "BBB", "name": "B", "verdict": "Avoid", "value_score": None, "return": None},
        ],
    }


def test_build_has_runs_groups_and_tokens():
    md = report.build([entry("2026-11-13", 7, 0.05), entry("2026-11-16", 4, -0.02, None)], "https://x.io/")
    assert "| 2026-11-13 | 7 | +5.0% | -10.0% | -1.0% | +2.0% | – | +0.30 |" in md
    assert "| 2026-11-16 |" in md and "https://x.io/" in md
    assert "A/A" in md                      # pipes can't break the table
    assert md.index("AAA") < md.index("BBB")  # best return first, unknown last


def test_build_empty():
    assert "No score snapshots" in report.build([])
