"""Small statistics helpers for honest reporting."""
from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a success rate. Better than +/- for small n."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def rate_with_interval(successes: int, n: int) -> str:
    lo, hi = wilson_interval(successes, n)
    pct = 100 * successes / n if n else 0.0
    return f"{pct:.0f}% ({successes}/{n}, 95% CI {100 * lo:.0f}-{100 * hi:.0f}%)"


def threshold_sweep(fail_errors: list[float], ok_errors: list[float], thresholds: list[float]) -> list[dict]:
    """For a list of thresholds, how many real failures are caught and how many good picks alarm.

    fail_errors: measured error on trials with an induced failure.
    ok_errors:   measured error on trials that actually succeeded.
    A trial alarms when its error exceeds the threshold.
    """
    rows = []
    for t in thresholds:
        caught = sum(e > t for e in fail_errors)
        false = sum(e > t for e in ok_errors)
        rows.append({
            "threshold": t,
            "caught": caught, "failures": len(fail_errors),
            "catch_rate": caught / len(fail_errors) if fail_errors else 0.0,
            "false_alarms": false, "good_picks": len(ok_errors),
            "false_alarm_rate": false / len(ok_errors) if ok_errors else 0.0,
        })
    return rows


def pick_threshold(rows: list[dict]) -> dict | None:
    """Largest threshold that still catches every failure (fewest false alarms at full recall)."""
    full = [r for r in rows if r["failures"] and r["caught"] == r["failures"]]
    return max(full, key=lambda r: r["threshold"]) if full else None
