"""Summarize each seed split of the unified low-speed evaluation."""
import argparse
import json
from pathlib import Path
from statistics import mean


def summarize(split):
    flat = split["flat"]
    moving = [row for row in flat if any(row["target"])]
    result = {
        "flat": {
            "episodes": len(flat),
            "held": sum(row["held"] for row in flat),
            "terminated": sum(row["terminated"] for row in flat),
            "contact_frames": sum(row["contact_frames"] for row in flat),
            "tracking_mean_m_s": mean(row["tracking_m_s"] for row in moving if row["tracking_m_s"] is not None),
            "hind_mean_of_p95_rad": mean(row["hind_p95_rad"] for row in flat),
            "pitch_mean_of_p95_deg": mean(row["pitch_p95_deg"] for row in flat),
        },
        "stairs": {},
    }
    for height, rows in split["stairs"].items():
        by_mode = {}
        for mode, interrupted in (("continuous", False), ("hold_and_resume", True)):
            subset = [row for row in rows if row["interrupted"] == interrupted]
            by_mode[mode] = {
                "episodes": len(subset),
                "passed": sum(row["passed"] for row in subset),
                "top_stable": sum(row["top_stable"] for row in subset),
                "fifth_wheel_contact": sum(max(row["levels"]) == 5 for row in subset),
                "impact_over_90_n": sum(row["peak_impact_n"] > 90. for row in subset),
                "hold_triggered": sum(row["hold_frames"] > 0 for row in subset),
                "held": sum(row["held"] for row in subset),
            }
        result["stairs"][height] = by_mode
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.reports:
        report = json.loads(path.read_text())
        print(path.name, json.dumps({label: summarize(split) for label, split in report.items()}))


if __name__ == "__main__":
    main()
