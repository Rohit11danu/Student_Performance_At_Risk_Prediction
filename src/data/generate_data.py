"""Generate a reproducible synthetic student performance dataset."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "student_performance.csv"
FIELDNAMES = [
    "student_id",
    "attendance_pct",
    "assignment_completion_pct",
    "previous_assessment_1_pct",
    "previous_assessment_2_pct",
    "previous_assessment_trend_pct",
    "at_risk",
]


def generate_records(n_records: int = 1000, seed: int = 42) -> list[dict[str, int | float | str]]:
    """Return synthetic records; the simulated future score is never exposed."""
    if n_records < 1:
        raise ValueError("n_records must be at least 1")

    rng = random.Random(seed)
    records: list[dict[str, int | float | str]] = []
    for index in range(1, n_records + 1):
        attendance = round(rng.uniform(45, 100), 1)
        completion = round(rng.uniform(30, 100), 1)
        previous_1 = round(rng.uniform(20, 95), 1)
        previous_2 = round(max(0, min(100, previous_1 + rng.gauss(0, 12))), 1)
        trend = round(previous_2 - previous_1, 1)

        # Only the binary outcome is retained. Predictors represent information
        # available before the next assessment; no sensitive attributes exist.
        next_score = (
            0.40 * previous_2
            + 0.25 * attendance
            + 0.20 * completion
            + 0.15 * previous_1
            + rng.gauss(0, 10)
        )
        next_score = max(0, min(100, next_score))
        records.append(
            {
                "student_id": f"S{index:05d}",
                "attendance_pct": attendance,
                "assignment_completion_pct": completion,
                "previous_assessment_1_pct": previous_1,
                "previous_assessment_2_pct": previous_2,
                "previous_assessment_trend_pct": trend,
                "at_risk": int(next_score < 40),
            }
        )
    return records


def write_dataset(output_path: Path = DEFAULT_OUTPUT, n_records: int = 1000, seed: int = 42) -> Path:
    """Generate records and write them as a CSV file."""
    records = generate_records(n_records=n_records, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=1000, help="Number of synthetic records (minimum 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    args = parser.parse_args()
    path = write_dataset(args.output, args.records, args.seed)
    print(f"Wrote {args.records} records to {path}")


if __name__ == "__main__":
    main()
