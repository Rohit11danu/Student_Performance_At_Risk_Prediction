"""Initial checks for synthetic dataset generation."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.data.generate_data import FIELDNAMES, generate_records, write_dataset


class GenerateDataTests(unittest.TestCase):
    def test_default_size_schema_ranges_and_label(self) -> None:
        records = generate_records()
        self.assertEqual(len(records), 1000)
        self.assertEqual(list(records[0]), FIELDNAMES)
        self.assertTrue({record["at_risk"] for record in records} <= {0, 1})
        for record in records:
            for field in (
                "attendance_pct",
                "assignment_completion_pct",
                "previous_assessment_1_pct",
                "previous_assessment_2_pct",
            ):
                self.assertGreaterEqual(record[field], 0)
                self.assertLessEqual(record[field], 100)
            self.assertEqual(
                record["previous_assessment_trend_pct"],
                round(record["previous_assessment_2_pct"] - record["previous_assessment_1_pct"], 1),
            )
        self.assertNotIn("next_assessment_score_pct", FIELDNAMES)

    def test_seed_is_reproducible_and_csv_is_written(self) -> None:
        self.assertEqual(generate_records(20, 7), generate_records(20, 7))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_dataset(Path(temp_dir) / "sample.csv", n_records=20, seed=7)
            with path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 20)
            self.assertEqual(list(rows[0]), FIELDNAMES)

    def test_rejects_zero_records(self) -> None:
        with self.assertRaises(ValueError):
            generate_records(0)


if __name__ == "__main__":
    unittest.main()
