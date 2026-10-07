"""Count sample coverage separately from prefix rates and missing tests."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

from experiment.test_tables import build_tables, render_tables, write_test_tables


def result(task, method, correct, kind="held_out_question", **extra):
    return {"task_id": task, "method": method, "question_kind": kind,
            "answer_match": correct, **extra}


class TestTablesTests(unittest.TestCase):
    def test_one_correct_prefix_counts_sample_once_and_uses_actual_denominator(self):
        rows = [result("040", "weighted", index < 49, first_token_match=False, full_match=False)
                for index in range(400)]
        rows += [result("021", "weighted", False), result("021", "weighted", False)]
        methods, samples = build_tables(rows, ["weighted"], ["021", "040", "missing"],
                                       ["held_out_question"], 3)
        self.assertEqual(methods[0]["successful_samples"], 1)
        self.assertEqual(methods[0]["tested_samples"], 2)
        self.assertEqual(methods[0]["planned_samples"], 3)
        self.assertEqual(methods[0]["sample_success_rate"], 0.5)
        self.assertEqual(samples[0]["prefix_success_rate"], 0)
        self.assertEqual(samples[1]["correct_prefixes"], 49)
        self.assertEqual(samples[1]["prefix_success_rate"], 0.1225)
        self.assertIsNone(samples[2]["prefix_success_rate"])
        self.assertIsNone(samples[2]["sample_success"])
        text = render_tables(methods, samples, ["weighted"], ["021", "040", "missing"],
                             ["held_out_question"])
        self.assertIn("12.25% (49/400)", text)
        self.assertIn("0.00% (0/2)", text)
        self.assertIn("| missing | 未测试 |", text)
        self.assertIn("50.00%", text)

    def test_question_kinds_and_methods_do_not_share_successes(self):
        rows = [result("one", "a", True, "training_question"),
                result("one", "a", False), result("one", "b", True)]
        methods, samples = build_tables(rows, ["a", "b"], ["one"],
                                       ["training_question", "held_out_question"], 1)
        counts = {(r["question_kind"], r["method"]): r["successful_samples"] for r in methods}
        self.assertEqual(counts["training_question", "a"], 1)
        self.assertEqual(counts["held_out_question", "a"], 0)
        self.assertEqual(counts["held_out_question", "b"], 1)
        missing = next(r for r in samples if r["question_kind"] == "training_question" and r["method"] == "b")
        self.assertIsNone(missing["prefix_success_rate"])

    def test_held_out_export_filters_stale_training_rows_and_silent_updates(self):
        rows = [result("one", "a", True, "training_question"), result("one", "a", False)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                write_test_tables(root, {"methods": ["a"], "test_question_mode": "held_out"},
                                  2, rows, ["one", "two"], announce=False)
            self.assertEqual(output.getvalue(), "")
            with (root / "test_method_success.csv").open(encoding="utf-8-sig", newline="") as source:
                methods = list(csv.DictReader(source))
            self.assertEqual(len(methods), 1)
            self.assertEqual(methods[0]["successful_samples"], "0")
            self.assertEqual(methods[0]["tested_samples"], "1")
            with (root / "test_sample_prefix_rates.csv").open(encoding="utf-8-sig", newline="") as source:
                samples = list(csv.DictReader(source))
            self.assertEqual(samples[1]["tested_prefixes"], "0")
            self.assertEqual(samples[1]["prefix_success_rate"], "")
            with contextlib.redirect_stdout(output):
                write_test_tables(root, {"methods": ["a"], "test_question_mode": "held_out"},
                                  2, rows, ["one", "two"], announce=True)
            self.assertIn("[统计表1/held_out_question]", output.getvalue())
            self.assertIn("[统计表2/held_out_question]", output.getvalue())
            self.assertNotIn("[统计表1/training_question]", output.getvalue())

    def test_no_results_marks_all_planned_samples_untested(self):
        methods, samples = build_tables([], ["a", "b"], ["one", "two"], ["held_out_question"], 2)
        self.assertTrue(all(row["tested_samples"] == 0 for row in methods))
        self.assertTrue(all(row["successful_samples"] == 0 for row in methods))
        self.assertTrue(all(row["sample_success_rate"] is None for row in methods))
        self.assertEqual(len(samples), 4)
        self.assertTrue(all(row["prefix_success_rate"] is None for row in samples))


if __name__ == "__main__":
    unittest.main()
