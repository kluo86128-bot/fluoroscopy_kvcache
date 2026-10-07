"""Final manifest reevaluation accepts new questions and never retrains."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import reevaluate_saved_run as reeval
from experiment.io import read_json, write_json
from test_experiment import Base, config


class ReevaluationTests(Base):
    def fixture(self, root):
        source = root / "source"
        source.mkdir()
        methods = ["baseline", "question_weighted_kv"]
        ids = ["one", "two"]
        data = root / "tasks.json"
        write_json(data, [{"task_id": task, "private_prefix": "private", "public_text": "public",
                           "question": "live training question changed", "held_out_question": "new test question " + task,
                           "answer": "12", "answer_format": "digits"} for task in ids])
        cfg = config(datasets=[str(data)], methods=methods, max_new_tokens=2)
        write_json(source / "resolved_config.json", cfg)
        write_json(source / "selected_tasks.json", ids)
        write_json(source / "result.json", {"status": "completed"})
        write_json(source / "status.json", {"status": "completed"})
        old_question, instruction = "original training question", cfg["output_instruction"]
        context = "private\0public\0" + old_question + "\0" + instruction
        for task in ids:
            sample_dir = source / "samples" / task
            write_json(sample_dir / "sample_info.json", {"task_id": task, "question": old_question,
                       "output_instruction": instruction, "context_signature": hashlib.sha256(context.encode()).hexdigest()})
            for method in methods:
                directory = sample_dir / method
                write_json(directory / "result.json", {"status": "completed"})
                prefixes = directory / "prefixes"
                prefixes.mkdir()
                items = []
                for rank in (1, 2):
                    filename = f"step_{rank:08d}.pt"
                    torch.save({"soft_prefix": self.initial, "step": rank}, prefixes / filename)
                    items.append({"path": filename, "rank": rank, "step": rank, "loss": rank / 100})
                torch.save({"soft_prefix": self.initial}, prefixes / "unlisted.pt")
                write_json(prefixes / "manifest.json", {"requested_topk": 2, "prefixes": items})
        return source, data

    def test_retests_only_final_manifest_with_one_model_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as log:
            root = Path(directory)
            source, data = self.fixture(root)
            original = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
            plan = reeval.build_plan(source, data)
            self.assertEqual(plan["problems"], [])
            self.assertEqual(reeval.plan_summary(plan)["manifest_prefixes"], 8)
            outputs = [torch.tensor([[1, 2]]) if index in (0, 1, 4, 5, 6) else torch.tensor([[1, 3]]) for index in range(8)]
            with patch("torch.optim.Adam", side_effect=AssertionError("must not train")), \
                    patch.object(self.backend, "setup_teacher", wraps=self.backend.setup_teacher) as teacher, \
                    patch.object(self.backend, "rollout", side_effect=outputs), \
                    patch("experiment.backend.load_backend", return_value=self.backend) as loader:
                self.assertEqual(reeval.run_reevaluation(plan, root / "output"), 0)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(teacher.call_count, 2)
            self.assertTrue(all(not call.kwargs["include_private"] for call in teacher.call_args_list))
            self.assertTrue(all(path.read_bytes() == content for path, content in original.items()))
            rows = read_json(root / "output/test_results.json")
            self.assertEqual(len(rows), 8)
            self.assertTrue(all("unlisted" not in row["prefix_path"] for row in rows))
            self.assertEqual(len((root / "output/test_results.jsonl").read_text(encoding="utf-8").splitlines()), 8)
            report = read_json(root / "output/summary.json")
            rates = {row["method"]: row for row in report["methods_summary"]}
            self.assertEqual(rates["baseline"]["successful_samples"], 2)
            self.assertEqual(rates["baseline"]["sample_success_rate"], 1)
            self.assertEqual(rates["question_weighted_kv"]["successful_samples"], 1)
            self.assertEqual(rates["question_weighted_kv"]["sample_success_rate"], 0.5)
            with (root / "output/test_sample_prefix_rates.csv").open(encoding="utf-8-sig", newline="") as stream:
                prefix_rates = {(r["task_id"], r["method"]): r for r in csv.DictReader(stream)}
            self.assertEqual(float(prefix_rates["two", "question_weighted_kv"]["prefix_success_rate"]), 0.5)
            self.assertNotIn("Top1", log.getvalue())
            questions = read_json(root / "output/questions_used.json")
            self.assertEqual(questions[0]["question"], "new test question one")

    def test_missing_prefix_and_method_are_reported_without_fake_zero_rates(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, data = self.fixture(root)
            (source / "samples/one/baseline/prefixes/step_00000001.pt").unlink()
            write_json(source / "samples/two/question_weighted_kv/result.json", {"status": "failed"})
            plan = reeval.build_plan(source, data)
            self.assertEqual(len(plan["problems"]), 2)
            with patch.object(self.backend, "rollout", return_value=torch.tensor([[1, 3]])):
                self.assertEqual(reeval.run_reevaluation(plan, root / "output", backend=self.backend), 1)
            rows = read_json(root / "output/test_results.json")
            self.assertEqual(sum(r["status"] == "invalid_prefix" for r in rows), 1)
            with (root / "output/test_sample_prefix_rates.csv").open(encoding="utf-8-sig", newline="") as stream:
                rates = {(r["task_id"], r["method"]): r for r in csv.DictReader(stream)}
            self.assertEqual(rates["one", "baseline"]["tested_prefixes"], "1")
            self.assertEqual(rates["two", "question_weighted_kv"]["prefix_success_rate"], "")

    def test_changed_private_context_is_skipped_and_running_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, data = self.fixture(root)
            rows = read_json(data)
            rows[0]["private_prefix"] = "different private dialogue"
            write_json(data, rows)
            plan = reeval.build_plan(source, data)
            self.assertEqual(len(plan["samples"]), 1)
            self.assertIn("上下文", plan["problems"][0]["error"])
            write_json(source / "status.json", {"status": "running"})
            with self.assertRaisesRegex(ValueError, "仍在执行"):
                reeval.build_plan(source, data)

    def test_inspect_has_no_model_load_and_rejects_manifest_escape(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, data = self.fixture(root)
            with patch.object(reeval, "run_reevaluation", side_effect=AssertionError("inspect cannot infer")):
                self.assertEqual(reeval.main(["--run-dir", str(source), "--dataset", str(data), "--inspect"]), 0)
            manifest = source / "samples/one/baseline/prefixes/manifest.json"
            payload = read_json(manifest)
            payload["prefixes"][0]["path"] = "../outside.pt"
            write_json(manifest, payload)
            with self.assertRaisesRegex(ValueError, "文件名"):
                reeval.build_plan(source, data)

    def test_inference_failure_retains_completed_results_and_marks_partial_counts(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, data = self.fixture(root)
            plan = reeval.build_plan(source, data)
            with patch.object(self.backend, "rollout", side_effect=[torch.tensor([[1, 2]]), RuntimeError("inference failed")]):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    reeval.run_reevaluation(plan, root / "output", backend=self.backend)
            report = read_json(root / "output/summary.json")
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["tested_prefixes"], 1)
            self.assertEqual(report["unprocessed_prefixes"], 7)
            self.assertEqual(len(read_json(root / "output/test_results.json")), 1)
            self.assertEqual(read_json(root / "output/status.json")["status"], "failed")


class RealQuestionFilesTests(unittest.TestCase):
    def test_all_40_test_questions_follow_requested_scope(self):
        root = Path(__file__).resolve().parents[1] / "test_samples"
        rows = [json.loads(line) for line in (root / "tasks.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        self.assertEqual(len(rows), 40)
        for index, row in enumerate(rows):
            with self.subTest(task_id=row["task_id"]):
                path = root / row["held_out_question_file"]
                if index < 20:
                    self.assertEqual(path.read_bytes(), (root / row["receiver_inputs"]["question_file"]).read_bytes())
                    self.assertEqual(row["test_question_version"], "copied_training_question_v1")
                else:
                    text = path.read_text(encoding="utf-8-sig")
                    self.assertEqual(row["test_question_version"], "background_single_field_v1")
                    self.assertTrue(text.startswith("Output format:"))
                    self.assertEqual(text.count("Question:"), 1)
                    self.assertNotIn("what is its name", text)
                    self.assertNotIn("followed by the restaurant name", text)
                    if row["slot"] == "city":
                        self.assertIn("In which city", text)
                        self.assertIn("complete English city name", text)
                    else:
                        self.assertIn("same", text)
                        self.assertIn("ordinal suffix", text)
