"""Verify paired inference, input integrity and source isolation with a tiny Qwen3."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiment.data import catalog, materialize, question_text
from experiment.io import write_json
from experiment.metrics import AnswerMonitor, test_prefix as evaluate_prefix
import retest_questions as retest
from test_experiment import backend, config


class QuestionRetestTests(unittest.TestCase):
    def fixture(self, root):
        source, bundle = root / "source", root / "questions"
        source.mkdir()
        bundle.mkdir()
        dataset = root / "tasks.jsonl"
        row = {"task_id": "sample", "private_prefix": "private", "public_text": "public",
               "question": "Current seats?", "held_out_question": "Current party?", "answer": "2", "answer_format": "digits"}
        dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
        cfg = config(datasets=[str(dataset)], methods=["question_weighted_kv"], score_probability="accepted_forms")
        write_json(source / "resolved_config.json", cfg)
        sample = materialize(catalog(cfg)[0], cfg["output_instruction"])
        questions = {}
        for kind, current, previous in (("training_question", sample.question, "Old seats?"),
                                        ("held_out_question", sample.held_out_question, "Old party?")):
            filename = kind + ".txt"
            (bundle / filename).write_text(previous, encoding="utf-8")
            questions[kind] = {"file": filename, "sha256": retest.digest(previous), "expected_current_sha256": retest.digest(current)}
        write_json(bundle / "manifest.json", {"question_version": retest.PREVIOUS, "current_question_version": retest.CURRENT,
                                              "samples": {"sample": questions}})
        write_json(source / "samples/sample/sample_info.json", {"question": sample.question,
                   "output_instruction": sample.output_instruction,
                   "context_signature": retest.context_signature(sample, sample.question, sample.output_instruction)})
        directory = source / "samples/sample/question_weighted_kv/prefixes"
        directory.mkdir(parents=True)
        model = backend()
        prefix = model.embedding(model.encode(sample.private_text)).detach().cpu()
        torch.save({"soft_prefix": prefix}, directory / "one.pt")
        torch.save({"soft_prefix": prefix + 0.01}, directory / "two.pt")
        write_json(directory / "manifest.json", {"prefixes": [{"rank": 1, "step": 2, "loss": .1, "path": "one.pt"},
                                                              {"rank": 2, "step": 1, "loss": .2, "path": "two.pt"}]})
        return source, bundle, dataset, model, prefix

    def test_paired_inference_never_trains_and_preserves_all_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, bundle, dataset, model, prefix = self.fixture(root)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            plans, metadata, problems = retest.build_plan([source], dataset, bundle)
            self.assertEqual(problems, [])
            with patch("torch.optim.Adam", side_effect=AssertionError("no training")), \
                 patch.object(model, "student", wraps=model.student) as student, \
                 patch.object(model, "setup_teacher", wraps=model.setup_teacher) as teacher, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(retest.run_retest(plans, root / "output", metadata, backend=model), 0)
            self.assertEqual(student.call_count, 2)  # one reconstruction per prefix, reused by all four queries
            self.assertFalse(teacher.call_args.kwargs["include_private"])
            rows = [json.loads(line) for line in (root / "output/test_results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual({r["question_version"] for r in rows}, {retest.PREVIOUS, retest.CURRENT})
            for step in (1, 2):
                self.assertEqual(len({r["prefix_sha256"] for r in rows if r["step"] == step}), 1)
            self.assertEqual(json.loads((root / "output/status.json").read_text())["paired_observations"], 4)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)
            with self.assertRaises(FileExistsError):
                retest.run_retest(plans, root / "output", metadata, backend=model)
            with self.assertRaises(ValueError):
                retest.run_retest(plans, source / "child", metadata, backend=model)

    def test_integrity_checks_reject_changed_context_and_questions(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, bundle, dataset, model, prefix = self.fixture(Path(temporary))
            original = dataset.read_text()
            dataset.write_text(original.replace('"private"', '"changed"'))
            with self.assertRaisesRegex(ValueError, "上下文"):
                retest.build_plan([source], dataset, bundle)
            dataset.write_text(original)
            (bundle / "training_question.txt").write_text("modified")
            with self.assertRaisesRegex(ValueError, "Git 快照"):
                retest.build_plan([source], dataset, bundle)

    def test_missing_prefix_is_reported_before_model_load_and_top_rank_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, bundle, dataset, model, prefix = self.fixture(Path(temporary))
            (source / "samples/sample/question_weighted_kv/prefixes/two.pt").unlink()
            plans, metadata, problems = retest.build_plan([source], dataset, bundle)
            self.assertEqual(len(problems), 1)
            self.assertFalse(retest.plan_summary(plans, problems)["ready"])
            plans, _, problems = retest.build_plan([source], dataset, bundle, max_prefixes=1)
            self.assertEqual(problems, [])
            self.assertEqual(retest.plan_summary(plans, problems)["runs"][0]["planned_outputs"], 4)

    def test_shared_cache_evaluator_matches_existing_test_and_probability_monitor(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, bundle, dataset, model, prefix = self.fixture(Path(temporary))
            cfg = retest.read_source_config(source)
            sample = materialize(catalog(cfg)[0], cfg["output_instruction"])
            public = model.encode(sample.public_text)
            observed, _, _ = model.setup_teacher(model.encode(sample.private_text), public, include_private=False)
            ids = model.encode_question(question_text(sample.question, sample.output_instruction, cfg["answer_boundary"]), cfg)
            self.assertTrue(question_text(sample.question, sample.output_instruction).endswith("Answer: "))
            monitor = AnswerMonitor(model, observed, ids, sample.answer, sample.aliases,
                                    probability_mode=cfg["score_probability"], leading_spaces=cfg["answer_leading_spaces"])
            with torch.no_grad():
                private, _ = model.student(prefix, public)
                result = retest.evaluate_question(model, private, observed, ids, sample.answer, sample.aliases, sample.answer_format, cfg, monitor)
            expected = evaluate_prefix(model, observed, public, ids, prefix, sample.answer, sample.aliases,
                                       cfg["max_new_tokens"], cfg["stop_strings"], sample.answer_format)
            for key, value in expected.items():
                self.assertEqual(result[key], value)
            self.assertEqual(result["answer_probability"], monitor(private)["answer_probability"])
            self.assertTrue(any(form["text"].startswith(" ") for form in result["answer_form_probabilities"]))


if __name__ == "__main__":
    unittest.main()
