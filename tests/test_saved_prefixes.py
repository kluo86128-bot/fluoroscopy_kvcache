"""Standalone inference scans all snapshots and preserves source artifacts."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from experiment.io import read_json, write_json
from test_experiment import Base, config

SCRIPT = Path(__file__).resolve().parents[1] / "test_saved_prefixes.py"
SPEC = importlib.util.spec_from_file_location("saved_prefix_cli", SCRIPT)
tester = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tester)


class StandaloneTests(Base):
    def fixture(self, root):
        sample_dir = root / "sample"
        sample_dir.mkdir()
        for name, text in (("prefix_a.txt", "private"), ("public_chunk.txt", "public"),
                           ("question_test.txt", "How many seats?")):
            (sample_dir / name).write_text(text, encoding="utf-8")
        write_json(sample_dir / "evaluation.json", {"task_id": "sample", "canonical_answer": "12", "accepted_aliases": ["twelve"]})
        prefix_dir = root / "prefixes"
        left, right = prefix_dir / "weighted", prefix_dir / "joint"
        left.mkdir(parents=True)
        right.mkdir()
        prefix = self.initial.detach().cpu()
        torch.save({"soft_prefix": prefix, "step": 1, "checkpoint_loss": 0.1}, left / "one.pt")
        torch.save({"soft_prefix_embeddings": prefix}, right / "old.pth")
        torch.save(prefix, right / "unlisted.pt")
        # All files are evaluated, including snapshots absent from this manifest.
        write_json(right / "manifest.json", {"prefixes": [{"path": "old.pth"}]})
        torch.save({"soft_prefix": prefix}, prefix_dir / "initial.pt")
        torch.save({"prefix": prefix, "optimizer": {}}, prefix_dir / "latest.pt")
        return sample_dir, prefix_dir

    def test_recursive_inference_all_formats_invalid_files_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            sample_dir, prefix_dir = self.fixture(root)
            torch.save({"soft_prefix": torch.zeros(1, 1, 16)}, prefix_dir / "wrong_shape.pt")
            torch.save({"soft_prefix": torch.full_like(self.initial, float("nan"))}, prefix_dir / "nan.pt")
            original = {p: p.read_bytes() for folder in (sample_dir, prefix_dir) for p in folder.rglob("*") if p.is_file()}
            cfg = config(max_new_tokens=4)
            sample, question, kind = tester.load_sample(sample_dir, cfg, answer_format="digits")
            files, excluded = tester.discover_prefixes(prefix_dir)
            self.assertEqual(len(files), 5)
            self.assertEqual(len(excluded), 2)
            with patch("torch.optim.Adam", side_effect=AssertionError("inference must not train")), \
                    patch.object(self.backend, "setup_teacher", wraps=self.backend.setup_teacher) as teacher, \
                    patch.object(self.backend, "rollout", side_effect=[torch.tensor([[1, 2]]), torch.tensor([[1, 3]]), torch.tensor([[1, 2]])]):
                self.assertEqual(tester.evaluate(sample, files, excluded, cfg, root / "output",
                                                 question_path=question, question_kind=kind, backend=self.backend), 1)
            self.assertEqual(teacher.call_count, 1)
            self.assertFalse(teacher.call_args.kwargs["include_private"])
            self.assertTrue(all(p.read_bytes() == data for p, data in original.items()))
            summary = read_json(root / "output/summary.json")
            self.assertEqual(summary["completed_prefixes"], 3)
            self.assertEqual(summary["successful_prefixes"], 2)
            self.assertEqual(summary["invalid_prefixes"], 2)
            self.assertEqual(summary["prefix_success_rate"], 2 / 3)
            self.assertTrue(summary["sample_success"])
            rows = read_json(root / "output/results.json")
            self.assertEqual(len(rows), 5)
            tested = [r for r in rows if r["status"] == "tested"]
            self.assertTrue(all(r["question_kind"] == "held_out_question" for r in tested))
            self.assertTrue(all(not r["first_token_match"] for r in tested))
            self.assertEqual([r["output"] for r in tested], ["12", "13", "12"])
            self.assertEqual(len((root / "output/results.jsonl").read_text(encoding="utf-8").splitlines()), 5)
            with (root / "output/results.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 5)
            self.assertTrue((root / "output/folder_summary.csv").is_file())

    def test_moved_sample_is_standalone_and_inspect_never_loads_model(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            sample_dir, prefix_dir = self.fixture(root)
            write_json(prefix_dir / "config.json", {"config": {"model_path": "saved_model", "cuda_devices": "1"}})
            with patch.object(tester, "evaluate", side_effect=AssertionError("inspect must not infer")):
                self.assertEqual(tester.main(["--sample-dir", str(sample_dir), "--prefix-dir", str(prefix_dir),
                                               "--cuda-devices", "3", "--output-dir", str(root / "output"), "--inspect"]), 0)
            self.assertFalse((root / "output").exists())
            cfg, source = tester.evaluation_config(prefix_dir, overrides={"cuda_devices": "3"})
            self.assertEqual(cfg["model_path"], "saved_model")
            self.assertEqual(cfg["cuda_devices"], "3")
            self.assertEqual(source, (prefix_dir / "config.json").resolve())

    def test_context_check_allows_new_test_question_but_rejects_different_dialogue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_dir, prefix_dir = self.fixture(root)
            sample, _, _ = tester.load_sample(sample_dir, config())
            context = sample.private_text + "\0" + sample.public_text + "\0old training question\0instruction"
            info = {"task_id": "sample", "question": "old training question", "output_instruction": "instruction",
                    "context_signature": hashlib.sha256(context.encode()).hexdigest()}
            write_json(prefix_dir / "sample_info.json", info)
            self.assertTrue(tester.check_provenance(prefix_dir / "weighted/one.pt", sample))
            write_json(prefix_dir / "sample_info.json", {**info, "task_id": "other"})
            with self.assertRaisesRegex(ValueError, "其他样本"):
                tester.check_provenance(prefix_dir / "weighted/one.pt", sample)
            write_json(prefix_dir / "sample_info.json", {**info, "context_signature": "different"})
            with self.assertRaisesRegex(ValueError, "上下文"):
                tester.check_provenance(prefix_dir / "weighted/one.pt", sample)

    def test_inference_failure_preserves_completed_rows(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            sample_dir, prefix_dir = self.fixture(root)
            sample, question, kind = tester.load_sample(sample_dir, config())
            files, excluded = tester.discover_prefixes(prefix_dir)
            with patch.object(self.backend, "rollout", side_effect=[torch.tensor([[1, 2]]), RuntimeError("inference failed")]):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    tester.evaluate(sample, files, excluded, config(), root / "output",
                                    question_path=question, question_kind=kind, backend=self.backend)
            summary = read_json(root / "output/summary.json")
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["completed_prefixes"], 1)
            self.assertEqual(summary["inference_errors"], 1)
            self.assertEqual(summary["unprocessed_prefixes"], 1)
            with (root / "output/folder_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                folders = list(csv.DictReader(stream))
            self.assertEqual(folders[0]["attempted_prefixes"], "2")
            self.assertEqual(folders[0]["invalid_prefixes"], "0")
            self.assertEqual(folders[0]["inference_errors"], "1")

    def test_output_collision_and_input_directory_protection(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            sample_dir, prefix_dir = self.fixture(root)
            for output in (sample_dir / "results", prefix_dir / "results", root):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "输出目录"):
                    tester.main(["--sample-dir", str(sample_dir), "--prefix-dir", str(prefix_dir),
                                 "--output-dir", str(output), "--inspect"])
