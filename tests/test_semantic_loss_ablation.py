"""Old losses with current questions: real training, resume and version isolation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiment.config import METHODS
from experiment.cli import main
from experiment.io import read_json, write_json
from experiment.metrics import AnswerMonitor
from experiment.objectives import Objective
from experiment.runner import run
from experiment.trainer import train
from prepare_questions import QUESTIONS
from test_experiment import Base, config


def ablation_config(**overrides):
    return config(**{
        "loss_profile": "semantic_enhanced_v1",
        "weighted_kv_version": 1, "semantic_query_mode": "uniform",
        "attention_loss_version": 1, "lambda_attention": 0,
        **overrides,
    })


class ProfileTests(unittest.TestCase):
    def test_profile_cannot_silently_run_new_losses_or_structured_weights(self):
        for values in ({"weighted_kv_version": 2}, {"attention_loss_version": 2},
                       {"semantic_query_mode": "structured"}, {"lambda_attention": 1}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "semantic_enhanced_v1"):
                ablation_config(**values)

    def test_actual_launch_configs_pass_foreground_worker_and_resume_guards(self):
        project = Path(__file__).resolve().parents[1]
        for filename in ("gpu0.json", "gpu2.json", "gpu3.json"):
            with self.subTest(config=filename), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                output = Path(directory) / "run"
                with patch("experiment.cli.execute", return_value=0) as execute:
                    self.assertEqual(main(["--config", str(project / "configs" / filename),
                                           "--run-dir", str(output), "--foreground"]), 0)
                    self.assertEqual(execute.call_args.args[0]["loss_profile"],
                                     "current" if filename == "gpu3.json" else "semantic_enhanced_v1")
                    write_json(output / "worker.json", {"resume": False, "test_only": False})
                    self.assertEqual(main(["--worker", str(output)]), 0)
                    self.assertEqual(main(["--resume", str(output), "--foreground"]), 0)


class AblationIntegrationTests(Base):
    def test_three_groups_train_with_current_questions_and_legacy_losses(self):
        training, held, answer_format = QUESTIONS["city"]
        for mode in ("token", "mean", "none"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                root = Path(directory)
                data = root / "tasks.json"
                write_json(data, [{"task_id": "case", "private_prefix": "private", "public_text": "public",
                                   "question": training, "held_out_question": held,
                                   "answer": "City", "answer_format": answer_format}])
                methods = [method for method in METHODS if mode != "none" or method != "baseline"]
                cfg = ablation_config(base_loss=mode, methods=methods, datasets=[str(data)],
                                      checkpoint_metric="total", test_question_mode="held_out",
                                      rounds=1, steps_per_round=2, save_topk=1)
                output = root / "run"
                with patch.object(self.backend, "semantic_query_groups", side_effect=AssertionError("v1 must use all queries")):
                    self.assertEqual(run(cfg, output, backend=self.backend), 0)
                self.assertEqual(read_json(output / "samples/case/sample_info.json")["question"], training)
                summaries = read_json(output / "test_summary.json")
                for method in methods:
                    self.assertEqual(set(summaries[method]), {"held_out_question"})
                    folder = output / "samples/case" / method
                    recorded = read_json(folder / "config.json")
                    self.assertNotIn("semantic_query_positions", recorded)
                    self.assertEqual(recorded["uses_private_teacher"], method == "oracle_private_prefix_distillation")
                    history = [json.loads(line) for line in (folder / "history.jsonl").read_text(encoding="utf-8").splitlines()]
                    row = history[-1]
                    self.assertNotIn("attention_distribution_loss", row)
                    self.assertEqual(row["step"], 2)
                    if mode == "none":
                        self.assertEqual(row["base_loss"], 0)
                    if method == "question_attention_reconstruction":
                        self.assertAlmostEqual(row["weighted_aux_loss"], row["attention_output_loss"] + row["lse_loss"], places=6)
                    if method == "question_weighted_kv":
                        self.assertAlmostEqual(row["weighted_aux_loss"], row["weighted_kv_loss"], places=6)

    def test_v1_resume_is_exact_and_cannot_cross_to_v2(self):
        for method in ("question_weighted_kv", "question_attention_reconstruction"):
            cfg = ablation_config(methods=[method])
            monitor = AnswerMonitor(self.backend, self.observed, self.question_ids, "12")
            with self.subTest(method=method), tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as resumed, contextlib.redirect_stdout(io.StringIO()):
                train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                      method, cfg, full, monitor)
                original_compute = Objective.compute
                calls = [0]

                def interrupt(objective, prefix):
                    calls[0] += 1
                    if calls[0] == 6:
                        raise KeyboardInterrupt()
                    return original_compute(objective, prefix)

                with patch.object(Objective, "compute", interrupt), self.assertRaises(KeyboardInterrupt):
                    train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                          method, cfg, resumed, monitor)
                train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                      method, cfg, resumed, monitor, resume=True)
                left = torch.load(Path(full) / "latest.pt", weights_only=True)
                right = torch.load(Path(resumed) / "latest.pt", weights_only=True)
                torch.testing.assert_close(left["prefix"], right["prefix"], rtol=0, atol=0)
                self.assertEqual(left["topk"], right["topk"])
                with self.assertRaisesRegex(ValueError, "恢复配置或样本上下文"):
                    train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                          method, config(methods=[method]), resumed, monitor, resume=True)


if __name__ == "__main__":
    unittest.main()
