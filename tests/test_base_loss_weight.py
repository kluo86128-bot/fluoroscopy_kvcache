"""Base-loss scaling must affect only the selected method and its gradient."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiment.config import method_config, saved_config
from experiment.io import read_json, write_json
from experiment.objectives import Objective
from experiment.runner import run
from experiment.trainer import fingerprint
from test_experiment import Base, config
from test_semantic_loss_ablation import ablation_config


class CoefficientConfigurationTests(unittest.TestCase):
    def test_invalid_coefficients_rejected_globally_and_per_method(self):
        for invalid in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "lambda_base"):
                    config(lambda_base=invalid)
                with self.assertRaisesRegex(ValueError, "lambda_base"):
                    config(method_options={"question_weighted_kv": {"lambda_base": invalid}})

    def test_default_coefficient_preserves_historical_fingerprint_but_changed_one_does_not(self):
        cfg = ablation_config(methods=["question_weighted_kv"])
        old = {key: value for key, value in cfg.items() if key != "lambda_base"}
        self.assertEqual(saved_config(old)["lambda_base"], 1)
        self.assertEqual(fingerprint(old, "question_weighted_kv", "context"),
                         fingerprint(cfg, "question_weighted_kv", "context"))
        self.assertNotEqual(fingerprint(cfg, "question_weighted_kv", "context"),
                            fingerprint({**cfg, "lambda_base": 0.1}, "question_weighted_kv", "context"))


class CoefficientIntegrationTests(Base):
    def test_mixed_gradient_is_scaled_base_plus_unchanged_weighted_gradient(self):
        results = {}
        for name, method, cfg in (
                ("base", "baseline", ablation_config(methods=["baseline"])),
                ("aux", "question_weighted_kv", ablation_config(
                    base_loss="none", methods=["question_weighted_kv"], checkpoint_metric="total")),
                ("mixed", "question_weighted_kv", ablation_config(
                    lambda_base=0.1, methods=["question_weighted_kv"]))):
            prefix = self.initial.clone().requires_grad_()
            objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids, method, cfg)
            objective.refresh(prefix, 0)
            loss, parts, _ = objective.compute(prefix)
            results[name] = (loss.detach(), torch.autograd.grad(loss, prefix)[0], parts)
        torch.testing.assert_close(results["mixed"][0], 0.1 * results["base"][0] + results["aux"][0])
        torch.testing.assert_close(results["mixed"][1], 0.1 * results["base"][1] + results["aux"][1], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(results["mixed"][2]["token_loss"], results["base"][2]["token_loss"], rtol=0, atol=0)
        self.assertEqual(float(results["aux"][2]["base_loss"]), 0)

    def test_runner_applies_coefficient_only_to_weighted_method_and_records_effective_loss(self):
        project = Path(__file__).resolve().parents[1]
        sample = project / "test_samples/sgd_test_040"
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = root / "data.json"
            write_json(data, [{"task_id": "sgd_test_040", "private_prefix": "private", "public_text": "public",
                               "question": (sample / "question_search.txt").read_text(encoding="utf-8"),
                               "held_out_question": (sample / "question_test.txt").read_text(encoding="utf-8"),
                               "answer": "City", "answer_format": "english_city"}])
            cfg = ablation_config(datasets=[str(data)], methods=["baseline", "question_weighted_kv"],
                                  method_options={"question_weighted_kv": {"lambda_base": 0.1}},
                                  rounds=1, steps_per_round=2, save_topk=1,
                                  checkpoint_metric="total", test_question_mode="held_out")
            self.assertEqual(method_config(cfg, "baseline")["lambda_base"], 1)
            with patch("experiment.trainer.render"), patch("experiment.runner.render_comparison"):
                self.assertEqual(run(cfg, root / "run", backend=self.backend), 0)
            summary = read_json(root / "run/test_summary.json")
            self.assertEqual(set(summary), {"baseline", "question_weighted_kv"})
            for method, scale in (("baseline", 1), ("question_weighted_kv", 0.1)):
                folder = root / "run/samples/sgd_test_040" / method
                self.assertEqual(read_json(folder / "config.json")["config"]["lambda_base"], scale)
                history = [json.loads(line) for line in (folder / "history.jsonl").read_text(encoding="utf-8").splitlines()]
                for row in history:
                    self.assertAlmostEqual(row["base_loss"], scale * row["token_loss"], places=7)
                    self.assertAlmostEqual(row["total_loss"], row["base_loss"] + row["weighted_aux_loss"], places=7)
                self.assertEqual(set(summary[method]), {"held_out_question"})


if __name__ == "__main__":
    unittest.main()
