"""Position-sensitive attention loss and historical-run boundaries."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiment.backend import attention_summary
from experiment.cli import main
from experiment.config import DEFAULTS, load_config, saved_config
from experiment.io import write_json
from experiment.objectives import Objective
from experiment.plots import render
from experiment.trainer import fingerprint, train
from test_experiment import Base, config


class DistributionTests(unittest.TestCase):
    def test_swapped_positions_detected_even_when_output_lse_and_average_match(self):
        # Opposite queries swap their attention: averaging queries hides the error.
        query = torch.tensor([[[[1., 0.], [-1., 0.]]]])
        key = torch.tensor([[[[2., 0.], [-2., 0.]]]], requires_grad=True)
        value = torch.tensor([[[[3., 4.], [3., 4.]]]], requires_grad=True)
        predicted_key = key.detach().flip(2).clone().requires_grad_()
        observed = ((key, value),)
        predicted = ((predicted_key, value.detach().clone()),)
        reference = attention_summary(query, observed[0], return_log_attention=True)
        student = attention_summary(query, predicted[0], return_log_attention=True)
        torch.testing.assert_close(reference[0], student[0])
        torch.testing.assert_close(reference[1], student[1])
        torch.testing.assert_close(reference[2].exp().mean(2), student[2].exp().mean(2))

        class Backend:
            def student(self, prefix, public_ids):
                private = ((torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2)),)
                return private, predicted

            def probe(self, cache, question_ids):
                return (query,), cache

        objective = Objective(Backend(), observed, torch.ones(1, 2, dtype=torch.long),
                              torch.ones(1, 2, dtype=torch.long),
                              "question_attention_reconstruction", config())
        objective.refresh(torch.zeros(1), 0)
        _, parts, _ = objective.compute(torch.zeros(1))
        self.assertLess(float(parts["attention_output_loss"].detach()), 1e-12)
        self.assertLess(float(parts["lse_loss"].detach()), 1e-12)
        self.assertGreater(float(parts["attention_distribution_loss"].detach()), 1)
        # Key-position sum, then head/query mean: no division by public length.
        expected = (reference[2].detach().exp() * (reference[2].detach() - student[2])).sum(-1).mean()
        torch.testing.assert_close(parts["attention_distribution_loss"], expected)
        parts["attention_distribution_loss"].backward()
        self.assertTrue(torch.isfinite(predicted_key.grad).all())
        self.assertGreater(float(predicted_key.grad.norm()), 0)
        self.assertIsNone(key.grad)
        self.assertIsNone(value.grad)
        self.assertFalse(objective.reference["targets"][0][2].requires_grad)

    def test_large_scores_have_finite_log_distribution(self):
        query = torch.tensor([[[[10000., 0.]]]])
        key = torch.tensor([[[[10000., 0.], [-10000., 0.]]]])
        output, lse, log_attention = attention_summary(query, (key, torch.ones_like(key)), return_log_attention=True)
        self.assertTrue(all(torch.isfinite(x).all() for x in (output, lse, log_attention)))
        torch.testing.assert_close(log_attention.exp().sum(-1), torch.ones(1, 1, 1))

    def test_new_coefficient_validation_and_method_override(self):
        for invalid in (-1, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                config(lambda_attention=invalid)
        for invalid in (0, 3, True):
            with self.subTest(version=invalid), self.assertRaises(ValueError):
                config(attention_loss_version=invalid)
        config(method_options={"question_attention_reconstruction": {"lambda_attention": 0.1}})
        with self.assertRaises(ValueError):
            config(method_options={"question_attention_reconstruction": {"lambda_attention": -1}})

    def test_new_loss_appears_in_curve(self):
        row = {"step": 0, "base_loss": 1., "total_loss": 2.,
               "attention_distribution_loss": 0.2, "answer_probability": 0.1,
               "first_token_probability": 0.1, "support_score": None, "ema_score": None}
        labels = []

        def capture(fig, path, **kwargs):
            if Path(path).name == "loss_probability.png":
                labels.extend(line.get_label() for line in fig.axes[1].lines)

        with tempfile.TemporaryDirectory() as directory, patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=capture):
            render(directory, {"question_attention_reconstruction": [row]})
        self.assertIn("Attention: attention_distribution_loss", labels)

    def test_shipped_configs_keep_token_main_and_mean_controls(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        expected = {"default": "token", "all_methods": "token", "gpu0": "token",
                    "gpu3": "token", "gpu1": "token", "gpu2": "token"}
        for name, mode in expected.items():
            with self.subTest(config=name):
                cfg = load_config(root / (name + ".json"))
                self.assertEqual(cfg["base_loss"], mode)
                ablation = name in ("gpu0", "gpu2", "gpu3")
                self.assertEqual(cfg["attention_loss_version"], 1 if ablation else 2)
                if ablation:
                    self.assertEqual(cfg["lambda_attention"], 0)
                else:
                    self.assertGreater(cfg["lambda_attention"], 0)

    def test_auxiliary_only_configs_have_no_base_term_and_one_method(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        expected = {"attention_restruct": "question_attention_reconstruction",
                    "weight_kv": "question_weighted_kv"}
        for name, method in expected.items():
            with self.subTest(config=name):
                cfg = load_config(root / (name + ".json"))
                self.assertEqual(cfg["base_loss"], "none")
                self.assertEqual(cfg["checkpoint_metric"], "total")
                self.assertEqual(cfg["methods"], [method])


class AttentionIntegrationTests(Base):
    def test_new_loss_has_its_own_prefix_gradient_and_configured_weight(self):
        for mode in ("token", "mean"):
            with self.subTest(mode=mode):
                prefix = self.initial.clone().requires_grad_()
                objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids,
                                      "question_attention_reconstruction", config(base_loss=mode, lambda_attention=0.3))
                objective.refresh(prefix, 0)
                loss, parts, _ = objective.compute(prefix)
                expected_aux = parts["attention_output_loss"] + parts["lse_loss"] + 0.3 * parts["attention_distribution_loss"]
                torch.testing.assert_close(parts["weighted_aux_loss"], expected_aux)
                torch.testing.assert_close(loss, parts[mode + "_loss"] + expected_aux)
                parts["attention_distribution_loss"].backward()
                self.assertTrue(torch.isfinite(prefix.grad).all())
                self.assertGreater(float(prefix.grad.norm()), 0)
                self.assertTrue(all(p.grad is None for p in self.backend.model.parameters()))

    def test_zero_coefficient_recovers_original_objective(self):
        old = Objective(self.backend, self.observed, self.public_ids, self.question_ids,
                        "question_attention_reconstruction", config(attention_loss_version=1))
        new = Objective(self.backend, self.observed, self.public_ids, self.question_ids,
                        "question_attention_reconstruction", config(lambda_attention=0))
        for objective in (old, new):
            objective.refresh(self.initial, 0)
        old_loss, old_parts, _ = old.compute(self.initial)
        new_loss, new_parts, _ = new.compute(self.initial)
        torch.testing.assert_close(old_loss, new_loss)
        self.assertNotIn("attention_distribution_loss", old_parts)
        self.assertIn("attention_distribution_loss", new_parts)

    def test_cross_version_reference_state_and_direct_training_rejected(self):
        objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids,
                              "question_attention_reconstruction", config())
        with self.assertRaisesRegex(ValueError, "损失版本不一致"):
            objective.load_state_dict({"reference": None, "reference_step": None})
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "旧 attention"):
            train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                  "question_attention_reconstruction", config(attention_loss_version=1), directory, None)


class HistoricalAttentionTests(unittest.TestCase):
    def legacy(self):
        cfg = deepcopy(DEFAULTS)
        cfg["methods"] = ["question_attention_reconstruction"]
        cfg.pop("attention_loss_version")
        cfg.pop("lambda_attention")
        return cfg

    def test_saved_missing_version_remains_legacy_and_cannot_resume(self):
        cfg = self.legacy()
        self.assertEqual(saved_config(cfg)["attention_loss_version"], 1)
        self.assertEqual(saved_config(cfg)["lambda_attention"], 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "resolved_config.json", cfg)
            with patch("experiment.cli.catalog", return_value=[]), patch("experiment.cli.execute", return_value=0) as execute:
                self.assertEqual(main(["test", "--run-dir", directory, "--foreground"]), 0)
                self.assertEqual(execute.call_args.args[0]["attention_loss_version"], 1)
                execute.reset_mock()
                with self.assertRaisesRegex(ValueError, "旧 attention"):
                    main(["--resume", directory, "--foreground"])
                execute.assert_not_called()
                self.assertEqual(main(["diagnose", "--run-dir", directory, "--foreground"]), 0)
                self.assertEqual(execute.call_args.args[0]["attention_loss_version"], 1)

    def test_worker_blocks_legacy_training_but_allows_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "resolved_config.json", self.legacy())
            for test_only, source in ((False, None), (True, None), (False, directory)):
                with self.subTest(test_only=test_only, source=source):
                    write_json(root / "worker.json", {"resume": True, "test_only": test_only, "source": source})
                    with patch("experiment.cli.execute", return_value=0) as execute:
                        if test_only or source:
                            self.assertEqual(main(["--worker", directory]), 0)
                            self.assertEqual(execute.call_args.args[0]["attention_loss_version"], 1)
                        else:
                            with self.assertRaisesRegex(ValueError, "旧 attention"):
                                main(["--worker", directory])
                            execute.assert_not_called()

    def test_other_methods_fingerprint_ignores_new_attention_fields(self):
        old = self.legacy()
        new = saved_config(old)
        for method in ("baseline", "question_weighted_kv", "oracle_private_prefix_distillation"):
            with self.subTest(method=method):
                self.assertEqual(fingerprint(old, method, "context"), fingerprint(new, method, "context"))
        self.assertNotEqual(fingerprint(old, "question_attention_reconstruction", "context"),
                            fingerprint(new, "question_attention_reconstruction", "context"))


if __name__ == "__main__":
    unittest.main()
