"""CPU integration tests use a tiny, real Qwen3 with random frozen weights."""
from copy import deepcopy
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiment.backend import Backend, attention_summary, join_cache, kv_losses, slice_cache
from experiment.checkpoints import TopK
from experiment.config import DEFAULTS, METHODS, load_config, validate
from experiment.data import catalog, materialize, question_text
from experiment.io import RunBusy, read_json, run_lock, write_json
from experiment.metrics import AnswerMonitor, SupportScore, answer_match, summary
from experiment.objectives import Objective
from experiment.runner import run
from experiment.trainer import train

torch.set_num_threads(1)


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1 + ord(character) % 30 for character in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def decode(self, values, skip_special_tokens=False):
        return "".join(str(value) for value in values if value != 0)

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + "\nassistant:"


def backend():
    torch.manual_seed(11)
    configuration = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                               max_position_embeddings=1024, attention_dropout=0.0)
    configuration._attn_implementation = "eager"
    return Backend(Qwen3ForCausalLM(configuration), Tokenizer())


def config(**overrides):
    value = deepcopy(DEFAULTS)
    value.update(device="cpu", dtype="float32", rounds=2, steps_per_round=2, save_topk=2,
                 eval_every=1, log_every=1, plot_every=4, resume_every=1, reference_refresh_steps=2,
                 rollout_tokens=2, max_new_tokens=2, score_window_steps=2, ema_span_steps=2,
                 background=False, continue_on_error=False, score_probability="sequence")
    # Existing generic fixtures have no structured fields; use the explicit ablation.
    value["semantic_query_mode"] = "uniform"
    value.update(overrides)
    return validate(value)


class Base(unittest.TestCase):
    def setUp(self):
        self.backend = backend()
        self.private_ids = self.backend.encode("private")
        self.public_ids = self.backend.encode("public")
        self.question_ids = self.backend.encode("question")
        self.observed, self.length, self.oracle = self.backend.setup_teacher(self.private_ids, self.public_ids, True)
        self.initial = torch.randn((1, self.length, 16), generator=torch.Generator().manual_seed(42)) * 0.02


class ConfigurationTests(unittest.TestCase):
    def test_run_lock_blocks_concurrent_resume_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            with run_lock(directory):
                with self.assertRaises(RunBusy), run_lock(directory):
                    pass
            with run_lock(directory):
                pass

    def test_two_base_losses_and_single_seed_only(self):
        for invalid in ({"base_loss": "alternating"}, {"base_loss": "joint"}, {"seed": [42, 43]},
                        {"save_topk": 0}, {"methods": []}, {"methods": ["baseline", "baseline"]},
                        {"method_options": {"baseline": {"seed": 3}}}, {"temperature": float("nan")}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                config(**invalid)

    def test_auxiliary_only_base_rejects_baseline_and_requires_total_topk(self):
        invalid = (
            {"base_loss": "none", "methods": ["baseline"], "checkpoint_metric": "total"},
            {"base_loss": "none", "methods": ["question_weighted_kv"], "checkpoint_metric": "base"},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                config(**values)
        config(base_loss="none", checkpoint_metric="total",
               methods=["question_weighted_kv", "question_attention_reconstruction", "oracle_private_prefix_distillation"])

    def test_method_overrides_validated(self):
        config(method_options={"oracle_private_prefix_distillation": {"lambda_kl": 0.2}})
        with self.assertRaises(ValueError):
            config(method_options={"oracle_private_prefix_distillation": {"lambda_kl": -1}})

    def test_config_relative_paths_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "data.json", [{"task_id": "one"}, {"task_id": "two"}])
            write_json(root / "config.json", {"datasets": ["data.json"], "include_tasks": ["two"]})
            loaded = load_config(root / "config.json")
            self.assertEqual([r["task_id"] for r in catalog(loaded)], ["two"])
            self.assertEqual(loaded["datasets"], [str((root / "data.json").resolve())])
            loaded["exclude_tasks"] = ["missing"]
            with self.assertRaises(ValueError):
                catalog(loaded)

    def test_inline_data_output_constraint_before_answer_boundary(self):
        row = {"task_id": "sample", "_manifest": str(Path.cwd() / "manifest.json"),
               "private_prefix": "hidden", "public_text": "public", "question": "How many?\nAnswer:",
               "answer": "12", "output_instruction": "Number only", "held_out_question": "Seats?"}
        sample = materialize(row, "default")
        self.assertEqual(sample.held_out_answer, "12")
        self.assertTrue(question_text(sample.question, sample.output_instruction).endswith("Answer: "))
        self.assertTrue(question_text(sample.question, sample.output_instruction).startswith("Number only"))


class CacheAndObjectiveTests(Base):
    def test_mean_and_token_modes_choose_only_configured_base(self):
        for mode in ("mean", "token"):
            objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids, "baseline", config(base_loss=mode))
            loss, parts, _ = objective.compute(self.initial)
            torch.testing.assert_close(loss, parts["mean_loss" if mode == "mean" else "token_loss"])

    def test_auxiliary_only_modes_exclude_base_loss_from_total(self):
        cases = (("question_weighted_kv", "weighted_kv_loss"),
                 ("question_attention_reconstruction", None))
        for method, expected_component in cases:
            with self.subTest(method=method):
                cfg = config(methods=[method], base_loss="none", checkpoint_metric="total")
                objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids, method, cfg)
                prefix = self.initial.clone().requires_grad_()
                objective.refresh(prefix, 0)
                loss, parts, _ = objective.compute(prefix)
                self.assertEqual(float(parts["base_loss"]), 0)
                if expected_component:
                    torch.testing.assert_close(loss, parts[expected_component])
                else:
                    expected = parts["attention_output_loss"] + parts["lse_loss"] + parts["attention_distribution_loss"]
                    torch.testing.assert_close(loss, expected)
                loss.backward()
                self.assertTrue(torch.isfinite(prefix.grad).all())
                self.assertGreater(float(prefix.grad.norm()), 0)

    def test_true_prefix_reproduces_public_cache(self):
        prefix = self.backend.embedding(self.private_ids).detach()
        _, predicted = self.backend.student(prefix, self.public_ids)
        losses = kv_losses(predicted, self.observed)
        self.assertLess(float(losses["mean_loss"]), 1e-12)
        self.assertLess(float(losses["token_loss"]), 1e-12)

    def test_differentiable_cache_does_not_mutate_observed(self):
        before = [(key.clone(), value.clone()) for key, value in self.observed]
        prefix = self.initial.clone().requires_grad_()
        private, _ = self.backend.student(prefix, self.public_ids)
        logits, _ = self.backend.forward(ids=self.question_ids, cache=join_cache(private, self.observed))
        logits[0, -1, 2].backward()
        self.assertGreater(float(prefix.grad.norm()), 0)
        for actual, expected in zip(self.observed, before):
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)

    def test_every_method_has_student_gradients_frozen_reference_and_model(self):
        for method in METHODS:
            with self.subTest(method=method):
                prefix = self.initial.clone().requires_grad_()
                objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids,
                                      method, config(), self.oracle if method.startswith("oracle") else None)
                objective.refresh(prefix, 0)
                loss, _, _ = objective.compute(prefix)
                loss.backward()
                self.assertTrue(torch.isfinite(prefix.grad).all())
                self.assertGreater(float(prefix.grad.norm()), 0)
                self.assertTrue(all(p.grad is None for p in self.backend.model.parameters()))
                if objective.reference:
                    def tensors(value):
                        if isinstance(value, torch.Tensor):
                            yield value
                        elif isinstance(value, dict):
                            for item in value.values():
                                yield from tensors(item)
                        elif isinstance(value, tuple):
                            for item in value:
                                yield from tensors(item)
                    self.assertTrue(all(not x.requires_grad for x in tensors(objective.reference)))

    def test_non_oracle_rejects_private_cache(self):
        with self.assertRaises(ValueError):
            Objective(self.backend, self.observed, self.public_ids, self.question_ids, "baseline", config(), self.oracle)

    def test_sdpa_backend_preserves_all_active_gradient_paths(self):
        self.backend.model.config._attn_implementation = "sdpa"
        for method in METHODS:
            with self.subTest(method=method):
                prefix = self.initial.clone().requires_grad_()
                objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids,
                                      method, config(), self.oracle if method.startswith("oracle") else None)
                objective.refresh(prefix, 0)
                loss, _, _ = objective.compute(prefix)
                loss.backward()
                self.assertTrue(torch.isfinite(prefix.grad).all())
                self.assertGreater(float(prefix.grad.norm()), 0)

    def test_weights_have_floor_unit_mean_and_fixed_window(self):
        objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids,
                              "question_weighted_kv", config(weight_floor=0.2))
        self.assertTrue(objective.refresh(self.initial, 0))
        for weight in objective.reference["weights"]:
            self.assertGreaterEqual(float(weight.min()), 0.2)
            torch.testing.assert_close(weight.mean(-1), torch.ones_like(weight.mean(-1)))
        self.assertFalse(objective.refresh(self.initial * 2, 1))
        self.assertTrue(objective.refresh(self.initial * 2, 2))

    def test_attention_loss_zero_for_original_cache(self):
        prefix = self.backend.embedding(self.private_ids).detach()
        objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids,
                              "question_attention_reconstruction", config())
        objective.refresh(prefix, 0)
        _, parts, _ = objective.compute(prefix)
        self.assertLess(float(parts["attention_output_loss"]), 1e-12)
        self.assertLess(float(parts["lse_loss"]), 1e-12)
        self.assertLess(float(parts["attention_distribution_loss"]), 1e-12)

    def test_oracle_reference_uses_true_prefix_and_fixed_distribution(self):
        objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids,
                              "oracle_private_prefix_distillation", config(), self.oracle)
        objective.refresh(self.initial, 0)
        true_prefix = self.backend.embedding(self.private_ids).detach()
        _, parts, _ = objective.compute(true_prefix)
        self.assertLess(abs(float(parts["kl_loss"])), 1e-6)
        self.assertFalse(objective.refresh(self.initial * 2, 200))

    def test_multitoken_probability_matches_autoregressive_product(self):
        private, _ = self.backend.student(self.initial, self.public_ids)
        result = AnswerMonitor(self.backend, self.observed, self.question_ids, "12")(private)
        ids = self.backend.encode("12")
        cache = join_cache(private, self.observed)
        question = self.question_ids
        expected = 1.0
        for token in ids[0]:
            logits, _ = self.backend.forward(ids=question, cache=cache)
            expected *= float(logits[0, -1].float().softmax(-1)[token])
            question = torch.cat((question, token.view(1, 1)), dim=1)
        self.assertAlmostEqual(result["answer_probability"], expected, places=8)
        self.assertEqual(result["answer_tokens"], 2)


class ScoreAndPoolTests(unittest.TestCase):
    def test_score_window_and_time_scaled_ema(self):
        score = SupportScore(config(), 0.1)
        self.assertIsNone(score.add(0, 0.1)["support_score"])
        self.assertIsNone(score.add(1, 0.2)["support_score"])
        complete = score.add(2, 0.3)
        self.assertGreater(complete["support_score"], 0)
        self.assertEqual(complete["ema_score"], complete["support_score"])
        following = score.add(3, 0.4)
        alpha = 1 - 0.5 ** 0.5
        expected = alpha * following["support_score"] + (1 - alpha) * complete["ema_score"]
        self.assertAlmostEqual(following["ema_score"], expected)

    def test_topk_is_loss_ranked_immutable_and_restorable_after_pruning(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = TopK(directory, 2, "base")
            prefix = torch.tensor([[[1.0]]])
            pool.consider(1, prefix, 3)
            pool.consider(2, prefix, 2)
            old = pool.state_dict()
            pool.commit_resume()
            prefix.add_(10)
            pool.consider(3, prefix, 1)
            self.assertEqual([r["step"] for r in pool.entries], [3, 2])
            self.assertTrue((Path(directory) / "step_00000001.pt").exists())
            pool.load_state_dict(old)
            self.assertEqual([r["step"] for r in pool.entries], [2, 1])
            payload = torch.load(Path(directory) / pool.entries[0]["path"], weights_only=True)
            self.assertEqual(float(payload["soft_prefix"][0, 0, 0]), 1)

    def test_statistics_distinguish_top1_and_topk_coverage(self):
        rows = [{"task_id": "one", "rank": 1, "first_token_match": False, "full_match": False, "answer_match": False},
                {"task_id": "one", "rank": 2, "first_token_match": True, "full_match": True, "answer_match": True}]
        value = summary(rows, 40)
        self.assertEqual(value["answer_match"]["top1_rate"], 0)
        self.assertEqual(value["answer_match"]["prefix_rate"], 0.5)
        self.assertEqual(value["answer_match"]["topk_coverage"], 1)
        self.assertEqual(value["completed_samples"], 1)

    def test_answer_extraction_without_forcing_correctness(self):
        self.assertTrue(answer_match("Agent: 2\nexplanation", "2", ["two"])["answer_match"])
        self.assertFalse(answer_match("1\nIt could be 2", "2", ["two"])["answer_match"])
        self.assertFalse(answer_match("Answer: 2", "2", ["two"])["format_compliant"])


class TrainingTests(Base):
    def test_unfinished_samples_are_not_reported_as_completed_tests(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            write_json(root / "tasks.json", [{"task_id": "case", "private_prefix": "private", "public_text": "public", "question": "question", "answer": "12"}])
            write_json(root / "selected_tasks.json", ["case"])
            cfg = config(datasets=[str(root / "tasks.json")])
            run(cfg, root, test_only=True, backend=self.backend)
            result = read_json(root / "result.json")
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["completed"], [])
            self.assertEqual(len(result["skipped"]), 1)

    def test_answer_labels_do_not_change_optimization_or_topk(self):
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right, contextlib.redirect_stdout(io.StringIO()):
            cfg = config(methods=["question_attention_reconstruction"], rounds=1, steps_per_round=2)
            for path, answer in ((left, "12"), (right, "99")):
                train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                      "question_attention_reconstruction", cfg, path,
                      AnswerMonitor(self.backend, self.observed, self.question_ids, answer))
            a = torch.load(Path(left) / "latest.pt", weights_only=True)
            b = torch.load(Path(right) / "latest.pt", weights_only=True)
            torch.testing.assert_close(a["prefix"], b["prefix"], atol=0, rtol=0)
            self.assertEqual([(r["step"], r["loss"]) for r in a["topk"]],
                             [(r["step"], r["loss"]) for r in b["topk"]])

    def test_budget_saved_losses_and_only_three_figure_types(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            cfg = config()
            monitor = AnswerMonitor(self.backend, self.observed, self.question_ids, "12")
            rows = train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                         "baseline", cfg, directory, monitor)
            self.assertEqual(rows[-1]["step"], 4)
            self.assertEqual([r["step"] for r in rows], [0, 1, 2, 3, 4])
            manifest = read_json(Path(directory) / "prefixes" / "manifest.json")
            self.assertEqual(manifest["prefix_count"], 2)
            for record in manifest["prefixes"]:
                payload = torch.load(Path(directory) / "prefixes" / record["path"], weights_only=True)
                _, public = self.backend.student(payload["soft_prefix"], self.public_ids)
                loss = float(kv_losses(public, self.observed)["token_loss"])
                self.assertAlmostEqual(loss, record["loss"], places=7)
            self.assertEqual({p.name for p in (Path(directory) / "figures").glob("*.png")},
                             {"loss_probability.png", "support_score.png", "support_ema.png"})

    def test_resume_matches_uninterrupted_optimizer_and_reference(self):
        cfg = config(methods=["question_attention_reconstruction"])
        monitor = AnswerMonitor(self.backend, self.observed, self.question_ids, "12")
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as resumed, contextlib.redirect_stdout(io.StringIO()):
            train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                  "question_attention_reconstruction", cfg, full, monitor)
            original = Objective.compute
            calls = [0]

            def interrupted(objective, prefix):
                calls[0] += 1
                if calls[0] == 6:  # Before third update; latest.pt is step two.
                    raise KeyboardInterrupt()
                return original(objective, prefix)

            with patch.object(Objective, "compute", interrupted), self.assertRaises(KeyboardInterrupt):
                train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                      "question_attention_reconstruction", cfg, resumed, monitor)
            train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                  "question_attention_reconstruction", cfg, resumed, monitor, resume=True)
            left = torch.load(Path(full) / "latest.pt", weights_only=True)
            right = torch.load(Path(resumed) / "latest.pt", weights_only=True)
            torch.testing.assert_close(left["prefix"], right["prefix"], rtol=0, atol=0)
            self.assertEqual(left["history"], right["history"])
            self.assertEqual(left["objective"]["attention_loss_version"], 2)
            self.assertTrue(all("attention_distribution_loss" in row for row in left["history"]))

    def test_all_active_methods_end_to_end_same_initial_and_heldout(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output_log:
            root = Path(directory)
            dataset = root / "tasks.json"
            write_json(dataset, [{"task_id": "case", "private_prefix": "private", "public_text": "public",
                                  "question": "question", "answer": "12", "held_out_question": "other question"}])
            cfg = config(datasets=[str(dataset)], methods=list(METHODS), rounds=1, steps_per_round=2)
            output = root / "run"
            output.mkdir()
            self.assertEqual(run(cfg, output, backend=self.backend), 0)
            summary_data = read_json(output / "test_summary.json")
            self.assertEqual(set(summary_data), set(METHODS))
            with (output / "test_method_success.csv").open(encoding="utf-8-sig", newline="") as source:
                method_counts = list(csv.DictReader(source))
            with (output / "test_sample_prefix_rates.csv").open(encoding="utf-8-sig", newline="") as source:
                prefix_rates = list(csv.DictReader(source))
            self.assertEqual(len(method_counts), len(METHODS) * 2)
            self.assertEqual(len(prefix_rates), len(METHODS) * 2)
            for row in method_counts:
                expected = summary_data[row["method"]][row["question_kind"]]["answer_match"]
                self.assertEqual(int(row["successful_samples"]), expected["covered_samples"])
                self.assertEqual(int(row["tested_samples"]), 1)
                self.assertEqual(int(row["planned_samples"]), 1)
            for row in prefix_rates:
                expected = summary_data[row["method"]][row["question_kind"]]["answer_match"]
                self.assertEqual(row["task_id"], "case")
                self.assertEqual(int(row["tested_prefixes"]), 2)
                self.assertEqual(float(row["prefix_success_rate"]), expected["prefix_rate"])
            for kind in ("training_question", "held_out_question"):
                self.assertEqual(output_log.getvalue().count(f"[统计表1/{kind}]"), 1)
                self.assertEqual(output_log.getvalue().count(f"[统计表2/{kind}]"), 1)
            initial_probabilities = []
            for method in METHODS:
                self.assertEqual(summary_data[method]["training_question"]["prefix_tests"], 2)
                self.assertEqual(summary_data[method]["held_out_question"]["prefix_tests"], 2)
                history = [json.loads(x) for x in (output / "samples" / "case" / method / "history.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(history[-1]["step"], 2)
                self.assertTrue(math.isfinite(history[-1]["answer_probability"]))
                initial_probabilities.append(history[0]["answer_probability"])
            self.assertEqual(len(set(initial_probabilities)), 1)
            self.assertEqual({p.name for p in (output / "samples" / "case" / "comparison").glob("*.png")},
                             {"answer_probability_comparison.png", "support_score_comparison.png", "support_ema_comparison.png"})


if __name__ == "__main__":
    unittest.main()
