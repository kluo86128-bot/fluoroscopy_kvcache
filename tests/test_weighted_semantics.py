"""Semantic query allocation, full-prompt alignment and weighted-KV integration."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from experiment.cli import main
from experiment.config import DEFAULTS, load_config, saved_config
from experiment.data import question_text
from experiment.io import read_json, write_json
from experiment.metrics import AnswerMonitor
from experiment.objectives import Objective
from experiment.question_semantics import RELATION_WEIGHTS, SEMANTIC_WEIGHTS, effective_shares, query_coefficients, semantic_spans
from experiment.runner import run
from experiment.trainer import fingerprint, train
from prepare_questions import QUESTIONS
from test_experiment import Base, backend, config


class MappingTests(unittest.TestCase):
    def test_group_shares_do_not_depend_on_token_counts(self):
        groups = {"object_state": [1, 2], "target_field": [4], "required_value": [6, 7, 8], "question": [10, 11]}
        coefficients = query_coefficients(14, groups, SEMANTIC_WEIGHTS)
        for group, indices in groups.items():
            self.assertAlmostEqual(sum(coefficients[i] for i in indices), SEMANTIC_WEIGHTS[group])
        self.assertAlmostEqual(sum(coefficients), 1)
        self.assertEqual([coefficients[i] for i in (0, 3, 5, 9, 12, 13)], [0] * 6)

    def test_unified_relation_merges_target_budget_and_accepts_explicit_three_groups(self):
        groups = {"target_relation": [1, 2, 3], "required_value": [6, 7], "question": [10]}
        merged = query_coefficients(12, groups, SEMANTIC_WEIGHTS)
        self.assertEqual(merged, query_coefficients(12, groups, RELATION_WEIGHTS))
        for group, positions in groups.items():
            self.assertAlmostEqual(sum(merged[i] for i in positions), RELATION_WEIGHTS[group])
        self.assertAlmostEqual(sum(merged), 1)
        config(semantic_query_weights=RELATION_WEIGHTS.copy())

    def test_real_qwen_tokenizer_aligns_raw_and_chat_without_label_or_format_queries(self):
        path = Path(__file__).resolve().parents[2] / "develop_experiment/tokenizer/Qwen3-4B"
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        encoding_backend = backend()
        encoding_backend.tokenizer = tokenizer
        for slot, (training, held, _) in QUESTIONS.items():
            # Held-out questions deliberately omit the training-only relation.
            with self.assertRaises(ValueError):
                semantic_spans(held)
            for question in (training,):
                for mode in ("raw", "chat"):
                    with self.subTest(slot=slot, mode=mode, held=question == held):
                        cfg = config(question_format=mode, semantic_query_mode="structured")
                        text = question_text(question, cfg["output_instruction"], cfg["answer_boundary"])
                        ids = encoding_backend.encode_question(text, cfg)
                        groups = encoding_backend.semantic_query_groups(text, cfg, ids)
                        coefficients = query_coefficients(ids.shape[1], groups, SEMANTIC_WEIGHTS)
                        prompt = encoding_backend.question_prompt(text, cfg)
                        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
                        selected = {i for positions in groups.values() for i in positions}
                        self.assertTrue(all(coefficients[i] == 0 for i in range(len(coefficients)) if i not in selected))
                        format_end = prompt.index("Target relation:")
                        self.assertTrue(all(coefficients[i] == 0 for i, (_, end) in enumerate(encoded["offset_mapping"]) if end <= format_end))
                        self.assertEqual(coefficients[-1], 0)  # Prefilled answer whitespace.
                        spans = semantic_spans(prompt)
                        shares = effective_shares(groups, SEMANTIC_WEIGHTS)
                        for group, indices in groups.items():
                            self.assertAlmostEqual(sum(coefficients[i] for i in indices), shares[group])
                            for i in indices:
                                a, b = encoded["offset_mapping"][i]
                                self.assertTrue(any(any(c.isalnum() for c in prompt[max(a, x):min(b, y)])
                                                    for x, y in spans[group] if max(a, x) < min(b, y)))
                        with self.assertRaisesRegex(ValueError, "实际 question 输入不一致"):
                            encoding_backend.semantic_query_groups(text, cfg, ids[:, :-1])

    def test_malformed_structure_and_positions_fail_instead_of_uniform_fallback(self):
        with self.assertRaisesRegex(ValueError, "Target object"):
            semantic_spans("Which city?")
        with self.assertRaisesRegex(ValueError, "Target relation"):
            semantic_spans(QUESTIONS["city"][0] + "\nTarget relation: Another relation.")
        with self.assertRaisesRegex(ValueError, "混用"):
            semantic_spans(QUESTIONS["city"][0] + "\nTarget field: Another field.")
        legacy = "Target object: Restaurant.\nTarget state: Confirmed.\nTarget field: City.\nRequired value: Complete name.\nQuestion: Which city?"
        self.assertEqual(set(semantic_spans(legacy)), set(SEMANTIC_WEIGHTS))
        groups = {"object_state": [0], "target_field": [1], "required_value": [2], "question": [3]}
        for invalid in ([], [1], [9], [True]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                query_coefficients(4, {**groups, "question": invalid}, SEMANTIC_WEIGHTS)

    def test_config_validates_shares_and_all_shipped_configs_enable_semantics(self):
        for invalid in ({}, {**SEMANTIC_WEIGHTS, "question": 0.5}, {**SEMANTIC_WEIGHTS, "question": float("nan")},
                        {**SEMANTIC_WEIGHTS, "question": -1}, {**SEMANTIC_WEIGHTS, "question": True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                config(semantic_query_weights=invalid)
        for version in (0, 3, True):
            with self.subTest(version=version), self.assertRaises(ValueError):
                config(weighted_kv_version=version)
        config(method_options={"question_weighted_kv": {"semantic_query_weights": SEMANTIC_WEIGHTS.copy()}})
        for path in (Path(__file__).resolve().parents[1] / "configs").glob("*.json"):
            cfg = load_config(path)
            self.assertEqual(cfg["weighted_kv_version"], 2)
            self.assertEqual(cfg["semantic_query_mode"], "structured")
            expected = RELATION_WEIGHTS if path.stem in ("gpu0", "gpu2", "gpu3") else SEMANTIC_WEIGHTS
            self.assertEqual(cfg["semantic_query_weights"], expected)


class AggregationTests(unittest.TestCase):
    def test_excluded_queries_cannot_directly_change_kv_weights_but_field_queries_can(self):
        groups = {"object_state": [0], "target_field": [1], "required_value": [2], "question": [3]}
        cfg = config(semantic_query_mode="structured")
        observed = ((torch.tensor([[[[-1.], [1.]]]]), torch.ones(1, 1, 2, 1)),)
        private = ((torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)),)
        keys = torch.cat((private[0][0], observed[0][0], torch.zeros(1, 1, 6, 1)), dim=2)
        queries = torch.tensor([[[[-1.], [2.], [1.], [-0.5], [0.1], [0.2]],
                                 [[1.], [3.], [0.5], [-1.], [0.3], [0.4]]]])

        def weights(q, version=2, mode="structured"):
            class ProbeBackend:
                def student(self, prefix, public_ids):
                    return private, observed
                def probe(self, cache, question_ids):
                    return (q,), ((keys, torch.zeros_like(keys)),)
            objective = Objective(ProbeBackend(), observed, torch.ones(1, 2, dtype=torch.long),
                                  torch.ones(1, 6, dtype=torch.long), "question_weighted_kv",
                                  {**cfg, "weighted_kv_version": version, "semantic_query_mode": mode}, question_groups=groups)
            objective.refresh(torch.zeros(1), 0)
            return objective.reference["weights"][0]

        original = weights(queries)
        altered = queries.clone()
        altered[:, :, 4:] = -1000
        torch.testing.assert_close(original, weights(altered), rtol=0, atol=0)
        altered[:, :, 1] *= -1
        self.assertFalse(torch.allclose(original, weights(altered)))
        torch.testing.assert_close(original.mean(-1), torch.ones(1, 1))
        self.assertGreaterEqual(float(original.min()), cfg["weight_floor"])
        self.assertFalse(original.requires_grad)
        torch.testing.assert_close(weights(queries, mode="uniform"), weights(queries, version=1, mode="uniform"), rtol=0, atol=0)


class SemanticIntegrationTests(Base):
    def inputs(self, cfg):
        text = question_text(QUESTIONS["party_size"][0], cfg["output_instruction"], cfg["answer_boundary"])
        ids = self.backend.encode_question(text, cfg)
        return ids, self.backend.semantic_query_groups(text, cfg, ids)

    def test_both_base_losses_have_gradients_and_detached_semantic_references(self):
        for mode in ("token", "mean"):
            cfg = config(base_loss=mode, semantic_query_mode="structured")
            ids, groups = self.inputs(cfg)
            prefix = self.initial.clone().requires_grad_()
            objective = Objective(self.backend, self.observed, self.public_ids, ids,
                                  "question_weighted_kv", cfg, question_groups=groups)
            objective.refresh(prefix, 0)
            loss, parts, _ = objective.compute(prefix)
            torch.testing.assert_close(loss, parts[mode + "_loss"] + parts["weighted_kv_loss"])
            parts["weighted_kv_loss"].backward()
            self.assertTrue(torch.isfinite(prefix.grad).all())
            self.assertGreater(float(prefix.grad.norm()), 0)
            self.assertTrue(all(not w.requires_grad for w in objective.reference["weights"]))
            self.assertTrue(all(p.grad is None for p in self.backend.model.parameters()))
            with self.assertRaisesRegex(ValueError, "四组结构化"):
                Objective(self.backend, self.observed, self.public_ids, ids, "question_weighted_kv", cfg)

    def test_new_semantic_run_resumes_exactly_and_records_groups_without_answer_supervision(self):
        cfg = config(methods=["question_weighted_kv"], semantic_query_mode="structured")
        ids, groups = self.inputs(cfg)
        monitor = AnswerMonitor(self.backend, self.observed, ids, "12")
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as resumed, contextlib.redirect_stdout(io.StringIO()):
            train(self.backend, self.observed, self.public_ids, ids, self.initial,
                  "question_weighted_kv", cfg, full, monitor, question_groups=groups)
            original_compute = Objective.compute
            calls = [0]
            def interrupt(objective, prefix):
                calls[0] += 1
                if calls[0] == 6:
                    raise KeyboardInterrupt()
                return original_compute(objective, prefix)
            with patch.object(Objective, "compute", interrupt), self.assertRaises(KeyboardInterrupt):
                train(self.backend, self.observed, self.public_ids, ids, self.initial,
                      "question_weighted_kv", cfg, resumed, monitor, question_groups=groups)
            other_monitor = AnswerMonitor(self.backend, self.observed, ids, "99")
            train(self.backend, self.observed, self.public_ids, ids, self.initial,
                  "question_weighted_kv", cfg, resumed, other_monitor, question_groups=groups, resume=True)
            left = torch.load(Path(full) / "latest.pt", weights_only=True)
            right = torch.load(Path(resumed) / "latest.pt", weights_only=True)
            torch.testing.assert_close(left["prefix"], right["prefix"], rtol=0, atol=0)
            self.assertEqual(left["topk"], right["topk"])
            self.assertEqual(left["objective"].get("weighted_kv_version"), 2)
            for a, b in zip(left["objective"]["reference"]["weights"], right["objective"]["reference"]["weights"]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            recorded = read_json(Path(full) / "config.json")
            self.assertEqual(recorded["semantic_query_positions"], groups)
            self.assertEqual(recorded["effective_semantic_query_weights"], RELATION_WEIGHTS)
            self.assertAlmostEqual(sum(recorded["query_token_coefficients"]), 1)

    def test_runner_routes_structured_query_positions_to_weighted_method(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            write_json(root / "data.json", [{"task_id": "case", "private_prefix": "private", "public_text": "public",
                                           "question": QUESTIONS["party_size"][0], "held_out_question": QUESTIONS["party_size"][1], "answer": "12"}])
            cfg = config(methods=["baseline", "question_weighted_kv"], semantic_query_mode="structured",
                         datasets=[str(root / "data.json")], rounds=1, steps_per_round=1)
            self.assertEqual(run(cfg, root / "run", backend=self.backend), 0)
            recorded = read_json(root / "run/samples/case/question_weighted_kv/config.json")
            self.assertEqual(set(recorded["semantic_query_positions"]), set(RELATION_WEIGHTS))
            baseline = read_json(root / "run/samples/case/baseline/config.json")
            self.assertNotIn("semantic_query_positions", baseline)

    def test_old_reference_or_changed_mapping_cannot_be_loaded(self):
        cfg = config(semantic_query_mode="structured")
        ids, groups = self.inputs(cfg)
        objective = Objective(self.backend, self.observed, self.public_ids, ids,
                              "question_weighted_kv", cfg, question_groups=groups)
        objective.refresh(self.initial, 0)
        with self.assertRaisesRegex(ValueError, "版本不一致"):
            objective.load_state_dict({"reference": None, "reference_step": None})
        state = objective.state_dict()
        state["reference"] = {**state["reference"], "query_token_weights": ()}
        with self.assertRaisesRegex(ValueError, "语义查询位置或贡献比例不一致"):
            objective.load_state_dict(state)
        changed = {**groups, "question": groups["question"][:-1]}
        self.assertNotEqual(fingerprint(cfg, "question_weighted_kv", "context", groups),
                            fingerprint(cfg, "question_weighted_kv", "context", changed))


class HistoricalTests(unittest.TestCase):
    def test_legacy_weighted_run_cannot_resume_but_remains_readable(self):
        cfg = deepcopy(DEFAULTS)
        cfg["methods"] = ["question_weighted_kv"]
        for key in ("weighted_kv_version", "semantic_query_mode", "semantic_query_weights"):
            cfg.pop(key)
        self.assertEqual(saved_config(cfg)["weighted_kv_version"], 1)
        self.assertEqual(saved_config(cfg)["semantic_query_mode"], "uniform")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "resolved_config.json", cfg)
            with patch("experiment.cli.catalog", return_value=[]), patch("experiment.cli.execute", return_value=0) as execute:
                with self.assertRaisesRegex(ValueError, "旧 weight_kv"):
                    main(["--resume", directory, "--foreground"])
                execute.assert_not_called()
                self.assertEqual(main(["test", "--run-dir", directory, "--foreground"]), 0)
                self.assertEqual(main(["diagnose", "--run-dir", directory, "--foreground"]), 0)
                for test_only in (False, True):
                    write_json(root / "worker.json", {"resume": True, "test_only": test_only})
                    if test_only:
                        self.assertEqual(main(["--worker", directory]), 0)
                    else:
                        with self.assertRaisesRegex(ValueError, "旧 weight_kv"):
                            main(["--worker", directory])


if __name__ == "__main__":
    unittest.main()
