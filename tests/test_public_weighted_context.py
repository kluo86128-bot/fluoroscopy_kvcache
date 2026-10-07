"""Weighted references exclude inferred private KV and retain absolute RoPE positions."""
import unittest
from unittest.mock import patch

import torch

from experiment.config import JOINT_METHOD, saved_config, require_current_weighted_kv
from experiment.objectives import Objective
from experiment.trainer import fingerprint
from test_experiment import Base, config


class PublicReferenceTests(Base):
    def test_public_probe_preserves_positions_and_strict_previous_mask(self):
        before = [(key.clone(), value.clone()) for key, value in self.observed]
        length, count = self.public_ids.shape[1], self.question_ids.shape[1]
        with patch.object(self.backend.model, "forward", wraps=self.backend.model.forward) as forward:
            queries, complete = self.backend.probe_public(self.observed, self.question_ids,
                                                          prefix_length=self.length)
        arguments = forward.call_args.kwargs
        torch.testing.assert_close(arguments["position_ids"][0], torch.arange(length, length + count) + self.length)
        torch.testing.assert_close(arguments["cache_position"], torch.arange(length, length + count))
        mask = arguments["attention_mask"]
        self.assertEqual(tuple(mask.shape), (1, 1, count, length + count))
        for token in range(count):
            self.assertTrue((mask[0, 0, token, :length + token] == 0).all())
            self.assertTrue((mask[0, 0, token, length + token:] < 0).all())
        for layer, original in zip(complete, before):
            self.assertEqual(layer[0].shape[2], length + count)
            for result, target in zip(layer, original):
                torch.testing.assert_close(result[:, :, :length], target, rtol=0, atol=0)
        self.assertTrue(all(not query.requires_grad for query in queries))
        for layer, original in zip(self.observed, before):
            for result, target in zip(layer, original):
                torch.testing.assert_close(result, target, rtol=0, atol=0)

    def test_weighted_references_are_invariant_to_prefix_contents_and_keep_training_gradient(self):
        cfg = config(methods=["question_weighted_kv"], base_loss="none", checkpoint_metric="total")
        objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids,
                              "question_weighted_kv", cfg)
        with patch.object(self.backend, "student", side_effect=AssertionError("reference cannot read soft prefix KV")):
            self.assertTrue(objective.refresh(self.initial, 0))
            original = [weight.clone() for weight in objective.reference["weights"]]
            changed = torch.randn_like(self.initial) * 100
            self.assertTrue(objective.refresh(changed, cfg["reference_refresh_steps"]))
        for weight, target in zip(objective.reference["weights"], original):
            torch.testing.assert_close(weight, target, rtol=0, atol=0)
        prefix = self.initial.clone().requires_grad_()
        loss, parts, _ = objective.compute(prefix)
        loss.backward()
        self.assertEqual(float(parts["base_loss"]), 0)
        self.assertTrue(torch.isfinite(prefix.grad).all())
        self.assertGreater(float(prefix.grad.norm()), 0)

    def test_future_question_tokens_cannot_change_earlier_queries(self):
        changed = self.question_ids.clone()
        changed[:, -1] = (changed[:, -1] + 1) % self.backend.embedding.weight.shape[0]
        first, _ = self.backend.probe_public(self.observed, self.question_ids, prefix_length=self.length)
        second, _ = self.backend.probe_public(self.observed, changed, prefix_length=self.length)
        for left, right in zip(first, second):
            torch.testing.assert_close(left[:, :, :-1], right[:, :, :-1], rtol=0, atol=0)

    def test_sdpa_uses_the_same_public_mask_and_has_weighted_gradients(self):
        eager, _ = self.backend.probe_public(self.observed, self.question_ids, prefix_length=self.length)
        self.backend.model.config._attn_implementation = "sdpa"
        sdpa, _ = self.backend.probe_public(self.observed, self.question_ids, prefix_length=self.length)
        for left, right in zip(eager, sdpa):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        cfg = config(methods=["question_weighted_kv"], base_loss="none", checkpoint_metric="total")
        objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids,
                              "question_weighted_kv", cfg)
        prefix = self.initial.clone().requires_grad_()
        objective.refresh(prefix, 0)
        loss, _, _ = objective.compute(prefix)
        loss.backward()
        self.assertGreater(float(prefix.grad.norm()), 0)

    def test_legacy_state_is_readable_but_cannot_resume_new_context(self):
        cfg = config(methods=["question_weighted_kv"])
        old = {key: value for key, value in cfg.items()
               if key not in ("weighted_kv_context_version", "weighted_kv_normalization_version")}
        legacy = saved_config(old)
        self.assertEqual(legacy["weighted_kv_context_version"], 1)
        with self.assertRaisesRegex(ValueError, "旧参考方式"):
            require_current_weighted_kv(legacy)
        old_objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids,
                                  "question_weighted_kv", legacy)
        old_objective.refresh(self.initial, 0)
        objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids,
                              "question_weighted_kv", cfg)
        with self.assertRaisesRegex(ValueError, "查询上下文版本"):
            objective.load_state_dict(old_objective.state_dict())
        for method in ("question_weighted_kv", JOINT_METHOD):
            self.assertNotEqual(fingerprint(cfg, method, "same"), fingerprint(legacy, method, "same"))
            self.assertEqual(fingerprint(old, method, "same"), fingerprint(legacy, method, "same"))
        for method in ("baseline", "question_attention_reconstruction"):
            self.assertEqual(fingerprint(cfg, method, "same"), fingerprint(legacy, method, "same"))

    def test_normalization_version_guards_old_states_and_fingerprints(self):
        cfg = config(methods=["question_weighted_kv"])
        old = {key: value for key, value in cfg.items() if key != "weighted_kv_normalization_version"}
        legacy = saved_config(old)
        self.assertEqual(legacy["weighted_kv_context_version"], 2)
        self.assertEqual(legacy["weighted_kv_normalization_version"], 1)
        with self.assertRaisesRegex(ValueError, "旧归一化方式"):
            require_current_weighted_kv(legacy)
        require_current_weighted_kv(cfg)
        objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids,
                              "question_weighted_kv", cfg)
        objective.refresh(self.initial, 0)
        self.assertEqual(objective.state_dict()["weighted_kv_normalization_version"], 2)
        historical_state = objective.state_dict()
        historical_state.pop("weighted_kv_normalization_version")
        with self.assertRaisesRegex(ValueError, "归一化版本"):
            objective.load_state_dict(historical_state)
        for method in ("question_weighted_kv", JOINT_METHOD):
            self.assertEqual(fingerprint(old, method, "same"), fingerprint(legacy, method, "same"))
            self.assertNotEqual(fingerprint(cfg, method, "same"), fingerprint(legacy, method, "same"))
        for method in ("baseline", "question_attention_reconstruction"):
            self.assertEqual(fingerprint(cfg, method, "same"), fingerprint(legacy, method, "same"))
        for invalid in (0, 3, True):
            with self.assertRaisesRegex(ValueError, "normalization_version"):
                config(weighted_kv_normalization_version=invalid)


class MaskCalculationTests(unittest.TestCase):
    def test_current_and_future_keys_are_excluded_but_previous_question_keys_participate(self):
        observed = ((torch.tensor([[[[0.], [1.]]]]), torch.ones(1, 1, 2, 1)),)
        queries = torch.tensor([[[[1.], [2.], [3.]]]])
        keys = torch.tensor([[[[0.], [1.], [2.], [5.], [100.]]]])
        cfg = config(methods=["question_weighted_kv"], weighted_kv_normalization_version=1)

        def weights(candidate_keys):
            class PublicBackend:
                def probe_public(self, cache, ids, *, prefix_length):
                    return (queries,), ((candidate_keys, torch.zeros_like(candidate_keys)),)
            objective = Objective(PublicBackend(), observed, torch.ones(1, 2, dtype=torch.long),
                                  torch.ones(1, 3, dtype=torch.long), "question_weighted_kv", cfg)
            objective.refresh(torch.zeros(1, 4, 1), 0)
            return objective.reference["weights"][0]

        actual = weights(keys)
        expected_attention = []
        for token in range(3):
            allowed_keys = keys[0, 0, :2 + token, 0]
            expected_attention.append((queries[0, 0, token, 0] * allowed_keys).softmax(-1)[:2])
        importance = torch.stack(expected_attention).mean(0)
        expected = cfg["weight_floor"] + (1 - cfg["weight_floor"]) * 2 * importance / importance.sum()
        torch.testing.assert_close(actual[0, 0], expected)
        changed = keys.clone()
        changed[:, :, -1] = -10000
        torch.testing.assert_close(actual, weights(changed), rtol=0, atol=0)
        changed = keys.clone()
        changed[:, :, 2] = -5
        self.assertFalse(torch.allclose(actual, weights(changed)))


class PublicNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.observed = ((torch.tensor([[[[0.], [1.]]]]), torch.ones(1, 1, 2, 1)),)
        # Two query heads share one KV head, and prefer different positions.
        self.queries = torch.tensor([[[[-2.], [1.], [3.], [-1.]],
                                      [[-1.], [2.], [4.], [-2.]]]])
        self.keys = torch.tensor([[[[0.], [1.], [20.], [30.], [40.], [50.]]]])

    def weights(self, *, keys=None, cfg=None, groups=None):
        candidate = self.keys if keys is None else keys
        queries = self.queries
        class PublicBackend:
            def probe_public(self, cache, ids, *, prefix_length):
                return (queries,), ((candidate, torch.zeros_like(candidate)),)
        objective = Objective(PublicBackend(), self.observed, torch.ones(1, 2, dtype=torch.long),
                              torch.ones(1, 4, dtype=torch.long), "question_weighted_kv",
                              cfg or config(), question_groups=groups)
        objective.refresh(torch.zeros(1, 3, 1), 0)
        return objective.reference["weights"][0], objective

    def test_each_query_competes_only_over_public_keys_before_head_and_query_averaging(self):
        actual, objective = self.weights()
        expected_queries = torch.cat((torch.zeros_like(self.queries), self.queries), dim=-1).softmax(-1)
        expected = 0.1 + 0.9 * 2 * expected_queries.mean(dim=(1, 2))
        torch.testing.assert_close(actual[:, 0], expected)
        torch.testing.assert_close(actual.mean(-1), torch.ones(1, 1))
        self.assertGreaterEqual(float(actual.min()), 0.1)
        self.assertFalse(actual.requires_grad)
        legacy, _ = self.weights(cfg=config(weighted_kv_normalization_version=1))
        self.assertFalse(torch.allclose(actual, legacy))

    def test_previous_current_and_future_question_keys_do_not_compete_for_weight_mass(self):
        actual, _ = self.weights()
        changed = self.keys.clone()
        changed[:, :, 2:] = torch.tensor([-10000., 10000., -5000., 5000.]).view(1, 1, 4, 1)
        new, _ = self.weights(keys=changed)
        torch.testing.assert_close(actual, new, rtol=0, atol=0)
        changed[:, :, 1] = -2
        public_changed, _ = self.weights(keys=changed)
        self.assertFalse(torch.allclose(actual, public_changed))

    def test_structured_coefficients_apply_after_per_query_public_softmax(self):
        groups = {"object_state": [0], "target_field": [1], "required_value": [2], "question": [3]}
        actual, objective = self.weights(cfg=config(semantic_query_mode="structured"), groups=groups)
        distribution = torch.cat((torch.zeros_like(self.queries), self.queries), dim=-1).softmax(-1)
        coefficients = distribution.new_tensor(objective.query_weights)[None, None, :, None]
        expected = 0.1 + 0.9 * 2 * (distribution * coefficients).sum(2).mean(1)
        torch.testing.assert_close(actual[:, 0], expected)
