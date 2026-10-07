"""The joint objective shares a prefix and preserves both loss components."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from experiment.config import JOINT_METHOD, load_config, require_current_attention_loss, require_current_weighted_kv
from experiment.data import question_text
from experiment.io import read_json, write_json
from experiment.metrics import AnswerMonitor
from experiment.objectives import Objective
from experiment.runner import run
from experiment.trainer import fingerprint, train
from prepare_questions import QUESTIONS
from test_experiment import Base, config
from test_semantic_loss_ablation import ablation_config


class JointTests(Base):
    def test_joint_loss_and_gradient_equal_sum_of_independent_losses_in_both_versions(self):
        for version in (1, 2):
            with self.subTest(version=version):
                factory = ablation_config if version == 1 else config
                cfg = factory(methods=[JOINT_METHOD], base_loss="none", checkpoint_metric="total",
                              lambda_weighted=1.3, lambda_output=0.7, lambda_lse=0.2)
                results = {}
                for method in ("question_weighted_kv", "question_attention_reconstruction", JOINT_METHOD):
                    prefix = self.initial.clone().requires_grad_()
                    objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids, method, cfg)
                    with patch.object(self.backend, "probe", wraps=self.backend.probe) as probe, \
                            patch.object(self.backend, "probe_public", wraps=self.backend.probe_public) as public_probe:
                        objective.refresh(prefix, 0)
                        self.assertEqual(probe.call_count, int(method != "question_weighted_kv"))
                        self.assertEqual(public_probe.call_count, int(method != "question_attention_reconstruction"))
                    loss, parts, _ = objective.compute(prefix)
                    results[method] = (loss.detach(), torch.autograd.grad(loss, prefix)[0], parts)
                weighted, attention, joint = (results[name] for name in
                                              ("question_weighted_kv", "question_attention_reconstruction", JOINT_METHOD))
                torch.testing.assert_close(joint[0], weighted[0] + attention[0])
                torch.testing.assert_close(joint[1], weighted[1] + attention[1], rtol=1e-5, atol=1e-7)
                self.assertEqual(float(joint[2]["base_loss"]), 0)
                torch.testing.assert_close(joint[2]["weighted_kv_component_loss"], weighted[0])
                torch.testing.assert_close(joint[2]["attention_reconstruction_loss"], attention[0])
                self.assertEqual("attention_distribution_loss" in joint[2], version == 2)

    def test_joint_reference_tracks_both_versions_and_rejects_private_supervision(self):
        cfg = config(methods=[JOINT_METHOD], base_loss="none", checkpoint_metric="total")
        objective = Objective(self.backend, self.observed, self.public_ids, self.question_ids, JOINT_METHOD, cfg)
        objective.refresh(self.initial, 0)
        state = objective.state_dict()
        self.assertEqual(state["attention_loss_version"], 2)
        self.assertEqual(state["weighted_kv_version"], 2)
        self.assertEqual(set(state["reference"]), {"weights", "queries", "targets"})
        self.assertTrue(all(not t.requires_grad for t in state["reference"]["queries"]))
        for key in ("attention_loss_version", "weighted_kv_version"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                objective.load_state_dict({**state, key: 1})
        with self.assertRaisesRegex(ValueError, "非 oracle"):
            Objective(self.backend, self.observed, self.public_ids, self.question_ids, JOINT_METHOD, cfg, self.oracle)
        for change in ({"lambda_weighted": 0.3}, {"lambda_output": 0.3}, {"attention_loss_version": 1},
                       {"weighted_kv_version": 1}, {"semantic_query_mode": "structured"}):
            self.assertNotEqual(fingerprint(cfg, JOINT_METHOD, "same"),
                                fingerprint({**cfg, **change}, JOINT_METHOD, "same"))

    def test_structured_queries_apply_to_joint_weighted_component(self):
        cfg = config(methods=[JOINT_METHOD], base_loss="none", checkpoint_metric="total",
                     semantic_query_mode="structured")
        text = question_text(QUESTIONS["city"][0], cfg["output_instruction"], cfg["answer_boundary"])
        ids = self.backend.encode_question(text, cfg)
        groups = self.backend.semantic_query_groups(text, cfg, ids)
        objective = Objective(self.backend, self.observed, self.public_ids, ids, JOINT_METHOD, cfg,
                              question_groups=groups)
        self.assertIsNotNone(objective.query_weights)
        objective.refresh(self.initial, 0)
        changed = {**cfg, "semantic_query_weights": {"target_relation": 0.4, "required_value": 0.3, "question": 0.3}}
        other = Objective(self.backend, self.observed, self.public_ids, ids, JOINT_METHOD, changed,
                          question_groups=groups)
        with self.assertRaisesRegex(ValueError, "语义查询"):
            other.load_state_dict(objective.state_dict())

    def test_joint_training_resume_matches_continuous_training(self):
        cfg = ablation_config(methods=[JOINT_METHOD], base_loss="none", checkpoint_metric="total")
        monitor = AnswerMonitor(self.backend, self.observed, self.question_ids, "12")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch("experiment.trainer.render"):
            full, resumed = (Path(directory) / name for name in ("full", "resumed"))
            train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                  JOINT_METHOD, cfg, full, monitor)
            compute = Objective.compute
            calls = 0

            def interrupted(objective, prefix):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise KeyboardInterrupt()
                return compute(objective, prefix)

            with patch.object(Objective, "compute", interrupted), self.assertRaises(KeyboardInterrupt):
                train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                      JOINT_METHOD, cfg, resumed, monitor)
            train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                  JOINT_METHOD, cfg, resumed, monitor, resume=True)
            left = torch.load(full / "latest.pt", weights_only=True)
            right = torch.load(resumed / "latest.pt", weights_only=True)
            torch.testing.assert_close(left["prefix"], right["prefix"], rtol=0, atol=0)
            self.assertEqual(left["history"], right["history"])

    def test_runner_and_tables_support_joint_method_and_actual_gpu_config(self):
        project = Path(__file__).resolve().parents[1]
        actual = load_config(project / "configs/joint_021_gpu3.json")
        require_current_attention_loss(actual)
        require_current_weighted_kv(actual)
        self.assertEqual(actual["cuda_devices"], "3")
        self.assertEqual(actual["methods"], [JOINT_METHOD])
        self.assertEqual(actual["base_loss"], "none")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch("experiment.trainer.render"), patch("experiment.runner.render_comparison"):
            root = Path(directory)
            data = root / "data.json"
            write_json(data, [{"task_id": "case", "private_prefix": "private", "public_text": "public",
                               "question": "question", "answer": "12", "held_out_question": "held question"}])
            cfg = ablation_config(datasets=[str(data)], methods=[JOINT_METHOD], base_loss="none",
                                  checkpoint_metric="total", rounds=1, steps_per_round=2,
                                  test_question_mode="held_out")
            self.assertEqual(run(cfg, root / "run", backend=self.backend), 0)
            self.assertEqual(set(read_json(root / "run/test_summary.json")), {JOINT_METHOD})
            saved = read_json(root / "run/samples/case" / JOINT_METHOD / "config.json")
            self.assertFalse(saved["uses_private_teacher"])
            with (root / "run/test_method_success.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["method"], JOINT_METHOD)
            self.assertEqual(rows[0]["tested_samples"], "1")
