"""Launch configurations evaluate only their designated held-out question."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from experiment.config import METHODS, load_config, saved_config, method_config
from experiment.io import read_json, write_json
from experiment.question_semantics import RELATION_WEIGHTS, SEMANTIC_WEIGHTS
from experiment.runner import run
from prepare_questions import QUESTIONS
from test_experiment import Base, config


class LaunchTests(Base):
    def test_three_base_groups_train_and_only_test_simplified_question(self):
        training, held, answer_format = QUESTIONS["city"]
        for mode in ("token", "mean", "none"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                root = Path(directory)
                data = root / "tasks.json"
                write_json(data, [{"task_id": "case", "private_prefix": "private", "public_text": "public",
                                   "question": training, "held_out_question": held,
                                   "answer": "City", "answer_format": answer_format}])
                methods = [method for method in METHODS if mode != "none" or method != "baseline"]
                cfg = config(base_loss=mode, methods=methods, datasets=[str(data)],
                             checkpoint_metric="total", test_question_mode="held_out",
                             semantic_query_mode="structured", semantic_query_weights=RELATION_WEIGHTS.copy(),
                             rounds=1, steps_per_round=1, save_topk=1)
                output = root / "run"
                self.assertEqual(run(cfg, output, backend=self.backend), 0)
                summaries = read_json(output / "test_summary.json")
                self.assertEqual(set(summaries), set(methods))
                for method in methods:
                    self.assertEqual(set(summaries[method]), {"held_out_question"})
                    self.assertEqual(summaries[method]["held_out_question"]["prefix_tests"], 1)
                    rows = read_json(output / "samples/case" / method / "test_results.json")
                    self.assertEqual([row["question_kind"] for row in rows], ["held_out_question"])
                    history = [json.loads(line) for line in
                               (output / "samples/case" / method / "history.jsonl").read_text(encoding="utf-8").splitlines()]
                    self.assertIn("answer_probability", history[-1])
                    if mode == "none":
                        self.assertEqual(history[-1]["base_loss"], 0)

    def test_missing_test_question_fails_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "tasks.json"
            write_json(data, [{"task_id": "case", "private_prefix": "private", "public_text": "public",
                               "question": "Question", "answer": "City"}])
            cfg = config(datasets=[str(data)], test_question_mode="held_out")
            with patch("experiment.runner.load_backend") as load:
                with self.assertRaisesRegex(ValueError, "question_test"):
                    run(cfg, root / "run")
                load.assert_not_called()

    def test_launch_configs_use_distinct_requested_gpus_and_results(self):
        root = Path(__file__).resolve().parents[1]
        outputs = set()
        for filename, task, gpu in (("gpu0.json", "021", "0"), ("gpu2.json", "033", "2"), ("gpu3.json", "040", "3")):
            cfg = load_config(root / "configs" / filename)
            self.assertEqual((cfg["base_loss"], cfg["cuda_devices"], cfg["device"]), ("token", gpu, "cuda:0"))
            manual = task == "040"
            self.assertEqual(cfg["semantic_query_weights"], SEMANTIC_WEIGHTS if manual else RELATION_WEIGHTS)
            self.assertEqual(cfg["loss_profile"], "current" if manual else "semantic_enhanced_v1")
            self.assertEqual((cfg["weighted_kv_version"], cfg["attention_loss_version"]), (2 if manual else 1, 1))
            self.assertEqual(cfg["semantic_query_mode"], "structured" if manual else "uniform")
            self.assertEqual(cfg["lambda_attention"], 0)
            self.assertEqual(cfg["test_question_mode"], "held_out")
            self.assertEqual(cfg["include_tasks"], ["sgd_test_" + task])
            self.assertEqual(cfg["limit"], 1)
            self.assertEqual(cfg["methods"], ["baseline", "question_weighted_kv"])
            self.assertEqual(method_config(cfg, "baseline")["lambda_base"], 1)
            self.assertEqual(method_config(cfg, "question_weighted_kv")["lambda_base"], 0.1 if task == "040" else 1)
            self.assertIn("question_rollback_" + task + "_token", cfg["output_dir"])
            outputs.add(cfg["output_dir"])
        self.assertEqual(len(outputs), 3)
        self.assertEqual(saved_config({})["test_question_mode"], "both")
