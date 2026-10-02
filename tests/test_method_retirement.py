"""Retired methods cannot re-enter training; historical reads remain supported."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiment.cli import main, parser
from experiment.config import DEFAULTS, METHODS, load_config, validate
from experiment.io import write_json


class RetirementTests(unittest.TestCase):
    def historical_config(self):
        return {**deepcopy(DEFAULTS), "methods": ["question_output_consistency"],
                "method_options": {"question_output_consistency": {"lambda_kl": 0.2}}}

    def test_new_training_rejects_retired_method(self):
        with self.assertRaisesRegex(ValueError, "已退出实验组"):
            validate(self.historical_config())

    def test_cli_training_choices_exclude_retired_method(self):
        action = next(a for a in parser()._actions if a.dest == "methods")
        self.assertNotIn("question_output_consistency", action.choices)
        self.assertIn("oracle_private_prefix_distillation", action.choices)

    def test_all_shipped_configs_use_active_methods(self):
        for path in (Path(__file__).resolve().parents[1] / "configs").glob("*.json"):
            with self.subTest(config=path.name):
                config = load_config(path)
                self.assertTrue(set(config["methods"]) <= set(METHODS))

    def test_historical_read_validates_options(self):
        validate(self.historical_config(), allow_retired=True)
        config = self.historical_config()
        config["method_options"]["question_output_consistency"]["lambda_kl"] = -1
        with self.assertRaises(ValueError):
            validate(config, allow_retired=True)

    def test_stale_retired_options_rejected_for_new_training(self):
        config = self.historical_config()
        config["methods"] = ["baseline"]
        with self.assertRaises(ValueError):
            validate(config)

    def test_cli_historical_test_allowed_but_resume_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "resolved_config.json", self.historical_config())
            with patch("experiment.cli.catalog", return_value=[]), \
                 patch("experiment.cli.execute", return_value=0) as execute:
                self.assertEqual(main(["test", "--run-dir", directory, "--foreground"]), 0)
                self.assertTrue(execute.call_args.args[3])
                execute.reset_mock()
                with self.assertRaisesRegex(ValueError, "已退出实验组"):
                    main(["--resume", directory, "--foreground"])
                execute.assert_not_called()

    def test_cli_historical_diagnose_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "resolved_config.json", self.historical_config())
            with patch("experiment.cli.catalog", return_value=[]), \
                 patch("experiment.cli.execute", return_value=0) as execute:
                self.assertEqual(main(["diagnose", "--run-dir", directory, "--foreground"]), 0)
                self.assertEqual(execute.call_args.kwargs["source"], root.resolve())

    def test_background_worker_preserves_retirement_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "resolved_config.json", self.historical_config())
            for test_only, source in [(True, None), (False, directory), (False, None)]:
                with self.subTest(test_only=test_only, source=source):
                    write_json(root / "worker.json", {"test_only": test_only,
                               "source": source, "resume": False})
                    with patch("experiment.cli.execute", return_value=0) as execute:
                        if test_only or source:
                            self.assertEqual(main(["--worker", directory]), 0)
                        else:
                            with self.assertRaisesRegex(ValueError, "已退出实验组"):
                                main(["--worker", directory])
                            execute.assert_not_called()

    def test_legacy_background_test_loads_missing_new_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.historical_config()
            config.pop("experiment_version")
            config.pop("answer_leading_spaces")
            config["score_probability"] = "sequence"
            write_json(root / "resolved_config.json", config)
            write_json(root / "worker.json", {"test_only": True, "resume": False})
            with patch("experiment.cli.execute", return_value=0) as execute:
                self.assertEqual(main(["--worker", directory]), 0)
                self.assertEqual(execute.call_args.args[0]["experiment_version"], 1)
                self.assertEqual(execute.call_args.args[0]["score_probability"], "sequence")

    def test_legacy_active_run_cannot_resume_with_new_probability_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.historical_config()
            config.pop("experiment_version")
            config["methods"] = ["baseline"]
            config["method_options"] = {}
            write_json(root / "resolved_config.json", config)
            with self.assertRaisesRegex(ValueError, "新版问题/概率口径"):
                main(["--resume", directory, "--foreground"])


if __name__ == "__main__":
    unittest.main()
