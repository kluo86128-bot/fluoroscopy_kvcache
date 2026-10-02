from pathlib import Path
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer as RealTokenizer

from experiment.answer_forms import (ACCEPTED_DEFINITION, answer_forms, prefix_free_paths,
                                     format_compliant, target_mention)
from experiment.config import DIRECT_OUTPUT
from experiment.io import read_json, write_json
from experiment.metrics import AnswerMonitor, answer_match
from experiment.runner import run
from prepare_questions import update_questions
from test_experiment import Base, config


class ProbabilityBackend:
    """Known normalized distributions; catches double counting and wrong token IDs."""
    forms = {"Orinda": [1, 2], " Orinda": [3, 2], "March 5": [6, 7],
             "March 5th": [6, 7, 8], "5th March": [9, 10]}
    probabilities = {(): {1: .02, 3: .8, 6: .1, 9: .04}, (1,): {2: .6}, (3,): {2: .9},
                     (6,): {7: .5}, (6, 7): {8: .8}, (9,): {10: .8}}

    def encode(self, text):
        return torch.tensor([self.forms[text]], dtype=torch.long)

    def continuation_logits(self, cache, question, continuation):
        values = []
        path = continuation[0].tolist()
        for i in range(len(path)):
            selected = self.probabilities[tuple(path[:i])]
            probs = torch.zeros(12, dtype=torch.float64)
            for token, value in selected.items():
                probs[token] = value
            probs[0] = 1 - sum(selected.values())
            values.append(probs.log())
        return torch.stack(values)[None]


class UnionProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.backend = ProbabilityBackend()
        self.cache = ((torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2)),)
        self.question = torch.tensor([[0]])

    def monitor(self, answer, aliases=(), spaces=(0, 1)):
        return AnswerMonitor(self.backend, self.cache, self.question, answer, aliases,
                             probability_mode="accepted_forms", leading_spaces=spaces)(self.cache)

    def test_leading_space_probability_is_included_without_masking_logits(self):
        value = self.monitor("Orinda")
        self.assertAlmostEqual(value["canonical_answer_probability"], .02 * .6, places=12)
        self.assertAlmostEqual(value["answer_probability"], .02 * .6 + .8 * .9, places=12)
        self.assertEqual(value["probability_definition"], ACCEPTED_DEFINITION)
        self.assertEqual(value["answer_union_path_count"], 2)

    def test_alias_ancestor_covers_longer_canonical_path_without_double_counting(self):
        value = self.monitor("March 5th", ["March 5", "5th March", "March 5"], (0,))
        self.assertAlmostEqual(value["canonical_answer_probability"], .1 * .5 * .8, places=12)
        self.assertAlmostEqual(value["answer_probability"], .1 * .5 + .04 * .8, places=12)
        self.assertEqual(value["answer_union_path_count"], 2)
        self.assertFalse(value["answer_form_probabilities"][0]["included_in_union"])

    def test_paths_deduplicated_and_prefix_union_bounded_by_one(self):
        self.assertEqual(prefix_free_paths([[1], [1, 2], [1], [3, 2]]), [(1,), (3, 2)])
        self.assertEqual(answer_forms("Orinda", [" Orinda", "Orinda"]), ["Orinda", " Orinda"])

    def test_qwen_space_and_date_paths_from_local_tokenizer(self):
        path = Path(__file__).resolve().parents[2] / "develop_experiment/tokenizer/Qwen3-4B/tokenizer.json"
        tokenizer = RealTokenizer.from_file(str(path))
        self.assertEqual(tokenizer.encode("Orinda", add_special_tokens=False).ids, [2195, 17416])
        self.assertEqual(tokenizer.encode(" Orinda", add_special_tokens=False).ids, [2521, 17416])
        paths = [tokenizer.encode(t, add_special_tokens=False).ids
                 for t in answer_forms("March 5th", ["March 5", "5 March", "5th March"])]
        self.assertEqual(len(prefix_free_paths(paths)), 6)


class FormatAndQuestionTests(unittest.TestCase):
    def test_fact_match_does_not_imply_month_first_format(self):
        value = answer_match("5th March", "March 5th", ["5th March"], "month_ordinal")
        self.assertTrue(value["full_match"])
        self.assertFalse(value["format_compliant"])
        self.assertTrue(format_compliant("March 5th", "month_ordinal"))
        self.assertFalse(format_compliant("March 11st", "month_ordinal"))

    def test_numbered_city_has_target_but_is_not_format_compliant(self):
        value = answer_match("2. Orinda", "Orinda", [], "english_city")
        self.assertTrue(value["target_mentioned"])
        self.assertFalse(value["format_compliant"])
        self.assertEqual(value["recovery_review_status"], "pending_semantic_review")
        self.assertFalse(format_compliant("丹ville", "english_city"))
        self.assertTrue(format_compliant(" Orinda", "english_city"))
        self.assertFalse(format_compliant("  Orinda", "english_city"))
        self.assertTrue(target_mention("not Orinda", "Orinda", [])["target_mentioned"])
        self.assertFalse(target_mention("Orindaville", "Orinda", [])["target_mentioned"])

    def test_in_place_question_update_preserves_paths_private_data_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old/tasks.jsonl"
            source.parent.mkdir()
            row = {"task_id": "one", "slot": "city", "source_file": "dev/dialogues_001.json", "teacher_only": {"prefix_a_file": "missing_private.txt"},
                   "receiver_inputs": {"question_file": "missing_question.txt", "public_chunk_file": "missing_public.txt"},
                   "evaluation_only_file": "missing_labels.json", "held_out_question_file": "missing_held.txt"}
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (source.parent / "missing_question.txt").write_text("old training", encoding="utf-8")
            (source.parent / "missing_held.txt").write_text("old held", encoding="utf-8")
            protected = {name: b"unchanged" for name in ("missing_private.txt", "missing_public.txt", "missing_labels.json")}
            for name, content in protected.items():
                (source.parent / name).write_bytes(content)
            self.assertEqual(update_questions(source), 1)
            generated = json.loads(source.read_text(encoding="utf-8"))
            self.assertEqual(generated["teacher_only"], row["teacher_only"])
            self.assertEqual(generated["receiver_inputs"], row["receiver_inputs"])
            self.assertEqual(generated["evaluation_only_file"], row["evaluation_only_file"])
            self.assertEqual(generated["answer_format"], "english_city")
            self.assertEqual(generated["source_file"], row["source_file"])
            text = (source.parent / generated["receiver_inputs"]["question_file"]).read_text(encoding="utf-8")
            self.assertIn("final confirmed", text)
            self.assertNotIn("Danville", text)
            for name, content in protected.items():
                self.assertEqual((source.parent / name).read_bytes(), content)
            before = {path: path.read_bytes() for path in source.parent.iterdir()}
            self.assertEqual(update_questions(source), 1)
            self.assertEqual(before, {path: path.read_bytes() for path in source.parent.iterdir()})


class AcceptedFormsIntegrationTests(Base):
    def test_new_answer_and_alias_diagnostics_do_not_change_training_or_topk(self):
        from experiment.trainer import train
        cfg = config(rounds=1, steps_per_round=2, score_probability="accepted_forms")
        states = []
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            for name, answer, aliases in (("a", "12", ["twelve"]), ("b", "99", ["ninety nine"])):
                root = Path(directory) / name
                monitor = AnswerMonitor(self.backend, self.observed, self.question_ids, answer, aliases,
                                        probability_mode="accepted_forms")
                train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                      "question_attention_reconstruction", cfg, root, monitor)
                states.append(torch.load(root / "latest.pt", weights_only=True))
        torch.testing.assert_close(states[0]["prefix"], states[1]["prefix"], atol=0, rtol=0)
        self.assertEqual([(r["step"], r["loss"]) for r in states[0]["topk"]],
                         [(r["step"], r["loss"]) for r in states[1]["topk"]])

    def test_all_active_methods_train_and_score_with_same_probability_definition(self):
        from experiment.config import METHODS
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = root / "data.json"
            write_json(data, [{"task_id": "case", "private_prefix": "private", "public_text": "public",
                               "question": "How many?", "held_out_question": "Party size?", "answer": "12",
                               "accepted_aliases": ["twelve"], "answer_format": "digits"}])
            cfg = config(datasets=[str(data)], methods=list(METHODS), rounds=1, steps_per_round=2,
                         score_probability="accepted_forms", output_instruction=DIRECT_OUTPUT)
            self.assertEqual(run(cfg, root / "run", backend=self.backend), 0)
            for method in METHODS:
                folder = root / "run/samples/case" / method
                history = [json.loads(line) for line in (folder / "history.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertTrue(all(row["probability_definition"] == ACCEPTED_DEFINITION for row in history))
                self.assertTrue(all(0 <= row["answer_probability"] <= 1 for row in history))
                last = history[-1]
                self.assertGreaterEqual(last["answer_probability"], last["canonical_answer_probability"] - 1e-12)
                latest = torch.load(folder / "latest.pt", weights_only=True)
                self.assertEqual(latest["score"]["observations"][-1][1], last["answer_probability"])
                self.assertEqual(read_json(folder / "result.json")["score_probability"], "accepted_forms")


if __name__ == "__main__":
    unittest.main()
