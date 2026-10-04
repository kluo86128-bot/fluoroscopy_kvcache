"""A city-only update must preserve other fields and their version metadata."""
import json
from pathlib import Path
import tempfile
import unittest

from prepare_questions import QUESTION_VERSION, update_questions


class ScopeTests(unittest.TestCase):
    def test_default_update_preserves_non_city_questions_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for slot in ("city", "party_size", "date"):
                folder = root / slot
                folder.mkdir()
                for name in ("search.txt", "test.txt"):
                    (folder / name).write_text("Original " + slot + "\n", encoding="utf-8")
                rows.append({"task_id": slot, "slot": slot, "question_version": "structured_field_semantic_v3",
                             "receiver_inputs": {"question_file": slot + "/search.txt"},
                             "held_out_question_file": slot + "/test.txt"})
            manifest = root / "tasks.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            preserved = {p: p.read_bytes() for slot in ("party_size", "date") for p in (root / slot).iterdir()}
            self.assertEqual(update_questions(manifest), 1)
            updated = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(updated[1:], rows[1:])
            self.assertEqual(updated[0]["question_version"], QUESTION_VERSION)
            self.assertTrue(all(p.read_bytes() == content for p, content in preserved.items()))
            self.assertIn("Target relation:", (root / "city/search.txt").read_text(encoding="utf-8"))
            held = (root / "city/test.txt").read_text(encoding="utf-8")
            self.assertTrue(held.startswith("Output format:"))
            self.assertIn("\n\nQuestion:", held)
            for label in ("Tracking rule:", "Target relation:", "Required value:"):
                self.assertNotIn(label, held)
            snapshot = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(update_questions(manifest), 1)
            self.assertTrue(all(p.read_bytes() == content for p, content in snapshot.items()))


if __name__ == "__main__":
    unittest.main()
