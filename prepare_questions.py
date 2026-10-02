"""Update existing question files in place; never reads private text or labels."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUESTIONS = {
    "party_size": (
        "How many people were included in the final confirmed restaurant reservation? Return only the number in digits.",
        "What was the party size of the final confirmed restaurant booking? Return only the number in digits.",
        "digits"),
    "city": (
        "In which city is the restaurant in the final confirmed reservation located? Return only the English city name, spelled exactly as in the conversation.",
        "What is the location city of the restaurant in the final confirmed booking? Return only the English city name, spelled exactly as in the conversation.",
        "english_city"),
    "date": (
        "What is the dining date of the final confirmed restaurant reservation? Return only the full English month name, one space, and the day number with its ordinal suffix. Do not include a year.",
        "On which date will the meal in the final confirmed restaurant booking take place? Return only the full English month name, one space, and the day number with its ordinal suffix. Do not include a year.",
        "month_ordinal"),
}


def update_questions(manifest=None):
    manifest = Path(manifest or ROOT / "test_samples" / "tasks.jsonl").resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows or any(row.get("slot") not in QUESTIONS for row in rows):
        raise ValueError("源样本包含未定义的问题字段")
    protected = {manifest}
    for row in rows:
        for key in ("evaluation_only_file", "held_out_evaluation_only_file", "source_snapshot_file"):
            if row.get(key):
                protected.add((manifest.parent / row[key]).resolve())
        for mapping in (row.get("teacher_only", {}), row.get("receiver_inputs", {})):
            for key, value in mapping.items():
                if key.endswith("_file") and key != "question_file":
                    protected.add((manifest.parent / value).resolve())
    updates = {}
    # Validate every destination before writing. Repeated questions share templates.
    for row in rows:
        training, held, answer_format = QUESTIONS[row["slot"]]
        for relative, text in ((row["receiver_inputs"]["question_file"], training),
                               (row["held_out_question_file"], held)):
            path = (manifest.parent / relative).resolve()
            if path in protected or not path.is_relative_to(manifest.parent) or path.suffix != ".txt" or not path.is_file():
                raise ValueError(f"问题文件路径无效: {path}")
            if path in updates and updates[path] != text + "\n":
                raise ValueError(f"同一问题文件被不同字段/问法共用: {path}")
            updates[path] = text + "\n"
        row["answer_format"] = answer_format
        row["question_version"] = "unique_field_english_v2"
    for path, text in updates.items():
        if path.read_text(encoding="utf-8-sig") != text:
            path.write_text(text, encoding="utf-8")
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if manifest.read_text(encoding="utf-8-sig") != text:
        manifest.write_text(text, encoding="utf-8")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量原地更新样本的问题，不读取或改写私有文本和评估答案")
    parser.add_argument("--manifest", type=Path, default=ROOT / "test_samples" / "tasks.jsonl")
    args = parser.parse_args()
    print(f"Updated {update_questions(args.manifest)} samples in place.")
